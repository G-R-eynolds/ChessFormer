import torch
import torch.nn as nn
import math

class ChessRelativeAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.relative_embedding_height = nn.Embedding(15, self.head_dim) # -7 to +7
        self.relative_embedding_width = nn.Embedding(15, self.head_dim)  # -7 to +7

    def forward(self, embedded_sequence: torch.Tensor):
        """
        Args:
            embedded_sequence: A tensor of shape (Batch, 64, d_model)
                               from the StateEmbedding module.
        Returns:
            output: (Batch, 64, d_model)
        """
        batch_size, seq_len, _ = embedded_sequence.shape # seq_len is 64
        assert seq_len == 64, "Input must be 8x8=64 tokens"
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"

        # 1. Project to Q, K, V and split into heads
        q = self.q_proj(embedded_sequence).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, S, D)
        k = self.k_proj(embedded_sequence).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(embedded_sequence).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 2. Content-based attention scores: (B, H, 64, 64)
        content_score = torch.matmul(q, k.transpose(-2, -1))

        # 3. APPLY POSITIONAL ENCODINGS AS A BIAS
        # Create a 64x64 matrix of relative positions (dx, dy)
        positions = torch.arange(64, device=q.device).view(8, 8)
        rel_coords = positions.view(1, -1) - positions.view(-1, 1) # (64, 64)
        rel_ranks = (rel_coords // 8).clamp(-7, 7) + 7  # (64, 64)
        rel_files = (rel_coords % 8).clamp(-7, 7) + 7   # (64, 64)
        h_emb = self.relative_embedding_height(rel_ranks) # (64, 64, head_dim)
        w_emb = self.relative_embedding_width(rel_files)  # (64, 64, head_dim)
        positional_embedding = h_emb + w_emb              # (64, 64, head_dim)

        # Calculate positional score: query interacts with the positional embeddings
        # q: (B, H, S, D), pos_emb: (S, S, D)
        # We'll expand q to (B, H, S, 1, D) and pos_emb to (1, 1, S, S, D)
        q_exp = q.unsqueeze(3)  # (B, H, S, 1, D)
        pos_emb_exp = positional_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S, D)
        # Elementwise multiply and sum over D
        positional_score = (q_exp * pos_emb_exp).sum(-1)  # (B, H, S, S)

        # 4. Add the positional bias to the content score
        total_score = (content_score + positional_score) / math.sqrt(self.head_dim)

        # 5. Standard attention
        attention_weights = torch.softmax(total_score, dim=-1)  # (B, H, S, S)
        output = torch.matmul(attention_weights, v)  # (B, H, S, D)
        # Merge heads: (B, S, H, D) -> (B, S, H*D)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)  # (B, 64, d_model)
        return output

# --- Example Usage ---
if __name__ == "__main__":
    D_MODEL = 512
    N_HEAD = 8
    attn = ChessRelativeAttention(d_model=D_MODEL, nhead=N_HEAD)
    # Dummy input: (Batch, 64, d_model)
    x = torch.randn(2, 64, D_MODEL)
    y = attn(x)
    print(f"Output shape: {y.shape}")  # (2, 64, 512) 