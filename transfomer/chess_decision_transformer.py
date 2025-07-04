import torch
import torch.nn as nn
from state_embedding import StateEmbedding
from chess_relative_attention import ChessRelativeAttention

class StateTopologyBlock(nn.Module):
    """Applies ChessRelativeAttention + norm + residual to a (Batch, 64, d_model) state."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = ChessRelativeAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: (Batch, 64, d_model)
        return self.norm(x + self.attn(x))

class TransformerBlock(nn.Module):
    """A single block of the transformer, using standard attention."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class ChessDecisionTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, max_moves, action_vocab_size):
        super().__init__()
        self.d_model = d_model
        self.state_embedder = StateEmbedding(input_channels=119, d_model=d_model)
        self.state_topology = StateTopologyBlock(d_model, nhead)
        self.rtg_embedder = nn.Linear(1, d_model)
        self.action_embedder = nn.Embedding(action_vocab_size, d_model)
        self.global_positional_embedding = nn.Embedding(max_moves * (64 + 2), d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.action_predictor = nn.Linear(d_model, action_vocab_size)

    def create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device
        # 1. --- EMBED THE INPUT SEQUENCE ---
        rtg_embeddings = self.rtg_embedder(returns_to_go) # (B, S, D)
        action_embeddings = self.action_embedder(actions) # (B, S, D)
        states_reshaped = states.view(-1, 119, 8, 8)
        state_embeddings_raw = self.state_embedder(states_reshaped) # (B*S, 64, D)
        # Apply topology block to each state
        state_embeddings_topo = self.state_topology(state_embeddings_raw) # (B*S, 64, D)
        state_embeddings = state_embeddings_topo.view(batch_size, seq_len, 64, self.d_model) # (B, S, 64, D)
        # 2. --- ARRANGE TOKENS INTO A SINGLE SEQUENCE ---
        tokens = []
        for t in range(seq_len):
            rtg_t = rtg_embeddings[:, t:t+1, :]         # (B, 1, D)
            state_t = state_embeddings[:, t, :, :]      # (B, 64, D)
            action_t = action_embeddings[:, t:t+1, :]   # (B, 1, D)
            tokens.append(torch.cat([rtg_t, state_t, action_t], dim=1))
        token_sequence = torch.cat(tokens, dim=1)  # (B, S*66, D)
        total_len = seq_len * 66
        global_positions = torch.arange(0, total_len, device=device)
        token_sequence = token_sequence + self.global_positional_embedding(global_positions)[None, :, :]
        # 3. --- PROCESS THROUGH THE TRANSFORMER ---
        causal_mask = self.create_causal_mask(total_len, device=device)
        processed_sequence = token_sequence
        for block in self.transformer_blocks:
            processed_sequence = block(processed_sequence, mask=causal_mask)
        # 4. --- PREDICT THE NEXT ACTION ---
        action_indices = torch.arange(65, total_len, 66, device=device)
        action_outputs = processed_sequence[:, action_indices, :] # (B, S, D)
        action_logits = self.action_predictor(action_outputs) # (B, S, vocab_size)
        return action_logits

# --- Example Usage ---
if __name__ == "__main__":
    D_MODEL = 512
    N_HEAD = 8
    NUM_LAYERS = 4
    MAX_MOVES = 60
    ACTION_VOCAB_SIZE = 4672  # Example: all legal UCI moves
    model = ChessDecisionTransformer(D_MODEL, N_HEAD, NUM_LAYERS, MAX_MOVES, ACTION_VOCAB_SIZE)
    batch = 2
    seq_len = 5
    states = torch.randn(batch, seq_len, 119, 8, 8)
    actions = torch.randint(0, ACTION_VOCAB_SIZE, (batch, seq_len))
    returns_to_go = torch.randn(batch, seq_len, 1)
    timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch, 1)
    logits = model(states, actions, returns_to_go, timesteps)
    print(f"Logits shape: {logits.shape}")  # (2, 5, 4672) 