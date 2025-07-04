import torch
import torch.nn as nn
import torch.nn.functional as F
from chess_relative_attention import ChessRelativeAttention

class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class StateTopologyBlock(nn.Module):
    """Chess-specific topology block using ChessRelativeAttention (Leela Zero 'smolgen' style)."""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = ChessRelativeAttention(d_model, nhead)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: (batch, 64, d_model)
        return self.norm(x + self.attn(x))

class BoardEmbeddingWithTopology(nn.Module):
    """Embeds a (batch, 119, 8, 8) board tensor to (batch, 64, d_model) with topology block."""
    def __init__(self, input_channels=119, d_model=256, nhead=8):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.embedding = nn.Linear(input_channels, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 64, d_model))  # learnable 2D positional encoding
        self.topology = StateTopologyBlock(d_model, nhead)
    def forward(self, board_tensor):
        # board_tensor: (batch, 119, 8, 8)
        batch = board_tensor.shape[0]
        x = board_tensor.view(batch, self.input_channels, 64).transpose(1, 2)  # (batch, 64, 119)
        x = self.embedding(x)  # (batch, 64, d_model)
        x = x + self.positional  # add positional encoding
        x = self.topology(x)     # apply chess-specific topology block
        return x  # (batch, 64, d_model)

class EncoderBlock(nn.Module):
    """Transformer encoder block with Mish nonlinearity."""
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            Mish(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class ChessEncoderTransformer(nn.Module):
    """
    Encoder-only transformer for chess board analysis, inspired by Leela Chess Zero's 'smolgen'.
    Input: (batch, 119, 8, 8) board tensor
    Output: (batch, 64, d_model) encoded board representation
    """
    def __init__(self, input_channels=119, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = BoardEmbeddingWithTopology(input_channels, d_model, nhead)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, board_tensor):
        x = self.embedding(board_tensor)  # (batch, 64, d_model)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        return x  # (batch, 64, d_model)

class PolicyHead(nn.Module):
    """Policy head: predicts move logits for each square."""
    def __init__(self, d_model, action_size):
        super().__init__()
        self.fc = nn.Linear(d_model, action_size)
    def forward(self, x):
        # x: (batch, 64, d_model)
        logits = self.fc(x)  # (batch, 64, action_size)
        return logits

class ValueHead(nn.Module):
    """Value head: predicts a scalar evaluation for the position."""
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(64 * d_model, d_model),
            Mish(),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):
        # x: (batch, 64, d_model)
        x = x.view(x.size(0), -1)
        value = self.fc(x)  # (batch, 1)
        return value

class EncoderOnlyChessTransformer(nn.Module):
    """
    Full encoder-only chess transformer with policy and value heads.
    Input: (batch, 119, 8, 8) board tensor (from encode_history_tensor)
    Output: policy logits (batch, 64, action_size), value (batch, 1)
    """
    def __init__(self, input_channels=119, d_model=256, nhead=8, num_layers=6, action_size=4672, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder = ChessEncoderTransformer(input_channels, d_model, nhead, num_layers, dim_feedforward, dropout)
        self.policy_head = PolicyHead(d_model, action_size)
        self.value_head = ValueHead(d_model)
    def forward(self, board_tensor):
        x = self.encoder(board_tensor)  # (batch, 64, d_model)
        policy_logits = self.policy_head(x)  # (batch, 64, action_size)
        value = self.value_head(x)           # (batch, 1)
        return policy_logits, value 