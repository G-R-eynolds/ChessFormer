import torch
import torch.nn as nn
import math

class StateEmbedding(nn.Module):
    """
    This module takes the multi-plane tensor and embeds it into the transformer's native dimension (d_model),
    including 2D positional encodings for each square.
    """
    def __init__(self, input_channels: int, d_model: int):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.embedding_layer = nn.Linear(input_channels, d_model)
        # Register 2D positional encodings as a buffer (not a parameter)
        self.register_buffer('positional_encoding', self._build_2d_positional_encoding(8, 8, d_model))

    def _build_2d_positional_encoding(self, height, width, d_model):
        """
        Create a (8, 8, d_model) tensor of 2D sine/cosine positional encodings.
        """
        pe = torch.zeros(height, width, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        for y in range(height):
            for x in range(width):
                pos = y * width + x
                pe[y, x, 0::2] = torch.sin(pos * div_term)
                pe[y, x, 1::2] = torch.cos(pos * div_term)
        return pe  # shape: (8, 8, d_model)

    def forward(self, board_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of board tensors.
        Args:
            board_tensor: A tensor of shape (Batch, Channels, 8, 8)
        Returns:
            A tensor of shape (Batch, 64, d_model), which is a sequence of
            64 embedded square tokens with positional encodings added.
        """
        batch_size = board_tensor.shape[0]
        # 1. Permute the tensor to group by squares: (Batch, Channels, 8, 8) -> (Batch, 8, 8, Channels)
        tensor_permuted = board_tensor.permute(0, 2, 3, 1)
        # 2. Flatten the spatial dimensions: (Batch, 8, 8, Channels) -> (Batch, 64, Channels)
        sequence_of_squares = tensor_permuted.reshape(batch_size, 64, self.input_channels)
        # 3. Apply the embedding layer: (Batch, 64, Channels) -> (Batch, 64, d_model)
        embedded_sequence = self.embedding_layer(sequence_of_squares)
        # 4. Add 2D positional encoding: (1, 64, d_model) broadcasted to (Batch, 64, d_model)
        pe = self.positional_encoding.view(1, 64, self.d_model)
        embedded_sequence = embedded_sequence + pe
        return embedded_sequence

# --- Example Usage ---
if __name__ == "__main__":
    # Example: 119 input channels, d_model=512
    TOTAL_INPUT_CHANNELS = 119
    D_MODEL = 512
    state_embedder = StateEmbedding(input_channels=TOTAL_INPUT_CHANNELS, d_model=D_MODEL)
    # Dummy batch of board tensors: (Batch, Channels, 8, 8)
    board_tensor = torch.randn(2, TOTAL_INPUT_CHANNELS, 8, 8)
    state_embedding_sequence = state_embedder(board_tensor)
    print(f"Output shape: {state_embedding_sequence.shape}")  # (2, 64, 512) 