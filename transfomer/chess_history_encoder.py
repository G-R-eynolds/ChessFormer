import chess
import torch
import numpy as np
from typing import List
from chess_state_encoder import encode_chess_state

def encode_history_tensor(
    boards: List[chess.Board],
    repetitions: List[int],
    max_history: int = 8
) -> torch.Tensor:
    """
    Encode the current and past 7 board states into a 119x8x8 tensor for neural networks.
    Args:
        boards: List of python-chess Board objects, most recent first (len <= 8)
        repetitions: List of repetition counts for each board (most recent first)
        max_history: Number of half-moves to encode (default 8)
    Returns:
        torch.Tensor: (119, 8, 8) tensor
    """
    # If not enough history, pad with the oldest board
    if len(boards) < max_history:
        pad_count = max_history - len(boards)
        boards = [boards[0]] * pad_count + boards
        repetitions = [repetitions[0]] * pad_count + repetitions
    else:
        boards = boards[-max_history:]
        repetitions = repetitions[-max_history:]

    # 112 planes for piece history and repetition
    planes = np.zeros((112, 8, 8), dtype=np.float32)

    # For each time step (t=0 is most recent)
    for t, board in enumerate(boards):
        enc = encode_chess_state(board)  # (18, 8, 8)
        # For the player to move: planes 0-5, for opponent: 6-11
        # Determine player color at this time step
        player_color = board.turn
        opp_color = not player_color
        # For each piece type (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        for i, piece_type in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
            # Player planes
            player_plane = enc[i*2] if player_color == chess.WHITE else enc[i*2+1]
            planes[i + 6*t] = player_plane
            # Opponent planes
            opp_plane = enc[i*2+1] if player_color == chess.WHITE else enc[i*2]
            planes[48 + i + 6*t] = opp_plane
        # Repetition planes (first and second repetition)
        rep_count = repetitions[t]
        if rep_count >= 2:
            planes[96 + 2*t + 1, :, :] = 1.0  # Second repetition
        if rep_count >= 1:
            planes[96 + 2*t, :, :] = 1.0      # First repetition

    # 7 context planes
    context_planes = np.zeros((7, 8, 8), dtype=np.float32)
    # Use the most recent board for context
    board = boards[-1]
    # Castling rights
    context_planes[0, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    context_planes[1, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    context_planes[2, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    context_planes[3, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # Side to move (all 1s if Black, 0s if White)
    context_planes[4, :, :] = 1.0 if board.turn == chess.BLACK else 0.0
    # 50-move rule counter (normalized)
    context_planes[5, :, :] = board.halfmove_clock / 100.0
    # Total move count (normalized by 100)
    context_planes[6, :, :] = board.fullmove_number / 100.0

    # Stack all planes
    tensor = np.concatenate([planes, context_planes], axis=0)  # (119, 8, 8)
    return torch.from_numpy(tensor).float()

def get_repetition_counts(history: List[chess.Board]) -> List[int]:
    """
    For each board in history, count how many times its position has occurred (for repetition planes).
    Args:
        history: List of python-chess Board objects, most recent first
    Returns:
        List of repetition counts (most recent first)
    """
    fen_counts = {}
    counts = []
    for board in reversed(history):
        fen = board.board_fen() + ' ' + ('w' if board.turn else 'b')
        fen_counts[fen] = fen_counts.get(fen, 0) + 1
        counts.append(fen_counts[fen])
    return list(reversed(counts))

# Example usage:
if __name__ == "__main__":
    import random
    boards = []
    board = chess.Board()
    boards.append(board.copy())
    for _ in range(7):
        moves = list(board.legal_moves)
        move = random.choice(moves)
        board.push(move)
        boards.append(board.copy())
    reps = get_repetition_counts(boards)
    tensor = encode_history_tensor(boards, reps)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor[0] nonzero: {torch.count_nonzero(tensor[0])}")
    print(f"Tensor[112:] context planes:\n{tensor[112:,0,0]}") 