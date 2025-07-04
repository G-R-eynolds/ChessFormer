import os
import torch
import chess
from encoder_only_chess_transformer import EncoderOnlyChessTransformer
from chess_history_encoder import encode_history_tensor, get_repetition_counts

# --- Load model and action vocab ---
def load_encoder_only_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    model = EncoderOnlyChessTransformer(
        input_channels=config['input_channels'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        action_size=config['action_size'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    uci_to_idx = checkpoint['uci_to_idx']
    idx_to_uci = checkpoint['idx_to_uci']
    if isinstance(idx_to_uci, list):
        idx_to_uci = {i: u for i, u in enumerate(idx_to_uci)}
    return model, uci_to_idx, idx_to_uci

# --- Inference function ---
def infer_next_move_encoder_only(model, uci_to_idx, idx_to_uci, fen, move_history_uci, max_history=8):
    board = chess.Board(fen) if fen else chess.Board()
    boards = [board.copy()]
    for move_uci in move_history_uci:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            boards.append(board.copy())
        else:
            raise ValueError(f"Illegal move in history: {move_uci}")
    history = boards[-max_history:][::-1]
    reps = get_repetition_counts(history)
    state_tensor = encode_history_tensor(history, reps, max_history=max_history)  # (119, 8, 8)
    board_tensor = state_tensor.unsqueeze(0)  # (1, 119, 8, 8)
    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)
        # policy_logits: (1, 64, action_size)
        # For each legal move, get the source square and move idx
        best_move = None
        best_prob = -float('inf')
        best_value = value_pred.item()
        for move in board.legal_moves:
            uci = move.uci()
            if uci not in uci_to_idx:
                continue
            move_idx = uci_to_idx[uci]
            source_square = chess.SQUARE_NAMES.index(uci[:2])
            logits = policy_logits[0, source_square, :]
            prob = torch.softmax(logits, dim=0)[move_idx].item()
            if prob > best_prob:
                best_prob = prob
                best_move = uci
        return best_move, best_prob, best_value

# --- CLI ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Encoder-Only Chess Transformer Inference")
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), 'encoder_only_chess_transformer.pt'), help='Path to saved model')
    parser.add_argument('--fen', type=str, required=True, help='FEN string of the current board state')
    parser.add_argument('--moves', type=str, nargs='*', default=[], help='List of previous moves in UCI format')
    args = parser.parse_args()

    model, uci_to_idx, idx_to_uci = load_encoder_only_model(args.model)
    best_move, prob, value = infer_next_move_encoder_only(model, uci_to_idx, idx_to_uci, args.fen, args.moves)
    print(f"Best next move: {best_move} (probability {prob:.4f}), value: {value:.4f}") 