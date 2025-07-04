import os
import torch
import chess
from chess_decision_transformer import ChessDecisionTransformer
from chess_history_encoder import encode_history_tensor, get_repetition_counts

# --- Load model and action vocab ---
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    model = ChessDecisionTransformer(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_moves=config['max_moves'],
        action_vocab_size=config['action_vocab_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Load action vocab from checkpoint if present
    if 'uci_to_idx' in checkpoint and 'idx_to_uci' in checkpoint:
        uci_to_idx = checkpoint['uci_to_idx']
        idx_to_uci = {int(k): v for k, v in checkpoint['idx_to_uci'].items()}
    else:
        raise ValueError("Model checkpoint does not contain action vocabulary. Retrain and save with vocab.")
    return model, uci_to_idx, idx_to_uci

# --- Inference function ---
def infer_next_move(model, uci_to_idx, idx_to_uci, fen, move_history_uci, returns_to_go=0.0, max_history=8):
    # Start from the initial position and replay the move history
    board = chess.Board(fen) if fen else chess.Board()
    boards = [board.copy()]
    for move_uci in move_history_uci:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            boards.append(board.copy())
        else:
            raise ValueError(f"Illegal move in history: {move_uci}")
    # Now, the current board is after all moves in move_history_uci
    # Use last max_history boards for encoding
    history = boards[-max_history:][::-1]
    reps = get_repetition_counts(history)
    state_tensor = encode_history_tensor(history, reps, max_history=max_history)  # (119, 8, 8)
    # Prepare model input (batch=1, seq_len=1)
    states = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 119, 8, 8)
    actions = torch.zeros((1, 1), dtype=torch.long)  # dummy previous action
    returns_to_go = torch.tensor([[[returns_to_go]]], dtype=torch.float32)  # (1, 1, 1)
    timesteps = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        logits = model(states, actions, returns_to_go, timesteps)  # (1, 1, vocab)
        probs = torch.softmax(logits[0, 0], dim=0)
        # Sort indices by descending probability
        sorted_indices = torch.argsort(probs, descending=True)
        for idx in sorted_indices:
            idx = int(idx.item())
            uci = idx_to_uci[idx]
            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                continue  # Skip invalid UCI strings
            if move in board.legal_moves:
                return uci, float(probs[idx].item())
        # If no legal move found, return None
        return None, 0.0

# --- CLI ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Decision Transformer Chess Inference")
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), 'decision_transformer_test.pt'), help='Path to saved model')
    parser.add_argument('--fen', type=str, required=True, help='FEN string of the current board state')
    parser.add_argument('--moves', type=str, nargs='*', default=[], help='List of previous moves in UCI format')
    parser.add_argument('--rtg', type=float, default=0.0, help='Returns-to-go (default 0.0)')
    args = parser.parse_args()

    model, uci_to_idx, idx_to_uci = load_model(args.model)
    best_move, prob = infer_next_move(model, uci_to_idx, idx_to_uci, args.fen, args.moves, returns_to_go=args.rtg)
    print(f"Best next move: {best_move} (probability {prob:.4f})") 