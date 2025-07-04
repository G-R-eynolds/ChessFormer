import chess
import os
from infer_decision_transformer import load_model, infer_next_move

def play_vs_model_text():
    # Load model and action vocab
    model_path = os.path.join(os.path.dirname(__file__), 'decision_transformer_test.pt')
    model, uci_to_idx, idx_to_uci = load_model(model_path)

    board = chess.Board()
    move_history = []
    user_color = chess.WHITE

    while not board.is_game_over():
        print(board)
        print("Move history:", move_history)
        if board.turn == user_color:
            move_uci = input("Your move (UCI): ").strip()
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    move_history.append(move_uci)
                else:
                    print("Illegal move. Try again.")
            except Exception:
                print("Invalid move format. Try again.")
        else:
            print("Model is thinking...")
            model_move_uci, prob = infer_next_move(model, uci_to_idx, idx_to_uci, None, move_history)
            if model_move_uci is not None:
                model_move = chess.Move.from_uci(model_move_uci)
                if model_move in board.legal_moves:
                    print(f"Model plays: {model_move_uci} (prob={prob:.4f})")
                    board.push(model_move)
                    move_history.append(model_move_uci)
                else:
                    print(f"Model tried illegal move: {model_move_uci}")
                    break
            else:
                print("Model could not find a legal move.")
                break
    print(board)
    outcome = board.outcome()
    if outcome is not None:
        print(f"Game over: {board.result()} - {outcome.termination.name}")
    else:
        print(f"Game over: {board.result()} - (unknown termination)")

if __name__ == "__main__":
    play_vs_model_text() 