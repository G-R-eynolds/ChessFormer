import chess
import chess.pgn
import torch
from typing import List, Dict, Any
from chess_history_encoder import encode_history_tensor, get_repetition_counts

RESULT_TO_REWARD = {
    "1-0": 1.0,    # White wins
    "0-1": -1.0,   # Black wins
    "1/2-1/2": 0.0 # Draw
}

def pgn_file_to_trajectories(pgn_path: str, max_history: int = 8) -> List[Dict[str, Any]]:
    """
    Parses a PGN file and returns a list of trajectories for training.
    Each trajectory is a dict with keys: 'states', 'actions', 'returns_to_go', 'result', 'headers'.
    """
    trajectories = []
    with open(pgn_path, encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in RESULT_TO_REWARD:
                continue  # skip unfinished or malformed games
            reward = RESULT_TO_REWARD[result]
            # --- Extract moves and build trajectory ---
            board = game.board()
            boards = [board.copy()]
            actions = []
            for move in game.mainline_moves():
                actions.append(move.uci())
                board.push(move)
                boards.append(board.copy())
            # For each ply, build the history tensor
            states = []
            action_indices = []
            for t in range(1, len(boards)):
                # Most recent first for history encoder
                history = boards[max(0, t-max_history):t][::-1]
                reps = get_repetition_counts(history)
                state_tensor = encode_history_tensor(history, reps, max_history=max_history)  # (119, 8, 8)
                states.append(state_tensor)
                # Action index: convert UCI to index (to be mapped by user)
                action_indices.append(actions[t-1])
            # Returns-to-go: +1 for win, 0 for draw, -1 for loss, same for all plies
            returns_to_go = torch.full((len(states), 1), reward, dtype=torch.float32)
            trajectory = {
                "states": torch.stack(states),  # (seq_len, 119, 8, 8)
                "actions": action_indices,      # (seq_len,) list of UCI strings
                "returns_to_go": returns_to_go, # (seq_len, 1)
                "result": result,
                "headers": dict(game.headers)
            }
            trajectories.append(trajectory)
    return trajectories

# --- Example Usage ---
if __name__ == "__main__":
    import sys
    import os
    # Example: parse the first 2 games from a PGN file
    pgn_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "Lichess Elite Database", "lichess_elite_2013-09.pgn")
    trajs = pgn_file_to_trajectories(pgn_path)
    print(f"Parsed {len(trajs)} trajectories from {pgn_path}")
    if trajs:
        t0 = trajs[0]
        print(f"First trajectory: {len(t0['states'])} plies, result={t0['result']}, headers={t0['headers']}")
        print(f"States shape: {t0['states'].shape}")
        print(f"Returns-to-go: {t0['returns_to_go'].flatten().tolist()}")
        print(f"First 5 actions: {t0['actions'][:5]}") 