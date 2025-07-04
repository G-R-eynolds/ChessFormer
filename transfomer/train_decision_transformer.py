import os
import torch
import torch.nn as nn
import torch.optim as optim
from chess_decision_transformer import ChessDecisionTransformer
from pgn_to_trajectories import pgn_file_to_trajectories
import time
import random
import chess.pgn
import glob
import zstandard as zstd
import io

# --- Config ---
D_MODEL = 256  # Scaled up for better performance
N_HEAD = 8
NUM_LAYERS = 8
MAX_MOVES = 80  # Allow for longer games
BATCH_SIZE = 2
SEQ_LEN = 10  # Use fewer plies for debugging
SAMPLE_GAMES = None  # Use all games in the file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streaming PGN from uncompressed PGN files ---
def stream_pgn_games(pgn_path):
    with open(pgn_path, encoding='utf-8') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield game

def build_action_vocab_from_pgn(pgn_paths):
    uci_set = set()
    for pgn_path in pgn_paths:
        print(f"Scanning moves in {pgn_path}...", flush=True)
        for game in stream_pgn_games(pgn_path):
            for move in game.mainline_moves():
                uci_set.add(move.uci())
    uci_moves = sorted(uci_set)
    uci_to_idx = {uci: i for i, uci in enumerate(uci_moves)}
    idx_to_uci = {i: uci for i, uci in enumerate(uci_moves)}
    return uci_to_idx, idx_to_uci

def stream_trajectories_from_pgn(pgn_path, uci_to_idx, seq_len, max_history=8):
    from pgn_to_trajectories import RESULT_TO_REWARD, encode_history_tensor, get_repetition_counts
    skipped_games = 0
    yielded_trajectories = 0
    for game_idx, game in enumerate(stream_pgn_games(pgn_path)):
        result = game.headers.get("Result", "*")
        if result not in RESULT_TO_REWARD:
            skipped_games += 1
            print(f"[TrajGen] Skipping game {game_idx} in {pgn_path} (result={result})")
            continue
        reward = RESULT_TO_REWARD[result]
        board = game.board()
        boards = [board.copy()]
        actions = []
        for move in game.mainline_moves():
            actions.append(move.uci())
            board.push(move)
            boards.append(board.copy())
        states = []
        action_indices = []
        for t in range(1, len(boards)):
            history = boards[max(0, t-max_history):t][::-1]
            reps = get_repetition_counts(history)
            state_tensor = encode_history_tensor(history, reps, max_history=max_history)
            states.append(state_tensor)
            action_indices.append(actions[t-1])
        returns_to_go = torch.full((len(states), 1), reward, dtype=torch.float32)
        # Truncate/pad
        actions_idx = [uci_to_idx.get(a, 0) for a in action_indices[:seq_len]]
        states = states[:seq_len]
        returns_to_go = returns_to_go[:seq_len]
        pad_len = seq_len - len(actions_idx)
        if pad_len > 0:
            actions_idx += [0] * pad_len
            states = list(states) + [states[-1]] * pad_len
            returns_to_go = torch.cat([returns_to_go, returns_to_go[-1:].repeat(pad_len, 1)], dim=0)
        if isinstance(states, list):
            states = torch.stack(states)
        yielded_trajectories += 1
        if yielded_trajectories % 1000 == 0:
            print(f"[TrajGen] Yielding trajectory {yielded_trajectories} from game {game_idx} in {pgn_path}")
        yield {
            "states": states,
            "actions": torch.tensor(actions_idx, dtype=torch.long),
            "returns_to_go": returns_to_go
        }
    print(f"[TrajGen] Finished {pgn_path}: Skipped {skipped_games}, Yielded {yielded_trajectories}")

def batch_generator(pgn_paths, uci_to_idx, batch_size, seq_len):
    for pgn_path in pgn_paths:
        print(f"[BatchGen] Streaming from {pgn_path}...", flush=True)
        batch = []
        skipped = 0
        processed = 0
        for traj in stream_trajectories_from_pgn(pgn_path, uci_to_idx, seq_len):
            batch.append(traj)
            processed += 1
            if len(batch) == batch_size:
                print(f"[BatchGen] Yielding batch of size {batch_size} from {pgn_path} (processed {processed})")
                yield batch
                batch = []
        if batch:
            print(f"[BatchGen] Yielding final batch of size {len(batch)} from {pgn_path} (processed {processed})")
            yield batch

def get_pgn_files_for_year(base_dir, year):
    pattern = os.path.join(base_dir, "Lichess Elite Database", f"lichess_elite_{year}-*.pgn")
    return sorted(glob.glob(pattern))

def split_pgn_files(pgn_files, val_ratio=0.1):
    random.shuffle(pgn_files)
    split_idx = int((1 - val_ratio) * len(pgn_files))
    return pgn_files[:split_idx], pgn_files[split_idx:]

# --- Shuffling batches in memory ---
def shuffled_batch_generator(pgn_paths, uci_to_idx, batch_size, seq_len):
    all_batches = []
    batch = []
    for pgn_path in pgn_paths:
        print(f"Streaming from {pgn_path}...", flush=True)
        for traj in stream_trajectories_from_pgn(pgn_path, uci_to_idx, seq_len):
            batch.append(traj)
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []
    if batch:
        all_batches.append(batch)
    random.shuffle(all_batches)
    for batch in all_batches:
        yield batch

# --- Validation batch generator (no shuffle) ---
def val_batch_generator(pgn_paths, uci_to_idx, batch_size, seq_len):
    batch = []
    for pgn_path in pgn_paths:
        print(f"[VAL] Streaming from {pgn_path}...", flush=True)
        for traj in stream_trajectories_from_pgn(pgn_path, uci_to_idx, seq_len):
            batch.append(traj)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

# --- Main training loop ---
def train():
    # Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Only use the first 2016 month file for fast testing
    pgn_files = [os.path.join(base_dir, "Lichess Elite Database", "lichess_elite_2016-01.pgn")]
    print(f"Building action vocabulary from {len(pgn_files)} files...", flush=True)
    global uci_to_idx, idx_to_uci, ACTION_VOCAB_SIZE
    uci_to_idx, idx_to_uci = build_action_vocab_from_pgn(pgn_files)
    ACTION_VOCAB_SIZE = len(uci_to_idx)
    print(f"Action vocab size: {ACTION_VOCAB_SIZE}", flush=True)
    # Split files for train/val (but only one file, so split manually)
    train_pgn_files = pgn_files
    val_pgn_files = pgn_files
    print(f"Train files: {len(train_pgn_files)}, Val files: {len(val_pgn_files)}", flush=True)
    print("Starting streaming training...", flush=True)
    # Model
    model = ChessDecisionTransformer(D_MODEL, N_HEAD, NUM_LAYERS, MAX_MOVES, ACTION_VOCAB_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    epoch_times = []
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\n[Train] Starting epoch {epoch+1}/{num_epochs}", flush=True)
        epoch_start = time.time()
        total_loss = 0.0
        total_batches = 0
        # --- Shuffle file order for stochasticity ---
        random.shuffle(train_pgn_files)
        # --- Training ---
        for batch_idx, batch in enumerate(batch_generator(train_pgn_files, uci_to_idx, BATCH_SIZE, SEQ_LEN)):
            states = torch.stack([d["states"] for d in batch]).to(DEVICE)
            actions = torch.stack([d["actions"] for d in batch]).to(DEVICE)
            returns_to_go = torch.stack([d["returns_to_go"] for d in batch]).to(DEVICE)
            timesteps = torch.arange(SEQ_LEN).unsqueeze(0).repeat(states.shape[0], 1).to(DEVICE)
            optimizer.zero_grad()
            logits = model(states, actions, returns_to_go, timesteps)
            loss = criterion(logits.view(-1, ACTION_VOCAB_SIZE), actions.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(1, total_batches)
        # --- Validation ---
        model.eval()
        val_total_loss = 0.0
        val_total_batches = 0
        for val_batch_idx, val_batch in enumerate(batch_generator(val_pgn_files, uci_to_idx, BATCH_SIZE, SEQ_LEN)):
            states = torch.stack([d["states"] for d in val_batch]).to(DEVICE)
            actions = torch.stack([d["actions"] for d in val_batch]).to(DEVICE)
            returns_to_go = torch.stack([d["returns_to_go"] for d in val_batch]).to(DEVICE)
            timesteps = torch.arange(SEQ_LEN).unsqueeze(0).repeat(states.shape[0], 1).to(DEVICE)
            logits = model(states, actions, returns_to_go, timesteps)
            val_loss = criterion(logits.view(-1, ACTION_VOCAB_SIZE), actions.view(-1))
            val_total_loss += val_loss.item()
            val_total_batches += 1
        model.train()
        avg_val_loss = val_total_loss / max(1, val_total_batches)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = num_epochs - (epoch + 1)
        eta = avg_epoch_time * epochs_left
        print(f"[Train] Finished epoch {epoch+1}/{num_epochs} | avg train loss={avg_loss:.4f} | avg val loss={avg_val_loss:.4f} | time/epoch={epoch_time:.2f}s | ETA={eta:.1f}s", flush=True)
    # Save model after training
    save_path = os.path.join(os.path.dirname(__file__), 'decision_transformer_test.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': D_MODEL,
            'nhead': N_HEAD,
            'num_layers': NUM_LAYERS,
            'max_moves': MAX_MOVES,
            'action_vocab_size': ACTION_VOCAB_SIZE
        },
        'uci_to_idx': uci_to_idx,
        'idx_to_uci': idx_to_uci
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train() 