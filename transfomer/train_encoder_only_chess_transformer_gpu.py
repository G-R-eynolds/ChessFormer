import os
import torch
import torch.nn as nn
import torch.optim as optim
import chess.pgn
import random
import time
from torch.utils.data import Dataset, DataLoader
from encoder_only_chess_transformer import EncoderOnlyChessTransformer
from chess_history_encoder import encode_history_tensor, get_repetition_counts
from tqdm import tqdm
import numpy as np

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

# --- Config ---
PGN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Lichess Elite Database", "lichess_elite_2016-01.pgn")
MAX_GAMES = 1000000  # Use as much data as possible
BATCH_SIZE = 64
EPOCHS = 8  # Will stop early if 12h reached
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 8
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
MAX_HISTORY = 8
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
VAL_SPLIT = 0.01  # 1% for validation
MAX_TRAIN_HOURS = 12

# --- Step 1: Parse PGN and collect positions (metadata only) ---
print("Parsing PGN and collecting position metadata...")
games = []
with open(PGN_PATH, encoding="utf-8") as pgn:
    for _ in tqdm(range(MAX_GAMES), desc="Reading games"):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        games.append(game)

uci_set = set()
position_metadata = []  # Each entry: (game_idx, t, move_uci, result)
for game_idx, game in tqdm(list(enumerate(games)), desc="Extracting positions"):
    result = game.headers.get("Result", "*")
    if result == "1-0":
        value = 1.0
    elif result == "0-1":
        value = -1.0
    elif result == "1/2-1/2":
        value = 0.0
    else:
        continue  # skip unfinished/malformed
    board = game.board()
    boards = [board.copy()]
    actions = []
    for move in game.mainline_moves():
        actions.append(move.uci())
        board.push(move)
        boards.append(board.copy())
    for t in range(1, len(boards)):
        move_uci = actions[t-1]
        uci_set.add(move_uci)
        position_metadata.append((game_idx, t, move_uci, value))

uci_list = sorted(uci_set)
uci_to_idx = {uci: i for i, uci in enumerate(uci_list)}
action_size = len(uci_list)

# --- Train/Val Split ---
np.random.seed(42)
indices = np.arange(len(position_metadata))
np.random.shuffle(indices)
split = int(len(indices) * (1 - VAL_SPLIT))
train_indices = indices[:split]
val_indices = indices[split:]

# --- Step 2: Custom Dataset for parallel encoding ---
class ChessPositionDataset(Dataset):
    def __init__(self, games, position_metadata, uci_to_idx, max_history=8):
        self.games = games
        self.position_metadata = position_metadata
        self.uci_to_idx = uci_to_idx
        self.max_history = max_history

    def __len__(self):
        return len(self.position_metadata)

    def __getitem__(self, idx):
        game_idx, t, move_uci, value = self.position_metadata[idx]
        game = self.games[game_idx]
        board = game.board()
        boards = [board.copy()]
        actions = []
        for move in game.mainline_moves():
            actions.append(move.uci())
            board.push(move)
            boards.append(board.copy())
        history = boards[max(0, t-self.max_history):t][::-1]
        reps = get_repetition_counts(history)
        board_tensor = encode_history_tensor(history, reps, max_history=self.max_history)  # (119, 8, 8)
        source_square = chess.SQUARE_NAMES.index(move_uci[:2])
        move_idx = self.uci_to_idx[move_uci]
        return (
            board_tensor,  # (119, 8, 8)
            source_square, # int
            move_idx,      # int
            value          # float
        )

train_dataset = ChessPositionDataset(games, [position_metadata[i] for i in train_indices], uci_to_idx, max_history=MAX_HISTORY)
val_dataset = ChessPositionDataset(games, [position_metadata[i] for i in val_indices], uci_to_idx, max_history=MAX_HISTORY)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True,
    pin_memory=True
)

# --- Step 3: Model, optimizer, loss ---
model = EncoderOnlyChessTransformer(
    input_channels=119,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    action_size=action_size,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=2e-4)
policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

# --- Step 4: Training loop with mixed precision, progress bars, diagnostics ---
print("Starting GPU training...")
best_val_loss = float('inf')
best_model_state = None
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_batches = 0
    batch_times = []
    model_times = []
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} [train]") as pbar:
        for batch in train_loader:
            batch_start = time.time()
            board_tensors, source_squares, move_indices, values = batch
            board_tensors = board_tensors.to(DEVICE, non_blocking=True)
            source_squares = source_squares.to(DEVICE, non_blocking=True)
            move_indices = move_indices.to(DEVICE, non_blocking=True)
            values = values.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            model_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                policy_logits, value_pred = model(board_tensors)
                selected_logits = policy_logits[torch.arange(board_tensors.shape[0]), source_squares, :]
                policy_loss = policy_criterion(selected_logits, move_indices)
                value_loss = value_criterion(value_pred, values)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_end = time.time()
            model_times.append(model_end - model_start)
            total_loss += loss.item()
            total_batches += 1
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated() / 2**30
            pbar.set_postfix({
                'avg_batch_prep_s': f"{sum(batch_times)/len(batch_times):.3f}",
                'avg_model_step_s': f"{sum(model_times)/len(model_times):.3f}",
                'loss': f"{loss.item():.4f}",
                'VRAM_GB': f"{vram:.2f}"
            })
            pbar.update(1)
            # Stop if we've trained for 12 hours
            if (time.time() - start_time) > MAX_TRAIN_HOURS * 3600:
                print("Reached 12 hour training limit. Stopping early.")
                break
    avg_loss = total_loss / max(1, total_batches)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | avg loss={avg_loss:.4f} | time/epoch={epoch_time:.2f}s")
    print(f"  Avg batch prep time: {sum(batch_times)/len(batch_times):.4f}s | Avg model step time: {sum(model_times)/len(model_times):.4f}s")
    if HAVE_PSUTIL:
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"  CPU Utilization: {cpu}%")
    # --- Validation ---
    model.eval()
    val_total_loss = 0.0
    val_total_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            board_tensors, source_squares, move_indices, values = batch
            board_tensors = board_tensors.to(DEVICE, non_blocking=True)
            source_squares = source_squares.to(DEVICE, non_blocking=True)
            move_indices = move_indices.to(DEVICE, non_blocking=True)
            values = values.float().unsqueeze(1).to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast():
                policy_logits, value_pred = model(board_tensors)
                selected_logits = policy_logits[torch.arange(board_tensors.shape[0]), source_squares, :]
                policy_loss = policy_criterion(selected_logits, move_indices)
                value_loss = value_criterion(value_pred, values)
                loss = policy_loss + value_loss
            val_total_loss += loss.item()
            val_total_batches += 1
    avg_val_loss = val_total_loss / max(1, val_total_batches)
    print(f"  Validation loss: {avg_val_loss:.4f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        print("  New best model found and saved.")
    # Stop if we've trained for 12 hours
    if (time.time() - start_time) > MAX_TRAIN_HOURS * 3600:
        print("Reached 12 hour training limit. Stopping early.")
        break

# --- Save best model and vocab after training ---
save_path = os.path.join(os.path.dirname(__file__), 'encoder_only_chess_transformer_gpu.pt')
torch.save({
    'model_state_dict': best_model_state,
    'config': {
        'input_channels': 119,
        'd_model': D_MODEL,
        'nhead': N_HEAD,
        'num_layers': NUM_LAYERS,
        'action_size': action_size,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT
    },
    'uci_to_idx': uci_to_idx,
    'idx_to_uci': uci_list
}, save_path)
print(f"Best model saved to {save_path}") 