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
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

# --- Config ---
PGN_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Lichess Elite Database", "lichess_elite_2016-01.pgn")
MAX_GAMES = 500
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device("cpu")
D_MODEL = 64
N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
MAX_HISTORY = 8
NUM_WORKERS = 20  # For your 22-thread CPU
PREFETCH_FACTOR = 4

# --- Step 1: Parse PGN and collect positions (metadata only) ---
print("Parsing PGN and collecting position metadata...")
games = []
with open(PGN_PATH, encoding="utf-8") as pgn:
    for _ in tqdm(range(MAX_GAMES), desc="Reading games"):
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        games.append(game)

# --- Step 2: Build action vocabulary ---
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

# --- Step 3: Custom Dataset for parallel encoding ---
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

dataset = ChessPositionDataset(games, position_metadata, uci_to_idx, max_history=MAX_HISTORY)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=True,
    pin_memory=False
)

# --- Step 4: Model, optimizer, loss ---
model = EncoderOnlyChessTransformer(
    input_channels=119,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=NUM_LAYERS,
    action_size=action_size,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
policy_criterion = nn.CrossEntropyLoss()
value_criterion = nn.MSELoss()

# --- Step 5: Training loop ---
print("Starting training...")
epoch_times = []
for epoch in range(EPOCHS):
    total_loss = 0.0
    total_batches = 0
    start_time = time.time()
    batch_times = []
    model_times = []
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for batch in data_loader:
            batch_start = time.time()
            board_tensors, source_squares, move_indices, values = batch
            board_tensors = board_tensors.to(DEVICE)  # (B, 119, 8, 8)
            source_squares = source_squares.to(DEVICE)
            move_indices = move_indices.to(DEVICE)
            values = values.float().unsqueeze(1).to(DEVICE)
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            model_start = time.time()
            optimizer.zero_grad()
            policy_logits, value_pred = model(board_tensors)  # policy_logits: (B, 64, action_size)
            selected_logits = policy_logits[torch.arange(board_tensors.shape[0]), source_squares, :]  # (B, action_size)
            policy_loss = policy_criterion(selected_logits, move_indices)
            value_loss = value_criterion(value_pred, values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            model_end = time.time()
            model_times.append(model_end - model_start)
            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({
                'avg_batch_prep_s': f"{sum(batch_times)/len(batch_times):.3f}",
                'avg_model_step_s': f"{sum(model_times)/len(model_times):.3f}",
                'loss': f"{loss.item():.4f}"
            })
            pbar.update(1)
    avg_loss = total_loss / max(1, total_batches)
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    eta = avg_epoch_time * (EPOCHS - (epoch + 1))
    print(f"Epoch {epoch+1}/{EPOCHS} | avg loss={avg_loss:.4f} | time/epoch={epoch_time:.2f}s | ETA={eta:.1f}s")
    print(f"  Avg batch prep time: {sum(batch_times)/len(batch_times):.4f}s | Avg model step time: {sum(model_times)/len(model_times):.4f}s")
    if HAVE_PSUTIL:
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"  CPU Utilization: {cpu}%")

# --- Save model and vocab after training ---
save_path = os.path.join(os.path.dirname(__file__), 'encoder_only_chess_transformer.pt')
torch.save({
    'model_state_dict': model.state_dict(),
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
print(f"Model saved to {save_path}") 