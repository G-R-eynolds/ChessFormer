import chess
import chess.pgn
import tkinter as tk
from tkinter import ttk, messagebox
import os

class ChessSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Simulator")
        self.root.geometry("800x600")
        
        # Initialize chess board
        self.board = chess.Board()
        self.selected_square = None
        self.game_history = []
        
        # Create GUI components
        self.create_widgets()
        self.update_display()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Control buttons
        ttk.Button(control_frame, text="New Game", command=self.new_game).grid(row=0, column=0, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Undo Move", command=self.undo_move).grid(row=1, column=0, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Reset Game", command=self.reset_game).grid(row=2, column=0, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Save Game", command=self.save_game).grid(row=3, column=0, pady=5, sticky="ew")
        ttk.Button(control_frame, text="Load Game", command=self.load_game).grid(row=4, column=0, pady=5, sticky="ew")
        
        # Game info
        info_frame = ttk.LabelFrame(control_frame, text="Game Info", padding="10")
        info_frame.grid(row=5, column=0, pady=(20, 0), sticky="ew")
        
        self.turn_label = ttk.Label(info_frame, text="Turn: White")
        self.turn_label.grid(row=0, column=0, pady=5)
        
        self.status_label = ttk.Label(info_frame, text="Status: Playing")
        self.status_label.grid(row=1, column=0, pady=5)
        
        self.move_count_label = ttk.Label(info_frame, text="Moves: 0")
        self.move_count_label.grid(row=2, column=0, pady=5)
        
        # Chess board frame (right side)
        board_frame = ttk.Frame(main_frame)
        board_frame.grid(row=0, column=1, sticky="nsew")
        
        # Create chess board canvas
        self.canvas = tk.Canvas(board_frame, width=400, height=400, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_square_click)
        
        # Move history
        history_frame = ttk.LabelFrame(main_frame, text="Move History", padding="10")
        history_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.history_text = tk.Text(history_frame, height=6, width=80)
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.grid(row=0, column=0, sticky="nsew")
        history_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure history frame grid weights
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
    def draw_board(self):
        """Draw the chess board with pieces"""
        self.canvas.delete("all")
        
        # Board colors
        light_square = "#F0D9B5"
        dark_square = "#B58863"
        selected_color = "#7B61FF"
        
        square_size = 50
        
        for rank in range(8):
            for file in range(8):
                x1 = file * square_size
                y1 = (7 - rank) * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                # Determine square color
                is_light = (rank + file) % 2 == 0
                color = light_square if is_light else dark_square
                
                # Highlight selected square
                square = chess.square(file, rank)
                if self.selected_square == square:
                    color = selected_color
                
                # Draw square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Draw piece
                piece = self.board.piece_at(square)
                if piece:
                    piece_symbol = self.get_piece_symbol(piece)
                    self.canvas.create_text(
                        x1 + square_size // 2,
                        y1 + square_size // 2,
                        text=piece_symbol,
                        font=("Arial", 24, "bold"),
                        fill="black" if piece.color == chess.WHITE else "white"
                    )
                
                # Draw coordinates
                if file == 0:  # Left edge
                    self.canvas.create_text(
                        5, y1 + square_size // 2,
                        text=str(8 - rank),
                        font=("Arial", 10),
                        anchor="w"
                    )
                if rank == 7:  # Bottom edge
                    self.canvas.create_text(
                        x1 + square_size // 2, 395,
                        text=chr(97 + file),
                        font=("Arial", 10),
                        anchor="s"
                    )
    
    def get_piece_symbol(self, piece):
        """Get Unicode symbol for chess piece"""
        symbols = {
            chess.PAWN: "♙" if piece.color == chess.WHITE else "♟",
            chess.ROOK: "♖" if piece.color == chess.WHITE else "♜",
            chess.KNIGHT: "♘" if piece.color == chess.WHITE else "♞",
            chess.BISHOP: "♗" if piece.color == chess.WHITE else "♝",
            chess.QUEEN: "♕" if piece.color == chess.WHITE else "♛",
            chess.KING: "♔" if piece.color == chess.WHITE else "♚"
        }
        return symbols.get(piece.piece_type, "")
    
    def on_square_click(self, event):
        """Handle square clicks for move selection"""
        square_size = 50
        file = event.x // square_size
        rank = 7 - (event.y // square_size)
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            square = chess.square(file, rank)
            
            if self.selected_square is None:
                # First click - select piece
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.update_display()
            else:
                # Second click - attempt move
                move = chess.Move(self.selected_square, square)
                
                # Check for pawn promotion
                selected_piece = self.board.piece_at(self.selected_square)
                if (selected_piece and 
                    selected_piece.piece_type == chess.PAWN and
                    ((self.board.turn == chess.WHITE and rank == 7) or 
                     (self.board.turn == chess.BLACK and rank == 0))):
                    # Promote to queen by default
                    move = chess.Move(self.selected_square, square, chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self.make_move(move)
                else:
                    # Try to select a different piece
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.board.turn:
                        self.selected_square = square
                    else:
                        self.selected_square = None
                    self.update_display()
    
    def make_move(self, move):
        """Make a move on the board"""
        # Store move for history
        san_move = self.board.san(move)
        self.game_history.append((san_move, self.board.copy()))
        
        # Make the move
        self.board.push(move)
        
        # Reset selection
        self.selected_square = None
        
        # Update display
        self.update_display()
        
        # Check game status
        self.check_game_status()
    
    def check_game_status(self):
        """Check and display game status"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            messagebox.showinfo("Game Over", f"Checkmate! {winner} wins!")
            self.status_label.config(text="Status: Checkmate")
        elif self.board.is_stalemate():
            messagebox.showinfo("Game Over", "Stalemate! The game is a draw.")
            self.status_label.config(text="Status: Stalemate")
        elif self.board.is_insufficient_material():
            messagebox.showinfo("Game Over", "Insufficient material! The game is a draw.")
            self.status_label.config(text="Status: Insufficient Material")
        elif self.board.is_check():
            self.status_label.config(text="Status: Check")
        else:
            self.status_label.config(text="Status: Playing")
    
    def update_display(self):
        """Update the display with current board state"""
        self.draw_board()
        
        # Update labels
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        self.turn_label.config(text=f"Turn: {turn}")
        
        move_count = len(self.board.move_stack)
        self.move_count_label.config(text=f"Moves: {move_count}")
        
        # Update move history
        self.update_move_history()
    
    def update_move_history(self):
        """Update the move history display"""
        self.history_text.delete(1.0, tk.END)
        
        # Get the game as PGN
        game = chess.pgn.Game()
        game.add_line(self.board.move_stack)
        
        # Display move history
        history_text = str(game.mainline())
        self.history_text.insert(tk.END, history_text)
        self.history_text.see(tk.END)
    
    def new_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.selected_square = None
        self.game_history = []
        self.update_display()
    
    def undo_move(self):
        """Undo the last move"""
        if self.board.move_stack:
            self.board.pop()
            if self.game_history:
                self.game_history.pop()
            self.selected_square = None
            self.update_display()
    
    def reset_game(self):
        """Reset the game to initial position"""
        self.board = chess.Board()
        self.selected_square = None
        self.game_history = []
        self.update_display()
    
    def save_game(self):
        """Save the current game to a PGN file"""
        try:
            game = chess.pgn.Game()
            game.add_line(self.board.move_stack)
            
            filename = "chess_game.pgn"
            with open(filename, "w") as f:
                f.write(str(game))
            
            messagebox.showinfo("Success", f"Game saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save game: {e}")
    
    def load_game(self):
        """Load a game from a PGN file"""
        try:
            filename = "chess_game.pgn"
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    game = chess.pgn.read_game(f)
                
                if game:
                    self.board = game.board()
                    self.selected_square = None
                    self.game_history = []
                    self.update_display()
                    messagebox.showinfo("Success", f"Game loaded from {filename}")
                else:
                    messagebox.showerror("Error", "No valid game found in file")
            else:
                messagebox.showerror("Error", f"File {filename} not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load game: {e}")

def main():
    root = tk.Tk()
    app = ChessSimulator(root)
    root.mainloop()

if __name__ == "__main__":
    main() 