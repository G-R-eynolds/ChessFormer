import chess
import numpy as np
from typing import Tuple, Optional

def encode_chess_state(board: chess.Board) -> np.ndarray:
    """
    Encode a chess board state into a stack of binary planes.
    
    Args:
        board: A python-chess Board object representing the current game state
        
    Returns:
        np.ndarray: A 3D array of shape (18, 8, 8) containing:
            - 12 piece location planes (6 piece types × 2 colors)
            - 1 turn plane
            - 4 castling rights planes
            - 1 en passant square plane
    """
    
    # Initialize the output array with zeros
    # Shape: (18, 8, 8) - 18 planes total
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    
    # Piece location planes (0-11): 6 piece types × 2 colors
    piece_planes = {
        chess.PAWN: 0,      # White pawns (plane 0)
        chess.KNIGHT: 2,    # White knights (plane 2)
        chess.BISHOP: 4,    # White bishops (plane 4)
        chess.ROOK: 6,      # White rooks (plane 6)
        chess.QUEEN: 8,     # White queens (plane 8)
        chess.KING: 10,     # White kings (plane 10)
    }
    
    # Debug: Print piece mapping
    # print("Piece plane mapping:")
    # for piece_type, plane_idx in piece_planes.items():
    #     piece_name = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop", 
    #                  chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}[piece_type]
    #     print(f"  {piece_name}: White=plane {plane_idx}, Black=plane {plane_idx+1}")
    
    # Fill piece location planes
    piece_count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_count += 1
            # Calculate plane index based on piece type and color
            if piece.color == chess.WHITE:
                plane_idx = piece_planes[piece.piece_type]
            else:
                plane_idx = piece_planes[piece.piece_type] + 1
            
            # Convert square to rank and file
            rank, file = divmod(square, 8)
            # Flip the rank so that rank 0 (bottom) maps to row 7 in visualization
            # and rank 7 (top) maps to row 0 in visualization
            visual_rank = 7 - rank
            planes[plane_idx, visual_rank, file] = 1.0
            
            # Debug: Print piece placement
            # piece_name = {chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop", 
            #              chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"}[piece.piece_type]
            # color_name = "White" if piece.color == chess.WHITE else "Black"
            # square_name = chess.square_name(square)
            # print(f"  {color_name} {piece_name} at {square_name} -> plane {plane_idx}, pos ({visual_rank}, {file})")
    
    # print(f"Total pieces placed: {piece_count}")
    
    # Turn plane (12): 1 if White's turn, 0 if Black's turn
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
        # print("Turn: White (plane 12 = all 1s)")
    else:
        # print("Turn: Black (plane 12 = all 0s)")
        pass
    
    # Castling rights planes (13-16)
    # Plane 13: White kingside castling
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
        # print("White kingside castling: allowed (plane 13 = all 1s)")
    else:
        # print("White kingside castling: not allowed (plane 13 = all 0s)")
        pass
    
    # Plane 14: White queenside castling
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
        # print("White queenside castling: allowed (plane 14 = all 1s)")
    else:
        # print("White queenside castling: not allowed (plane 14 = all 0s)")
        pass
    
    # Plane 15: Black kingside castling
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
        # print("Black kingside castling: allowed (plane 15 = all 1s)")
    else:
        # print("Black kingside castling: not allowed (plane 15 = all 0s)")
        pass
    
    # Plane 16: Black queenside castling
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0
        # print("Black queenside castling: allowed (plane 16 = all 1s)")
    else:
        # print("Black queenside castling: not allowed (plane 16 = all 0s)")
        pass
    
    # En passant square plane (17): 1 on the en passant target square
    if board.has_legal_en_passant():
        ep_square = board.ep_square
        if ep_square is not None:
            rank, file = divmod(ep_square, 8)
            visual_rank = 7 - rank
            planes[17, visual_rank, file] = 1.0
            # print(f"En passant target: {chess.square_name(ep_square)} (plane 17, pos ({visual_rank}, {file}))")
    else:
        # print("En passant: not available (plane 17 = all 0s)")
        pass
    
    return planes

def decode_chess_state(planes: np.ndarray) -> chess.Board:
    """
    Decode binary planes back to a chess board state.
    
    Args:
        planes: A 3D array of shape (18, 8, 8) containing the encoded state
        
    Returns:
        chess.Board: A python-chess Board object representing the decoded state
    """
    
    board = chess.Board()
    board.clear()  # Start with empty board
    
    # Piece mapping for decoding
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Decode piece locations
    for piece_idx, piece_type in enumerate(piece_types):
        # White pieces (even planes: 0, 2, 4, 6, 8, 10)
        white_plane = planes[piece_idx * 2]
        for rank in range(8):
            for file in range(8):
                if white_plane[rank, file] > 0.5:  # Threshold for binary values
                    # Flip the rank back to chess coordinates
                    chess_rank = 7 - rank
                    square = chess_rank * 8 + file
                    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
        
        # Black pieces (odd planes: 1, 3, 5, 7, 9, 11)
        black_plane = planes[piece_idx * 2 + 1]
        for rank in range(8):
            for file in range(8):
                if black_plane[rank, file] > 0.5:
                    # Flip the rank back to chess coordinates
                    chess_rank = 7 - rank
                    square = chess_rank * 8 + file
                    board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
    
    # Decode turn
    if planes[12, 0, 0] > 0.5:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK
    
    # Decode castling rights
    board.castling_rights = 0
    if planes[13, 0, 0] > 0.5:  # White kingside
        board.castling_rights |= chess.BB_H1
    if planes[14, 0, 0] > 0.5:  # White queenside
        board.castling_rights |= chess.BB_A1
    if planes[15, 0, 0] > 0.5:  # Black kingside
        board.castling_rights |= chess.BB_H8
    if planes[16, 0, 0] > 0.5:  # Black queenside
        board.castling_rights |= chess.BB_A8
    
    # Decode en passant square
    ep_square = None
    for rank in range(8):
        for file in range(8):
            if planes[17, rank, file] > 0.5:
                # Flip the rank back to chess coordinates
                chess_rank = 7 - rank
                ep_square = chess_rank * 8 + file
                break
        if ep_square is not None:
            break
    
    board.ep_square = ep_square
    
    return board

def get_plane_descriptions() -> list:
    """
    Get descriptions of each plane in the encoding.
    
    Returns:
        list: List of strings describing each plane
    """
    return [
        "White Pawns",
        "Black Pawns", 
        "White Knights",
        "Black Knights",
        "White Bishops",
        "Black Bishops",
        "White Rooks",
        "Black Rooks",
        "White Queens",
        "Black Queens",
        "White Kings",
        "Black Kings",
        "Turn (White=1, Black=0)",
        "White Kingside Castling",
        "White Queenside Castling",
        "Black Kingside Castling",
        "Black Queenside Castling",
        "En Passant Square"
    ]

def visualize_planes(planes: np.ndarray, title: str = "Chess State Planes") -> None:
    """
    Visualize the binary planes as a grid of images.
    
    Args:
        planes: A 3D array of shape (18, 8, 8) containing the encoded state
        title: Title for the visualization
    """
    
    import matplotlib.pyplot as plt
    
    descriptions = get_plane_descriptions()
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)
    
    for i in range(18):
        row = i // 6
        col = i % 6
        
        if row < 3:
            ax = axes[row, col]
            
            # Show the plane with binary colormap
            im = ax.imshow(planes[i], cmap='binary', vmin=0, vmax=1)
            ax.set_title(descriptions[i], fontsize=9, pad=5)
            
            # Always show grid lines
            ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
            
            # Set major ticks for coordinates
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
            ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
            
            # Add colorbar for each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Count non-zero elements for debugging
            nonzero_count = np.count_nonzero(planes[i])
            ax.text(0.02, 0.98, f'Non-zero: {nonzero_count}', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
        


def test_encoding_decoding():
    """
    Test the encoding and decoding functions with a sample game.
    """
    print("Testing chess state encoding and decoding...")
    
    # Create a board with some moves
    board = chess.Board()
    
    # Make some moves
    moves = [
        "e2e4",  # White pawn to e4
        "e7e5",  # Black pawn to e5
        "g1f3",  # White knight to f3
        "b8c6",  # Black knight to c6
    ]
    
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        print(f"After {move_uci}: {board.fen()}")
    
    # Encode the state
    planes = encode_chess_state(board)
    print(f"\nEncoded state shape: {planes.shape}")
    
    # Decode back to board
    decoded_board = decode_chess_state(planes)
    print(f"Decoded board FEN: {decoded_board.fen()}")
    
    # Verify they match
    if board.fen() == decoded_board.fen():
        print("✓ Encoding/decoding test passed!")
    else:
        print("✗ Encoding/decoding test failed!")
        print(f"Original: {board.fen()}")
        print(f"Decoded:  {decoded_board.fen()}")
    
    # Print plane descriptions
    print("\nPlane descriptions:")
    for i, desc in enumerate(get_plane_descriptions()):
        print(f"Plane {i:2d}: {desc}")
    
    return planes

if __name__ == "__main__":
    # Run the test
    planes = test_encoding_decoding()
    
    # Try to visualize (if matplotlib is available)
    try:
        visualize_planes(planes, "Test Chess State Encoding")
    except:
        print("\nVisualization skipped (matplotlib not available)") 