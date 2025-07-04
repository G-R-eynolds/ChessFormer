# Chess Simulator

A clean and efficient chess simulation environment built with `python-chess` and `tkinter`.

## Features

- **Visual Chess Board**: Clean, intuitive interface with piece symbols
- **Move Validation**: All moves are validated according to chess rules
- **Game History**: Complete move history with PGN notation
- **Game Management**: New game, undo moves, reset, save/load functionality
- **Game Status**: Real-time status updates (check, checkmate, stalemate, etc.)
- **Coordinate System**: File and rank coordinates displayed on the board

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Note: `tkinter` is usually included with Python installations, but if you encounter issues, you may need to install it separately for your system.

2. **Run the Simulator**:
   ```bash
   python chess_simulator.py
   ```

## Usage

### Basic Gameplay

1. **Making Moves**: Click on a piece to select it, then click on the destination square
2. **Pawn Promotion**: Pawns automatically promote to queens when reaching the opposite end
3. **Move Validation**: Only legal moves are allowed

### Controls

- **New Game**: Start a fresh game
- **Undo Move**: Undo the last move made
- **Reset Game**: Return to the initial board position
- **Save Game**: Save the current game to `chess_game.pgn`
- **Load Game**: Load a previously saved game from `chess_game.pgn`

### Game Information

- **Turn Indicator**: Shows whose turn it is (White/Black)
- **Game Status**: Displays current game state (Playing, Check, Checkmate, etc.)
- **Move Counter**: Shows the number of moves made
- **Move History**: Complete move history in PGN notation

## Technical Details

### Dependencies

- `python-chess==1.10.0`: Core chess engine and move validation
- `tkinter`: GUI framework (usually included with Python)

### Architecture

The simulator is built with a clean, modular design:

- **ChessSimulator Class**: Main application class handling GUI and game logic
- **Board Visualization**: Custom canvas-based board with Unicode piece symbols
- **Move Handling**: Integrated with python-chess for robust move validation
- **Game Persistence**: PGN format for saving and loading games

### Key Features

- **Efficient Move Validation**: Uses python-chess's built-in legal move generation
- **Clean UI**: Simple, intuitive interface without clutter
- **Complete Chess Rules**: Supports all standard chess rules including special moves
- **Extensible Design**: Easy to add new features or modify existing functionality

## File Structure

```
ChessFormer/
├── chess_simulator.py    # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Future Enhancements

Potential improvements that could be added:

- AI opponent integration
- Move hints and analysis
- Custom board themes
- Network multiplayer support
- Opening book integration
- Game analysis tools

## License

This project is open source and available under the MIT License. 