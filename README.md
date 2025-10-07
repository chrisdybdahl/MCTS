# Monte Carlo Tree Search (MCTS) Implementation

This project provides a comprehensive Python implementation of the Monte Carlo Tree Search algorithm with neural network integration for training AI agents to play board games Hex and Nim competitively.

## Features

- **Monte Carlo Tree Search**: UCB1-based tree policy with neural network guided rollouts
- **Neural Network Integration**: Actor and Critic networks for move prediction and position evaluation
- **Reinforcement Learning**: Self-play training with experience replay and epsilon decay
- **Multiple Games**: Hex (connection game) and Nim (subtraction game) implementations
- **Tournament System**: Automated model evaluation and comparison
- **Online Play**: Client for connecting to competitive gaming servers

## Project Structure

- `src/MCTS.py` — Core Monte Carlo Tree Search algorithm implementation
- `src/RLSystem.py` — Reinforcement learning training system and self-play logic
- `src/NeuralNet.py` — TensorFlow/Keras neural network wrapper
- `src/Node.py` — MCTS tree node implementation with UCB1 scoring
- `src/TwoPlayerGame.py` — Abstract base class for two-player games
- `src/Hex.py` — Hex game implementation with hexagonal board
- `src/Nim.py` — Nim game implementation
- `src/Board.py` — Hex board representation and visualization
- `src/Tournament.py` — Model evaluation and tournament system
- `src/MyHexActor.py` — Hex-specific actor wrapper for online play
- `src/MyActorClient.py` — Online tournament client
- `src/config.py` — Centralized configuration for all parameters
- `src/helpers.py` — Utility functions for UCT scoring and action selection
- `ActorClient.py` — Base client for server communication
- `play_online.py` — Script for online tournament participation
- `server.crt` — SSL certificate for server connection

## Games Implemented

### Hex
- **Objective**: Connect opposite sides of the board with your pieces
- **Board**: Hexagonal grid (default 4x4, configurable)
- **Players**: 2 (Player 1: top-to-bottom, Player 2: left-to-right)
- **Win Condition**: First player to create an unbroken chain wins

### Nim
- **Objective**: Force opponent to take the last piece
- **Setup**: Start with N pieces, players take 1-K pieces per turn
- **Strategy**: Classic combinatorial game theory example

## Requirements

- **Python**: ^3.11
- **TensorFlow**: ^2.20.0 (neural networks)
- **NumPy**: ^2.3.3 (numerical computations)
- **Matplotlib**: ^3.10.6 (visualization)

## Installation

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd monte-carlo-tree-search
   ```

2. **Install dependencies using Poetry:**
   ```powershell
   poetry install
   ```

   Or using pip:
   ```powershell
   pip install tensorflow numpy matplotlib
   ```

## Usage

### Quick Start

Run training to see the MCTS algorithm in action:

```powershell
# Train an agent on Hex
python src/RLSystem.py

# Play online with a trained model
python play_online.py
```

### Configuration

All simulation parameters are centralized in `src/config.py`:

```python
# Game parameters
HEX_HEIGHT = 4                    # Hex board height
HEX_WIDTH = 4                     # Hex board width  
NIM_N = 10                        # Initial pieces in Nim
NIM_K = 3                         # Maximum pieces per turn

# MCTS parameters
M = 100                           # Simulations per move
C = 1.4                          # UCB1 exploration constant
TIMELIMIT = 10                   # Time limit per move (seconds)

# Training parameters
EPISODES = 8                      # Self-play episodes
BATCH_SIZE = 200                 # Neural network batch size
ACTOR_EPOCHS = 50                # Training epochs per update
```

### Example Output

Each training session produces:
- **Model Checkpoints**: Saved in `trained_models/` directory
- **Training Statistics**: Win rates and performance metrics
- **Console Output**: Real-time training progress and timing information

## Algorithm Details

### Monte Carlo Tree Search (MCTS)
1. **Selection**: Navigate tree using UCB1 formula: `Q + C * √(ln(parent_visits) / visits)`
2. **Expansion**: Add new child nodes to the tree
3. **Simulation**: Rollout games using neural network guidance
4. **Backpropagation**: Update node statistics back to root

### Neural Network Integration
- **Actor Network**: Maps game states to action probability distributions
- **Critic Network**: Evaluates position strength (optional)
- **Training**: Self-play with experience replay and epsilon-greedy exploration

## Extending the Project

- Add new games by implementing the `TwoPlayerGame` interface
- Experiment with different neural network architectures in `src/config.py`
- Implement new tree policies by modifying the UCB1 formula in `src/helpers.py`
- Add new regression approaches for rollout policies

## Development

When adding new features:

1. **New Games**: Inherit from `TwoPlayerGame` base class
2. **New Controllers**: Modify the neural network architecture in `config.py`
3. **New Algorithms**: Extend the MCTS implementation in `src/MCTS.py`
4. **Dependencies**: Update using `poetry add <package>` or `pip install <package>`

## Dependencies

- **Python**: 3.11+
- **TensorFlow**: 2.20.0 (neural networks)
- **NumPy**: 2.3.3 (numerical computations)
- **Matplotlib**: 3.10.6 (visualization)

Install with: `poetry install`

## Notes

- This implementation is designed for educational and research purposes
- For production use, consider optimizing the neural network training pipeline
- The online client requires a valid SSL certificate and authentication token
- Trained models are saved in the `trained_models/` directory
