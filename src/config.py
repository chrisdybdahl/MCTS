# Path name parameters
FILENAME = 'MCTS_results'
FOLDER_NAME = 'trained_models'  # Global parameter

# Training - Hex game parameters
HEX_STARTING_PLAYER = 1
HEX_HEIGHT = 5
HEX_WIDTH = 5

# Training - Nim game parameters
NIM_STARTING_PLAYER = 1
NIM_N = 10
NIM_K = 5

# Training - Monte Carlo parameters
M = 10
C = 1
ROLLOUT_EPSILON = 0.8  # Probability p to make random move and (1 - p) to make move from actor in rollout game
EPSILON_DECAY = 0.99  # Global parameter
MIN_EPSILON = 0.1  # Global parameter
TIMELIMIT = 10
D_POLICY = 'stochastic'  # Global parameter - 'greedy' vs. 'stochastic'

# Training - RL system parameters
EPISODES = 200
RECORD_FREQUENCY = 50
BATCH_SIZE = 20
VISUALIZE = False
VERBOSE = 0
LAST_WINNERS_NUM = 20

# Training - General neural network parameters
OPTIMIZER = 'adam'
METRICS = ['accuracy']
LEARNING_RATE = None

# Training - Actor neural network parameters
ACTOR_EPOCHS = 50
ACTOR_LAYERS = [(128, 'relu'), (128, 'sigmoid'), (HEX_HEIGHT * HEX_WIDTH, 'softmax')]
ACTOR_LOSS_FUNCTION = 'kl_divergence'

# Training - Critic neural network parameters
CRITIC_EPOCHS = 50
CRITIC_LAYERS = [(128, 'relu'), (128, 'sigmoid'), (1, 'tanh')]
CRITIC_LOSS_FUNCTION = 'mse'
STATE_DISCOUNT = 1
EVAL_EPSILON = 1  # Probability p to evaluate using rollout and (1 - p) using critic if it exists

# Offline tournament - Parameters
TOURNAMENT_VISUALIZE = False

# Online tournament - Parameters
CLIENT_D_POLICY = 'greedy'  # 'greedy' vs. 'stochastic'
CLIENT_50_PATH = f'{FOLDER_NAME}\\actor_model_49_50.keras'
TOKEN = '986b618d6bc948f59ab4a8aab319ddd4'  # Global parameter
