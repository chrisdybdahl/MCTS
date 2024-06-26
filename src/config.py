# Path name parameters
FILENAME = 'MCTS_results'
FOLDER_NAME = 'trained_models'  # Global parameter

# Training - Hex game parameters
HEX_STARTING_PLAYER = 1
HEX_HEIGHT = 4
HEX_WIDTH = 4

# Training - Nim game parameters
NIM_STARTING_PLAYER = 1
NIM_N = 10
NIM_K = 3

# Training - Monte Carlo parameters
M = 100
C = 1.4
TIMELIMIT = 10
ROLLOUT_EPSILON = 1  # Probability p to make random move and (1 - p) to make move from actor in rollout game
ROLLOUT_EPSILON_DECAY = 0.9  # Global parameter
MIN_ROLLOUT_EPSILON = 0.1  # Global parameter
D_POLICY = 'stochastic'  # Global parameter - 'greedy' vs. 'stochastic'

# Training - RL system parameters
EPISODES = 8
RECORD_FREQUENCY = 2
BATCH_SIZE = 200
VISUALIZE = False
VERBOSE = 0
LAST_WINNERS_NUM = 20
START = 0

# Training - General neural network parameters
OPTIMIZER = 'adam'
METRICS = ['accuracy', 'mae']

# Training - Actor neural network parameters
ACTOR_EPOCHS = 50
ACTOR_PATH = None
ACTOR_LAYERS = [(64, 'relu'), (64, 'relu'), (64, 'relu'), (HEX_HEIGHT * HEX_WIDTH, 'softmax')]
NIM_ACTOR_LAYERS = [(128, 'relu'), (128, 'relu'), (NIM_K, 'softmax')]
ACTOR_LOSS_FUNCTION = 'kl_divergence'
ACTOR_LEARNING_RATE = None

# Training - Critic neural network parameters
CRITIC_EPOCHS = 50
CRITIC_PATH = None
CRITIC_LAYERS = [(64, 'relu'), (64, 'relu'), (1, 'tanh')]
CRITIC_LOSS_FUNCTION = 'mse'
STATE_DISCOUNT = 0.95
EVAL_EPSILON = 1  # Probability p to evaluate using rollout and (1 - p) using critic if it exists
EVAL_EPSILON_DECAY = 0.95  # Global parameter
MIN_EVAL_EPSILON = 0.1  # Global parameter
CRITIC_LEARNING_RATE = None

# Offline tournament - Parameters
TOURNAMENT_FOLDER = 'topp_models'
NUM_MATCHES = 10
TOURNAMENT_VISUALIZE = False
TOURNAMENT_FILENAME = 'tournament_results'
TOURNAMENT_D_POLICY = 'greedy'

# Online tournament - Parameters
CLIENT_D_POLICY = 'greedy'  # 'greedy' vs. 'stochastic'
CLIENT_50_PATH = f'C:\\Users\\chris\\PycharmProjects\\MCTS\\src\\oht\\actor_model_49_10.keras'
TOKEN = '986b618d6bc948f59ab4a8aab319ddd4'  # Global parameter
CLIENT_VISUALIZE = False
