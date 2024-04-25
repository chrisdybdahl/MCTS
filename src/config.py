# Path name parameters
FILENAME = 'MCTS_results'
FOLDER_NAME = 'trained_models'  # Global parameter

# Training - Hex game parameters
HEX_STARTING_PLAYER = 1
HEX_HEIGHT = 7
HEX_WIDTH = 7

# Training - Nim game parameters
NIM_STARTING_PLAYER = 1
NIM_N = 10
NIM_K = 5

# Training - Monte Carlo parameters
M = 5
C = 1
ROLLOUT_EPSILON = 1
EPSILON_DECAY = 0.99  # Global parameter
MIN_EPSILON = 0.2  # Global parameter
TIMELIMIT = 10
D_POLICY = 'greedy'  # Global parameter - 'greedy' vs. 'stochastic'

# Training - RL system parameters
EPISODES = 50
RECORD_FREQUENCY = 50
BATCH_SIZE = 20
VISUALIZE = True
VERBOSE = 2
LAST_WINNERS_NUM = 20

# Training - General neural network parameters
OPTIMIZER = 'adam'
METRICS = ['accuracy']
LEARNING_RATE = None

# Training - Actor neural network parameters
ACTOR_EPOCHS = 100
ACTOR_LAYERS = [(128, 'relu'), (128, 'sigmoid'), (HEX_HEIGHT * HEX_WIDTH, 'softmax')]
ACTOR_LOSS_FUNCTION = 'kl_divergence'

# Training - Critic neural network parameters
CRITIC_EPOCHS = 80
CRITIC_LAYERS = [(128, 'relu'), (128, 'sigmoid'), (1, 'tanh')]
CRITIC_LOSS_FUNCTION = 'mse'
STATE_DISCOUNT = 1
EVAL_EPSILON = 1

# Tournament - Parameters
CLIENT_OPTIMIZER = 'adam'
CLIENT_METRICS = ['accuracy']
CLIENT_LEARNING_RATE = None
CLIENT_LAYERS = [(128, 'relu'), (128, 'sigmoid'), (HEX_HEIGHT * HEX_WIDTH, 'softmax')]
CLIENT_LOSS_FUNCTION = 'kl_divergence'
CLIENT_D_POLICY = 'greedy'  # 'greedy' vs. 'stochastic'
CLIENT_50_PATH = f'{FOLDER_NAME}/actor_model_50.keras'