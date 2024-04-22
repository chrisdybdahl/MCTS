from keras.layers import Dense
from keras.models import Sequential

from HEX import Hex
from MCTS import MCTS, uct_score
from config import FILENAME


def train(episodes, size, record_freq, m, epochs):
    """
    Trains the MCTS

    :return:
    """
    # Parameters for neural network
    num_features = size ** 2 + 1  # Number of input features - n ^ 2 board squares + 1 current player indicator
    num_classes = size ** 2  # Number of output probabilities - n ^ 2 board squares

    # Initialize the neural network model
    # TODO: Change how one chooses the NN parameters and activation functions
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])

    # Compile the model using KL divergence as the loss
    # Measuring how one probability distribution diverges from a second, expected probability distribution
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['accuracy'])

    # Initialize statistics
    last_20_winners = []
    player_1_wins = 0
    player_2_wins = 0

    # Initialize the Monte Carlo Tree Search class
    mcts = MCTS(uct_score)

    # Run episodes
    for episode in range(episodes):
        # Initialize new game
        game = Hex(size, size)

        if episode % record_freq == 0:
            # TODO: Record neural network? See task description
            pass

        while not game.get_win_state():
            # Run Monte Carlo Tree Search from the current game
            mc_action, mc_action_probabilities = mcts.run(game, m=m, value_lambda=0.2)  # Combining the evaluations
            game.do_action(mc_action)

            # Train actor neural network
            game_state = game.get_board_state()
            model.fit(game_state, mc_action_probabilities, epochs=epochs)

        # Record statistics
        if game.get_win_state() == 1:
            player_1_wins += 1
        else:
            player_2_wins += 1

        last_20_winners.append(game.get_win_state())
        if len(last_20_winners) > 0:
            last_20_winners.pop(0)

    with open(f'{FILENAME}.txt', 'w') as f:
        f.write(f'Player 1 wins: {player_1_wins}\n')
        f.write(f'Player 2 wins: {player_2_wins}\n')
        f.write(f'Last 20 winners: {last_20_winners}\n')


if __name__ == '__main__':
    # Make predictions (outputting probabilities)
    train(10, 3, 2, 100, 2)
