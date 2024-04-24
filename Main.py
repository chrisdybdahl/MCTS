import numpy as np
import tensorflow as tf

from HEX import Hex
from MCTS import MCTS, uct_score
from config import FILENAME


def train_hex(episodes, size, record_freq, m, actor_epochs, actor_layers: list[tuple[int, str]], critic_epochs,
              critic_layers: list[tuple[int, str]]):
    """
    Trains the MCTS

    :return:
    """
    # Parameters for actor neural network
    num_classes = size ** 2  # Number of output probabilities - n ^ 2 board squares

    # Initialize the actor neural network model
    # TODO: Same neural net for actor as for critic?
    actor_model = tf.keras.models.Sequential()
    for layer, activation_func in actor_layers:
        actor_model.add(tf.keras.layers.Dense(layer, activation=activation_func))
    actor_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    actor_model.compile(optimizer='adam', loss='kl_divergence', metrics=['accuracy'])  # Using KL divergence as the loss

    # Initialize the critic neural network model
    critic_model = tf.keras.models.Sequential()
    for layer, activation_func in critic_layers:
        critic_model.add(tf.keras.layers.Dense(layer, activation=activation_func))
    critic_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    critic_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Using MSE as the loss

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
            # TODO: Train critic NN
            mc_action, mc_action_probabilities = mcts.run(game, critic_net=critic_model, m=m, c=1.4, value_lambda=0.2)
            print(f'Execute move {mc_action} in game')
            game.do_action(mc_action)
            print(f'MC Action Probabilities: {mc_action_probabilities}')

            # Train actor neural network
            game_state = game.get_board_state()
            actor_model.fit(game_state, mc_action_probabilities, epochs=actor_epochs, verbose=2)

        # Extract game result
        game_result = game.get_win_state()

        # Train critic neural network
        game_state = game.get_board_state()
        critic_model.fit(game_state, np.array([[game_result]]), epochs=critic_epochs, verbose=2)

        # Record statistics
        if game_result == 1:
            player_1_wins += 1
        else:
            player_2_wins += 1

        last_20_winners.append(game_result)
        if len(last_20_winners) > 20:
            last_20_winners.pop(0)

    with open(f'{FILENAME}.txt', 'w') as f:
        f.write(f'Player 1 wins: {player_1_wins}\n')
        f.write(f'Player 2 wins: {player_2_wins}\n')
        f.write(f'Last 20 winners: {last_20_winners}\n')


if __name__ == '__main__':
    # Make predictions (outputting probabilities)
    actor_layers = [(32, 'sigmoid'), (32, 'sigmoid')]
    critic_layers = [(32, 'sigmoid')]
    train_hex(10, 3, 2, 10, 100, actor_layers, 50, critic_layers)
