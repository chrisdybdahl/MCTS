import numpy as np
import tensorflow as tf

from Hex import Hex
from MCTS import MCTS
from Nim import Nim
from TwoPlayerGame import TwoPlayerGame
from config import FILENAME, EPISODES, RECORD_FREQUENCY, M, ACTOR_EPOCHS, ACTOR_LAYERS, CRITIC_EPOCHS, CRITIC_LAYERS, \
    HEX_HEIGHT, HEX_WIDTH, NIM_N, NIM_K, BATCH_SIZE, STATE_DISCOUNT, VERBOSE
from helpers import uct_score, minibatch_indices


def train_mcts(game: TwoPlayerGame, m: int, episodes: int, record_freq: int, batch_size: int,
               actor_epochs: int, actor_layers: list[tuple[int, str]],
               critic_epochs: int, critic_layers: list[tuple[int, str]],
               state_discount: float = 1, verbose: int = 0):
    """
    Trains the MCTS

    :param verbose:
    :param game: game to train on
    :param m: number of monte carlo simulations
    :param episodes: number of actual games to play
    :param record_freq: frequency to record neural network weights
    :param batch_size: size of random batch of actual cases to train actor neural network
    :param actor_epochs: number of epochs to train actor neural network
    :param actor_layers: layers for actor neural network
    :param critic_epochs: number of epochs to train critic neural network
    :param critic_layers: layers for critic neural network
    :param state_discount: discount factor for state values to train critic neural network
    """
    # Initialize replay buffer
    replay_buffer = [[], []]

    # Number of outputs for actor neural network
    num_actions = len(game.get_all_actions())

    # Initialize the actor neural network model
    # TODO: Same neural net for actor as for critic?
    # TODO: Create own class to initialize neural nets
    actor_model = tf.keras.models.Sequential()
    for layer, activation_func in actor_layers:
        actor_model.add(tf.keras.layers.Dense(layer, activation=activation_func))
    actor_model.add(tf.keras.layers.Dense(num_actions, activation='softmax'))
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
    for episode in range(1, episodes + 1):
        # Initialize new game
        game.reset()

        # Initialize list of episode states and number of moves
        episode_states = []
        num_moves = 0
        while not game.get_win_state():
            num_moves += 1

            # Run Monte Carlo Tree Search from the current game TODO: Change how one chooses the evaluation policy
            mc_action, mc_action_probabilities = mcts.run(game, actor_net=actor_model, critic_net=critic_model, m=m,
                                                          c=1.4, rollout_epsilon=0.5, value_lambda=0.2)

            # Record case in replay buffer
            game_state = game.get_board_state()
            episode_states.append(game_state)
            replay_buffer[0].append(game_state)
            replay_buffer[1].append(mc_action_probabilities)

            game.do_action(mc_action)

        # Extract game result
        game_result = game.get_win_state()

        # Create list of (optional) discounted states
        target_discounted_results = game_result * state_discount ** np.asarray(range(num_moves, 0, -1))

        # Retrieve random minibatch of cases for actor
        states, target_probabilities = minibatch_indices(replay_buffer, batch_size)

        print(f'state: {states[0]}')
        print(f'target probability = {target_probabilities[0]}')
        print(f'estimated probability: {actor_model.predict(np.array([states[0]]), verbose=0)[0]}')

        # Train actor neural network
        actor_model.fit(states, target_probabilities, epochs=actor_epochs, verbose=verbose)

        # Train critic neural network
        critic_model.fit(np.asarray(episode_states), target_discounted_results, epochs=critic_epochs, verbose=verbose)

        # Save neural network weights with given frequency
        if episode % record_freq == 0:
            actor_model.save_weights(f'trained_models/actor_model_{episode}.weights.h5', overwrite=True)
            critic_model.save_weights(f'trained_models/critic_weights_{episode}.weights.h5', overwrite=True)

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
    Hex_game = Hex(HEX_HEIGHT, HEX_WIDTH)
    Nim_game = Nim(NIM_N, NIM_K)
    train_mcts(Hex_game, M, EPISODES, RECORD_FREQUENCY, BATCH_SIZE, ACTOR_EPOCHS, ACTOR_LAYERS, CRITIC_EPOCHS,
               CRITIC_LAYERS, STATE_DISCOUNT, VERBOSE)
