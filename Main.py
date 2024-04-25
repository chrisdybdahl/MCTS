import numpy as np

from Hex import Hex
from MCTS import MCTS
from NeuralNet import NeuralNet
from Nim import Nim
from TwoPlayerGame import TwoPlayerGame
from config import FILENAME, EPISODES, RECORD_FREQUENCY, M, ACTOR_EPOCHS, ACTOR_LAYERS, CRITIC_EPOCHS, CRITIC_LAYERS, \
    HEX_HEIGHT, HEX_WIDTH, NIM_N, NIM_K, BATCH_SIZE, STATE_DISCOUNT, VERBOSE, OPTIMIZER, ACTOR_LOSS_FUNCTION, METRICS, \
    CRITIC_LOSS_FUNCTION, LAST_WINNERS_NUM, ROLLOUT_EPSILON, TIMELIMIT, C, EVAL_LAMBDA
from helpers import uct_score, minibatch_indices


def train_mcts(game: TwoPlayerGame, m: int, episodes: int, record_freq: int, batch_size: int, filename: str,
               actor_epochs: int, actor_net: NeuralNet, c: float = 1, rollout_epsilon: float = 1, timelimit: int = 100,
               verbose: int = 0, last_winners_num: int = 20,
               critic_epochs: int = None, critic_net: NeuralNet = None, state_discount: float = 1, **critic_param):
    """
    Trains the MCTS

    :param game: game to train on
    :param m: number of monte carlo simulations
    :param episodes: number of actual games to play
    :param record_freq: frequency to record neural network weights
    :param batch_size: size of random batch of actual cases to train actor neural network
    :param filename: filepath to results
    :param actor_epochs: number of epochs to train actor neural network
    :param actor_net: actor neural network
    :param c: parameter for tree policy
    :param rollout_epsilon: epsilon for epsilon greedy strategy to predict action probabilities
    :param timelimit: timelimit for monte carlo tree search
    :param verbose: verbosity level
    :param last_winners_num: number of last winners to record
    :param critic_epochs: number of epochs to train critic neural network
    :param critic_net: critic neural network
    :param state_discount: discount factor for state values to train critic neural network
    """
    # Initialize replay buffer
    replay_buffer = [[], []]

    # Initialize statistics
    last_winners = []
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

            # Run Monte Carlo Tree Search from the current game
            # TODO: improve rollout_epsilon over time making rollouts less likely
            # TODO: improve eval_epsilon over time to make more
            mc_action, mc_action_probabilities = mcts.run(game, m=m, actor_net=actor_net, c=c,
                                                          rollout_epsilon=rollout_epsilon, timelimit=timelimit,
                                                          critic_net=critic_net, **critic_param)

            # Record case in replay buffer
            game_state = game.get_board_state()
            episode_states.append(game_state)
            replay_buffer[0].append(game_state)
            replay_buffer[1].append(mc_action_probabilities)

            game.do_action(mc_action)

        # Extract game result
        game_result = game.get_win_state()

        # Retrieve random minibatch of cases for actor
        states, target_probabilities = minibatch_indices(replay_buffer, batch_size)

        # Train actor neural network
        actor_net.fit(states, target_probabilities, actor_epochs, verbose)

        # Train critic neural network TODO: check if critic neural network is correct
        if critic_net:
            # Create list of (optional) discounted states
            target_discounted_results = game_result * state_discount ** np.asarray(range(num_moves, 0, -1))

            critic_net.fit(np.asarray(episode_states), target_discounted_results, critic_epochs, verbose)

        print(f'state: {states[0]}')
        print(f'target probability = {target_probabilities[0]}')
        print(f'estimated probability: {actor_net.predict(np.array([states[0]]), 0)[0]}')
        print(f'estimated evaluations: {critic_net.predict(np.array([states[0]]), 0)[0]}')

        # Save neural network weights with given frequency
        if episode % record_freq == 0:
            actor_net.save_weights(f'trained_models/actor_model_{episode}.weights.h5', True)

            if critic_net:
                critic_net.save_weights(f'trained_models/critic_weights_{episode}.weights.h5', True)

        # Record statistics
        if game_result == 1:
            player_1_wins += 1
        else:
            player_2_wins += 1

        last_winners.append(game_result)
        if len(last_winners) > last_winners_num:
            last_winners.pop(0)

    with open(f'{filename}.txt', 'w') as f:
        f.write(f'Player 1 wins: {player_1_wins}\n')
        f.write(f'Player 2 wins: {player_2_wins}\n')
        f.write(f'Last 20 winners: {last_winners}\n')


if __name__ == '__main__':
    # Make predictions (outputting probabilities)
    hex_game = Hex(HEX_HEIGHT, HEX_WIDTH)
    nim_game = Nim(NIM_N, NIM_K)

    actor_net = NeuralNet(ACTOR_LAYERS, OPTIMIZER, ACTOR_LOSS_FUNCTION, METRICS)
    critic_net = NeuralNet(CRITIC_LAYERS, OPTIMIZER, CRITIC_LOSS_FUNCTION, METRICS)

    train_mcts(hex_game, M, EPISODES, RECORD_FREQUENCY, BATCH_SIZE, FILENAME, ACTOR_EPOCHS, actor_net, C,
               ROLLOUT_EPSILON, TIMELIMIT, VERBOSE, LAST_WINNERS_NUM, CRITIC_EPOCHS, critic_net, STATE_DISCOUNT,
               eval_lambda=EVAL_LAMBDA)
