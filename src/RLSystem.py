import numpy as np

from Hex import Hex
from MCTS import MCTS
from NeuralNet import NeuralNet
from Nim import Nim
from TwoPlayerGame import TwoPlayerGame
from config import (FILENAME, EPISODES, RECORD_FREQUENCY, M, ACTOR_EPOCHS, ACTOR_LAYERS, CRITIC_EPOCHS,
                    CRITIC_LAYERS, HEX_HEIGHT, HEX_WIDTH, NIM_N, NIM_K, BATCH_SIZE, STATE_DISCOUNT, VERBOSE,
                    OPTIMIZER, ACTOR_LOSS_FUNCTION, METRICS, CRITIC_LOSS_FUNCTION, LAST_WINNERS_NUM,
                    ROLLOUT_EPSILON, TIMELIMIT, C, EPSILON_DECAY, MIN_EPSILON, EVAL_EPSILON, VISUALIZE,
                    LEARNING_RATE, HEX_STARTING_PLAYER, NIM_STARTING_PLAYER, FOLDER_NAME)
from helpers import uct_score, minibatch_indices


def rl_mcts(game: TwoPlayerGame, m: int, episodes: int, record_freq: int, batch_size: int, filename: str,
            actor_epochs: int, actor_net: NeuralNet, c: float = 1, rollout_epsilon: float = 1, timelimit: int = 10,
            visualize: bool = False, verbose: int = 0, last_winners_num: int = 20, critic_epochs: int = None,
            critic_net: NeuralNet = None, state_discount: float = 1, eval_epsilon: float = 1):
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
    :param rollout_epsilon: epsilon for epsilon greedy strategy to execute action in rollouts based on actor net
    :param timelimit: timelimit for monte carlo tree search
    :param visualize: whether to visualize game after each actual move
    :param verbose: verbosity level
    :param last_winners_num: number of last winners to record
    :param critic_epochs: number of epochs to train critic neural network
    :param critic_net: critic neural network
    :param state_discount: discount factor for state values to train critic neural network
    :param eval_epsilon: epsilon for epsilon greedy strategy to estimate evaluation based on critic net
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
    for episode in range(episodes):
        # Initialize new game
        game.reset()

        # Initialize list of episode states and number of moves
        episode_states = []
        num_moves = 0

        # Improve epsilon over time to execute actions in rollouts more based on actor neural network and less random
        rollout_epsilon = max(rollout_epsilon * EPSILON_DECAY ** episode, MIN_EPSILON)

        # Improve epsilon over time to estimate evaluations more based on critic neural network and from rollouts
        eval_epsilon = max(eval_epsilon * EPSILON_DECAY ** episode, MIN_EPSILON)

        if visualize:
            print(f"Player_{game.get_current_player()}'s turn")
            game.visualize()

        while not game.get_win_state():
            num_moves += 1

            # Run Monte Carlo Tree Search from the current game
            mc_action, mc_action_probabilities = mcts.run(game, m=m, actor_net=actor_net, c=c,
                                                          rollout_epsilon=rollout_epsilon, timelimit=timelimit,
                                                          critic_net=critic_net, eval_epsilon=eval_epsilon)

            # Record case in replay buffer
            game_state = game.get_board_state()
            episode_states.append(game_state)
            replay_buffer[0].append(game_state)
            replay_buffer[1].append(mc_action_probabilities)

            game.do_action(mc_action)

            if visualize:
                print(f"Player_{game.get_current_player()}'s turn")
                game.visualize()

        # Extract game result
        winner = game.get_win_state()
        if winner == 1:
            game_result = 1
            player_1_wins += 1
        else:
            game_result = -1
            player_2_wins += 1

        # Retrieve random minibatch of cases for actor
        states, target_probabilities = minibatch_indices(replay_buffer, batch_size)

        # Train actor neural network
        actor_net.fit(states, target_probabilities, actor_epochs, verbose)

        # Train critic neural network
        if critic_net:
            # Create list of (optional) discounted states
            target_discounted_results = game_result * state_discount ** np.asarray(range(num_moves, 0, -1))

            critic_net.fit(np.asarray(episode_states), target_discounted_results, critic_epochs, verbose)

            print(f'estimated evaluations: {critic_net.predict(np.array([episode_states[0]]), verbose=0)[0]}')
            print(f'target evaluations: {target_discounted_results[0]}')

        print(f'state: {states[0]}')
        print(f'target probability = {target_probabilities[0]}')
        print(f'estimated probability: {actor_net.predict(np.array([states[0]]), verbose=0)[0]}')

        # Save neural network weights with given frequency
        if episode + 1 % record_freq == 0:
            actor_net.save(path=f'{FOLDER_NAME}/actor_model_{episode + 1}.keras', overwrite=True)

            if critic_net:
                critic_net.save(path=f'{FOLDER_NAME}/critic_model_{episode + 1}.keras', overwrite=True)

        last_winners.append(winner)
        if len(last_winners) > last_winners_num:
            last_winners.pop(0)

    with open(f'{filename}.txt', 'w') as f:
        f.write(f'Player 1 wins: {player_1_wins}\n')
        f.write(f'Player 2 wins: {player_2_wins}\n')
        f.write(f'Last 20 winners: {last_winners}\n')


if __name__ == '__main__':
    hex_game = Hex(HEX_STARTING_PLAYER, HEX_HEIGHT, HEX_WIDTH)
    nim_game = Nim(NIM_STARTING_PLAYER, NIM_N, NIM_K)

    actor_net = NeuralNet(None, ACTOR_LAYERS, OPTIMIZER, ACTOR_LOSS_FUNCTION, LEARNING_RATE, METRICS)
    critic_net = NeuralNet(None, CRITIC_LAYERS, OPTIMIZER, CRITIC_LOSS_FUNCTION, LEARNING_RATE, METRICS)

    rl_mcts(hex_game, M, EPISODES, RECORD_FREQUENCY, BATCH_SIZE, FILENAME, ACTOR_EPOCHS, actor_net, C, ROLLOUT_EPSILON,
            TIMELIMIT, VISUALIZE, VERBOSE, LAST_WINNERS_NUM, CRITIC_EPOCHS, critic_net, STATE_DISCOUNT, EVAL_EPSILON)
