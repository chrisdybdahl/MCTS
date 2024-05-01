import os
import time

from Hex import Hex
from MCTS import MCTS
from NeuralNet import NeuralNet
from Nim import Nim
from TwoPlayerGame import TwoPlayerGame
from config import (FILENAME, EPISODES, RECORD_FREQUENCY, M, ACTOR_EPOCHS, ACTOR_LAYERS, CRITIC_EPOCHS,
                    CRITIC_LAYERS, HEX_HEIGHT, HEX_WIDTH, NIM_N, NIM_K, BATCH_SIZE, STATE_DISCOUNT, VERBOSE,
                    OPTIMIZER, ACTOR_LOSS_FUNCTION, METRICS, CRITIC_LOSS_FUNCTION, LAST_WINNERS_NUM,
                    ROLLOUT_EPSILON, TIMELIMIT, C, ROLLOUT_EPSILON_DECAY, MIN_ROLLOUT_EPSILON, EVAL_EPSILON, VISUALIZE,
                    ACTOR_LEARNING_RATE, HEX_STARTING_PLAYER, NIM_STARTING_PLAYER, FOLDER_NAME, CRITIC_LEARNING_RATE,
                    EVAL_EPSILON_DECAY, MIN_EVAL_EPSILON, ACTOR_PATH, CRITIC_PATH)
from helpers import uct_score, minibatch_indices

folder_path = f'{os.getcwd()}\\{FOLDER_NAME}'


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
    replay_buffer_actor = [[], []]
    replay_buffer_critic = [[], []]

    # Initialize statistics
    last_winners = []
    player_1_wins = 0
    player_2_wins = 0

    # Initialize the Monte Carlo Tree Search class
    mcts = MCTS(uct_score)

    # Run episodes
    for episode in range(episodes):
        # Start timer
        timer = time.time()

        print(f'Episode: {episode + 1}/{episodes}')

        # Initialize new game
        game.reset()

        # Initialize list number of moves
        num_moves = 0

        # Improve epsilon over time to execute actions in rollouts more based on actor neural network and less random
        rollout_epsilon = max(rollout_epsilon * ROLLOUT_EPSILON_DECAY ** episode, MIN_ROLLOUT_EPSILON)

        # Improve epsilon over time to estimate evaluations more based on critic neural network and from rollouts
        eval_epsilon = max(eval_epsilon * EVAL_EPSILON_DECAY ** episode, MIN_EVAL_EPSILON)

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
            replay_buffer_actor[0].append(game_state)
            replay_buffer_actor[1].append(mc_action_probabilities)
            if critic_net:
                replay_buffer_critic[0].append(game_state)

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
        states, target_probabilities = minibatch_indices(replay_buffer_actor, batch_size)

        # Train actor neural network
        actor_net.fit(states, target_probabilities, actor_epochs, verbose)

        if critic_net:
            # Create list of discounted states
            [replay_buffer_critic[1].append([game_result * state_discount ** episode_left])
             for episode_left in range(num_moves - 1, -1, -1)]

            # Retrieve random minibatch of cases for critic
            states, target_discounted_results = minibatch_indices(replay_buffer_critic, batch_size)

            # Train critic neural network
            critic_net.fit(states, target_discounted_results, critic_epochs, verbose)

        # Save neural network weights with given frequency
        if episode == 0 or (episode + 1) % record_freq == 0:
            num_moves = len(game.get_all_actions())
            print(f'Saving actor model to: {folder_path}\\actor_model_{num_moves}_{episode + 1}.keras')
            actor_net.save(path=f'{folder_path}\\actor_model_{num_moves}_{episode + 1}.keras', overwrite=True)

            if critic_net:
                print(
                    f'Saving critic model to: {folder_path}\\critic_model_{num_moves}_{episode + 1}.keras')
                critic_net.save(path=f'{folder_path}\\critic_model_{num_moves}_{episode + 1}.keras', overwrite=True)

        last_winners.append(winner)
        if len(last_winners) > last_winners_num:
            last_winners.pop(0)

        print(f'Runtime: {time.time() - timer} seconds')

    with open(f'{filename}.txt', 'w') as f:
        f.write(f'Player 1 wins: {player_1_wins}\n')
        f.write(f'Player 2 wins: {player_2_wins}\n')
        f.write(f'Last 20 winners: {last_winners}\n')


if __name__ == '__main__':
    hex_game = Hex(HEX_STARTING_PLAYER, HEX_HEIGHT, HEX_WIDTH)
    nim_game = Nim(NIM_STARTING_PLAYER, NIM_N, NIM_K)

    actor_net = NeuralNet(ACTOR_PATH, ACTOR_LAYERS, OPTIMIZER, ACTOR_LOSS_FUNCTION, ACTOR_LEARNING_RATE, METRICS)
    critic_net = NeuralNet(CRITIC_PATH, CRITIC_LAYERS, OPTIMIZER, CRITIC_LOSS_FUNCTION, CRITIC_LEARNING_RATE, METRICS)

    rl_mcts(hex_game, M, EPISODES, RECORD_FREQUENCY, BATCH_SIZE, FILENAME, ACTOR_EPOCHS, actor_net, C,
            ROLLOUT_EPSILON, TIMELIMIT, VISUALIZE, VERBOSE, LAST_WINNERS_NUM, CRITIC_EPOCHS, critic_net, STATE_DISCOUNT,
            EVAL_EPSILON)
