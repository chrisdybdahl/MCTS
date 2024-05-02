import random
import time
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from NeuralNet import NeuralNet
from Node import Node
from TwoPlayerGame import TwoPlayerGame
from config import D_POLICY
from helpers import choose_actor_action


class MCTS:
    def __init__(self, tree_policy_func: Callable[[Node, float], tuple[Node, ...]]):
        """
        Initialize MCTS parameters here

        :param tree_policy_func: tree policy function
        """
        self.tree_policy_func = tree_policy_func
        self.M = None

    @staticmethod
    def rollout(node: Node, actor_net: NeuralNet, epsilon: float) -> float:
        """
        Rollout simulation from node

        :param node: node to run a rollout simulation from
        :param actor_net: actor neural network
        :param epsilon: epsilon for epsilon greedy choice between random and neural network action
        :return: game result from rollout simulation
        """
        rollout_game = node.get_game()

        # Get all possible actions
        possible_actions = rollout_game.get_all_actions()

        while not rollout_game.get_win_state():
            # Get the valid actions
            valid_actions = rollout_game.get_actions()

            # Probability p to make random move and (1 - p) to make move from actor in rollout game
            if random.random() > epsilon:
                # Extract game state
                game_state = rollout_game.get_board_state()

                # Estimate probabilities using actor neural network, either greedily or stochastically
                action = choose_actor_action(actor_net, possible_actions, valid_actions, game_state, D_POLICY)
            else:
                action = random.choice(valid_actions)
            rollout_game.do_action(action)

        game_win_state = rollout_game.get_win_state()

        # Assume all evaluations are from the perspective of player_1
        if game_win_state == 1:  # If player_1 won return 1
            return 1
        else:  # If player_2 won return -1
            return -1

    @staticmethod
    def critic(node: Node, critic_net: NeuralNet) -> float:
        """
        Produce evaluation of leaf node using the critic-RL neural network

        :param node: node to evaluate
        :param critic_net:
        :return: leaf evaluation
        """
        critic_game = node.get_game()
        game_state = critic_game.get_board_state()
        critic_evaluation = critic_net.predict(np.array([game_state]), verbose=0)[0]
        return critic_evaluation

    def run(self, game: TwoPlayerGame, m: int, actor_net: NeuralNet, c: float = 1, rollout_epsilon: float = 1,
            timelimit: int = None, critic_net: NeuralNet = None, eval_epsilon: float = 1) \
            -> tuple[tuple[int, int], NDArray]:
        """
        Run the MCTS

        :param game:
        :param m:
        :param actor_net:
        :param c:
        :param rollout_epsilon:
        :param timelimit:
        :param critic_net:
        :param eval_epsilon:
        :return: best action
        """

        # Start timer
        timer = time.time()

        # Initialize new node with the given game
        root_node = Node(game)

        tree_time = 0
        expansion_time = 0
        evaluation_time = 0
        backpropogation_time = 0

        for i in range(m):
            # print(i)
            timer_i = time.time_ns()
            # Tree search - choose node from root to leaf node using tree policy
            policy_node, policy_action = self.tree_policy_func(root_node, c)
            tree_time += (time.time_ns() - timer_i)
            # print(f'tree search: {time.time_ns() - timer_i}')

            timer_i = time.time_ns()
            # Node expansion - generate a child state of the parent state
            children_dict = policy_node.get_children_dict()

            """valid_actions = policy_node.get_valid_actions()
            untried_actions = [action for action in valid_actions if action not in children_dict.keys()]
            if untried_actions:
                action = random.choice(untried_actions)
                expanded_node = policy_node.expand(action)
            else:
                action = policy_action
                expanded_node = children_dict[action]
            """

            if policy_action in children_dict.keys():
                expanded_node = children_dict[policy_action]
            else:
                expanded_node = policy_node.expand(policy_action)
            expansion_time += (time.time_ns() - timer_i)
            # print(f'node expansion: {time.time_ns() - timer_i}')

            timer_i = time.time_ns()
            # Leaf evaluation - estimate the value of a leaf node using the default policy
            if critic_net:  # Probability p to evaluate using rollout and (1 - p) using critic if it exists
                if random.random() > eval_epsilon:
                    leaf_evaluation = self.critic(expanded_node, critic_net)
                else:
                    leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)
            else:
                leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)
            evaluation_time += (time.time_ns() - timer_i)
            # print(f'leaf evaluation: {time.time_ns() - timer_i}')

            timer_i = time.time_ns()
            # Backpropagation - passing the evaluation back up the tree, updating relevant data
            expanded_node.update(leaf_evaluation)
            backpropogation_time += (time.time_ns() - timer_i)
            # print(f'backpropagation: {time.time_ns() - timer_i}')

            # Stop monte carlo search if runtime exceeds timelimit
            if timelimit and time.time() - timer > timelimit:
                print(f'Timed out in iteration {i + 1}/{m}, after {time.time() - timer} seconds')
                break

        print(f'Tree search time: {tree_time} seconds')
        print(f'Expansion time: {expansion_time} seconds')
        print(f'Evaluation time: {evaluation_time} seconds')
        print(f'Backpropogation time: {backpropogation_time} seconds')

        # Choose best action from the root by the highest visit count
        best_child = max(root_node.get_children(), key=lambda child: child.get_visits())
        best_action = best_child.get_parent_action()

        # Action probabilities from the root
        all_actions = game.get_all_actions()
        valid_actions = game.get_actions()
        children_dict = root_node.get_children_dict()

        all_actions_visits = [
            children_dict[action].get_visits() if action in children_dict.keys() and action in valid_actions else 0 for
            action in all_actions]
        action_probabilities = np.divide(all_actions_visits, root_node.get_visits())

        return best_action, action_probabilities
