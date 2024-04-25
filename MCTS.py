import random
from typing import Callable

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from Node import Node
from TwoPlayerGame import TwoPlayerGame


class MCTS:
    def __init__(self, tree_policy_func: Callable[[Node, float], Node]):
        """
        Initialize MCTS parameters here

        :param tree_policy_func: tree policy function
        """
        self.tree_policy_func = tree_policy_func
        self.M = None

    @staticmethod
    def rollout(node: Node, actor_net: tf.keras.models, epsilon: float) -> float:
        """
        Rollout simulation from node

        :param node: node to run a rollout simulation from
        :param actor_net: actor neural network
        :param epsilon: epsilon for epsilon greedy choice between random and neural network action
        :return: game result from rollout simulation
        """
        rollout_game = node.get_game()

        # Get all possible actions
        all_possible_actions = rollout_game.get_all_actions()

        while not rollout_game.get_win_state():
            # Choose randomly among the possible valid actions
            possible_actions = rollout_game.get_actions()

            # Epsilon greedy choice
            if random.random() > epsilon:
                # Estimate probabilities using actor neural network
                game_state = rollout_game.get_board_state()
                probabilities = actor_net.predict(np.array([game_state]), verbose=0)[0]
                probabilities_valid = [probability if action in possible_actions else 0
                                       for probability, action in zip(probabilities, all_possible_actions)]
                probabilities_valid_scaled = np.divide(probabilities_valid, sum(probabilities_valid))

                # Draw action using the estimated probabilities TODO: Check if only use the action with highest prob
                action_index = np.random.choice(len(all_possible_actions), p=probabilities_valid_scaled, replace=False)
                action = all_possible_actions[action_index]
            else:
                action = random.choice(possible_actions)
            rollout_game.do_action(action)

        game_win_state = rollout_game.get_win_state()  # TODO: check if correct or always if player_1 wins, return 1

        # Assume all evaluations are from the perspective of player_1
        if game_win_state == 1:  # If player_1 won return 1
            return 1
        if game_win_state == 2:  # If player_2 won return -1
            return -1
        return 0  # If draw return 0

    @staticmethod
    def critic(node: Node, critic_net: tf.keras.models) -> float:
        """
        Produce evaluation of leaf node using the critic-RL neural network

        :param node: node to evaluate
        :param critic_net:
        :return: leaf evaluation
        """
        critic_game = node.get_game()
        game_state = critic_game.get_board_state()
        critic_evaluation = critic_net.predict(np.array([game_state]), verbose=0)
        return critic_evaluation[0, 0]

    def run(self, game: TwoPlayerGame, actor_net: tf.keras.models, critic_net: tf.keras.models = None, m: int = 500,
            c: float = 1, rollout_epsilon: float = 0.8, **kwargs) -> tuple[tuple[int, int], NDArray]:
        """
        Run the MCTS

        :param game:
        :param actor_net:
        :param critic_net:
        :param m:
        :param c:
        :param rollout_epsilon:
        :param kwargs:
        :return: best action
        """
        # Initialize new node with the given game
        root_node = Node(game)

        eval_lambda = None
        eval_epsilon = None
        if 'eval_lambda' in kwargs:
            eval_lambda = kwargs['eval_lambda']
        elif 'eval_epsilon' in kwargs:
            eval_epsilon = kwargs['eval_epsilon']

        for i in range(m):
            # Tree search - choose node from root to leaf node using tree policy
            policy_node = self.tree_policy_func(root_node, c)

            # Node expansion - generate some or all child states of a parent state
            children_dict = policy_node.get_children_dict()
            valid_actions = policy_node.get_unattempted_actions() + list(children_dict.keys())
            if valid_actions:
                action = random.choice(valid_actions)
                expanded_node = children_dict[action] if action in children_dict.keys() else policy_node.expand(action)
            else:
                expanded_node = policy_node

            # Leaf evaluation - estimate the value of a leaf node using the default policy
            # TODO: Code clean-up, keep having two options - lambda and epsilon?
            if eval_lambda and critic_net:
                critic_result = self.critic(expanded_node, critic_net)
                rollout_result = self.rollout(expanded_node, actor_net, rollout_epsilon)
                leaf_evaluation = (1 - eval_lambda) * critic_result + eval_lambda * rollout_result
            elif eval_epsilon:  # TODO: improve value_epsilon over time making rollouts less likely
                if random.random() > eval_epsilon and critic_net:
                    leaf_evaluation = self.critic(expanded_node, critic_net)
                else:
                    leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)
            else:
                leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)

            # Backpropagation - passing the evaluation back up the tree, updating relevant data
            expanded_node.update(leaf_evaluation)

            # TODO: Implement timelimit

        # Choose best action from the root by the highest visit count
        best_child = max(root_node.get_children(), key=lambda child: child.get_visits())
        best_action = best_child.get_parent_action()

        # Action probabilities from the root
        all_actions = game.get_all_actions()
        children_dict = root_node.get_children_dict()

        all_actions_visits = [children_dict[action].get_visits() if action in children_dict.keys()
                              else 0 for action in all_actions]
        action_probabilities = np.divide(all_actions_visits, root_node.get_visits())

        return best_action, action_probabilities
