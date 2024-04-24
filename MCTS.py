import copy
import random
from typing import Callable

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from TwoPlayerGame import TwoPlayerGame


class Node:
    def __init__(self, game: TwoPlayerGame, parent=None, parent_action=None):
        """
        Initializes the node

        :param game: game in its current state
        :param parent_action: action leading to this node
        :param parent: parent node leading to this node
        """
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.parent_action = parent_action
        self.unattempted_actions = game.get_actions()
        self.current_player = game.get_current_player()
        self.children = []
        self.visits = 0
        self.sum_evaluation = 0
        self.q = 0

    def expand(self, action):
        """
        Expands the current node with the given action

        :param action: action to be performed
        :return: expanded node using the given action on the current game
        """
        # Execute move in copied game
        expanded_game = self.get_game()
        expanded_game.do_action(action)

        # Create new node with the copied changed game
        expanded_node = Node(expanded_game, self, action)

        # Add node to children
        self.children.append(expanded_node)

        # Remove action from unattempted actions
        self.unattempted_actions.remove(action)

        return expanded_node

    def update(self, evaluation):
        """
        Updates the current node's, and its ancestors', number of visits, evaluation sum, and q score

        :param evaluation: evaluation from rollout or critic
        """
        self.visits += 1
        self.sum_evaluation += evaluation
        self.q = self.sum_evaluation / self.visits

        if self.parent:
            self.parent.update(evaluation)

    def get_parent(self):
        """
        Returns the current node's parent

        :return: parent node
        """
        return self.parent

    def get_parent_action(self):
        """
        Returns the action leading to this node

        :return: action leading to this node
        """
        return self.parent_action

    def get_game(self) -> TwoPlayerGame:
        """
        Returns the node's current game

        :return: copy of the node's current game
        """
        return copy.deepcopy(self.game)

    def get_unattempted_actions(self) -> list:
        """
        Returns the current node's possible actions

        :return: list of the node's possible actions
        """
        return self.unattempted_actions

    def get_children(self) -> list:
        """
        Returns the current node's children

        :return: list of node's children
        """
        return self.children

    def get_visits(self) -> int:
        """
        Returns the current node's visits

        :return: number of visits
        """
        return self.visits

    def get_current_player(self) -> int:
        """
        Returns the current node's current player

        :return: current player
        """
        return self.current_player

    def get_q(self) -> float:
        """
        Returns the current node's q score

        :return: node q score
        """
        return self.q


def uct_score(root_node: Node, c: float) -> Node | None:
    """uct = lambda child_wins, child_visits, parent_visits, a: (child_wins / child_visits) + a * np.sqrt(
        np.log(parent_visits) / (child_visits + 1))

    children_nodes = parent_node.get_children()
    if children_nodes:
        return max(children_nodes,
                   key=lambda child: uct(child.get_wins(), child.get_visits(), parent_node.get_visits(), c))
    return parent_node"""
    # Lambda function to retrieve the exploration bonus of a certain node
    uct = lambda child_visits, parent_visits, a: a * np.sqrt(np.log(parent_visits) / (child_visits + 1))

    # BFS search for finding nodes which can be expanded further
    non_exhausted_nodes = []
    queue = [root_node]

    while queue:
        current_node = queue.pop(0)
        # Check if the node has unattempted actions
        if current_node.get_unattempted_actions():
            non_exhausted_nodes.append(current_node)
        # Enqueue child nodes
        for child in current_node.get_children():
            queue.append(child)

    # Record which player's turn it is
    current_player = root_node.get_current_player()

    # Find the greedy best action choice by assessing the combination of q score and exploration bonus
    if non_exhausted_nodes:  # TODO: check if correct, what if tree is fully expanded? Consider all pre-existing nodes?
        if current_player == 1:
            return max(non_exhausted_nodes,
                       key=lambda node: node.get_q() + uct(node.get_visits(), root_node.get_visits(), c))
        elif current_player == 2:
            return min(non_exhausted_nodes,
                       key=lambda node: node.get_q() - uct(node.get_visits(), root_node.get_visits(), c))
    return None


class MCTS:
    def __init__(self, tree_policy_func: Callable[[Node, float], Node]):
        """
        Initialize MCTS parameters here

        :param tree_policy_func: tree policy function
        """
        self.tree_policy_func = tree_policy_func
        self.M = None

    @staticmethod
    def rollout(node: Node) -> float:
        """
        Rollout simulation from node

        :param node: node to run a rollout simulation from
        :return: game result from rollout simulation
        """
        rollout_game = node.get_game()

        while not rollout_game.get_win_state():
            possible_actions = rollout_game.get_actions()
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
        critic_evaluation = critic_net.predict(game_state, verbose=0)
        return critic_evaluation[0, 0]

    def run(self, game: TwoPlayerGame, critic_net: tf.keras.models = None, m: int = 500, c: float = 1, **kwargs) \
            -> tuple[tuple[int, int], NDArray]:
        """
        Run the MCTS

        :param game:
        :param critic_net:
        :param m:
        :param c:
        :param kwargs:
        :return: best action
        """
        root_node = Node(game)

        value_lambda = None
        value_epsilon = None
        if 'value_lambda' in kwargs:
            value_lambda = kwargs['value_lambda']
        elif 'value_epsilon' in kwargs:
            value_epsilon = kwargs['value_epsilon']

        for i in range(m):
            # Tree search - choose node from root to leaf node using tree policy
            policy_node = self.tree_policy_func(root_node, c)

            if not policy_node:  # Terminate if all nodes are explored TODO: Check if correct
                break

            # Node expansion - generate some or all child states of a parent state
            unattempted_actions = policy_node.get_unattempted_actions()
            action = random.choice(unattempted_actions)
            expanded_node = policy_node.expand(action)

            # Leaf evaluation - estimate the value of a leaf node using the default policy
            leaf_evaluation = None  # TODO: Code clean-up, keep having two options - lambda and epsilon?
            if value_lambda and critic_net:
                critic_result = self.critic(expanded_node, critic_net)
                rollout_result = self.rollout(expanded_node)
                leaf_evaluation = (1 - value_lambda) * critic_result + value_lambda * rollout_result
            elif value_epsilon:  # TODO: improve value_epsilon over time making rollouts less likely
                if random.random() <= value_epsilon and critic_net:
                    leaf_evaluation = self.critic(expanded_node, critic_net)
                else:
                    leaf_evaluation = self.rollout(expanded_node)

            # Backpropagation - passing the evaluation back up the tree, updating relevant data
            expanded_node.update(leaf_evaluation)

            # TODO: Implement timelimit

        # Choose best action from the root by the highest visit count
        best_child = max(root_node.get_children(), key=lambda child: child.get_visits())
        best_action = best_child.get_parent_action()

        # Action probabilities from the root
        all_actions = game.get_all_actions()
        action_visits_dict = {}
        for child in root_node.get_children():
            action_visits_dict[child.get_parent_action()] = child.get_visits()

        all_actions_visits = [action_visits_dict[action] if action in action_visits_dict.keys() else 0 for action in all_actions]
        action_probabilities = np.divide([all_actions_visits], root_node.get_visits())

        return best_action, action_probabilities
