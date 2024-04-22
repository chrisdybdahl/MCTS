import copy
import random
from typing import Callable, Tuple, Any, List

import numpy as np

from TwoPlayerGame import TwoPlayerGame


class Node:
    def __init__(self, game: TwoPlayerGame, parent=None, parent_action=None):
        """
        Initializes the node

        :param game: game in its current state
        :param parent_action: action leading to this node
        :param parent: parent node leading to this node
        """
        self.game = copy.deepcopy(game)  # TODO: Check whether game need to be copy.deepcopy
        self.parent = parent
        self.parent_action = parent_action
        self.possible_actions = game.get_actions()
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
        expanded_game = self.get_game()
        expanded_game.do_action(action)
        expanded_node = Node(expanded_game, self, action)
        self.children.append(expanded_node)
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

    def get_game(self):
        """
        Returns the node's current game

        :return: copy of the node's current game
        """
        return copy.deepcopy(self.game)

    def get_possible_actions(self):
        """
        Returns the current node's possible actions

        :return: list of the node's possible actions
        """
        return self.possible_actions

    def get_children(self):
        """
        Returns the current node's children

        :return: list of node's children
        """
        return self.children

    def get_visits(self):
        """
        Returns the current node's visits

        :return: number of visits
        """
        return self.visits

    def get_q(self):
        """
        Returns the current node's q score

        :return: node q score
        """
        return self.q


def uct_score(parent_node: Node, **kwargs) -> Node:
    c = kwargs['c'] if 'c' in kwargs.keys() else 1.41

    uct: lambda child_wins, child_visits, parent_visits, a: (child_wins / child_visits) + a * np.sqrt(
        np.log(parent_visits) / (child_visits + 1))

    # TODO: Check whether need if statement regarding which player's turn it is
    # TODO: Check whether choice based on uct only happens when there are no more possible actions for parent node
    children_nodes = parent_node.get_children()
    if children_nodes:
        return max(children_nodes,
                   key=lambda child: uct(child.get_wins(), child.get_visits(), parent_node.get_visits(), c))
    return parent_node


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


class MCTS:
    def __init__(self, tree_policy_func: Callable[[Node], Node]):
        """
        Initialize MCTS parameters here

        :param tree_policy_func: tree policy function
        """
        self.tree_policy_func = tree_policy_func
        self.M = None

    def critic(self, node: Node) -> float:
        """
        Produce evaluation of leaf node using the critic-RL neural network

        :param node: node to evaluate
        :return: leaf evaluation
        """
        critic_game = node.get_game()
        game_state = critic_game.get_board_state()
        critic_evaluation = 1  # TODO: Implement neural net critic_evaluation with self
        return critic_evaluation

    def run(self, game: TwoPlayerGame, m: int = 500, **kwargs) -> tuple[int, list[float]]:
        """
        Run the MCTS

        :param game:
        :param m:
        :param kwargs:
        :return: best action
        """
        root_node = Node(game)

        value_lambda = None
        value_epsilon = None
        if 'lambda' in kwargs:
            value_lambda = kwargs['lambda']
        elif 'epsilon' in kwargs:
            value_epsilon = kwargs['epsilon']

        for _ in range(m):
            # Tree search - choose node from root to leaf node using tree policy
            policy_node = self.tree_policy_func(root_node, kwargs)  # TODO: Tree policy chose among all created nodes? Depth-First-Search?

            # Node expansion - generate some or all child states of a parent state
            possible_actions = policy_node.get_possible_actions()
            action = random.choice(possible_actions)
            expanded_node = policy_node.expand(action)

            # Leaf evaluation - estimate the value of a leaf node using the default policy
            leaf_evaluation = None  # TODO: Code clean-up, keep having two options - lambda and epsilon?
            if value_lambda:
                critic_result = self.critic(expanded_node)
                rollout_result = rollout(expanded_node)
                leaf_evaluation = (1 - value_lambda) * critic_result + value_lambda * rollout_result
            elif value_epsilon:  # TODO: improve value_epsilon over time making rollouts less likely
                if random.random() > value_epsilon:
                    leaf_evaluation = rollout(expanded_node)
                else:
                    leaf_evaluation = self.critic(expanded_node)

            # Backpropagation - passing the evaluation back up the tree, updating relevant data
            expanded_node.update(leaf_evaluation)

            # TODO: Implement timelimit

        # Choose best action from the root by the highest visit count
        best_action = max(root_node.get_children(), key=lambda child: child.get_value())

        # Action probabilities from the root
        action_probabilities = [child.get_visits() / root_node.get_visits() for child in root_node.get_children()]

        return best_action, action_probabilities
