import copy
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


class MonteCarloTreeSearchNode:
    def __init__(self, game: TwoPlayerGame, parent=None, parent_action: tuple[int, int] = None):
        """
        Initializes the node

        :param game: game in its current state
        :param parent_action: action leading to this node
        :param parent: parent node leading to this node
        """
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.parent_action = parent_action
        self.valid_actions = game.get_actions()
        self.untried_actions = self.valid_actions
        self.current_player = game.get_current_player()
        self.children = []
        self.children_dict = {}
        self.visits = 0
        self.sum_evaluation = 0
        self.q = 0

    def get_untried_actions(self) -> list:
        self.untried_actions = self.game.get_actions()
        return self.untried_actions

    def get_current_player(self) -> int:
        return self.current_player

    def get_parent_action(self):
        return self.parent_action

    def get_q(self) -> float:
        return self.q

    def get_visits(self) -> int:
        return self.visits

    def expand(self):
        # Execute move in copied game
        expanded_game = copy.deepcopy(self.game)
        action = self.untried_actions.pop()
        expanded_game.do_action(action)

        # Create new node with the copied changed game
        expanded_node = MonteCarloTreeSearchNode(expanded_game, self, action)

        # Add node to children
        self.children.append(expanded_node)
        self.children_dict[action] = expanded_node
        self.untried_actions.remove(action)

        return expanded_node

    def is_terminal(self) -> bool:
        return self.game.get_win_state() != 0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def rollout(self, actor_net: NeuralNet, epsilon: float) -> float:
        rollout_game = copy.deepcopy(self.game)

        while rollout_game.get_win_state():
            action = self.rollout_policy(rollout_game, actor_net, epsilon)

            rollout_game.do_action(action)

        return rollout_game.get_win_state()

    def backpropagate(self, result):
        self.visits += 1
        self.sum_evaluation += result
        self.q = self.sum_evaluation / self.visits

        if self.parent:
            self.parent.update(result)

    def best_child(self, c, current_player):
        if current_player == 1:
            choices_weights = [child.get_q() + c * np.sqrt(np.log(self.get_visits()) / (child.get_visits() + 1)) for child
                               in self.children]
            return self.children[np.argmax(choices_weights)]
        else:
            choices_weights = [child.get_q() - c * np.sqrt(np.log(self.get_visits()) / (child.get_visits() + 1)) for child
                               in self.children]
            return self.children[np.argmin(choices_weights)]

    @staticmethod
    def rollout_policy(game: TwoPlayerGame, actor_net: NeuralNet, epsilon: float):
        # Get the valid actions
        valid_actions = game.get_actions()
        possible_actions = game.get_all_actions()

        # Probability p to make random move and (1 - p) to make move from actor in rollout game
        if random.random() > epsilon:
            # Extract game state
            game_state = game.get_board_state()

            # Estimate probabilities using actor neural network, either greedily or stochastically
            action = choose_actor_action(actor_net, possible_actions, valid_actions, game_state, D_POLICY)
        else:
            action = random.choice(valid_actions)
        return action

    def tree_policy(self):
        current_node = self

        while not current_node.is_terminal():
            if not current_node.is_terminal():
                return current_node.expand()

        return current_node

    def best_action(self, m: int, c: float, e: float, actor_net: NeuralNet, player: int):
        for _ in range(m):
            v = self.tree_policy()
            reward = v.rollout(actor_net, e)
            v.backpropagate(reward)

        return self.best_child(c, player)


def main(game: TwoPlayerGame, m: int, actor_net: NeuralNet, c: float = 1, e: float = 1):
    root = MonteCarloTreeSearchNode(game)
    current_player = root.get_current_player()
    selected_node = root.best_action(m, c, e, actor_net, current_player)

    return selected_node


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

        for i in range(m):
            x = self.tree_policy_func(root_node, c)

            # Tree search - choose node from root to leaf node using tree policy
            if x is None:
                continue
            else:
                policy_node, policy_action = x

            # Node expansion - generate a child state of the parent state
            children_dict = policy_node.get_children_dict()

            """valid_actions = policy_node.get_valid_actions()
            untried_actions = [action for action in valid_actions if action not in children_dict.keys()]
            if untried_actions:
                action = random.choice(untried_actions)
                expanded_node = policy_node.expand(action)
            else:
                action = policy_action
                expanded_node = children_dict[action]"""

            expanded_node = children_dict[
                policy_action] if policy_action in children_dict.keys() else policy_node.expand(policy_action)

            # Leaf evaluation - estimate the value of a leaf node using the default policy
            if critic_net:  # Probability p to evaluate using rollout and (1 - p) using critic if it exists
                if random.random() > eval_epsilon:
                    leaf_evaluation = self.critic(expanded_node, critic_net)
                else:
                    leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)
            else:
                leaf_evaluation = self.rollout(expanded_node, actor_net, rollout_epsilon)

            # Backpropagation - passing the evaluation back up the tree, updating relevant data
            expanded_node.update(leaf_evaluation)

            # Stop monte carlo search if runtime exceeds timelimit
            if timelimit and time.time() - timer > timelimit:
                print(f'Timed out in iteration {i + 1}/{m}, after {time.time() - timer} seconds')
                break

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
