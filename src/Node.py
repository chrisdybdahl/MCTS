import copy

from TwoPlayerGame import TwoPlayerGame


class Node:
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
        self.children_dict[action] = expanded_node
        self.untried_actions.remove(action)

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

    def get_valid_actions(self) -> list:
        """
        Returns the current node's possible actions

        :return: list of the node's possible actions
        """
        return self.valid_actions

    def get_untried_actions(self) -> list:
        """
        Returns the current node's untried actions

        :return: list of the node's untried actions
        """
        return self.untried_actions

    def get_children(self) -> list:
        """
        Returns the current node's children

        :return: list of node's children
        """
        return self.children

    def get_children_dict(self) -> dict:
        """
        Returns a dictionary with action as keys containing the node's children

        :return: dictionary with action as keys and child nodes as values
        """
        return self.children_dict

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
