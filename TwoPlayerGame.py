class TwoPlayerGame:
    def __init__(self):
        """
        Creates an instance of game

        """
        self.current_player = 1  # Assume player_1 begins
        self.state = 0  # 0 if no winner, 1 if player_1 won, 2 if player_2 won

    def get_actions(self):
        """
        Returns the possible action for the state
        """
        pass

    def do_action(self, action):
        """
        Removes pieces from the board

        :param action: action to be performed
        """
        pass

    def get_state(self):
        """
        Returns the state of the game

        :return: the current state
        """
        return self.state

    def current_player(self):
        """
        Returns the current player of the game

        :return: the current player
        """
        return self.current_player

    def visualize(self):
        """
        Visualizes the game
        """
        pass

    def play(self):
        """
        Plays the game
        """
        pass
