class TwoPlayerGame:
    def __init__(self, starting_player: int):
        """
        Creates an instance of the game

        """
        self.starting_player = starting_player
        self.current_player = starting_player
        self.win_state = 0  # 0 if no winner, 1 if player_1 won, 2 if player_2 won

    def get_current_player(self) -> int:
        """
        Returns the current player of the game

        :return: the current player
        """
        return self.current_player

    def get_win_state(self) -> int:
        """
        Returns the win state of the game

        :return: the current win state
        """
        return self.win_state

    def get_starting_player(self):
        """
        Returns the starting player of the game

        :return: starting player
        """
        return self.starting_player

    def get_actions(self) -> list:
        """
        Returns the valid actions for the current game state

        :return: list of the valid actions for the current game state
        """
        pass

    def get_all_actions(self) -> list:
        """
        Returns all the actions for the game

        :return: list of all the actions for the game
        """
        pass

    def reset(self):
        """
        Resets the game state

        """
        self.current_player = self.starting_player
        self.win_state = 0

    def choose_move(self) -> object:
        """
        Takes in an input from the user

        :return: returns the input from the user
        """
        pass

    def do_action(self, action) -> bool:
        """
        Performs the action

        :param action: action to be performed
        :return: True if action is valid, False otherwise
        """
        pass

    def get_board_state(self) -> list:
        """
        Returns the board state of the game

        :return: the board state
        """
        pass

    def visualize(self):
        """
        Visualizes the game
        """
        pass

    def play(self):
        """
        Plays the game
        """

        self.visualize()
        while not self.win_state:
            print(f"Player_{self.get_current_player()}'s turn")

            invalid_action = True
            while invalid_action:
                action = self.choose_move()
                valid = self.do_action(action)
                if valid:
                    invalid_action = False

            self.visualize()

            if self.win_state:
                print(f'Player {self.win_state} won!')
