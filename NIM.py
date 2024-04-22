from TwoPlayerGame import TwoPlayerGame


class NIM(TwoPlayerGame):
    def __init__(self, N, K):
        """
        Creates an instance of the NIM board and initializes parameters

        :param N: number of pieces on the board
        :param K: number of pieces a player maximum can take off the board
        """
        super().__init__()
        self.N = N
        self.K = K

    def get_actions(self) -> list:
        """
        Returns the possible action for the state
        """
        return list(range(self.N + 1))

    def choose_move(self) -> object:
        return int(input('Number of pieces to remove: '))

    def do_action(self, action) -> bool:
        """
        Removes pieces from the board

        :param action: list of tuples which represent the coordinates of the pieces to remove
        :return: True if action is valid, False otherwise
        """
        if 1 <= action <= self.K and 0 <= self.N - action:
            self.N -= action

            if not self.N:
                self.win_state = self.current_player

            self.current_player = 3 - self.current_player
            return True
        print(f'Invalid action: {action}')
        return False

    def get_board_state(self) -> object:
        """
        Returns the board state

        :return: number of pieces left
        """
        return [self.N, self.current_player]

    def visualize(self):
        """
        Visualizes the board
        """
        print(f'Pieces left: {self.N}')


if __name__ == '__main__':
    game = NIM(10, 4)
    game.play()
