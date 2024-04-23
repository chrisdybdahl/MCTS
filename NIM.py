import numpy as np
from numpy.typing import NDArray

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
        self.pieces_left = N

    def get_actions(self) -> list:
        return list(range(self.pieces_left + 1))

    def get_all_actions(self) -> list:
        return list(range(self.N + 1))

    def choose_move(self) -> object:
        return int(input(f'Number of pieces to remove (K = {self.K}): '))

    def do_action(self, action) -> bool:
        """
        Removes pieces from the board

        :param action: number of pieces to remove
        :return: True if action is valid, False otherwise
        """
        if 1 <= action <= self.K and 0 <= self.pieces_left - action:
            self.pieces_left -= action

            if not self.pieces_left:
                self.win_state = self.current_player

            self.current_player = 3 - self.current_player
            return True
        print(f'Invalid action: {action}')
        return False

    def get_board_state(self) -> NDArray:
        """
        Returns the board state

        :return: number of pieces left
        """
        return np.array([[self.pieces_left, self.current_player]])

    def visualize(self):
        """
        Visualizes the board
        """
        print(f'Pieces left: {self.pieces_left} / {self.N}')


if __name__ == '__main__':
    game = NIM(10, 4)
    game.play()
