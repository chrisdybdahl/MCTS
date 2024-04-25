from TwoPlayerGame import TwoPlayerGame


class Nim(TwoPlayerGame):
    def __init__(self, starting_player, n, k):
        """
        Creates an instance of the NIM board and initializes parameters

        :param n: number of pieces on the board
        :param k: number of pieces a player maximum can take off the board
        """
        super().__init__(starting_player)
        self.N = n
        self.K = k
        self.pieces_left = n

    def get_actions(self) -> list:
        return list(range(1, min(self.pieces_left, self.K) + 1))

    def get_all_actions(self) -> list:
        return list(range(1, self.N + 1))

    def reset(self):
        super().reset()
        self.pieces_left = self.N

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

    def get_board_state(self) -> list:
        """
        Returns the board state

        :return: number of pieces left
        """
        return [self.pieces_left, self.current_player]

    def visualize(self):
        """
        Visualizes the board
        """
        print(f'Pieces left: {self.pieces_left} / {self.N}')


if __name__ == '__main__':
    game = Nim(1, 10, 4)
    game.play()
