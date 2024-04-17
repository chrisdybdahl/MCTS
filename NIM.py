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

    def get_actions(self):
        """
        Returns the possible action for the state
        """
        pass

    def do_action(self, removed_pieces):
        """
        Removes pieces from the board

        :param removed_pieces: list of tuples which represent the coordinates of the pieces to remove
        """
        pass

    def visualize(self):
        """
        Visualizes the board
        """
        pass

    def play(self):
        super().play()
