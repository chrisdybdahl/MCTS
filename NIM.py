class NIM:
    def __init__(self, N, K):
        """
        Creates an instance of the NIM board and initializes parameters

        :param N: number of pieces on the board
        :param K: number of pieces a player maximum can take off the board
        """
        self.N = N
        self.K = K
        self.current_player = 1

    def do_action(self, removed_pieces):
        """
        Removes pieces from the board

        :param removed_pieces: list of tuples which represent the coordinates of the pieces to remove
        """
        self.N = removed_pieces
        self.current_player = 3 - self.current_player

    def get_state(self):
        """
        Returns the state of the game

        :return: 0 no win yet, 1 player_1 win, 2 player_1 win
        """
        return ValueError

    def current_player(self):
        """
        Returns the current player of the game

        :return: integer representing the current player
        """
        return ValueError
