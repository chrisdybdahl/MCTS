class NIM:
    def __init__(self, N, K):
        """
        Creates an instance of the NIM board and initializes parameters

        Parameters:
            N (int): number of pieces on the board
            K (int): number of pieces a player maximum can take off the board
        N is the number of pieces on the board

        """
        self.N = N
        self.K = K

    def remove(self, removed_pieces):
        """
        Removes pieces from the board

        Parameters:
            removed_pieces: list of touples"""
        self.N = removed_pieces
