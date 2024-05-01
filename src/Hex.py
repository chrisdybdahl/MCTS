import numpy as np

from Board import Board
from TwoPlayerGame import TwoPlayerGame


class Hex(TwoPlayerGame):
    def __init__(self, starting_player, height, width):
        super().__init__(starting_player)
        self.height = height
        self.width = width
        self.board = Board(height, width)

    def __dfs(self, y, x, player):
        """
        Performs DFS search to determine whether a player has won

        :param x: x coordinate
        :param y: y coordinate
        :return: 0 if no player has won, 1 if player_1 has won, 2 if player_2 has won
        """

        front = [(y, x)]
        explored = set()

        while front:
            current_y, current_x = front.pop()

            if current_y + 1 == self.height and player == 1:  # Player_1 goes "top-down"
                return 1
            elif current_x + 1 == self.width and player == 2:  # Player_2 goes "left-right"
                return 2

            explored.add((current_y, current_x))

            for neigh_y, neigh_x in self.board.neighbors[(current_y, current_x)]:
                if (neigh_y, neigh_x) in explored or self.board.get_cell(neigh_y, neigh_x) != player:
                    continue
                front.append((neigh_y, neigh_x))

        return 0

    def __update_state(self):
        """
        Updates the state by running DFS for player_1 (from x = 0) and player_2 (from y = 0) where they have pieces

        :return: state of the game
        """
        # Find the starting coordinates for player_1 (top) and player_2 (left)
        player_1_start_coords = [(0, x) for x in range(self.width) if self.board.get_cell(0, x) == 1]
        player_2_start_coords = [(y, 0) for y in range(self.height) if self.board.get_cell(y, 0) == 2]

        for y, x in player_1_start_coords:
            leaf = self.__dfs(y, x, 1)
            if leaf:
                self.win_state = leaf
                return self.win_state

        for y, x in player_2_start_coords:
            leaf = self.__dfs(y, x, 2)
            if leaf:
                self.win_state = leaf
                return self.win_state

        return self.win_state

    def get_actions(self) -> list:
        return [(y, x) for y in range(self.height) for x in range(self.width) if not self.board.get_cell(y, x)]

    def get_all_actions(self) -> list:
        return [(y, x) for y in range(self.height) for x in range(self.width)]

    def reset(self):
        super().reset()
        self.board = Board(self.height, self.width)

    def choose_move(self) -> object:
        x = int(input('x: '))
        y = int(input('y: '))
        return y, x

    def do_action(self, action) -> bool:
        """
        Executes the given action and updates the state of the game

        :param action: tuple of (x, y) coordinates
        :return: True if action is valid, False otherwise
        """
        y, x = action
        if not self.board.valid_position(y, x):
            print(f'Coordinate ({y},{x}) is out of bounds')
            return False
        if self.board.get_cell(y, x):
            print(f'Coordinate ({y},{x}) is already placed')
            return False
        self.board.set_cell(y, x, self.current_player)
        self.current_player = 3 - self.current_player

        self.__update_state()
        return True

    def get_board_state(self) -> list:
        """
        Returns the board state and current player

        :return: flattened array of board and current player
        """
        board_player_list = self.board.get_board().flatten().tolist()
        board_player_list.insert(0, self.current_player)
        return board_player_list

    def set_board_state(self, board_state: list):
        """
        Sets the board state

        :param board_state: the board state to set the board
        """
        self.current_player = board_state[0]
        self.board.set_board(np.reshape(board_state[1:], (self.height, self.width)))

    def visualize(self):
        self.board.print_board()


if __name__ == '__main__':
    game = Hex(1, 3, 3)
    game.play()
