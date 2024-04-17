import numpy as np

from TwoPlayerGame import TwoPlayerGame


class Board:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width), dtype=int)
        self.neighbors = self.connect_neighbors()

    def get_cell(self, y, x):
        """
        Returns the cell at the given position

        :param y: y coordinate
        :param x: x coordinate
        :return: value of the cell
        """
        return self.board[y, x]

    def set_cell(self, y, x, value):
        """
        Sets the cell at the given position

        :param y: y coordinate
        :param x: x coordinate
        :param value: value to set
        :return: None
        """
        self.board[y, x] = value

    def valid_position(self, y, x):
        """
        Checks if the given coordinate is valid

        :param y: y coordinate
        :param x: x coordinate
        :return: boolean indicating whether the given coordinate is valid
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def connect_neighbors(self):
        """
        Creates a dictionary of all the connected neighbors

        """
        offsets = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]

        neighbors = {}
        for x in range(self.width):
            for y in range(self.height):
                neighbors_coords = [(y + offset_y, x + offset_x) for offset_x, offset_y in offsets if
                                    self.valid_position(y + offset_y, x + offset_x)]
                neighbors[(y, x)] = neighbors_coords
        return neighbors

    def print_board(self):
        """
        Prints the board in hex format

        """
        hex_height = self.height * 2 - 1
        hex_width = self.width

        for row in range(hex_height):
            row_string = ""
            for space in range(abs(hex_width - row - 1)):
                row_string += " "

            for x in range(max([row - hex_width + 1, 0]), min([row + 1, hex_width])):
                y = row - x
                row_string += f" {self.get_cell(y, x)}"

            print(row_string)


class Hex(TwoPlayerGame):
    def __init__(self, height, width):
        super().__init__()
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

            if current_x + 1 == self.width and player == 1:
                return 1
            elif current_y + 1 == self.height and player == 2:
                return 2

            explored.add((current_y, current_x))

            for neigh_y, neigh_x in self.board.neighbors[(current_y, current_x)]:
                if (neigh_y, neigh_x) in explored or self.board.get_cell(neigh_y, neigh_x) != player:
                    continue
                front.append((neigh_y, neigh_x))

        return 0

    def update_state(self):
        """
        Updates the state by running DFS for player_1 (from x = 0) and player_2 (from y = 0) where they have pieces

        :return: state of the game
        """
        player_1_start_coords = [(y, 0) for y in range(self.height) if self.board.get_cell(y, 0) == 1]
        player_2_start_coords = [(0, x) for x in range(self.width) if self.board.get_cell(0, x) == 2]

        for y, x in player_1_start_coords:
            leaf = self.__dfs(y, x, 1)
            if leaf:
                self.state = leaf
                return self.state

        for y, x in player_2_start_coords:
            leaf = self.__dfs(y, x, 2)
            if leaf:
                self.state = leaf
                return self.state

        return self.state

    def do_action(self, action):
        """
        Executes the given action and updates the state of the game

        :param action: tuple of (x, y) coordinates
        :return: resulting board if action is valid, None otherwise
        """

        y, x = action
        if not self.board.valid_position(y, x):
            print(f'Coordinate ({y},{x}) is out of bounds')
            return None
        if self.board.get_cell(y, x):
            print(f'Coordinate ({y},{x}) is already placed')
            return None
        self.board.set_cell(y, x, self.current_player)
        self.current_player = 3 - self.current_player

        self.update_state()

        return self.board

    def get_actions(self):
        valid_actions = [(y, x) for y in range(self.height) for x in range(self.width) if not self.board.get_cell(y, x)]
        return valid_actions

    def visualize(self):
        self.board.print_board()

    def play(self):
        while not self.state:
            invalid_action = True
            while invalid_action:
                x = int(input('x: '))
                y = int(input('y: '))
                new_board = self.do_action((y, x))
                if new_board is not None:
                    invalid_action = False

            self.visualize()

            if self.state:
                print(f'Player {self.state} won!')


if __name__ == '__main__':
    game = Hex(4, 4)
    game.play()
