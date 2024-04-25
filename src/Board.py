import numpy as np
from numpy.typing import NDArray


class Board:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.board = np.zeros((height, width), dtype=int)
        self.neighbors = self.__connect_neighbors()

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

    def valid_position(self, y, x) -> bool:
        """
        Checks if the given coordinate is valid

        :param y: y coordinate
        :param x: x coordinate
        :return: boolean indicating whether the given coordinate is valid
        """
        return 0 <= x < self.width and 0 <= y < self.height

    def __connect_neighbors(self):
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

    def get_board(self) -> NDArray:
        """
        Returns the board array

        :return: board array
        """
        return self.board.copy()
