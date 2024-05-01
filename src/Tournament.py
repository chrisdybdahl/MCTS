import copy
import os

import numpy as np
from matplotlib import pyplot as plt

from Hex import Hex
from NeuralNet import NeuralNet
from TwoPlayerGame import TwoPlayerGame
from config import HEX_STARTING_PLAYER, HEX_HEIGHT, HEX_WIDTH, TOURNAMENT_FOLDER, NUM_MATCHES, \
    TOURNAMENT_VISUALIZE, TOURNAMENT_FILENAME, TOURNAMENT_D_POLICY
from helpers import choose_actor_action


class Tournament:
    def __init__(self, game: TwoPlayerGame):
        self.game = copy.deepcopy(game)
        self.folder_name = None
        self.folder_path = None
        self.models = None
        self.matches = None
        self.results = None

    def __play_match(self, red_actor: NeuralNet, blue_actor: NeuralNet, visualize: bool = False) -> int:

        game = copy.deepcopy(self.game)

        if visualize:
            print(f"Player_{game.get_current_player()}'s turn")
            game.visualize()

        # Get all possible actions
        possible_actions = game.get_all_actions()

        while not game.win_state:
            # Get the valid actions
            valid_actions = game.get_actions()

            # Extract game state
            game_state = game.get_board_state()

            # Get current player
            current_player = game.get_current_player()

            # Current actor choose move
            current_actor = red_actor if current_player == 1 else blue_actor
            action = choose_actor_action(current_actor, possible_actions, valid_actions, game_state,
                                         TOURNAMENT_D_POLICY)

            # Execute action
            game.do_action(action)

            if visualize:
                print(f"Player_{game.get_current_player()}'s turn")
                game.visualize()

        winner = game.get_win_state()

        return winner

    def play_tournament(self, folder_name: str, num_matches: int = 25, visualize: bool = False, filename: str = None):
        self.folder_name = folder_name
        self.folder_path = f'{os.getcwd()}\\{self.folder_name}'
        self.models = sorted([str(filename) for filename in os.listdir(self.folder_path)],
                             key=lambda file: int(file.split('_')[-1].split('.')[0]))
        print(self.models)
        starting_player = self.game.get_starting_player()

        num_models = len(self.models)
        results = np.zeros((num_models, num_models))

        for i in range(num_models):
            for j in range(num_models):
                if i != j:
                    player_1 = self.models[i]
                    player_2 = self.models[j]

                    player_1_actor = NeuralNet(f'{self.folder_path}\\{player_1}')
                    player_2_actor = NeuralNet(f'{self.folder_path}\\{player_2}')

                    player_1_wins = 0
                    player_2_wins = 0

                    if visualize:
                        print(f'{player_1 if starting_player == 1 else player_2} begins')
                        print(f'{player_1} versus {player_2}')

                    for k in range(num_matches):
                        if visualize:
                            print(f'Match: {k + 1}/{num_matches}')

                        winner = self.__play_match(player_1_actor, player_2_actor, visualize=visualize)

                        if visualize:
                            print(f'{player_1 if winner == 1 else player_2} wins!')

                        if winner == 1:
                            player_1_wins += 1
                        else:
                            player_2_wins += 1

                    if filename:
                        with open(f'{filename}.txt', 'a') as f:
                            f.write(f'{player_1} versus {player_2}'
                                    f' - starting player: {player_1 if starting_player == 1 else player_2}'
                                    f'\n{player_1_wins} - {player_2_wins}\n')

                    w = player_1_wins / num_matches

                    results[i, j] = w

            np.fill_diagonal(results, np.nan)

            self.results = results

    def visualize_results(self):

        num_models = len(self.models)
        models = self.models

        fig, ax = plt.subplots()
        ax.imshow(self.results)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(num_models), labels=models)
        ax.set_yticks(np.arange(num_models), labels=models)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(num_models):
            for j in range(num_models):
                ax.text(j, i, self.results[i, j], ha="center", va="center", color="w")

        ax.set_title("Tournament results")
        fig.tight_layout()
        plt.show()

    def reset(self):
        self.folder_name = None
        self.folder_path = None
        self.models = None
        self.matches = None
        self.results = None


if __name__ == "__main__":
    hex_game = Hex(HEX_STARTING_PLAYER, HEX_HEIGHT, HEX_WIDTH)
    tournament = Tournament(hex_game)
    tournament.play_tournament(TOURNAMENT_FOLDER, num_matches=NUM_MATCHES, visualize=TOURNAMENT_VISUALIZE,
                               filename=TOURNAMENT_FILENAME)
    tournament.visualize_results()
