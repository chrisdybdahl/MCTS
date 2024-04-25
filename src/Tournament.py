import copy
import os

from NeuralNet import NeuralNet
from TwoPlayerGame import TwoPlayerGame
from helpers import choose_actor_action


class Tournament:
    def __init__(self, game: TwoPlayerGame):
        self.game = copy.deepcopy(game)
        self.results = []

    @staticmethod
    def __match_making(models: list[str]) -> list[tuple[str, str]]:
        num_models = len(models)
        matches = [(models[0], models[1]) for i in range(num_models) for j in range(num_models) if i != j]
        return matches

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
            action = choose_actor_action(current_actor, possible_actions, valid_actions, game_state)

            # Execute action
            game.do_action(action)

            if visualize:
                print(f"Player_{game.get_current_player()}'s turn")
                game.visualize()

        winner = game.get_current_player()

        return winner

    def play_tournament(self, folder_name: str, num_matches: int = 25, visualize: bool = False, filename: str = None):
        folder_name = folder_name
        folder_path = f'{os.getcwd()}\\{folder_name}'
        models = [str(filename) for filename in os.listdir(folder_path)]
        matches = self.__match_making(models)

        if visualize:
            print(f'Player {self.game.get_starting_player()} begins')

        for red_str, blue_str in matches:
            red_filename = red_str.split('\\')[-1]
            blue_filename = blue_str.split('\\')[-1]

            red_actor = NeuralNet(red_str)
            blue_actor = NeuralNet(blue_str)

            red_wins = 0
            blue_wins = 0

            if visualize:
                print(f'Player 1 - {red_filename} versus Player 2 - {blue_filename}')
            for i in range(num_matches):
                if visualize:
                    print(f'Match: {i + 1}/{num_matches}')

                winner = self.__play_match(red_actor, blue_actor, visualize=visualize)

                if visualize:
                    print(f'\nPlayer {winner} - {red_filename if winner == 1 else blue_filename} wins!')

                if winner == 1:
                    red_wins += 1
                else:
                    blue_wins += 1

            if filename:
                with open(f'{filename}.txt', 'a') as f:
                    f.write(f'Player 1 - {red_filename} versus Player 2 - {blue_filename}'
                            f'\nPlayer 1 wins: {red_wins} - Player 2 wins: {blue_wins}\n')

            w = 1 if red_wins > blue_wins else -1 if blue_wins > red_wins else 0

            self.results.append((red_filename, blue_filename, w))

    def visualize_results(self):
        pass
