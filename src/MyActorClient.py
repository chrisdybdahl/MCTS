import copy

from ActorClient import ActorClient
from Hex import Hex
from MyHexActor import MyHexActor
from config import TOKEN


class MyActionClient(ActorClient):
    def __init__(self, hex_actor: MyHexActor):
        super().__init__(auth=TOKEN, qualify=False)
        self.actor = copy.deepcopy(hex_actor)
        self.unique_id = None
        self.series_id = None
        self.player_map = None
        self.num_games = None
        self.size = None
        self.hex_game = None

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        super().handle_series_start(unique_id, series_id, player_map, num_games, game_params)
        self.unique_id = unique_id
        self.series_id = series_id
        self.player_map = player_map
        self.num_games = num_games
        self.size = game_params[0]

    def handle_game_start(self, start_player):
        super().handle_game_start(start_player)
        self.hex_game = Hex(start_player, self.size, self.size)

    def handle_get_action(self, state):
        # Use simulation of Hex game to retrieve possible and valid moves
        possible_actions = self.hex_game.get_all_actions()
        valid_actions = self.hex_game.get_actions()

        # Use the actor to choose an action
        row, col = self.actor.get_action(state, possible_actions, valid_actions)

        # Update the Hex game
        self.hex_game.do_action((row, col))
        return row, col
