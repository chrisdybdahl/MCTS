from NeuralNet import NeuralNet
from config import CLIENT_D_POLICY
from helpers import choose_actor_action


class MyHexActor:
    def __init__(self, path: str):
        self.actor_net = NeuralNet(path)
        self.possible_actions = []
        self.valid_actions = []

    def get_action(self, game_state: list, possible_actions: list, valid_actions: list):
        action = choose_actor_action(self.actor_net, possible_actions, valid_actions, game_state, CLIENT_D_POLICY)
        return action
