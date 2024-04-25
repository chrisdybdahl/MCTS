import numpy as np

from Node import Node
from NeuralNet import NeuralNet


def uct_score(root_node: Node, c: float) -> Node | None:
    # Lambda function to retrieve the exploration bonus of a certain node
    uct = lambda child_visits, parent_visits, a: a * np.sqrt(np.log(parent_visits) / (child_visits + 1))

    # BFS search for finding nodes which can be expanded further
    pre_existing_nodes = []
    queue = [root_node]

    while queue:
        current_node = queue.pop(0)
        pre_existing_nodes.append(current_node)
        # Enqueue child nodes
        for child_node in current_node.get_children():
            queue.append(child_node)

    # Record which player's turn it is
    current_player = root_node.get_current_player()

    # Find the greedy best action choice by assessing the combination of q score and exploration bonus
    if current_player == 1:
        return max(pre_existing_nodes,
                   key=lambda child: child.get_q() + uct(child.get_visits(), root_node.get_visits(), c))
    else:
        return min(pre_existing_nodes,
                   key=lambda child: child.get_q() - uct(child.get_visits(), root_node.get_visits(), c))


def minibatch_indices(replay_buffer, batch_size):
    num_cases = len(replay_buffer[0])
    indices = np.random.choice(num_cases, size=min(num_cases, batch_size), replace=False)
    states = np.array(replay_buffer[0])[indices]
    target_probabilities = np.array(replay_buffer[1])[indices]
    return states, target_probabilities


def choose_actor_action(actor_net: NeuralNet, possible_actions: list[tuple[int, int]],
                        valid_actions: list[tuple[int, int]], game_state: list[int], d_policy: str = 'greedy'):
    """
    Choose an action given a state from actor neural network

    :param actor_net: actor neural network
    :param possible_actions: list of possible actions
    :param valid_actions: list of valid actions
    :param game_state: current game state
    :param d_policy: policy to use for choosing action given probabilities
    :return: action to execute
    """
    probabilities = actor_net.predict(np.array([game_state]), verbose=0)[0]
    probabilities_valid = [probability if action in valid_actions else 0
                           for probability, action in zip(probabilities, possible_actions)]
    probabilities_valid_scaled = np.divide(probabilities_valid, sum(probabilities_valid))

    if d_policy == 'greedy':
        # Execute action greedily by choosing the one with the highest probability
        action_index = np.argmax(probabilities_valid_scaled)
    else:
        # Execute action stochastically based on estimated probabilities
        action_index = np.random.choice(len(possible_actions), p=probabilities_valid_scaled)

    return possible_actions[action_index]
