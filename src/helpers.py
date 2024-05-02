import numpy as np

from NeuralNet import NeuralNet
from Node import Node


def __uct(child_visits: int, parent_visits: int, c: float) -> float:
    return c * np.sqrt(np.log(parent_visits) / (child_visits + 1))


def __final_score(parent_node: Node, action: object, c: float, current_player: int) -> float:
    children_dict = parent_node.get_children_dict()

    if action in children_dict.keys():
        child_node = children_dict[action]
        q_value = child_node.get_q()
        uct = __uct(child_node.get_visits(), parent_node.get_visits(), c)
        score = (q_value + uct) if current_player == 1 else (q_value - uct)
    else:
        score = __uct(0, parent_node.get_visits(), c)

    return score


def minibatch_indices(replay_buffer: list[...], batch_size: int):
    num_cases = len(replay_buffer[0])
    indices = np.random.choice(num_cases, size=min(num_cases, batch_size), replace=False)
    states = np.array(replay_buffer[0])[indices]
    targets = np.array(replay_buffer[1])[indices]
    return states, targets


def uct_score(root_node: Node, c: float) -> tuple[Node, ...]:
    # BFS search for finding nodes which can be expanded further
    pre_existing_nodes = []
    queue = [root_node]

    while queue:
        current_node = queue.pop(0)

        """# If current node has untried action
        untried_actions = current_node.get_untried_actions()
        if untried_actions:
            action = random.choice(untried_actions)
            # print(f'pieces left: {current_node.get_game().get_board_state()}, action: {action}')
            return current_node, action"""

        # Enqueue existing child nodes to search through existing nodes in tree
        for node in current_node.get_children():
            queue.append(node)

        # Add tuple of parent node and action in list
        valid_actions = current_node.get_valid_actions()
        for valid_action in valid_actions:
            pre_existing_nodes.append((current_node, valid_action))

    # Record which player's turn it is
    current_player = root_node.get_current_player()

    if len(pre_existing_nodes) == 0:
        return None

    # Find the greedy best action choice by assessing the combination of q score and exploration bonus
    if current_player == 1:
        return max(pre_existing_nodes, key=lambda x: __final_score(x[0], x[1], c, current_player))
    else:
        return min(pre_existing_nodes, key=lambda x: __final_score(x[0], x[1], c, current_player))


def choose_actor_action(actor_net: NeuralNet, possible_actions: list[...],
                        valid_actions: list[...], game_state: list[int], d_policy: str = 'greedy'):
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
