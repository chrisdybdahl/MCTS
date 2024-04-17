class Node:
    def __init__(self, parent=None, game: object = None):
        """
        Initializes the node

        :param parent: Parent node
        """
        self.parent = parent
        self.game = game
        self.children = []
        self.visits = 0
        self.value = 0  # Action value?

    def expand(self):
        """
        Expands the current node

        """
        return ValueError

    def parent(self):
        """
        Returns the current node's parent

        :return: parent node
        """
        return self.parent

    def update_value(self, new_value):
        """
        Updates the current node's value'

        :param new_value: new value to update
        """
        self.value = new_value

    def get_value(self):
        """
        Returns the current node's value'

        :return: float representing current node's value'
        """
        return self.value


class MCTS:
    def __init__(self, M):
        """
        Initialize MCTS parameters here

        :param M (int) the number simulations each move in the actual game stems
        from, often 500 suffices (can be dynamic / time limited):
        """
        self.M = M

    def tree_search(self):
        """
        Traverse the tree from the root to a leaf node using tree policy

        """
        self.root = 1

    def expand_node(self, node: Node):
        """
        Generate some or all child states of a parent node , and connect the tree node housing the parent state
        (aka. parent node) to the nodes housing the child states (aka. child nodes)

        :param node: node to expand
        """
        self.root = 1

    def evaluate_leaf(self, node: Node) -> float:
        """
        Estimate the value of the leaf node in the tree by doing a rollout simulation using the default policy from
        the leaf node's state to a final state.
        It the vale of the node should be evaluated using an entire rollout game

        :return: value of the leaf node
        :param node: the node to evaluate
        """
        self.node = node
        return node.get_value()

    def backpropagation(self, node):
        """
        Passing the evaluation of a final state back up the tree, updating the relevant data at all nodes and edges
        on the path from the final state to the tree root

        :param node: the node to backpropagate from
        """
        self.node = node

    def run(self):
        """
        Run the MCTS

        :return: best action
        """
        return ValueError
