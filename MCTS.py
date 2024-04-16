class MCTS:
    def __init__(self, M):
        """
        Initialize your data here

        Parameters:
            M (int): the number simulations each move in the actual game stems from,
            often 500 suffices (can be dynamic / time limited)
        """
        self.M = M

    def tree_search(self):
        """
        Traverse the tree from the root to a leaf node using tree policy


        """
        self.root = 1

    def expand_node(self, node):
        """

        Parameters:
            node: node to expand

        Generate some or all child states of a parent node , and connect the tree node housing the parent state
        (aka. parent node) to the nodes housing the child states (aka. child nodes)

        """
        self.root = 1

    def evaluate_leaf(self, node):
        """
        Estimate the value of the leaf node in the tree by doing a rollout simulation using the default olicy from
        the leaf node's state to a final state.
        It the vale of the node should be evaluated using an entire rollout game

        Parameters:
            node (Node): the node to evaluate

        """
        self.node = node
        return node.value

    def backpropagation(self, node):
        """
        Passing the evaluation of a final state back up the tree, updating the relevant data at all nodes and edges
        on the path from the final state to the tree root

        Parameters:
            node (Node): the node to backpropagate from

        """
        self.node = node
