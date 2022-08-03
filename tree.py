import numpy as np


class Node:
    def __init__(self, parent_node, point_loc):
        self.parent_node = parent_node
        self.point_loc = point_loc

        self.children_nodes_list = []

    def add_children(self, children_node):
        self.children_nodes_list.append(children_node)

    def get_parent_node(self):
        return self.parent_node


if __name__ == "__main__":
    dummy_node = Node(parent_node=None, point_loc=None)
    node_queue = []

    Q1 = [np.array([0., 0.]), np.array([0., 0.1]), np.array([0.1, 0.]), np.array([0.1, 0.1])]
    for point_location in Q1:
        new_node = Node(parent_node=dummy_node, point_loc=point_location)
        dummy_node.add_children(new_node)
        node_queue.append(new_node)

    while len(node_queue) > 0:
        expanding_node = node_queue.pop()
        node_location = expanding_node.point_loc
        # next_Q_points_list = generate_next_Q_points(node_location)
        # for point in next_Q_points_list:
        #     new_node = Node(parent_node=dummy_node, point_loc=point)
        #     expanding_node.add_children(new_node)
        #     node_queue.append(new_node)


