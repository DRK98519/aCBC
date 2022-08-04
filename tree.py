import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry


def reach_set_calc(x, reach_range): # With given x and reach_range, generate the Polygon that represents R(x)
    p1 = geometry.Point(x[0]-reach_range, x[1]-reach_range)
    p2 = geometry.Point(x[0]+reach_range, x[1]-reach_range)
    p3 = geometry.Point(x[0]+reach_range, x[1]+reach_range)
    p4 = geometry.Point(x[0]-reach_range, x[1]+reach_range)
    vertex_list = [p1, p2, p3, p4]
    reach_set = geometry.Polygon(vertex_list)
    return reach_set


class OpNode:
    def __init__(self, node_loc, parent_opnode):    # Set node loc i_t and parent_node i_t-1 as the attributes to newly defined OpNode object
        self.state = node_loc
        self.parent_OpNode = parent_opnode
        self.children_OpNode_list = []

    def add_child_opnode(self):    # Define methods that determines child nodes list with current node (Specific to graph)
        if self.state == 1:
            self.children_OpNode_list = [1, 3]
        elif self.state == 2:
            self.children_OpNode_list = [1, 3]
        elif self.state == 3:
            self.children_OpNode_list = [2, 3]
        else:
            print('Invalid node.')

    def get_par_opnode(self):
        return self.parent_OpNode


if __name__ == "__main__":
    # An example of plotting multiple sets in one plot
    x = [np.array([1, 2]), np.array([2, 4]), np.array([4, 1])]
    figure = []
    # indx = 0
    # print(x[2])
    for x_coord in x:
        print(x_coord)
        R_set = reach_set_calc(x_coord, 8)
        hcoord, vcoord = R_set.exterior.xy
        print(hcoord)
        figure += plt.fill(hcoord, vcoord, alpha=0.25, facecolor='red', edgecolor='red')

    plt.show()

    # print(hcoord)



    # R_set = reach_set_calc(x, 8)
    # hcoord, vcoord = R_set.exterior.xy
    # fig, axs = plt.subplots()
    # axs.fill(hcoord, vcoord, alpha=0.25, facecolor='red', edgecolor='red')
    # print(f'hcoord: {hcoord} vcoord: {vcoord}')
    # plt.show()


# class Node:
#     def __init__(self, parent_node, point_loc):
#         self.parent_node = parent_node
#         self.point_loc = point_loc
#
#         self.children_nodes_list = []
#
#     def add_children(self, children_node):
#         self.children_nodes_list.append(children_node)
#
#     def get_parent_node(self):
#         return self.parent_node
#
#
# if __name__ == "__main__":  # When directly run the program, compiler start reading the code from here instead of the very beginning.
#     dummy_node = Node(parent_node=None, point_loc=None)
#     node_queue = []
#
#     Q1 = [np.array([0., 0.]), np.array([0., 0.1]), np.array([0.1, 0.]), np.array([0.1, 0.1])]
#     for point_location in Q1:
#         new_node = Node(parent_node=dummy_node, point_loc=point_location)
#         dummy_node.add_children(new_node)
#         node_queue.append(new_node)
#
#     while len(node_queue) > 0:
#         expanding_node = node_queue.pop()
#         node_location = expanding_node.point_loc
#         # next_Q_points_list = generate_next_Q_points(node_location)
#         # for point in next_Q_points_list:
#         #     new_node = Node(parent_node=dummy_node, point_loc=point)
#         #     expanding_node.add_children(new_node)
#         #     node_queue.append(new_node)


