import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry


def reach_set_calc(x_val, reach_range):     # With given x and reach_range, generate the Polygon that represents R(x)
    p1 = geometry.Point(x_val[0] - reach_range, x_val[1] - reach_range)
    p2 = geometry.Point(x_val[0] + reach_range, x_val[1] - reach_range)
    p3 = geometry.Point(x_val[0] + reach_range, x_val[1] + reach_range)
    p4 = geometry.Point(x_val[0] - reach_range, x_val[1] + reach_range)
    vertex_list = [p1, p2, p3, p4]
    reach_set = geometry.Polygon(vertex_list)
    return reach_set


# Given a rectangular set, return discrete points inside the set
def discrete_x_calc(poly, t_step, node_val, approx_para):
    """
    :type approx_para: int
    :type poly: Polygon
    :type t_step: int
    :type node_val: int
    """
    [hcoord_val, vcoord_val] = poly.exterior.xy     # Find the horizontal and vertical coordinates of poly's vertices
    discrete_x = []
    for x_hcoord in np.linspace(min(hcoord_val), max(hcoord_val), approx_para):
        for x_vcoord in np.linspace(min(vcoord_val), max(vcoord_val), approx_para):
            discrete_x += [x_hcoord, x_vcoord]
    discrete_x.t_step = t_step
    discrete_x.t_node = node_val
    return discrete_x


class OpNode:
    # Set node loc i_t and parent_node i_t-1 as the attributes to newly defined OpNode object
    def __init__(self, node_loc, parent_opnode):
        self.state = node_loc
        self.parent_OpNode = parent_opnode
        self.children_OpNode_list = []

    # Define methods that determine child nodes list with current node (Specific to graph)
    def add_child_opnode(self):
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


class ConvexBody:
    def __init__(self, t_step, op_node, vertices):
        self.t_step = t_step
        self.t_node = op_node
        self.region = geometry.Polygon(vertices)


if __name__ == "__main__":
    # Define given convex bodies Q
    Q = dict()
    T = 2   # Finite horizon
    vertices_list = {
        "t=0, i=1": [np.array([1, 1]), np.array([2, 1]), np.array([2, 2]), np.array([1, 2])],
        "t=0, i=2": [np.array([3, 3]), np.array([3, 2]), np.array([5, 2]), np.array([5, 3])],
        "t=0, i=3": [np.array([1, 7]), np.array([1, 4]), np.array([5, 4]), np.array([5, 7])],
        "t=1, i=1": [np.array([8, 3]), np.array([8, 1]), np.array([11, 1]), np.array([11, 3])],
        "t=1, i=2": [np.array([12, 6]), np.array([12, 3]), np.array([14, 3]), np.array([14, 6])],
        "t=1, i=3": [np.array([7, 6]), np.array([7, 4]), np.array([9, 4]), np.array([9, 6])],
        "t=2, i=1": [np.array([10, 6]), np.array([10, 5]), np.array([11, 5]), np.array([11, 6])],
        "t=2, i=2": [np.array([6, 10]), np.array([6, 7]), np.array([11, 7]), np.array([11, 10])],
        "t=2, i=3": [np.array([12, 10]), np.array([12, 8]), np.array([14, 8]), np.array([14, 10])]
    }
    colors = ['red', 'blue', 'green']
    indx = 0
    for t in range(T+1):
        nodes = [1, 2, 3]
        for node in nodes:
            Q[f'Q_t={t}^i={node}'] = ConvexBody(t_step=t, op_node=node, vertices=vertices_list[f't={t}, i={node}'])
            hcoord, vcoord = Q[f'Q_t={t}^i={node}'].region.exterior.xy
            plt.fill(hcoord, vcoord, alpha=0.25, facecolor=colors[indx], edgecolor=colors[indx])
        indx += 1

    plt.grid(True)
    plt.show()


    # Plot multiple reachable set together in a figure
    # x = [np.array([1, 2]), np.array([2, 4]), np.array([43, 13])]
    # for x_coord in x:
    #     R_set = reach_set_calc(x_coord, 8)
    #     hcoord, vcoord = R_set.exterior.xy
    #     print(R_set.exterior.xy)
    #     plt.fill(hcoord, vcoord, alpha=0.25, facecolor='red', edgecolor='red')
    #
    # plt.axis('equal')
    # plt.grid()
    # plt.show()


    # fig, axs = plt.subplots(1, 1)
    #
    # hcoord1, vcoord1 = reach_set_calc(x[0], 8).exterior.xy
    # hcoord2, vcoord2 = reach_set_calc(x[1], 8).exterior.xy
    # hcoord3, vcoord3 = reach_set_calc(x[2], 8).exterior.xy
    #
    # axs.fill(hcoord1, vcoord1, hcoord2, vcoord2, alpha=0.25, facecolor='red', edgecolor='red')
    # axs.grid(True)





class Node:
    def __init__(self, parent_node, point_loc):
        self.parent_node = parent_node
        self.point_loc = point_loc

        self.children_nodes_list = []

    def add_children(self, children_node):
        self.children_nodes_list.append(children_node)

    def get_parent_node(self):
        return self.parent_node


if __name__ == "__main__":  # When directly run the program, compiler start reading the code from here instead of the very beginning.
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
        next_Q_points_list = generate_next_Q_points(node_location)
        for point in next_Q_points_list:
            new_node = Node(parent_node=expanding_node, point_loc=point)
            expanding_node.add_children(new_node)
            node_queue.append(new_node)


