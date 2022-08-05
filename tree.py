import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry


# Treat Player state and Opponent state both in State class with different state.value
def reach_set_calc(x_val, reach_range):     # With given x and reach_range, generate the Polygon that represents R(x)
    p1 = geometry.Point(x_val[0] - reach_range, x_val[1] - reach_range)
    p2 = geometry.Point(x_val[0] + reach_range, x_val[1] - reach_range)
    p3 = geometry.Point(x_val[0] + reach_range, x_val[1] + reach_range)
    p4 = geometry.Point(x_val[0] - reach_range, x_val[1] + reach_range)
    vertex_list = [p1, p2, p3, p4]
    reach_set = geometry.Polygon(vertex_list)
    return reach_set


# Given a rectangular set, return discrete points inside the set
# def discrete_x_calc(poly, t_node, approx_para):
def discrete_x_calc(poly, approx_para):
    """
    :type approx_para: int
    :type poly: Polygon
    """
    [hcoord_val, vcoord_val] = poly.exterior.xy     # Find the horizontal and vertical coordinates of poly's vertices
    discrete_x = []
    for x_hcoord in np.linspace(min(hcoord_val), max(hcoord_val), approx_para):
        for x_vcoord in np.linspace(min(vcoord_val), max(vcoord_val), approx_para):
            discrete_x += [[x_hcoord, x_vcoord]]
            # t_node.add_child_state(discrete_x)
    # discrete_x.t_step = t_step
    # discrete_x.t_node = t_node
    # return t_node.children_state_list
    return discrete_x

class State:
    # Set node loc i_t and parent_node i_t-1 as the attributes to newly defined OpNode object
    def __init__(self, state_val, parent_state, t_step, side):
        """
            :type state_val: int (Opponent), list (Player)
            :type parent_state: State / None (dummy_i)
            :type t_step: int, None(dummy i)
            :type side: str ('Player'/'Opponent')
            """
        self.value = state_val
        self.side = side
        self.parent_state = parent_state
        self.children_state_list = []
        self.t_step = t_step

    # Define methods that determine child nodes list with current node (Specific to graph)
    def add_child_state(self, child_state):
        # if self.value == 1:
        #     self.children_OpNode_list = [1, 3]
        # elif self.value == 2:
        #     self.children_OpNode_list = [1, 3]
        # elif self.value == 3:
        #     self.children_OpNode_list = [2, 3]
        # else:
        #     print('Invalid node.')
        self.children_state_list.append(child_state)

    def get_par_state(self):
        return self.parent_state


class ConvexBody:
    def __init__(self, t_step, node, vertices):
        """
        :type t_step: int
        :type node: int
        :type vertices: list
        """
        self.t_step = t_step
        self.t_node = node
        self.region = geometry.Polygon(vertices)


if __name__ == "__main__":
    state_queue = []
    dummy_i = State(state_val=None, parent_state=None, t_step=None, side=None)
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

    node_val = [1, 2, 3]
    colors = ['red', 'blue', 'green']
    indx = 0

    # Define given Convex Bodies
    for t in range(T+1):
        for node in node_val:
            Q[f'Q_t={t}^i={node}'] = ConvexBody(t_step=t, node=node, vertices=vertices_list[f't={t}, i={node}'])
            hcoord, vcoord = Q[f'Q_t={t}^i={node}'].region.exterior.xy
            plt.figure(1)
            plt.fill(hcoord, vcoord, alpha=0.25, facecolor=colors[indx], edgecolor=colors[indx])
            disc_x0_list = discrete_x_calc(Q[f'Q_t={t}^i={node}'].region, 2)
            # print(disc_x0_list)
            # Plot discrete x0 in Q0 sets (For test)
            if t == 0:
                for disc_x0 in disc_x0_list:
                    plt.scatter(disc_x0[0], disc_x0[1], color=colors[indx], linewidths=0.1, marker='.')
                # plt.plot(disc_x0[0], disc_x0[1], color=colors[indx])
        indx += 1
    plt.grid(True)
    plt.show()
    plt.close()

    # Define initial Opponent states i0 as State objects
    i0_vals = [1, 2, 3]
    for i0_val in i0_vals:
        new_state = State(state_val=i0_val, parent_state=dummy_i, t_step=0, side='Opponent')
        dummy_i.add_child_state(new_state)
        state_queue.append(new_state)   # Add all initial i_0 into state_queue, start expanding here
    print(state_queue)

    while len(state_queue)>0:
        # t_val = 0
        # while t_val < T:
        expanding_state = state_queue.pop() # Choose the last element in state_queue by default
        # Find children states of initial Opponent state
        if expanding_state.t_step == 0 and expanding_state.side.lower() == 'opponent':
            i_t_val = expanding_state.value
            t_val = expanding_state.t_step
            print(f'i_t_val={i_t_val}, t_val={t_val}, side={expanding_state.side}')
            disc_x_list = discrete_x_calc(Q[f'Q_t={t_val}^i={i_t_val}'].region, approx_para=2)
            for disc_x in disc_x_list:
                new_state = State(state_val=disc_x, parent_state=expanding_state, t_step=t_val, side='Player')
                expanding_state.add_child_state(new_state)
                state_queue.append(new_state)
        # Find children states of a Player state
        elif expanding_state.side.lower() == 'player':
            i_t_val = expanding_state.parent_state.value
            t_val = expanding_state.t_step
            print(f'x_t_val={expanding_state.value}, t_val={t_val}, side={expanding_state.side}')
            if t_val != T:  # x_t is not terminal state if t != T
                if i_t_val == 1:
                    new_state_list = [1, 3]
                elif i_t_val == 2:
                    new_state_list = [1, 3]
                else:   # i_t_val == 3
                    new_state_list = [2, 3]

                for state_val in new_state_list:
                    new_state = State(state_val, parent_state=expanding_state, t_step=t_val+1, side='Opponent')
                    expanding_state.add_child_state(new_state)
                    state_queue.append(new_state)
            # x_t is terminal state if t = T, no children state for x_T
        elif expanding_state.side.lower() == 'opponent':
            i_t_val = expanding_state.value
            x_t_minus_1_val = expanding_state.parent_state.value
            t_val = expanding_state.t_step
            print(f'x_t_minus_1_val={x_t_minus_1_val}, i_t_val={i_t_val}, t_val={t_val}, side={expanding_state.side}')
            R_set = reach_set_calc(x_t_minus_1_val, reach_range=20)
            R_intersect_Q = Q[f'Q_t={t_val}^i={i_t_val}'].region.intersection(R_set)
            # print(list(R_intersect_Q.exterior.coords))
            # R_intersect_Q = ConvexBody(t_val, i_t_val, list(R_intersect_Q.exterior.coords))
            print(type(R_intersect_Q))
            disc_x_list = discrete_x_calc(R_intersect_Q, approx_para=2)
            for disc_x in disc_x_list:
                # print(disc_x)
                new_state = State(state_val=disc_x, parent_state=expanding_state, t_step=t_val, side='Player')
                expanding_state.add_child_state(new_state)
                state_queue.append(new_state)
        # print(state_queue)
        print('Running')
    print('Done')
    print(state_queue)
    ################################################ Continue Here ################################################

    # node1 = State(state_val=1, parent_state=dummy_i, t_step=0, side='Opponent')
    # node2 = State(state_val=2, parent_state=dummy_i, t_step=0, side='Opponent')
    # node3 = State(state_val=3, parent_state=dummy_i, t_step=0, side='Opponent')
    # node_set = [node1, node2, node3]
    #
    # # print(node_set)
    #
    # # Generate list of discrete x0 in all convex bodies Q_0
    # colors = ['red', 'blue', 'green']
    # indx = 0
    # for t in range(T+1):
    #     for node in node_set:
    #         # Construct all given convex bodies as ConvexBody objects
    #         Q[f'Q_t={t}^i={node.value}'] = ConvexBody(t_step=t, node=node, vertices=vertices_list[f't={t}, i={node.value}'])
    #         hcoord, vcoord = Q[f'Q_t={t}^i={node.value}'].region.exterior.xy
    #         plt.fill(hcoord, vcoord, alpha=0.25, facecolor=colors[indx], edgecolor=colors[indx])
    #         # Generate list of discrete x0 in Q_0 sets
    #         if t == 0:
    #             disc_x0_list = discrete_x_calc(Q[f'Q_t={t}^i={node.value}'], node, 2)
    #             # print(f'({disc_x0[0]}, {disc_x0[1]})')
    #             print(disc_x0_list)
    #             for disc_x0 in disc_x0_list:
    #                 plt.scatter(disc_x0[0], disc_x0[1], color=colors[indx], linewidths=0.1, marker='.')
    #             # plt.plot(disc_x0[0], disc_x0[1], color=colors[indx])
    #     indx += 1
    # plt.grid(True)
    # plt.show()
    #
    # # Tree Structure (Game Dynamics) Start with first move of Player
    # state_queue += node_set
    # # for t in range(T+1):
    #
    # for disc_x0 in disc_x0_list:
    #     new_state = State(state_val=disc_x0,parent_state=)


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
#         next_Q_points_list = generate_next_Q_points(node_location)
#         for point in next_Q_points_list:
#             new_node = Node(parent_node=expanding_node, point_loc=point)
#             expanding_node.add_children(new_node)
#             node_queue.append(new_node)


