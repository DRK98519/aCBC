import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from numpy.linalg import norm
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]
# })


# Treat Player state and Opponent state both in State class with different state.value
def reach_set_calc(x_val, reach_range):  # With given x and reach_range, generate the Polygon that represents R(x)
    """
    :type x_val: list
    :type reach_range: float
    :return: reach_set: Polygon
    """
    p1 = geometry.Point(x_val[0] - reach_range, x_val[1] - reach_range)
    p2 = geometry.Point(x_val[0] + reach_range, x_val[1] - reach_range)
    p3 = geometry.Point(x_val[0] + reach_range, x_val[1] + reach_range)
    p4 = geometry.Point(x_val[0] - reach_range, x_val[1] + reach_range)
    vertex_list = [p1, p2, p3, p4]
    reach_set = geometry.Polygon(vertex_list)
    return reach_set


def graph_constraint(i_t):
    """
    :type i_t: int
    :return: reachable_nodes: list
    """
    if i_t == 1:
        reachable_nodes = [1, 3]
    elif i_t == 2:
        reachable_nodes = [1, 3]
    else:
        reachable_nodes = [2, 3]
    return reachable_nodes


# Given a rectangular set, return discrete points inside the set
# def discrete_x_calc(poly, t_node, approx_para):
def discrete_x_calc(poly, approx_para):
    """
    :type approx_para: int
    :type poly: Polygon
    :return discrete_x: list
    """
    [hcoord_val, vcoord_val] = poly.exterior.xy  # Find the horizontal and vertical coordinates of poly's vertices
    discrete_x = []
    for x_hcoord in np.linspace(min(hcoord_val), max(hcoord_val), approx_para):
        for x_vcoord in np.linspace(min(vcoord_val), max(vcoord_val), approx_para):
            discrete_x += [[x_hcoord, x_vcoord]]
            # t_node.add_child_state(discrete_x)
    # discrete_x.t_step = t_step
    # discrete_x.t_node = t_node
    # return t_node.children_state_list
    return discrete_x


def cost_calc(i_t, x_t_minus_1):
    """
    :type i_t: State
    :type x_t_minus_1: State
    :return: cost_list: list
    """
    cost_list = []
    for x_t in i_t.children_state_list:
        x_t_vec = np.array(x_t.state)
        x_t_minus_1_vec = np.array(x_t_minus_1.state)
        cost_list.append(norm(x_t_vec - x_t_minus_1_vec, 2))
    return cost_list


# def cost_calc(previous_x_val, x_val):
#     """
#     :type previous_x_val: list
#     :type x_val: list
#     :return: cost_val: float
#     """
#     previous_x_val = np.array(previous_x_val)
#     x_val = np.array(x_val)
#     return norm(previous_x_val - x_val, 2)  # Use 2-norm as cost structure


class State:
    # Set node loc i_t and parent_node i_t-1 as the attributes to newly defined OpNode object
    def __init__(self, state_value, parent_state, t_step, side):
        """
        :type state_value: int (Opponent), list (Player)
        :type parent_state: State / None (dummy_i)
        :type t_step: int, None(dummy i)
        :type side: str ('Player'/'Opponent')
        """
        self.state = state_value
        self.side = side
        self.parent_state = parent_state
        self.children_state_list = []
        self.t_step = t_step

    # Define methods that determine child nodes list with current node (Specific to graph)
    def add_child_state(self, child_state):
        """
        :type child_state: State
        """
        # if self.value == 1:
        #     self.children_OpNode_list = [1, 3]
        # elif self.value == 2:
        #     self.children_OpNode_list = [1, 3]
        # elif self.value == 3:
        #     self.children_OpNode_list = [2, 3]
        # else:
        #     print('Invalid node.')
        self.children_state_list.append(child_state)

    # def cost_calc(self, previous_state):
    #     x_t_minus_1_val = previous_state.value
    #     x_t_val = self.value
    #
    #     return self.parent_state


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


class Value:  # Value function
    def __init__(self, player_state, oppo_state, t_step, side, value, action):
        """
        :type player_state: State (None for U0)
        :type oppo_state: State (None for U0)
        :type t_step: int
        :type side: string
        :type value: float
        :type action: State
        """
        self.side = side
        self.player_state = player_state
        self.oppo_state = oppo_state
        self.t_step = t_step
        self.value = value
        self.action = action


# class Cost:
#     def __init__(self, x_t_minus_1_val, x_t_val, cost_val):
#         """
#         :type x_t_minus_1_val: State
#         :type x_t_val: State
#         :type cost_val: float
#         """
#         self.x_t_minus_1_val = x_t_minus_1_val
#         self.x_t_val = x_t_val
#         self.cost_val = cost_val


if __name__ == "__main__":
    ################################################ Discretization ################################################
    # Initialization
    state_queue = []
    value_eval_queue = []  # My idea: Keep adding x and i states in discretization loop for later optimal value approximation. When running optimal value approximation, input opponent action, return optimal player action. Or, directly compute optimal player value and optimal opponent value function together with corresponding optimal player and opponent actions.
    dummy_i = State(state_value=None, parent_state=None, t_step=None, side=None)
    Q = dict()
    T = 2  # Finite horizon
    node_vals = [1, 2, 3]
    colors = ['red', 'blue', 'green']
    line_style_list = ['-', '--', '-.']
    indx = 0
    disc_para = 10   # Constant reach range for any x here. Done for simplicity.
    num_nodes = 3
    # vertices_list = {
    #     "t=0, i=1": [np.array([1, 1]), np.array([2, 1]), np.array([2, 2]), np.array([1, 2])],
    #     "t=0, i=2": [np.array([3, 3]), np.array([3, 2]), np.array([5, 2]), np.array([5, 3])],
    #     "t=0, i=3": [np.array([4, 7]), np.array([4, 4]), np.array([5, 4]), np.array([5, 7])],
    #     "t=1, i=1": [np.array([8, 3]), np.array([8, 1]), np.array([11, 1]), np.array([11, 3])],
    #     "t=1, i=2": [np.array([12, 6]), np.array([12, 3]), np.array([14, 3]), np.array([14, 6])],
    #     "t=1, i=3": [np.array([7, 6]), np.array([7, 4]), np.array([9, 4]), np.array([9, 6])],
    #     "t=2, i=1": [np.array([10, 6]), np.array([10, 5]), np.array([11, 5]), np.array([11, 6])],
    #     "t=2, i=2": [np.array([6, 10]), np.array([6, 7]), np.array([11, 7]), np.array([11, 10])],
    #     "t=2, i=3": [np.array([12, 10]), np.array([12, 8]), np.array([14, 8]), np.array([14, 10])]
    # }
    # R = 8.5  # reach_range value (The value would vary depending on the given convex bodies)

    vertices_list = {
        "t=0, i=1": [np.array([8, 6]), np.array([10, 6]), np.array([10, 11]), np.array([8, 11])],
        "t=0, i=2": [np.array([26, 20]), np.array([32, 20]), np.array([32, 24]), np.array([26, 24])],
        "t=0, i=3": [np.array([9, 36]), np.array([11.5, 36]), np.array([11.5, 46]), np.array([9, 46])],
        "t=1, i=1": [np.array([10, 25]), np.array([15, 25]), np.array([15, 30]), np.array([10, 30])],
        "t=1, i=2": [np.array([34, 38]), np.array([36, 38]), np.array([36, 40]), np.array([34, 40])],
        "t=1, i=3": [np.array([44, 8]), np.array([48, 8]), np.array([48, 12]), np.array([44, 12])],
        "t=2, i=1": [np.array([14, 38]), np.array([22, 38]), np.array([22, 40]), np.array([14, 40])],
        "t=2, i=2": [np.array([14, 2]), np.array([28, 2]), np.array([28, 6]), np.array([14, 6])],
        "t=2, i=3": [np.array([45, 25]), np.array([50, 25]), np.array([50, 45]), np.array([45, 45])]
    }
    R = 36.7

    # Find a way to save all value functions and optimal strategy approximation and discrete x's for testing purpose.

    # Define given Convex Bodies
    fig1 = plt.figure(1)
    for t in range(T + 1):
        for node in node_vals:
            Q[f'Q_t={t}^i={node}'] = ConvexBody(t_step=t, node=node, vertices=vertices_list[f't={t}, i={node}'])
            hcoord, vcoord = Q[f'Q_t={t}^i={node}'].region.exterior.xy
            # plt.figure()
            plt.fill(hcoord, vcoord, alpha=0.25, facecolor=colors[indx], edgecolor=colors[indx],
                     linestyle=line_style_list[node-1], linewidth=2, label=fr"$Q_{t}^{{({node})}}$")
            disc_x0_list = discrete_x_calc(Q[f'Q_t={t}^i={node}'].region, disc_para)
            # print(disc_x0_list)

            # Plot discrete x0 in Q0 sets (For test)
            if t == 0:
                for disc_x0 in disc_x0_list:
                    plt.scatter(disc_x0[0], disc_x0[1], color=colors[indx], linewidths=0.1, marker='.')
                    plt.plot(disc_x0[0], disc_x0[1], color=colors[indx])
        indx += 1
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.title(fr"Given Convex Bodies (T = {T})")
    plt.savefig(f"Given Convex Bodies.png")
    plt.axis('equal')
    plt.show()

    # Define initial Opponent states i0 as State objects
    i0_vals = range(1, num_nodes+1)
    # i0_vals = [1, 2, 3]
    for i0_val in i0_vals:
        new_state = State(state_value=i0_val, parent_state=dummy_i, t_step=0, side='Opponent')
        dummy_i.add_child_state(new_state)
        state_queue.append(new_state)  # Add all initial i_0 into state_queue, start expanding here
    # print(state_queue)

    while len(state_queue) > 0:
        # t_val = 0
        # while t_val < T:
        expanding_state = state_queue.pop()  # Choose the last element in state_queue by default
        value_eval_queue.append(expanding_state)
        # i_t_val = expanding_state.state
        t_val = expanding_state.t_step
        # Find children states of initial Opponent state
        if expanding_state.t_step == 0 and expanding_state.side.lower() == 'opponent':
            i_t_val = expanding_state.state
            # t_val = expanding_state.t_step
            print(f'i_t_val={i_t_val}, t_val={t_val}, side={expanding_state.side}')
            disc_x_list = discrete_x_calc(Q[f'Q_t={t_val}^i={i_t_val}'].region, approx_para=disc_para)
            for disc_x in disc_x_list:
                # print(disc_x)
                new_state = State(state_value=disc_x, parent_state=expanding_state, t_step=t_val, side='Player')
                expanding_state.add_child_state(new_state)
                state_queue.append(new_state)
        # Find children states of a Player state
        elif expanding_state.side.lower() == 'player':
            i_t_val = expanding_state.parent_state.state
            # t_val = expanding_state.t_step
            print(f'x_t_val={expanding_state.state}, t_val={t_val}, side={expanding_state.side}')
            if t_val != T:  # x_t is not terminal state if t != T
                # Particular to given graph
                # if i_t_val == 1:
                #     new_state_list = [1, 3]
                # elif i_t_val == 2:
                #     new_state_list = [1, 3]
                # else:  # i_t_val == 3
                #     new_state_list = [2, 3]
                new_state_list = graph_constraint(i_t_val)

                for state_val in new_state_list:
                    new_state = State(state_val, parent_state=expanding_state, t_step=t_val + 1, side='Opponent')
                    expanding_state.add_child_state(new_state)
                    state_queue.append(new_state)
            # x_t is terminal state if t = T, no children state for x_T
            # if t_val > 0:

        elif expanding_state.side.lower() == 'opponent':
            i_t_val = expanding_state.state
            x_t_minus_1_val = expanding_state.parent_state.state
            # t_val = expanding_state.t_step
            # print(type(x_t_minus_1_val))
            print(f'x_t_minus_1_val={x_t_minus_1_val}, i_t_val={i_t_val}, t_val={t_val}, side={expanding_state.side}')
            R_set = reach_set_calc(x_t_minus_1_val, reach_range=R)     # 20
            R_intersect_Q = Q[f'Q_t={t_val}^i={i_t_val}'].region.intersection(R_set)
            # print(list(R_intersect_Q.exterior.coords))
            # R_intersect_Q = ConvexBody(t_val, i_t_val, list(R_intersect_Q.exterior.coords))
            # print(type(R_intersect_Q))
            disc_x_list = discrete_x_calc(R_intersect_Q, approx_para=disc_para)
            for disc_x in disc_x_list:
                # print(disc_x)
                new_state = State(state_value=disc_x, parent_state=expanding_state, t_step=t_val, side='Player')
                expanding_state.add_child_state(new_state)
                state_queue.append(new_state)
        # print(state_queue)
        print('Running')
    print('Discretization Done')
    # print(value_eval_queue.pop().parent_state.state)
    # print(value_eval_queue.pop().parent_state.parent_state.state)
    ################################################ End Here ################################################

    ####################################### Optimal Value Approximation #######################################
    # Get all leaf branches in the tree just for observation
    out = [state for state in value_eval_queue if state.t_step == T and state.side.lower() == 'player']
    print(len(out))
    # Remove all leaf branches in the tree
    value_eval_queue = [state for state in value_eval_queue if state.t_step != T or state.side.lower() != 'player']
    value_list = []
    UV_dict = dict()
    # for state in out:
    #     print(state.children_state_list)
    #     print(1)

    # Test section for the code only (Delete later)
    # last_val_list = []
    # # We need to reorder value_eval_queue (i0's, x0's, i1's, x1's, i2's)
    #
    value_eval_queue1 = value_eval_queue[:]
    # # value_eval_queue2 = value_eval_queue
    # for count in range(10):
    #     taken_val = value_eval_queue1.pop()
    #     last_val_list.append(taken_val)
    # for last_val in last_val_list:
    #     print(last_val.state)
    #     print(f'time step: {last_val.t_step}')
    #     print(last_val.side)
    print('\n\n\n')

    # print(len(value_eval_queue))
    counter = 0
    while len(value_eval_queue) > 0:
        # Two given states in value functions
        print("\n")
        counter += 1
        print(f"Iteration Number: {counter}")

        given_state_1 = value_eval_queue.pop()
        given_state_2 = given_state_1.parent_state
        t_val = given_state_1.t_step
        print(f"State 1: {given_state_1.state} t_val: {t_val} side: {given_state_1.side}")
        print(f'State 2: {given_state_2.state} t_val: {given_state_2.t_step} side: {given_state_2.side}')

        if t_val == T:  # Case of given x_T_minus_1 and i_T, select x_T
            c_list = cost_calc(given_state_1, given_state_2)
            # c_list = []
            # # x_T_list = []
            # for x_T in given_state_1.children_state_list:
            #     cost_val = cost_calc(given_state_2.state, x_T.state)
            #     # cost_val = norm(x_T.state - given_state_2.state, 2)
            #     c_list.append(cost_val)
            #     # cost = Value(given_state_2, given_state_1, t_val, 'Player', cost_val, action=x_T)
            opt_val = min(c_list)  # Value of V_T
            opt_indx = c_list.index(opt_val)
            UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'] = \
                Value(player_state=given_state_2, oppo_state=given_state_1, t_step=t_val, side='Player', value=opt_val,
                      action=given_state_1.children_state_list[opt_indx])

            print(f"Value side: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].side}, "
                  f"Value t_step: {t_val}, "
                  f"V_{t_val}: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].value}")
            # print(f"V_2: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].value}")

        elif given_state_1.side.lower() == 'player':    # Case of given x_t-1 and i_t-1, select i_t (1 <= t <= T)
            V_t_val_list = []
            for i_t in given_state_1.children_state_list:
                V_t = UV_dict[f'V_t={t_val+1} ({given_state_1.state}, {i_t.state})']

                # print(f"V_{t_val+1}: {V_t.value}")

                V_t_val_list.append(V_t.value)

            opt_val = max(V_t_val_list)
            opt_indx = V_t_val_list.index(opt_val)
            UV_dict[f'U_t={t_val+1} ({given_state_1.state}, {given_state_2.state})'] = \
                Value(player_state=given_state_1, oppo_state=given_state_2, t_step=t_val+1, side='Opponent',
                      value=opt_val, action=given_state_1.children_state_list[opt_indx])

            print(f"Value side: {UV_dict[f'U_t={t_val+1} ({given_state_1.state}, {given_state_2.state})'].side}, "
                  f"Value t_step: {t_val+1}, "
                  f"U_{t_val+1}: {UV_dict[f'U_t={t_val+1} ({given_state_1.state}, {given_state_2.state})'].value}, "
                  f"opt_i_{t_val+1}: "
                  f"{UV_dict[f'U_t={t_val+1} ({given_state_1.state}, {given_state_2.state})'].action.state}")

        elif given_state_1.side.lower() == 'opponent':  # Case of given x_t-1 and i_t, select x_t. t!=T
            t_val = given_state_1.t_step
            if t_val != 0:  # 1 <= t < T (Discuss t = 0 case separately)
                # V_t = min(c + U_t+1)
                c_list = cost_calc(given_state_1, given_state_2)
                U_t_plus_1_list = []
                for x_t in given_state_1.children_state_list:
                    U_t_plus_1_list.append(UV_dict[f'U_t={t_val+1} ({x_t.state}, {given_state_1.state})'].value)

                val_array = np.array(c_list) + np.array(U_t_plus_1_list)
                opt_val = min(val_array)     # V_t = min(c + U_t+1)
                opt_indx = val_array.argmin()
                UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'] = \
                    Value(player_state=given_state_2, oppo_state=given_state_1, t_step=t_val, side='Player',
                          value=opt_val, action=given_state_1.children_state_list[opt_indx])

                print(f"Value side: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].side}, "
                      f"Value t_step: {t_val}, "
                      f"V_{t_val}: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].value},"
                      f"opt_x{t_val}: "
                      f"{UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].action.state}")
                # print(f"V_{t_val}: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].value}")

            else:   # t = 0 Given i_0, (Here, i_0.parent_state=dummy_i) find optimal x_0
                U_1_list = []
                for x_0 in given_state_1.children_state_list:
                    U_1_list.append(UV_dict[f"U_t={t_val+1} ({x_0.state}, {given_state_1.state})"].value)

                    # For test purpose
                    # print(f"U_{1}: {UV_dict[f'U_t={t_val+1} ({x_0.state}, {given_state_1.state})'].value}, "
                    #       f"x0 state: {x_0.state}")
                    # print(f"x0 check: "
                    #       f"{UV_dict[f'U_t={t_val+1} ({x_0.state}, {given_state_1.state})'].player_state.state}")

                opt_val = min(U_1_list)
                opt_indx = U_1_list.index(opt_val)
                UV_dict[f"V_t={t_val} ({given_state_2.state}, {given_state_1.state})"] = \
                    Value(player_state=given_state_2, oppo_state=given_state_1, t_step=t_val, side='Player',
                          value=opt_val, action=given_state_1.children_state_list[opt_indx])

                # For test (delete later)
                if given_state_1.state == 2:
                    opt_x0 = UV_dict[f"V_t={t_val} ({given_state_2.state}, {given_state_1.state})"].action
                    print(f"i_{t_val}: {given_state_1.state}, opt x_{t_val}: {opt_x0.state}")

                print(f"Value side: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].side}, "
                      f"Value t_step: {t_val}, "
                      f"V_{t_val}: {UV_dict[f'V_t={t_val} ({given_state_2.state}, {given_state_1.state})'].value}")

    print('Optimal value approximation loop done. Evaluate U0 next.\n\n')

    # Case of no given states, referring to U_0
    V_0_list = []
    for i_0 in dummy_i.children_state_list:
        V_0_list.append(UV_dict[f'V_t={0} ({dummy_i.state}, {i_0.state})'].value)
    opt_val = max(V_0_list)     # U0 = max(V0)
    opt_indx = V_0_list.index(opt_val)
    UV_dict[f"U_t={0} ({dummy_i.state}, {None})"] = \
        Value(player_state=None, oppo_state=None, t_step=0, side="Opponent",
              value=opt_val, action=dummy_i.children_state_list[opt_indx])

    print(f"Value side: {UV_dict[f'U_t={0} ({dummy_i.state}, {None})'].side}, "
          f"Value t_step: {0}, U_{0}: {UV_dict[f'U_t={0} ({dummy_i.state}, {None})'].value}, "
          f"opt_i0: {UV_dict[f'U_t={0} ({dummy_i.state}, {None})'].action.state}")

    print("Optimal Value Approximation Done")
    ################################################ End Here ################################################

    ################################################ Display ################################################
    # Let user be opponent, show player optimal action approximation for demo (Plot them)
    # t = 0
    msg = ''
    oppo_hist = dict()
    while msg.lower() != 'n':
        t = 0
        opt_player_action = None
        opt_player_state = None
        tot_cost = 0
        while t <= T:
            fig2 = plt.figure(2)
            print(f"\nt={t}")

            # I reassigned opt_player_action to avoid warning about potentially undefined opt_player_action in else
            # statements
            prev_x_action = opt_player_action
            prev_x_state = opt_player_state     # Reassignment needed for later generation of R_intersect_Q

            # try:
            #     oppo_node = int(input("Enter opponent action: "))
            # except ValueError:
            #     print("Invalid selection of node. Try again.")
            oppo_node = int(input("Enter opponent action: "))
            if t == 0:
                if oppo_node not in [1, 2, 3]:
                    print("Invalid selection of node. Try again.")
                else:
                    oppo_action = [state for state in value_eval_queue1 if state.state == oppo_node and state.t_step == t]
                    oppo_action = oppo_action[0]

                    oppo_hist[f"i{t}"] = oppo_action.state

                    # Plot Q0
                    hcoord, vcoord = Q[f"Q_t={t}^i={oppo_node}"].region.exterior.xy
                    plt.fill(hcoord, vcoord, alpha=0.5, facecolor='red', edgecolor='red')
                    # Plot discrete x0 in Q0
                    disc_x0_list = discrete_x_calc(Q[f'Q_t={t}^i={oppo_node}'].region, disc_para)
                    for disc_x0 in disc_x0_list:
                        plt.scatter(disc_x0[0], disc_x0[1], color=colors[t], linewidths=0.1, marker='.')

                    opt_player_action = UV_dict[f"V_t={t} ({oppo_action.parent_state.state}, {oppo_action.state})"].action
                    opt_player_state = opt_player_action.state  # value of optimal x0 approximation in list form

                    print(f"Optimal Player State Approximation: {opt_player_state}")

                    # previous_x_state = opt_player_state
                    plt.scatter(opt_player_state[0], opt_player_state[1], color='black', linewidths=0.1, marker='.')
                    t += 1
            else:   # t != 0
                if oppo_node not in [state.state for state in prev_x_action.children_state_list]:
                    print("Invalid selection of node. Try again.")
                else:
                    oppo_action = \
                        [state for state in value_eval_queue1 if state.state == oppo_node and state.parent_state ==
                         prev_x_action][0]

                    oppo_hist[f"i{t}"] = oppo_action.state

                    # oppo_action = oppo_action[0]
                    # Find optimal x_t state approximation
                    opt_player_action = UV_dict[f"V_t={t} ({prev_x_action.state}, {oppo_action.state})"].action
                    opt_player_state = opt_player_action.state

                    print(f"Optimal Player State Approximation: {opt_player_state}")

                    # Plot Q
                    hcoord, vcoord = Q[f"Q_t={t}^i={oppo_action.state}"].region.exterior.xy
                    plt.fill(hcoord, vcoord, alpha=0.25, facecolor=colors[t], edgecolor=colors[t])

                    # Plot R(previous_x) intersect Q
                    R_set = reach_set_calc(prev_x_state, R)
                    R_intersect_Q = Q[f"Q_t={t}^i={oppo_action.state}"].region.intersection(R_set)
                    hcoord_inter, vcoord_inter = R_intersect_Q.exterior.xy
                    plt.fill(hcoord_inter, vcoord_inter, alpha=0.5, facecolor=colors[t], edgecolor=colors[t])
                    # Plot discrete x in R_intersect_Q
                    disc_x_list = discrete_x_calc(R_intersect_Q, approx_para=disc_para)
                    for disc_x in disc_x_list:
                        plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.1, marker='.')
                    # Plot optimal x_t state approximation
                    plt.scatter(opt_player_state[0], opt_player_state[1], facecolor='black', linewidths=0.1, marker='.')
                    # Connect optimal x_t state approximation to prev_x_state
                    plt.plot([prev_x_state[0], opt_player_state[0]], [prev_x_state[1], opt_player_state[1]],color='black')

                    tot_cost += norm(np.array(prev_x_state) - np.array(opt_player_state), 2)
                    print(f"Total Cost: {tot_cost}")
                    t += 1
        plt.grid(True)
        plt.axis('equal')
        plt.title(fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, $i_2={oppo_hist['i2']}$ "
                  f"\n{disc_para}x{disc_para} Discretization, Total Cost={round(tot_cost, 4)}")
        # plt.show()
        plt.savefig(f"Opponent History {oppo_hist['i0']}{oppo_hist['i1']}{oppo_hist['i2']}, disc_para={disc_para}")
        plt.show()
        msg = input("Rerun? [Y/N] ")

    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

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
