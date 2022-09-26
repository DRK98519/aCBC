import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
from numpy.linalg import norm
from random import *
import pickle


def reach_set_calc(x_val, reach_range):
    """
    :type x_val: list
    :type reach_range: float
    :return: reach_set: Polygon
    Description: With given x and reach_range, generate a rectangular set centering at x with side length
                 2 * reach_range
    """
    p1 = geometry.Point(x_val[0] - reach_range, x_val[1] - reach_range)
    p2 = geometry.Point(x_val[0] + reach_range, x_val[1] - reach_range)
    p3 = geometry.Point(x_val[0] + reach_range, x_val[1] + reach_range)
    p4 = geometry.Point(x_val[0] - reach_range, x_val[1] + reach_range)
    vertex_list = [p1, p2, p3, p4]
    reach_set = geometry.Polygon(vertex_list)
    return reach_set


def all_Q_plt(Q, node_num, color_set, line_style_set, T, plt_scale):
    """
    :param Q: dict
    :param node_num: int
    :param color_set: list
    :param line_style_set: list
    :param T: int
    :return: None
    """
    # Plot all given convex sets
    for t_val in range(T + 1):
        for node in range(1, node_num + 1):
            hcoord_q, vcoord_q = Q[f"Q_t={t_val}^i={node}"].region.exterior.xy
            plt.fill(hcoord_q, vcoord_q, alpha=0.1, facecolor=color_set[t_val], edgecolor=color_set[t_val],
                     linewidth=2,
                     linestyle=line_style_set[node - 1], label=fr"$Q_{t_val}^{{({node})}}$")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.axis(plt_scale)
    return None


def set_plotter(set, plt_color, alpha_val):
    """
    :type set: Polygon
    :type plt_color: string
    :type alpha_val: float
    :return: None
    """
    hcoord, vcoord = set.exterior.xy
    plt.fill(hcoord, vcoord, alpha=alpha_val, facecolor=plt_color, edgecolor=plt_color)


def game_plt(full_tree, oppo_action, Q, colors, UV_dict, t, prev_x_action, R, control):
    """
    :param full_tree: list
    :param oppo_action: State
    :param Q: dict
    :param colors: list
    :param UV_dict: dict
    :param t: int
    :param prev_x_action: State
    :param R: float
    :return: player_action: State
    """
    prev_x_state = prev_x_action.state
    # Plot selected Qt
    Qt = Q[f"Q_t={t}^i={oppo_action.state}"].region
    set_plotter(Qt, colors[t], alpha_val=0.25)

    # Plot the set discretized over
    if t == 0:
        set = Qt
    else:
        R_set = reach_set_calc(prev_x_action.state, R)
        set = Qt.intersection(R_set)
    set_plotter(set, colors[t], alpha_val=0.5)

    # Find disc xt in the set
    disc_x_list = [action.state for action in full_tree if action.parent_state == oppo_action]

    # Plot disc xt in the set
    for disc_x in disc_x_list:
        plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.5, marker='.')

    if control in ['1', '2']:   # Opt pl vs. Opt op or Opt pl vs. Sub-opt op
        # Find optimal player action xt
        player_action = UV_dict[f"V_t={t} ({prev_x_action.state}, {oppo_action.state})"].action
        player_state = player_action.state

    else:   # Control == '3', Sub-opt pl vs. Opt op
        # Randomly pick player action xt
        player_action = choice([action for action in full_tree if action.parent_state == oppo_action])
        player_state = player_action.state

    # Plot optimal xt in the set
    plt.scatter(player_state[0], player_state[1], color='black', linewidths=0.1, marker='.')

    if t != 0:
        # Connect optimal xt state approximation to prev_x_state
        plt.plot([prev_x_state[0], player_state[0]], [prev_x_state[1], player_state[1]], color='black')
    return player_action


# Given a rectangular set, return discrete points inside the set
# def discrete_x_calc(poly, t_node, approx_para):
def discrete_x_calc(poly, approx_para, bound_rmv):
    """
    :type approx_para: int
    :type poly: Polygon
    :type bound_rmv: string
    :return discrete_x: list
    """
    [hcoord_val, vcoord_val] = poly.exterior.xy  # Find the horizontal and vertical coordinates of poly's vertices
    discrete_x = []
    for x_hcoord in np.linspace(min(hcoord_val), max(hcoord_val), approx_para):
        for x_vcoord in np.linspace(min(vcoord_val), max(vcoord_val), approx_para):
            discrete_x += [[x_hcoord, x_vcoord]]

    discrete_x_copy = discrete_x[:]  # Back up original discrete list
    if bound_rmv.lower() == 'y':
        # Find discrete x on the boundary
        bound_x = []
        for x_eval in discrete_x_copy:
            if x_eval[0] in hcoord_val or x_eval[1] in vcoord_val:
                bound_x.append(x_eval)

                # Remove discrete x on the boundary from original discrete list
                discrete_x.pop(discrete_x.index(x_eval))
        print(bound_x)
    return discrete_x


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
        self.children_state_list.append(child_state)


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


if __name__ == "__main__":
    #################################### Display ####################################
    """
    Plot trajectory result. Allow user to control opponent, while player always applies its optimal strategy by 
    computer. Allow re-run functionality.
    IDEA: Separate game computation section (discretization, optimal value approximation) and game play section
    (display) as two different .py files
    """
    method = ''
    while method != 'terminate':
        method = input("Which method to use or terminate the program? [Old/New/Terminate]: ")
        if method.lower() == 'new':

            # Load tree info new files into the program
            tree_file = open('tree_info_new', 'rb')
            tree_info = pickle.load(tree_file)
            tree_file.close()

            # Assign keys from tree_info to variables in this program
            Q = tree_info['Q']
            full_tree = tree_info['full_tree']
            UV_dict = tree_info['UV_dict']
            T = tree_info['T']
            num_nodes = tree_info['num_nodes']
            colors = tree_info['colors']
            line_style_list = tree_info['line_style_list']
            plt_scale = tree_info['plt_scale']
            extra_disc_para = tree_info['extra_disc_para']
            scale_para = tree_info['scale_para']
            dummy_i = tree_info['dummy_i']
            performance_bound = tree_info['performance_bound']
            R = tree_info['R']
            method = tree_info['method']

            msg = ''
            oppo_hist = dict()
            tot_cost = 0

            while msg.lower() != 'n':
                all_Q_plt(Q, num_nodes, colors, line_style_list, T, plt_scale)
                ## Still need to add opt player vs. opt opponent
                control = input("Opt Player vs. Opt Opponent [1] / Opt Player vs. Sub-opt Opponent [2] / Sub-opt "
                                "Player vs. Opt Opponent [3]? ")
                if control not in ['1', '2', '3']:
                    print('Invalid game setting. Select again.')
                else:  # Valid game setting
                    if control == '2':  # Case of Player (PC) vs. Opponent (User)
                        # Initialize the game
                        t = 0
                        opt_player_action = dummy_i
                        opt_player_state = dummy_i.state

                        while t <= T:
                            prev_x_action = opt_player_action
                            prev_x_state = opt_player_state

                            oppo_node = int(input("Enter opponent action: "))
                            if t == 0:
                                if oppo_node not in range(num_nodes + 1):
                                    print("Invalid selection of node. Try again.")
                                else:  # oppo_node is valid with given graph
                                    oppo_hist[f"i{t}"] = oppo_node  # Store selected oppo_node to oppo_hist

                                    oppo_action = [action for action in full_tree if action.state == oppo_node and
                                                   action.parent_state == prev_x_action][0]

                                    # Plot the game process
                                    opt_player_action = game_plt(full_tree, oppo_action, Q, colors, UV_dict, t,
                                                                 prev_x_action, R, control)
                                    opt_player_state = opt_player_action.state
                                    # # Plot selected Q0
                                    # Q0 = Q[f"Q_t={t}^i={oppo_action.state}"].region
                                    # set_plotter(Q0, colors[t], alpha_val=0.25)
                                    # set_plotter(Q0, colors[t], alpha_val=0.5)
                                    #
                                    # # Find disc x0 in Q0
                                    # disc_x_list = [action.state for action in full_tree if action.parent_state ==
                                    #                oppo_action]
                                    #
                                    # # Plot disc x0 in Q0
                                    # for disc_x in disc_x_list:
                                    #     plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.5, marker='.')
                                    #
                                    # # Find optimal player action x0 (Can be made UDF)
                                    # opt_player_action = UV_dict[f"V_t={t} ({prev_x_action.state}, "
                                    #                             f"{oppo_action.state})"].action
                                    # opt_player_state = opt_player_action.state
                                    #
                                    # # Plot optimal x0 in Q0
                                    # plt.scatter(opt_player_state[0], opt_player_state[1], color='black', linewidths=0.1,
                                    #             marker='.')
                                    t += 1  # Update t value

                                    # Display
                                    print(f"Optimal Player State Approximation: {opt_player_state}")

                            else:  # t != 0
                                if oppo_node not in [action.state for action in prev_x_action.children_state_list]:
                                    print("Invalid selection of node. Try again.")
                                else:  # selected oppo_node is a reachable node
                                    oppo_hist[f"i{t}"] = oppo_node

                                    oppo_action = [action for action in full_tree if action.state == oppo_node and
                                                   action.parent_state == prev_x_action][0]

                                    # Plot the game process
                                    opt_player_action = game_plt(full_tree, oppo_action, Q, colors, UV_dict, t,
                                                                 prev_x_action, R, control)
                                    opt_player_state = opt_player_action.state
                                    # # Plot selected Qt
                                    # Qt = Q[f"Q_t={t}^i={oppo_action.state}"].region
                                    # set_plotter(Qt, colors[t], alpha_val=0.25)
                                    #
                                    # # Plot R(previous_x) intersect Qt
                                    # R_set = reach_set_calc(prev_x_state, R)
                                    # R_intersect_Q = Qt.intersection(R_set)
                                    # set_plotter(R_intersect_Q, colors[t], alpha_val=0.5)
                                    #
                                    # # Find disc xt in R(previous_x) intersect Qt
                                    # disc_x_list = [action.state for action in full_tree if action.parent_state ==
                                    #                oppo_action]
                                    #
                                    # # Plot disc xt in R(previous_x) intersect Qt
                                    # for disc_x in disc_x_list:
                                    #     plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.5, marker='.')
                                    #
                                    # # Find optimal player action xt in R(previous_x) intersect Qt
                                    # opt_player_action = UV_dict[f"V_t={t} ({prev_x_action.state}, {oppo_action.state}"
                                    #                             f")"].action
                                    # opt_player_state = opt_player_action.state
                                    #
                                    # # Plot optimal x_t in R(previous_x) intersect Qt
                                    # plt.scatter(opt_player_state[0], opt_player_state[1], color='black', linewidths=0.1,
                                    #             marker='.')
                                    #
                                    # # Connect optimal x_t state approximation to prev_x_state
                                    # plt.plot([prev_x_state[0], opt_player_state[0]],
                                    #          [prev_x_state[1], opt_player_state[1]], color='black')

                                    # Update cost
                                    tot_cost += norm(np.array(prev_x_state) - np.array(opt_player_state), 2)
                                    t += 1  # Update t value

                                    # Display
                                    print(f"Optimal Player State Approximation: {opt_player_state}")
                                    print(f"Cost: {tot_cost}")
                                    print(f"Running {method} method")

                        plt.title(fr"Sub-optimal Opponent vs. Optimal Player " + '\n' +
                                  fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, "
                                  fr"$i_2={oppo_hist['i2']}$" + "\n" + fr"$\epsilon$={performance_bound}"
                                  fr"(Without Boundary), Total Cost={round(tot_cost, 4)}")

                    elif control == '1':  # Case of Player (PC) vs. Opponent (PC)
                        for t in range(T+1):
                            if t == 0:
                                opt_oppo_action = UV_dict[f"U_t={t} ({dummy_i.state}, {None})"].action
                                prev_x_action = opt_oppo_action.parent_state

                                ### Need to check line 846 - 848 correctness!!!! Continue Here
                                # Plot game process
                                opt_player_action = game_plt(full_tree, opt_oppo_action, Q, colors, UV_dict, t,
                                                             prev_x_action, R, control)
                                opt_player_state = opt_player_action.state
                                prev_x_action = opt_player_action  # Reassign prev_x_action for next iteration use

                                # Update oppo_hist
                                oppo_hist[f"i{t}"] = opt_oppo_action.state

                            else:   # When t != 0
                                opt_oppo_action = UV_dict[f"U_t={t} ({prev_x_action.state}, {opt_oppo_action.state})"].\
                                    action

                                # Plot game process
                                opt_player_action = game_plt(full_tree, opt_oppo_action, Q, colors, UV_dict, t,
                                                             prev_x_action, R, control)
                                opt_player_state = opt_player_action.state

                                tot_cost += norm(np.array(prev_x_action.state) - np.array(opt_player_state), 2)
                                prev_x_action = opt_player_action

                                # Update oppo_hist
                                oppo_hist[f"i{t}"] = opt_oppo_action.state

                            # Display
                            print(f"\nt={t}")
                            print(f"Optimal i{t}: {opt_oppo_action.state}")
                            print(f"Optimal Player State Approximation: {opt_player_action.state}")
                            print(f"Total Cost: {tot_cost}")
                            print(f"Running {method} method")

                        plt.title(fr"Optimal Opponent vs. Optimal Player " + '\n' +
                                  fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, "
                                  fr"$i_2={oppo_hist['i2']}$" + "\n" + fr"$\epsilon$={performance_bound}"
                                  fr"(Without Boundary), Total Cost={round(tot_cost, 4)}")

                    elif control == '3':
                        for t in range(T+1):
                            if t == 0:
                                opt_oppo_action = UV_dict[f"U_t={t} ({dummy_i.state}, {None})"].action
                                prev_x_action = opt_oppo_action.parent_state

                                # Plot game process
                                ram_player_action = game_plt(full_tree, opt_oppo_action, Q, colors, UV_dict, t,
                                                             prev_x_action, R, control)
                                ram_player_state = ram_player_action.state
                                prev_x_action = ram_player_action

                                # Update oppo_hist
                                oppo_hist[f"i{t}"] = opt_oppo_action.state
                            else:
                                opt_oppo_action = UV_dict[f"U_t={t} ({prev_x_action.state}, {opt_oppo_action.state}"
                                                          f")"].action

                                ram_player_action = game_plt(full_tree, opt_oppo_action, Q, colors, UV_dict, t,
                                                             prev_x_action, R, control)
                                ram_player_state = ram_player_action.state

                                tot_cost += norm(np.array(prev_x_action.state) - np.array(ram_player_state), 2)
                                prev_x_action = ram_player_action

                                # Update oppo_hist
                                oppo_hist[f"i{t}"] = opt_oppo_action.state

                            # Display
                            print(f"\nt={t}")
                            print(f"Optimal i{t}: {opt_oppo_action.state}")
                            print(f"Sub-optimal Player State Approximation: {ram_player_state}")
                            print(f"Total Cost: {tot_cost}")
                            print(f"Running {method} method")

                        plt.title(fr"Optimal Opponent vs. Sub-optimal Player " + '\n' +
                                  fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, "
                                  fr"$i_2={oppo_hist['i2']}$" + "\n" + fr"$\epsilon$={performance_bound}"
                                  fr"(Without Boundary), Total Cost={round(tot_cost, 4)}")
                    plt.show()
                msg = input(f"Rerun (Method: {method})? [Y/N] ")
            #################################### Display ####################################
            # Save Simulation Results
            sim_result = {
                'tot_cost': tot_cost,
                'performance_bound': performance_bound,
                'extra_disc_para':  extra_disc_para
            }
            sim_file = open('sim_result_new', 'ab')


        elif method.lower() == 'old':

            # Load tree info old files into the program
            tree_file = open('tree_info_old', 'rb')
            tree_info = pickle.load(tree_file)
            tree_file.close()

            # Assign keys from tree_info to variables in this program
            Q = tree_info['Q']
            tree_no_lf_copy = tree_info['tree_no_lf_copy']
            UV_dict = tree_info['UV_dict']
            T = tree_info['T']
            num_nodes = tree_info['num_nodes']
            colors = tree_info['colors']
            line_style_list = tree_info['line_style_list']
            plt_scale = tree_info['plt_scale']
            disc_para = tree_info['disc_para']
            scale_para = tree_info['scale_para']
            dummy_i = tree_info['dummy_i']
            R = tree_info['R']
            method = tree_info['method']
            boundary_rmv = tree_info['boundary_rmv']

            msg = ''
            oppo_hist = dict()
            while msg.lower() != 'n':
                control = input("Player (PC) vs. Opponent (PC) [1] / Player (PC) vs. Opponent (User) [2]? ")
                if control not in ['1', '2']:
                    print('Invalid game setting. Select again.')
                else:  # control is in ['1', '2']
                    if control == '2':
                        # Let user be opponent, show player optimal action approximation for demo (Plot them)
                        t = 0
                        opt_player_action = None
                        opt_player_state = None
                        tot_cost = 0

                        while t <= T:
                            print(f"\nt={t}")

                            # I reassigned opt_player_action to avoid warning about potentially undefined
                            # opt_player_action in else statements
                            prev_x_action = opt_player_action
                            prev_x_state = opt_player_state  # Reassignment needed for later generation of R_intersect_Q

                            oppo_node = int(input("Enter opponent action: "))
                            if t == 0:
                                if oppo_node not in range(num_nodes + 1):
                                    print("Invalid selection of node. Try again.")
                                else:
                                    oppo_action = [action for action in tree_no_lf_copy if
                                                   action.state == oppo_node and action.t_step == t]
                                    oppo_action = oppo_action[0]

                                    oppo_hist[f"i{t}"] = oppo_action.state

                                    # Plot selected Q0
                                    Q0 = Q[f"Q_t={t}^i={oppo_node}"]
                                    set_plotter(Q0.region, colors[t], alpha_val=0.5)

                                    # Plot discrete x0 in Q0
                                    """
                                    disc_x0_list = [action.state for action in tree_no_lf_copy if
                                                    action.parent_state == oppo_action]
                                    """

                                    disc_x0_list = discrete_x_calc(Q[f'Q_t={t}^i={oppo_node}'].region, disc_para,
                                                                   bound_rmv=boundary_rmv)
                                    for disc_x0 in disc_x0_list:
                                        plt.scatter(disc_x0[0], disc_x0[1], color=colors[t], linewidths=0.5, marker='.')

                                    opt_player_action = UV_dict[
                                        f"V_t={t} ({oppo_action.parent_state.state}, {oppo_action.state})"].action
                                    opt_player_state = opt_player_action.state  # value of optimal x0 approximation

                                    print(f"Optimal Player State Approximation: {opt_player_state}")

                                    plt.scatter(opt_player_state[0], opt_player_state[1], color='black', linewidths=0.1,
                                                marker='.')
                                    t += 1
                            else:  # t != 0
                                if oppo_node not in [action.state for action in prev_x_action.children_state_list]:
                                    print("Invalid selection of node. Try again.")
                                else:
                                    oppo_action = \
                                        [state for state in tree_no_lf_copy if
                                         state.state == oppo_node and state.parent_state ==
                                         prev_x_action][0]

                                    oppo_hist[f"i{t}"] = oppo_action.state

                                    opt_player_action = UV_dict[
                                        f"V_t={t} ({prev_x_action.state}, {oppo_action.state})"].action
                                    opt_player_state = opt_player_action.state

                                    print(f"Optimal Player State Approximation: {opt_player_state}")

                                    # Plot Qt
                                    Qt = Q[f"Q_t={t}^i={oppo_action.state}"]
                                    set_plotter(Qt.region, colors[t], alpha_val=0.25)

                                    # Plot R(previous_x) intersect Q
                                    R_set = reach_set_calc(prev_x_state, R)
                                    R_intersect_Q = Q[f"Q_t={t}^i={oppo_action.state}"].region.intersection(R_set)
                                    set_plotter(R_intersect_Q, colors[t], alpha_val=0.5)

                                    # Plot discrete x in R_intersect_Q
                                    disc_x_list = discrete_x_calc(R_intersect_Q, approx_para=disc_para,
                                                                  bound_rmv=boundary_rmv)
                                    for disc_x in disc_x_list:
                                        plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.1, marker='.')
                                    # Plot optimal x_t state approximation
                                    plt.scatter(opt_player_state[0], opt_player_state[1], facecolor='black',
                                                linewidths=0.1, marker='.')
                                    # Connect optimal x_t state approximation to prev_x_state
                                    plt.plot([prev_x_state[0], opt_player_state[0]],
                                             [prev_x_state[1], opt_player_state[1]], color='black')

                                    tot_cost += norm(np.array(prev_x_state) - np.array(opt_player_state), 2)
                                    print(f"Total Cost: {tot_cost}")
                                    t += 1
                        if boundary_rmv.lower() == 'n':
                            plt.title(
                                fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, $i_2={oppo_hist['i2']}$ "
                                f"\n{disc_para}x{disc_para} Discretization, Total Cost={round(tot_cost, 4)}")
                        else:
                            plt.title(
                                fr"Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, $i_2={oppo_hist['i2']}$ "
                                f"\n{disc_para}x{disc_para} Discretization (Without Boundary), Total Cost={round(tot_cost, 4)}")

                    elif control == '1':
                        opt_oppo_action = dummy_i
                        prev_x_action = opt_oppo_action.parent_state
                        tot_cost = 0
                        for t in range(T + 1):
                            if t == 0:
                                # Find optimal i_0
                                opt_oppo_action = UV_dict[f"U_t={t} ({dummy_i.state}, {None})"].action
                                # Plot Q0
                                Q0 = Q[f'Q_t={t}^i={opt_oppo_action.state}']
                                set_plotter(Q0.region, colors[t], alpha_val=0.25)
                                set_plotter(Q0.region, colors[t], alpha_val=0.5)

                                # Find discrete x0 in Q0
                                disc_x_list = [action.state for action in tree_no_lf_copy if action.parent_state ==
                                               opt_oppo_action]

                            else:  # when t is not 0
                                # Find optimal i_t
                                opt_oppo_action = UV_dict[
                                    f"U_t={t} ({prev_x_action.state}, {opt_oppo_action.state})"].action

                                # Plot selected Qt
                                Qt = Q[f"Q_t={t}^i={opt_oppo_action.state}"]
                                set_plotter(Qt.region, colors[t], alpha_val=0.25)

                                # Plot R(previous_x) intersect Q
                                R_set = reach_set_calc(prev_x_action.state, R)
                                R_intersect_Q = Qt.region.intersection(R_set)
                                set_plotter(R_intersect_Q, colors[t], alpha_val=0.5)

                                # Find discrete x in R_intersect_Q
                                disc_x_list = discrete_x_calc(R_intersect_Q, disc_para, bound_rmv=boundary_rmv)

                            # Output message
                            print(f"\nt={t}")
                            print(f"Optimal i{t}: {opt_oppo_action.state}")

                            # Plot discrete x in sets
                            for disc_x in disc_x_list:
                                plt.scatter(disc_x[0], disc_x[1], color=colors[t], linewidths=0.1, marker='.')

                            # Given x_t-1 and i_t, find approximation of optimal x_t
                            opt_player_action = \
                                UV_dict[
                                    f"V_t={t} ({opt_oppo_action.parent_state.state}, {opt_oppo_action.state})"].action

                            print(f"Optimal Player State Approximation: {opt_player_action.state}")

                            # Plot optimal x_t state approximation
                            plt.scatter(opt_player_action.state[0], opt_player_action.state[1], facecolor='black',
                                        linewidth=0.1,
                                        marker='.')

                            # Connect optimal x_t state approximation to prev_x_state
                            if t != 0:
                                plt.plot([prev_x_action.state[0], opt_player_action.state[0]],
                                         [prev_x_action.state[1], opt_player_action.state[1]], color='black')
                                # Update total cost
                                tot_cost += norm(np.array(prev_x_action.state) - np.array(opt_player_action.state), 2)
                                print(f"Total Cost: {tot_cost}")

                            prev_x_action = opt_player_action
                            # Store optimal opponent history
                            oppo_hist[f"i{t}"] = opt_oppo_action.state

                        # Plot display
                        if boundary_rmv.lower() == 'n':
                            plt.title(fr"Optimal Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, "
                                      fr"$i_2={oppo_hist['i2']}$ "
                                      f"\n{disc_para}x{disc_para} Discretization, Total Cost={round(tot_cost, 4)}")
                        else:
                            plt.title(fr"Optimal Opponent History: $i_0={oppo_hist['i0']}$, $i_1={oppo_hist['i1']}$, "
                                      fr"$i_2={oppo_hist['i2']}$ "
                                      f"\n{disc_para}x{disc_para} Discretization (Without Boundary), Total Cost={round(tot_cost, 4)}")

                    # Plot all given convex sets
                    for t_val in range(T + 1):
                        for node in range(1, num_nodes + 1):
                            hcoord_q, vcoord_q = Q[f"Q_t={t_val}^i={node}"].region.exterior.xy
                            plt.fill(hcoord_q, vcoord_q, alpha=0.1, facecolor=colors[t_val], edgecolor=colors[t_val],
                                     linewidth=2,
                                     linestyle=line_style_list[node - 1], label=fr"$Q_{t_val}^{{({node})}}$")
                    plt.legend(fontsize=8)
                    plt.grid(True)
                    plt.axis(plt_scale)
                    if control == '1':
                        plt.savefig(f"Optimal Opponent History {oppo_hist['i0']}{oppo_hist['i1']}{oppo_hist['i2']}, "
                                    f"disc_para={disc_para}")
                    else:
                        plt.savefig(
                            f"Opponent History {oppo_hist['i0']}{oppo_hist['i1']}{oppo_hist['i2']}, disc_para={disc_para}")
                    plt.show()
                msg = input("Rerun? [Y/N] ")
            pass
