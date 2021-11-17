import os
from networkx.classes.function import degree
from networkx.generators.atlas import graph_atlas
from pyvis.network import Network
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import time
import sys
from scipy import stats
from numpy.linalg import matrix_rank


graph_name_in_focus = ""
fig_num = 0
seed_num = 0


FILEPATH_TO_GRAPHS = "./firebreak-master/graph_files"
FILEPATH_TO_SAVE_GRAPHS = "./graph_images"
FILEPATH_TO_SAVE_FIGS = "./figures"

NODE_STATE_LABEL = "NODE_STATE"
AT_RISK_NODE_LABEL = "AT_RISK_NODE"
NODE_ON_FIRE_LABEL = "FIRE_NODE"
SAFE_LABEL = "SAFE_NODE"


def main():
    command_line_args = sys.argv[1].split()

    graph_filename = str(command_line_args[0])
    seed_number = int(command_line_args[1])
    strategy = str(command_line_args[2])
    p_fire_initial = float(command_line_args[3])
    p_def_per_round = float(command_line_args[4])
    num_sims_per_combination = int(command_line_args[5])
    first_seed_run = bool(command_line_args[6])

    run_single_instance(
        graph_filename,
        seed_number,
        strategy,
        p_fire_initial,
        p_def_per_round,
        num_sims_per_combination,
        first_seed_run
    )


def setup_and_run_experiment_with_repo_graphs_running_sequentially():
    # Create graph want to run the simulation on
    # set which nodes the fire starts at and set the
    # defender nodes for the first round
    seed_nums = [100, 45, 500, 90, 80]
    strategy_functions = {
        "random_allocation": randomly_assigned_nodes_to_be_defended,
        "highest_degree_allocation": assign_nodes_with_highest_degree_to_be_defended,
        "node_most_occured_on_shortest_paths": defend_nodes_appearing_most_in_shortest_path_between_all_nodes_on_fire_and_other_nodes,
        "adjacent_to_nodes_on_fire": defend_nodes_neighbouring_nodes_on_fire
    }

    file_name_to_use = "27-10-2021_full_experiment_n_40.csv"

    run_experiment_with_repo_graphs(seed_nums,
                                    strategy_functions,
                                    [0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                                     0.11, 0.12, 0.13, 0.14, 0.15
                                     ],
                                    [0.01, 0.02, 0.03, 0.04, 0.05,
                                     0.06, 0.07, 0.08, 0.09, 0.1
                                     ],
                                    40,
                                    file_name_to_use)


def run_single_instance(graph_filename, seed_number, strategy, p_fire_initial, p_def_per_round, num_sims_per_combination, first_seed_run):

    if (first_seed_run):
        random.seed(seed_number)

    strategy_functions = {
        "random_allocation": randomly_assigned_nodes_to_be_defended,
        "highest_degree_allocation": assign_nodes_with_highest_degree_to_be_defended,
        "node_most_occured_on_shortest_paths": defend_nodes_appearing_most_in_shortest_path_between_all_nodes_on_fire_and_other_nodes,
        "adjacent_to_nodes_on_fire": defend_nodes_neighbouring_nodes_on_fire
    }

    time_take_obs = []
    num_saved_obs = []
    num_simulations_obs = []

    graph_to_use = create_graph_from_path(
        FILEPATH_TO_GRAPHS + "/" + graph_filename)

    for i in range(num_sims_per_combination):
        num_node = graph_to_use.number_of_nodes()
        num_node_on_fire_to_start = int(
            math.ceil(p_fire_initial * num_node))

        graph_to_use = set_graph_initial_state(graph_to_use)
        graph_to_use = randomly_assign_nodes_to_catch_fire(
            graph_to_use, num_node_on_fire_to_start)

        num_nodes_per_round_defended = int(
            math.ceil(p_def_per_round * num_node))
        time_start = time.time()
        num_saved, num_simulations = run_simulation_no_graph_generation(graph_to_use,
                                                                        strategy_functions[strategy],
                                                                        num_nodes_per_round_defended)
        time_end = time.time()
        time_taken = time_end - time_start
        time_take_obs.append(time_taken)
        num_saved_obs.append(num_saved)
        num_simulations_obs.append(num_simulations)

    column_avg_time_take = str(strategy) + "*" + \
        str(seed_number) + "*" + \
        str(p_fire_initial) + "*" + \
        str(p_def_per_round) + "*" + \
        "avg_time_taken"

    column_avg_num_saved = str(strategy) + "*" + \
        str(seed_number) + "*" + \
        str(p_fire_initial) + "*" + \
        str(p_def_per_round) + "*" + \
        "avg_num_saved"

    column_avg_num_simulations = str(strategy) + "*" + \
        str(seed_number) + "*" + \
        str(p_fire_initial) + "*" + \
        str(p_def_per_round) + "*" + \
        "avg_num_simulations"

    avg_time_taken = []
    avg_num_saved = []
    avg_num_simulations = []

    avg_time_taken.append(statistics.mean(time_take_obs))
    avg_num_saved.append(statistics.mean(num_saved_obs))
    avg_num_simulations.append(statistics.mean(num_simulations_obs))

    data = {column_avg_time_take: avg_time_taken,
            column_avg_num_saved: avg_num_saved,
            column_avg_num_simulations: avg_num_simulations}

    df = pd.DataFrame(data)

    df.to_pickle("./data_gen_host/" +
                 str(graph_filename) + "*" +
                 str(seed_number) + "*" +
                 str(strategy) + "*" +
                 str(p_fire_initial) + "*" +
                 str(p_def_per_round) + ".pkl")


def create_graph_from_path(graph_file_path):
    try:
        file = open(graph_file_path)
        graph_as_adj_list = file.read()
        file.close()
        graph = create_graph_from_string(graph_as_adj_list)
    except:
        file.close()
    return graph


# Produces a png of a graph
def generate_graph(graph, file_path_to_save_to):
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    colouring_scheme = []
    for node in graph:
        if nodes_state[node] == AT_RISK_NODE_LABEL:
            colouring_scheme.append("green")
        elif nodes_state[node] == SAFE_LABEL:
            colouring_scheme.append("blue")
        else:
            colouring_scheme.append("red")

    pos = nx.spring_layout(graph, seed=100)

    nx.draw(graph, pos=pos, node_color=colouring_scheme, with_labels=True)

    plt.savefig(file_path_to_save_to + ".png")
    plt.clf()

# Run simulation with a picture of the state of the graph saved to a location at each round


def run_simulation(graph, strategy_function, strategy_as_string, num_fire_initial, num_defender_per_round):
    global fig_num
    global graph_name_in_focus
    global seed_num

    num_simulations = 0
    graph = set_graph_initial_state(graph)
    graph = randomly_assign_nodes_to_catch_fire(graph, num_fire_initial)

    generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" +
                           str(seed_num) + "*" +
                           str(fig_num) + "*" +
                           str(strategy_as_string)))

    fig_num = fig_num + 1

    graph, num_nodes_defended = strategy_function(
        graph, num_defender_per_round)

    generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" +
                           str(seed_num) + "*" +
                           str(fig_num) + "*" +
                           str(strategy_as_string)))
    fig_num = fig_num + 1

    graph, num_nodes_lost = spread_fire_one_round(graph)

    if (num_nodes_lost == 0):
        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                        for value in nodes_state.values())
        return num_saved, num_simulations
    else:
        generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" + str(seed_num) +
                               "*" + str(fig_num) +
                               "*" + str(strategy_as_string)))
        fig_num = fig_num + 1
        num_simulations = num_simulations + 1

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)

    num_at_risk = sum(
        value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    while (num_nodes_lost != 0 and num_at_risk != 0):
        # Set the new nodes to be defended for the next round
        graph, num_nodes_defended = strategy_function(
            graph, num_defender_per_round)

        generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" + str(seed_num) +
                               "*" + str(fig_num) +
                               "*" + str(strategy_as_string)))
        fig_num = fig_num + 1

        # run simulation for another round
        graph, num_nodes_lost = spread_fire_one_round(graph)

        if (num_nodes_lost == 0 and num_nodes_defended == 0):
            nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
            num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                            for value in nodes_state.values())
            return num_saved, num_simulations
        elif (num_nodes_lost == 0):
            generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" + str(seed_num) +
                                   "*" + str(fig_num) +
                                   "*" + str(strategy_as_string)))
            nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
            num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                            for value in nodes_state.values())
            return num_saved, (num_simulations - 1)
        else:
            generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "*" +
                                   str(seed_num) + "*" +
                                   str(fig_num) + "*" +
                                   str(strategy_as_string)))
            fig_num = fig_num + 1
            num_simulations = num_simulations + 1

        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_at_risk = sum(
            value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                    for value in nodes_state.values())

    return num_saved, num_simulations

# Run simulation, but with no graph generation


def run_simulation_no_graph_generation(graph, strategy_function, num_defender_per_round):
    num_simulations = 0

    graph, num_nodes_defended = strategy_function(
        graph, num_defender_per_round)

    graph, num_nodes_lost = spread_fire_one_round(graph)

    if (num_nodes_lost == 0):
        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                        for value in nodes_state.values())
        return num_saved, num_simulations
    else:
        num_simulations = num_simulations + 1

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)

    num_at_risk = sum(
        value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    while (num_nodes_lost != 0 and num_at_risk != 0):
        # Set the new nodes to be defended for the next round
        graph, num_nodes_defended = strategy_function(
            graph, num_defender_per_round)

        # run simulation for another round
        graph, num_nodes_lost = spread_fire_one_round(graph)

        if (num_nodes_lost == 0 and num_nodes_defended == 0):
            nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
            num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                            for value in nodes_state.values())
            return num_saved, num_simulations

        elif (num_nodes_lost == 0):
            nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
            num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                            for value in nodes_state.values())
            return num_saved, (num_simulations - 1)
        else:
            num_simulations = num_simulations + 1

        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_at_risk = sum(
            value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL)
                    for value in nodes_state.values())

    return num_saved, num_simulations


def run_experiment_with_repo_graphs(seed_numbers, strategy_functions, probs_fire_initial, probs_def_per_round, num_sims_per_combination, csv_file_name_to_use):

    total_bar = len(seed_numbers) * len(strategy_functions)
    total_bar = total_bar * len(probs_fire_initial)
    total_bar = total_bar * len(probs_def_per_round)
    total_bar = total_bar * num_sims_per_combination
    current_bar = 0

    # Add each observation to the respective array that is the value for a specific key (which will be the
    # combination of the seed, strategy and what the simulation statistics (num of sims, num saved, etc.) is)
    columns_to_add_as_dict = {}

    #df, graph_files = gen_graph_data()
    # below lines up until the for loop replace the above for now
    #df = pd.read_csv("./data/graph_data.csv")

    filepath_to_graphs = "./firebreak-master/graph_files"
    graph_files = os.listdir(filepath_to_graphs)

    d = {'graph_name': graph_files}

    df = pd.DataFrame(data=d)

    total_bar = total_bar * len(graph_files)

    for seed_number in seed_numbers:
        for strategy in strategy_functions.keys():
            for prob_fire_initial in probs_fire_initial:
                for prob_def_per_round in probs_def_per_round:
                    column_name_avg_num_saved = str(strategy) + "*" + \
                        str(seed_number) + "*" + \
                        str(prob_fire_initial) + "*" + \
                        str(prob_def_per_round) + "*" + \
                        "avg_num_saved"

                    column_name_avg_num_simulations = str(strategy) + "*" + \
                        str(seed_number) + "*" + \
                        str(prob_fire_initial) + "*" + \
                        str(prob_def_per_round) + "*" + \
                        "avg_num_simulations"

                    column_name_avg_time_take = str(strategy) + "*" + \
                        str(seed_number) + "*" + \
                        str(prob_fire_initial) + "*" + \
                        str(prob_def_per_round) + "*" + \
                        "avg_time_take"

                    columns_to_add_as_dict[column_name_avg_num_saved] = []
                    columns_to_add_as_dict[column_name_avg_num_simulations] = [
                    ]
                    columns_to_add_as_dict[column_name_avg_time_take] = []

    for seed_number in seed_numbers:
        random.seed(seed_number)
        for graph_file in graph_files:
            graph_to_use = create_graph_from_path(
                FILEPATH_TO_GRAPHS + "/" + graph_file)
            for strategy in strategy_functions.keys():
                for p_fire_initial in probs_fire_initial:
                    for p_def_per_round in probs_def_per_round:
                        time_take_obs = []
                        num_saved_obs = []
                        num_simulations_obs = []
                        for i in range(num_sims_per_combination):
                            num_node = graph_to_use.number_of_nodes()
                            num_node_on_fire_to_start = int(
                                math.ceil(p_fire_initial * num_node))

                            graph_to_use = set_graph_initial_state(
                                graph_to_use)
                            graph_to_use = randomly_assign_nodes_to_catch_fire(
                                graph_to_use, num_node_on_fire_to_start)

                            num_nodes_per_round_defended = int(
                                math.ceil(p_def_per_round * num_node))
                            time_start = time.time()
                            num_saved, num_simulations = run_simulation_no_graph_generation(graph_to_use,
                                                                                            strategy_functions[strategy],
                                                                                            num_nodes_per_round_defended)
                            time_end = time.time()
                            time_taken = time_end - time_start
                            time_take_obs.append(time_taken)
                            num_saved_obs.append(num_saved)
                            num_simulations_obs.append(num_simulations)

                            current_bar += 1
                            progressBar(current_bar, total_bar, barLength=100)

                        column_to_add_avg_time_take = str(strategy) + "*" + \
                            str(seed_number) + "*" + \
                            str(p_fire_initial) + "*" + \
                            str(p_def_per_round) + "*" + \
                            "avg_time_take"

                        column_to_add_avg_num_saved = str(strategy) + "*" + \
                            str(seed_number) + "*" + \
                            str(p_fire_initial) + "*" + \
                            str(p_def_per_round) + "*" + \
                            "avg_num_saved"

                        column_to_add_avg_num_simulations = str(strategy) + "*" + \
                            str(seed_number) + "*" + \
                            str(p_fire_initial) + "*" + \
                            str(p_def_per_round) + "*" + \
                            "avg_num_simulations"

                        columns_to_add_as_dict[column_to_add_avg_time_take].append(
                            statistics.mean(time_take_obs))
                        columns_to_add_as_dict[column_to_add_avg_num_saved].append(
                            statistics.mean(num_saved_obs))
                        columns_to_add_as_dict[column_to_add_avg_num_simulations].append(
                            statistics.mean(num_simulations_obs))

    # Below ensures that columns are in-order
    columns_to_add_as_dict_items = columns_to_add_as_dict.items()
    sorted_columns_to_add_as_dict_items = sorted(columns_to_add_as_dict_items)

    columns_to_add_as_dict_items_keys_in_order = [
        col_key_val[0] for col_key_val in sorted_columns_to_add_as_dict_items]

    for column in columns_to_add_as_dict_items_keys_in_order:
        df[column] = columns_to_add_as_dict[column]

    df.to_csv(csv_file_name_to_use, index=False)


# https://stackoverflow.com/questions/6169217/replace-console-output-in-python
def progressBar(current, total, barLength=100):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


def run_experiment(seed_numbers, graphs_to_test):
    strategy_functions = {
        "random_allocation": randomly_assigned_nodes_to_be_defended,
        "highest_degree_allocation": assign_nodes_with_highest_degree_to_be_defended,
        "node_most_occured_on_shortest_paths": defend_nodes_appearing_most_in_shortest_path_between_all_nodes_on_fire_and_other_nodes,
        "adjacent_to_nodes_on_fire": defend_nodes_neighbouring_nodes_on_fire}

    d = {"graph_name": [], "strategy": [],
         "num_saved": [], "num_simulations": [], "seed": []}
    global graph_name_in_focus
    global fig_num
    global seed_num

    for seed in seed_numbers:
        seed_num = seed_num + 1
        random.seed(seed)
        for graph_file_name in graphs_to_test:
            for strategy_function_name, strategy_python_function in strategy_functions.items():
                fig_num = 0
                graph_name_in_focus = graph_file_name
                file_path_to_use = FILEPATH_TO_GRAPHS + "/" + graph_file_name
                graph = create_graph_from_path(file_path_to_use)
                num_saved, num_simulations = run_simulation(
                    graph, strategy_python_function, strategy_function_name, 2, 2)
                d["graph_name"].append(graph_file_name)
                d["strategy"].append(str(strategy_function_name))
                d["num_saved"].append(num_saved)
                d["num_simulations"].append(num_simulations)
                d["seed"].append(seed)

    df = pd.DataFrame(data=d)

    return df


# Runs a simulation of the fire spreading for one round, returning
# the graph after the round and the number of nodes that have caught
# fire
def spread_fire_one_round(graph):
    num_nodes_caught_fire = 0
    node_neighbor_seen = []
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    on_fire_nodes = {key: value for key,
                     value in nodes_state.items() if value == NODE_ON_FIRE_LABEL}
    defended_nodes = {key: value for key,
                      value in nodes_state.items() if value == SAFE_LABEL}

    for node in on_fire_nodes.keys():
        node_on_fire_neighbors = graph.neighbors(node)
        for node_neighbor in node_on_fire_neighbors:
            if (node_neighbor in defended_nodes.keys() or
                node_neighbor in on_fire_nodes.keys() or
                    node_neighbor in node_neighbor_seen):
                continue
            else:
                nodes_state[node_neighbor] = NODE_ON_FIRE_LABEL
                node_neighbor_seen.append(node_neighbor)
                num_nodes_caught_fire = num_nodes_caught_fire + 1

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph, num_nodes_caught_fire


def randomly_assigned_nodes_to_be_defended(graph, num_to_assign):
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    num_at_risk_nodes = len(list(at_risk_nodes.keys()))

    for i in range(min(num_at_risk_nodes, num_to_assign)):
        node_to_be_defender = random.choice(list(at_risk_nodes.keys()))
        nodes_state[node_to_be_defender] = SAFE_LABEL
        del at_risk_nodes[node_to_be_defender]

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph, min(num_at_risk_nodes, num_to_assign)


def defend_nodes_neighbouring_nodes_on_fire(graph, num_to_assign):

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)

    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}

    nodes_on_fire = {key: value for key,
                     value in nodes_state.items() if value == NODE_ON_FIRE_LABEL}

    num_at_risk_nodes = len(list(at_risk_nodes.keys()))

    node_found = False

    # Ensures all defender nodes are allocated, even if not neighbouring a node on fire
    nodes_defended = 0
    node_neighboring_focus_node_on_fire = None

    for i in range(min(num_at_risk_nodes, num_to_assign)):
        node_found = False
        node_neighboring_focus_node_on_fire = []
        for node_on_fire in nodes_on_fire.keys():
            node_neighboring_focus_node_on_fire = graph.neighbors(node_on_fire)
            for neighbouring_node_to_fire in node_neighboring_focus_node_on_fire:
                if neighbouring_node_to_fire in at_risk_nodes.keys():
                    node_found = True
                    nodes_defended += 1
                    nodes_state[neighbouring_node_to_fire] = SAFE_LABEL
                    del at_risk_nodes[neighbouring_node_to_fire]
                    break
            if (node_found):
                break

    # Simply randomly allocates a node to defender, if there's still defender nodes to allocate

    while (nodes_defended < min(num_at_risk_nodes, num_to_assign)):
        nodes_defended += 1
        node_to_be_defender = random.choice(list(at_risk_nodes.keys()))
        nodes_state[node_to_be_defender] = SAFE_LABEL
        del at_risk_nodes[node_to_be_defender]

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph, min(num_at_risk_nodes, num_to_assign)


def defend_nodes_appearing_most_in_shortest_path_between_all_nodes_on_fire_and_other_nodes(graph, num_to_assign):
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    node_occurence = {}

    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}

    nodes_on_fire = {key: value for key,
                     value in nodes_state.items() if value == NODE_ON_FIRE_LABEL}

    num_at_risk_nodes = len(list(at_risk_nodes.keys()))

    for node_on_fire in nodes_on_fire.keys():
        for at_risk_node in at_risk_nodes.keys():
            if (nx.has_path(graph, node_on_fire, at_risk_node)):
                shortest_path = nx.shortest_path(
                    graph, source=node_on_fire, target=at_risk_node)
                for i in range(1, len(shortest_path) - 1):
                    if shortest_path[i] in node_occurence.keys():
                        node_occurence[shortest_path[i]] += 1
                    else:
                        node_occurence[shortest_path[i]] = 1

    node_occurence_sorted_by_val = sorted(
        node_occurence.items(), key=lambda x: x[1], reverse=True)

    num_defender_nodes_to_allocate = min(num_at_risk_nodes, num_to_assign)
    num_defender_nodes_allocated = 0

    for node_to_defend, node_val in node_occurence_sorted_by_val:
        if node_to_defend in at_risk_nodes.keys():
            num_defender_nodes_allocated += 1
            nodes_state[node_to_defend] = SAFE_LABEL
            del at_risk_nodes[node_to_defend]
        else:
            continue

        if (num_defender_nodes_allocated == num_defender_nodes_to_allocate):
            break

    # Simply randomly allocates a node to defender, if there's still defender nodes to allocate

    while (num_defender_nodes_allocated < min(num_at_risk_nodes, num_to_assign)):
        num_defender_nodes_allocated += 1
        node_to_be_defender = random.choice(list(at_risk_nodes.keys()))
        nodes_state[node_to_be_defender] = SAFE_LABEL
        del at_risk_nodes[node_to_be_defender]

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph, min(num_at_risk_nodes, num_to_assign)


def assign_nodes_with_highest_degree_to_be_defended(graph, num_to_assign):
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    num_at_risk_nodes = len(list(at_risk_nodes.keys()))
    node_to_be_defender = None

    for i in range(min(num_at_risk_nodes, num_to_assign)):
        nodes_sorted_by_degree = sorted(
            graph.degree, key=lambda x: x[1], reverse=True)
        for node_degree_pair in nodes_sorted_by_degree:
            if node_degree_pair[0] in at_risk_nodes.keys():
                node_to_be_defender = node_degree_pair[0]
                break

        if (node_to_be_defender == None):
            raise Exception("No node to defend")

        nodes_state[node_to_be_defender] = SAFE_LABEL
        del at_risk_nodes[node_to_be_defender]

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph, min(num_at_risk_nodes, num_to_assign)


def randomly_assign_nodes_to_catch_fire(graph, num_to_assign):

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key,
                     value in nodes_state.items() if value == AT_RISK_NODE_LABEL}

    num_at_risk_nodes = len(list(at_risk_nodes.keys()))

    for i in range(min(num_at_risk_nodes, num_to_assign)):
        node_to_catch_fire = random.choice(list(at_risk_nodes.keys()))
        nodes_state[node_to_catch_fire] = NODE_ON_FIRE_LABEL
        del at_risk_nodes[node_to_catch_fire]

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph


def set_graph_initial_state(graph):
    nodes_state = {}

    for node in graph.nodes:
        nodes_state[node] = AT_RISK_NODE_LABEL

    nx.set_node_attributes(graph, values=nodes_state, name=NODE_STATE_LABEL)

    return graph


def create_graph_from_string(string_to_read_from):
    G = nx.Graph()
    string_to_read_from_array = string_to_read_from.splitlines()

    for line in string_to_read_from_array:
        nodes_on_line = line.split(" ")
        num_nodes = len(nodes_on_line)

        try:
            first_node = int(nodes_on_line[0])
        except:
            continue

        for i in range(1, num_nodes):
            node_to_add = int(nodes_on_line[i])
            G.add_edge(first_node, node_to_add)

    return G


def gen_graph_data():
    filepath_to_graphs = "./firebreak-master/graph_files"
    graph_files = os.listdir(filepath_to_graphs)
    graph = None
    df = pd.DataFrame(columns=['graph_name',
                               'num_nodes',
                               'num_edges',
                               'graph_density',
                               'mean_degree',
                               'median_degree',
                               'degree_std',
                               'iqr_degree',
                               'degree_amd',
                               'node_connectivity',
                               'average_clustering',
                               'adjacency_matrix_rank',
                               'incidence_matrix_rank',
                               'laplacian_matrix_rank',
                               'bethe_hessian_matrix_rank',
                               'modularity_matrix_rank',
                               'wiener_index',
                               'is_AT_free',
                               'has_bridge',
                               'is_chordal',
                               'clique_number',
                               'transitivity',
                               'edge_connectivity',
                               'node_connectivity',
                               'is_distance_regular',
                               'local_efficiency',
                               'global_efficiency',
                               'is_eulerian',
                               'is_semi_eulerian'])

    for graph_file in graph_files:
        try:
            file = open(filepath_to_graphs + "/" + graph_file)
            graph_as_adj_list = file.read()
            file.close()
            graph = create_graph_from_string(graph_as_adj_list)
            df = add_data_from_graphs_to_df(graph, graph_file, df)
        except:
            print(graph_file)
            file.close()

    return df, graph_files


def add_data_from_graphs_to_df(graph, graph_name, df):

    num_nodes = nx.number_of_nodes(graph)
    num_edges = nx.number_of_edges(graph)

    # https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.density.html#networkx.classes.function.density
    graph_density = nx.density(graph)

    # May get statistics like the mode and median
    # https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.degree_histogram.html#networkx.classes.function.degree_histogram
    graph_stats = calculate_graph_stat(graph)
    graph_mean_degree = graph_stats[0]
    graph_median_degree = graph_stats[1]
    graph_degree_std = graph_stats[2]
    graph_iqr_degree = graph_stats[3]
    graph_degree_amd = graph_stats[4]

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.connectivity.node_connectivity.html#networkx.algorithms.approximation.connectivity.node_connectivity
    graph_node_connectivity = nx.node_connectivity(graph)

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.clustering_coefficient.average_clustering.html#networkx.algorithms.approximation.clustering_coefficient.average_clustering
    graph_average_clustering = nx.average_clustering(graph)

    graph_adjacency_matrix_rank = matrix_rank(nx.adjacency_matrix(graph))
    graph_incidence_matrix_rank = matrix_rank(nx.incidence_matrix(graph))
    graph_laplacian_matrix_rank = matrix_rank(nx.laplacian_matrix(graph))
    graph_bethe_hessian_matrix_rank = matrix_rank(
        nx.bethe_hessian_matrix(graph))

    graph_modularity_matrix_rank = matrix_rank(nx.modularity_matrix(graph))

    # Sum of the shortest-path distances between each pair of reachable nodes
    graph_wiener_index = nx.wiener_index(graph)

    graph_is_AT_free = nx.is_at_free(graph)
    graph_has_bridge = nx.has_bridges(graph)

    graph_is_chordal = nx.is_chordal(graph)
    graph_clique_number = nx.graph_clique_number(graph)
    graph_transitivity = nx.transitivity(graph)

    graph_edge_connectivity = nx.edge_connectivity(graph)
    graph_node_connectivity = nx.node_connectivity(graph)

    graph_is_distance_regular = nx.is_distance_regular(graph)

    graph_local_efficiency = nx.local_efficiency(graph)
    graph_global_efficiency = nx.global_efficiency(graph)

    graph_is_eulerian = nx.is_eulerian(graph)
    graph_is_semi_eulerian = nx.is_semieulerian(graph)

    # Below takes too long to calculate

    # graph_small_world_coefficient_sigma = nx.sigma(graph)
    # graph_small_world_coefficient_omega = nx.omega(graph)
    # https://en.wikipedia.org/wiki/Small-world_network
    # graph_algebraic_connectivity = nx.algebraic_connectivity(graph)

    # Can't calculate below if graph is not connected

    # graph_average_shortest_path_length = nx.average_shortest_path_length(graph)
    # graph_extreme_distance_metric = nx.extrema_bounding(graph)
    # graph_radius = nx.radius(graph)
    # graph_average_node_connectivity = nx.average_node_connectivity(graph)
    # graph_diameter = nx.diameter(graph)

    # Below Returns run time error

    # graph_degree_assortativity_coefficient = nx.degree_assortativity_coefficient(graph)
    # graph_degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(graph)

    data_row = [graph_name,
                num_nodes,
                num_edges,
                graph_density,
                graph_mean_degree,
                graph_median_degree,
                graph_degree_std,
                graph_iqr_degree,
                graph_degree_amd,
                graph_node_connectivity,
                graph_average_clustering,
                graph_adjacency_matrix_rank,
                graph_incidence_matrix_rank,
                graph_laplacian_matrix_rank,
                graph_bethe_hessian_matrix_rank,
                graph_modularity_matrix_rank,
                graph_wiener_index,
                graph_is_AT_free,
                graph_has_bridge,
                graph_is_chordal,
                graph_clique_number,
                graph_transitivity,
                graph_edge_connectivity,
                graph_node_connectivity,
                graph_is_distance_regular,
                graph_local_efficiency,
                graph_global_efficiency,
                graph_is_eulerian,
                graph_is_semi_eulerian]

    row_series = pd.Series(data_row, index=df.columns)

    df = df.append(row_series, ignore_index=True)

    return df

# This might contain relevant information:
# https://networkx.org/documentation/stable/reference/algorithms/community.html


def calculate_graph_stat(graph):
    graph_histogram_data = nx.degree_histogram(graph)
    degree_data_for_nodes = []
    for i in range(len(graph_histogram_data)):
        for j in range(graph_histogram_data[i]):
            degree_data_for_nodes.append(i)

    graph_mean_degree = np.mean(degree_data_for_nodes)
    graph_median_degree = np.median(degree_data_for_nodes)

    # can't calculate mode as no gaurentee it'll be unique
    # graph_mode_degree = statistics.mode(degree_data_for_nodes)

    graph_sd_degree = statistics.pvariance(degree_data_for_nodes)
    graph_iqr_degree = stats.iqr(degree_data_for_nodes)
    graph_degree_amd = np.mean(
        (np.abs(degree_data_for_nodes - np.mean(degree_data_for_nodes))))

    return [graph_mean_degree,
            graph_median_degree,
            graph_sd_degree,
            graph_iqr_degree,
            graph_degree_amd
            ]


if __name__ == "__main__":
    main()
