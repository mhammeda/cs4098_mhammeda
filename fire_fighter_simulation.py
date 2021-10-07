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
    # Create graph want to run the simulation on
    # set which nodes the fire starts at and set the
    # defender nodes for the first round

    graph_files = ["fastGnp_20_0.1_18.adjlist",
                    "fastGnp_20_0.2_13.adjlist",
                    "fastGnp_100_0.01_6.adjlist",
                    "fastGnp_100_0.01_14.adjlist",
                    "randomRegular_20_5_20.adjlist",
                    "powerLawTree_20_4_24.adjlist",
                    "randomRegular_20_3_15.adjlist"]
    
    seed_nums = [100, 45, 500, 90, 80]

    df = run_experiment(seed_nums, graph_files)

    df.to_csv('sim_data.csv', index=False)


def create_graph_from_path(graph_file_path):
    try:
        file = open(graph_file_path)
        graph_as_adj_list = file.read()
        file.close()
        graph = create_graph_from_string(graph_as_adj_list)
        nx.spring_layout(graph, seed = 100)
    except:
        file.close()
    return graph

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
    
    nx.draw(graph, node_color=colouring_scheme, with_labels=True)

    plt.savefig(file_path_to_save_to + ".png")
    plt.clf()


def run_simulation(graph):
    global fig_num
    global graph_name_in_focus
    global seed_num

    num_simulations = 0
    graph = set_graph_initial_state(graph)
    graph = randomly_assign_nodes_to_catch_fire(graph, 2)
    graph = randomly_assigned_nodes_to_be_defended(graph, 2)

    generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "_" + str(seed_num) + "_" + str(fig_num)))
    fig_num = fig_num + 1

    graph, num_nodes_lost = spread_fire_one_round(graph)

    if (num_nodes_lost == 0):
        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL) for value in nodes_state.values())
        return num_simulations
    else:
        generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "_" + str(seed_num) + "_" + str(fig_num)))
        fig_num = fig_num + 1
        num_simulations = num_simulations + 1

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)

    num_at_risk = sum(value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    while (num_nodes_lost != 0 and num_at_risk != 0):
        # Set the new nodes to be defended for the next round
        randomly_assigned_nodes_to_be_defended(graph, 2)

        # run simulation for another round
        graph, num_nodes_lost = spread_fire_one_round(graph)

        if (num_nodes_lost == 0):
            nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
            num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL) for value in nodes_state.values())
            return num_saved, num_simulations
        else:
            generate_graph(graph, (FILEPATH_TO_SAVE_FIGS + "/" + graph_name_in_focus + "_" + str(seed_num) + "_" + str(fig_num)))
            fig_num = fig_num + 1
            num_simulations = num_simulations + 1
        
        nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
        num_at_risk = sum(value == AT_RISK_NODE_LABEL for value in nodes_state.values())
    
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    num_saved = sum((value == AT_RISK_NODE_LABEL or value == SAFE_LABEL) for value in nodes_state.values())

    return num_saved, num_simulations


def run_experiment(seed_numbers, graphs_to_test):
    d = {"graph_name": [], "num_saved": [], "num_simulations": [], "seed": []}
    global graph_name_in_focus
    global fig_num
    global seed_num

    for seed in seed_numbers:
        seed_num = seed_num + 1
        random.seed(seed)
        for graph_file_name in graphs_to_test:
            fig_num = 0
            graph_name_in_focus = graph_file_name
            file_path_to_use = FILEPATH_TO_GRAPHS + "/" + graph_file_name
            graph = create_graph_from_path(file_path_to_use)
            num_saved, num_simulations = run_simulation(graph)
            d["graph_name"].append(graph_file_name)
            d["num_saved"].append(num_saved)
            d["num_simulations"].append(num_simulations)
            d["seed"].append(seed)

    df = pd.DataFrame(data = d)

    return df







# Runs a simulation of the fire spreading for one round, returning
# the graph after the round and the number of nodes that have caught
# fire
def spread_fire_one_round(graph):
    num_nodes_caught_fire = 0
    node_neighbor_seen = []
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key, value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    on_fire_nodes = {key: value for key, value in nodes_state.items() if value == NODE_ON_FIRE_LABEL}
    defended_nodes = {key: value for key, value in nodes_state.items() if value == SAFE_LABEL}

    for node in on_fire_nodes.keys():
        node_on_fire_neighbors = graph.neighbors(node)
        for node_neighbor in node_on_fire_neighbors:
            if (node_neighbor in defended_nodes.keys() or
                node_neighbor in on_fire_nodes.keys() or
                node_neighbor in node_neighbor_seen):
                continue
            else:
                print("Caught fire: " + str(node_neighbor))
                nodes_state[node_neighbor] = NODE_ON_FIRE_LABEL
                node_neighbor_seen.append(node_neighbor)
                num_nodes_caught_fire = num_nodes_caught_fire + 1

    nx.set_node_attributes(graph, values = nodes_state, name = NODE_STATE_LABEL)

    return graph, num_nodes_caught_fire


def randomly_assigned_nodes_to_be_defended(graph, num_to_assign):
    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key, value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    num_at_risk_nodes = len(list(at_risk_nodes.keys()))
    
    for i in range(min(num_at_risk_nodes, num_to_assign)):
        node_to_be_defender = random.choice(list(at_risk_nodes.keys()))
        print("Defended: " + str(node_to_be_defender))
        nodes_state[node_to_be_defender] = SAFE_LABEL
        del at_risk_nodes[node_to_be_defender]
    
    nx.set_node_attributes(graph, values = nodes_state, name = NODE_STATE_LABEL)

    return graph



def randomly_assign_nodes_to_catch_fire(graph, num_to_assign):

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)
    at_risk_nodes = {key: value for key, value in nodes_state.items() if value == AT_RISK_NODE_LABEL}
    num_at_risk_nodes = len(list(at_risk_nodes.keys()))
    
    for i in range(min(num_at_risk_nodes, num_to_assign)):
        node_to_catch_fire = random.choice(list(at_risk_nodes.keys()))
        print("Caught fire: " + str(node_to_catch_fire))
        nodes_state[node_to_catch_fire] = NODE_ON_FIRE_LABEL
        del at_risk_nodes[node_to_catch_fire]
    
    nx.set_node_attributes(graph, values = nodes_state, name = NODE_STATE_LABEL)

    return graph

def set_graph_initial_state(graph):
    nodes_state = {}

    for node in graph.nodes:
        nodes_state[node] = AT_RISK_NODE_LABEL
    
    nx.set_node_attributes(graph, values = nodes_state, name = NODE_STATE_LABEL)
    
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


if __name__ == "__main__":
    main()
