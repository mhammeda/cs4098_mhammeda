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

random.seed(100)

filepath_to_graphs = "./firebreak-master/graph_files"

filepath_to_save_graphs = "./graph_images"

NODE_STATE_LABEL = "NODE_STATE"

AT_RISK_NODE_LABEL = "AT_RISK_NODE"
NODE_ON_FIRE_LABEL = "FIRE_NODE"
SAFE_LABEL = "SAFE_NODE"


def main():
    # Create graph want to run the simulation on
    # set which nodes the fire starts at and set the
    # defender nodes for the first round
    
    graph_file = "fastGnp_20_0.1_18.adjlist"

    try:
        file = open(filepath_to_graphs + "/" + graph_file)
        graph_as_adj_list = file.read()
        file.close()
        graph = create_graph_from_string(graph_as_adj_list)
    except:
        file.close()


    graph = set_graph_initial_state(graph)

    graph = randomly_assign_nodes_to_catch_fire(graph, 2)

    graph = randomly_assigned_nodes_to_be_defended(graph, 2)

    graph, num_nodes_lost = spread_fire_one_round(graph)

    nodes_state = nx.get_node_attributes(graph, NODE_STATE_LABEL)

    num_at_risk = sum(value == AT_RISK_NODE_LABEL for value in nodes_state.values())

    while (num_nodes_lost != 0 and num_at_risk != 0):
        # Set the new nodes to be defended for the next round
        randomly_assigned_nodes_to_be_defended(graph, 2)

        # run simulation for another round
        graph, num_nodes_lost = spread_fire_one_round(graph)
        
        num_at_risk = sum(value == AT_RISK_NODE_LABEL for value in nodes_state.values())

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

    plt.savefig("simple_sim.png")
    # check attributes of graph, i.e: number of nodes not caught fire
    


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

    updated_graph = graph
    
    return updated_graph

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
