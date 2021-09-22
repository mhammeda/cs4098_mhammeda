import os
from pyvis.network import Network

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filepath_to_graphs = ".\\firebreak-master\\graph_files"

filepath_to_save_graphs = ".\graph_images"

def main():
    graph_num = 0
    graph_files = os.listdir(filepath_to_graphs)
    
    for graph_file in graph_files:
        file = open(filepath_to_graphs + "\\" + graph_file)
        graph_as_adj_list = file.read()
        file.close()
        graph = create_graph_from_string(graph_as_adj_list)
        save_graph(graph, graph_num)
        graph_num = graph_num + 1


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
    
def save_graph(graph, graph_num):
    net = Network(height='700px', width='1100px',)
    net.from_nx(graph)
    net.show_buttons()
    net.save_graph(filepath_to_save_graphs + "\\" + "graph_" + str(graph_num) + ".html")
    
if __name__ == "__main__":
    main()