import os
from networkx.classes.function import degree
from networkx.generators.atlas import graph_atlas
from pyvis.network import Network
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import matrix_rank

filepath_to_graphs = "./firebreak-master/graph_files"

filepath_to_save_graphs = "./graph_images"


def main():
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

    i = 0

    for graph_file in graph_files:
        print(i)
        i = i + 1
        try:
            file = open(filepath_to_graphs + "/" + graph_file)
            graph_as_adj_list = file.read()
            file.close()
            graph = create_graph_from_string(graph_as_adj_list)
            df = add_data_from_graphs_to_df(graph, graph_file, df)
        except:
            file.close()

    df.to_csv('graph_data.csv', index=False)

    #save_graph(graph, graph_num)
    #graph_num = graph_num + 1


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

    # Below doesn't work for unconnected graphs
    # graph_diameter = nx.diameter(graph)

    graph_adjacency_matrix_rank = matrix_rank(nx.adjacency_matrix(graph))
    graph_incidence_matrix_rank = matrix_rank(nx.incidence_matrix(graph))
    graph_laplacian_matrix_rank = matrix_rank(nx.laplacian_matrix(graph))
    graph_bethe_hessian_matrix_rank = matrix_rank(
        nx.bethe_hessian_matrix(graph))

    # Below takes VERY long to calculate
    # graph_algebraic_connectivity = nx.algebraic_connectivity(graph)
    graph_modularity_matrix_rank = matrix_rank(nx.modularity_matrix(graph))

    # Sum of the shortest-path distances between each pair of reachable nodes
    graph_wiener_index = nx.wiener_index(graph)

    # Below Returns run time error
    # graph_degree_assortativity_coefficient = nx.degree_assortativity_coefficient(
    #    graph)
    # graph_degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(
    #    graph)

    graph_is_AT_free = nx.is_at_free(graph)
    graph_has_bridge = nx.has_bridges(graph)

    graph_is_chordal = nx.is_chordal(graph)
    graph_clique_number = nx.graph_clique_number(graph)
    graph_transitivity = nx.transitivity(graph)

    # Below takes too long to calculate
    #graph_average_node_connectivity = nx.average_node_connectivity(graph)

    graph_edge_connectivity = nx.edge_connectivity(graph)
    graph_node_connectivity = nx.node_connectivity(graph)

    graph_is_distance_regular = nx.is_distance_regular(graph)

    graph_local_efficiency = nx.local_efficiency(graph)
    graph_global_efficiency = nx.global_efficiency(graph)

    # https://en.wikipedia.org/wiki/Small-world_network
    # Taking too long to calculate
    #graph_small_world_coefficient_sigma = nx.sigma(graph)
    #graph_small_world_coefficient_omega = nx.omega(graph)

    # Can't calculate below if graph is not connected
    # graph_average_shortest_path_length = nx.average_shortest_path_length(
    #    graph)
    #graph_extreme_distance_metric = nx.extrema_bounding(graph)
    #graph_radius = nx.radius(graph)

    graph_is_eulerian = nx.is_eulerian(graph)
    graph_is_semi_eulerian = nx.is_semieulerian(graph)

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
            graph_degree_amd]


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
    net.save_graph(filepath_to_save_graphs + "/" +
                   "graph_" + str(graph_num) + ".html")


# Probably want to return the number of nodes that have been updated, to identify when a fire can no longer
# spread
def spread_fire_one_round(graph):
    nodes_on_fire_dict = nx.get_node_attributes(graph, "on_fire")
    nodes_defended_dict = nx.get_node_attributes(graph, "defended")

    for node in nodes_on_fire_dict.keys():
        node_on_fire_neighbors = graph.neighbors(node)
        for node_neighbor in node_on_fire_neighbors:
            if (node_neighbor in nodes_defended_dict.keys()):
                continue
            else:
                nodes_on_fire_dict[node_neighbor] = True

    nx.set_node_attributes(graph, 'on_fire', nodes_on_fire_dict)


if __name__ == "__main__":
    main()
