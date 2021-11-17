import csv

seed_nums = [100, 45]

probs_fire_initial = [0.05, 0.06, 0.07]

probs_def_per_round = [0.01, 0.02, 0.03]


num_sims_per_combination = 5

strategy_functions = ["random_allocation",
                      "highest_degree_allocation",
                      "node_most_occured_on_shortest_paths",
                      "adjacent_to_nodes_on_fire"
                      ]


graph_file_names = ['randomRegular_1000_5_18.adjlist', 'fastGnp_20_0.01_2.adjlist',
                    'powerLawTree_100_3_4.adjlist', 'fastGnp_100_0.1_3.adjlist']


with open('./gnu_arguments/prelim_arguments.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    for seed_num in seed_nums:
        for graph_file_name in graph_file_names:
            for strategy in strategy_functions:
                for prob_fire_initial in probs_fire_initial:
                    for prob_def_per_round in probs_def_per_round:
                        writer.writerow([graph_file_name,
                                        seed_num,
                                        strategy,
                                        prob_fire_initial,
                                        prob_def_per_round,
                                        num_sims_per_combination
                                         ]
                                        )
