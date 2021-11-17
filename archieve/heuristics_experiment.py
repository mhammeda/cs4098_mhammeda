def run_heuristics_experiment():
    random.seed(100)
    num_nodes_found = False
    num_trials_found = False

    while (not num_nodes_found):
        try:
            max_num_nodes = input(
                "How many nodes should the experiment use (minimum 5) \n")
            max_num_nodes = int(max_num_nodes)
            num_nodes_found = True
        except ValueError:
            num_nodes_found = False

    while(not num_trials_found):
        try:
            num_trials = input(
                "How many trials per node number should be conducted? \n")
            num_trials = int(num_trials)
            num_trials_found = True
        except ValueError:
            num_trials_found = False

    strategy_functions = {
        "random_allocation": randomly_assigned_nodes_to_be_defended,
        "highest_degree_allocation": assign_nodes_with_highest_degree_to_be_defended,
        "node_most_occured_on_shortest_paths": defend_nodes_appearing_most_in_shortest_path_between_all_nodes_on_fire_and_other_nodes,
        "adjacent_to_nodes_on_fire": defend_nodes_neighbouring_nodes_on_fire}

    time_taken_per_heuristic_per_num_nodes = {
        "heuristic_name": [],
        "num_nodes": [],
        "proportion_of_defender_nodes_per_round": [],
        "proportion_of_nodes_on_fire_initial": [],
        "mean_time": []
    }

    num_nodes_saved_per_heuristic_per_num_nodes = {
        "heuristic_name": [],
        "num_nodes": [],
        "proportion_of_defender_nodes_per_round": [],
        "proportion_of_nodes_on_fire_initial": [],
        "mean_num_nodes_saved": []
    }

    num_simulations_per_heuristic_per_num_nodes = {
        "heuristic_name": [],
        "num_nodes": [],
        "proportion_of_defender_nodes_per_round": [],
        "proportion_of_nodes_on_fire_initial": [],
        "mean_num_sims": []
    }

    for heuristic in strategy_functions.keys():
        for num_node in range(5, max_num_nodes):
            for p_fire_initial in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]:
                for p_def_per_round in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
                    num_simulations_sum = 0
                    num_saved_sum = 0
                    time_sum = 0
                    for i in range(num_trials):
                        p = random.uniform(0.2, 0.8)
                        graph = nx.gnp_random_graph(num_node, p)
                        print("num_node = " + str(num_node) +
                              ", p_fire_initial = " + str(p_fire_initial) +
                              ", p_def_per_round = " + str(p_def_per_round) +
                              ", trial = " + str(i))
                        graph = set_graph_initial_state(graph)
                        num_nodes_per_round_defended = int(
                            math.ceil(p_def_per_round * num_node))
                        num_nodes_on_fire_initial = int(
                            math.ceil(p_fire_initial * num_node))
                        graph = randomly_assign_nodes_to_catch_fire(
                            graph, num_nodes_on_fire_initial)
                        start_time = time.time()
                        num_saved, num_simulations = run_simulation_no_graph_generation(graph,
                                                                                        strategy_functions[heuristic],
                                                                                        num_nodes_per_round_defended)
                        end_time = time.time()
                        time_sum = time_sum + (end_time - start_time)
                        num_simulations_sum = num_simulations_sum + num_simulations
                        num_saved_sum = num_saved_sum + num_saved

                    num_simulations_mean = num_simulations_sum / num_trials
                    num_saved_mean = num_saved_sum / num_trials
                    time_mean = time_sum / num_trials

                    time_taken_per_heuristic_per_num_nodes["heuristic_name"].append(
                        heuristic)
                    num_nodes_saved_per_heuristic_per_num_nodes["heuristic_name"].append(
                        heuristic)
                    num_simulations_per_heuristic_per_num_nodes["heuristic_name"].append(
                        heuristic)
                    time_taken_per_heuristic_per_num_nodes["num_nodes"].append(
                        num_node)
                    num_nodes_saved_per_heuristic_per_num_nodes["num_nodes"].append(
                        num_node)
                    num_simulations_per_heuristic_per_num_nodes["num_nodes"].append(
                        num_node)
                    time_taken_per_heuristic_per_num_nodes["proportion_of_defender_nodes_per_round"].append(
                        p_def_per_round)
                    num_nodes_saved_per_heuristic_per_num_nodes["proportion_of_defender_nodes_per_round"].append(
                        p_def_per_round)
                    num_simulations_per_heuristic_per_num_nodes["proportion_of_defender_nodes_per_round"].append(
                        p_def_per_round)
                    time_taken_per_heuristic_per_num_nodes["proportion_of_nodes_on_fire_initial"].append(
                        p_fire_initial)
                    num_nodes_saved_per_heuristic_per_num_nodes["proportion_of_nodes_on_fire_initial"].append(
                        p_fire_initial)
                    num_simulations_per_heuristic_per_num_nodes["proportion_of_nodes_on_fire_initial"].append(
                        p_fire_initial)

                    time_taken_per_heuristic_per_num_nodes["mean_time"].append(
                        time_mean)
                    num_nodes_saved_per_heuristic_per_num_nodes["mean_num_nodes_saved"].append(
                        num_saved_mean)
                    num_simulations_per_heuristic_per_num_nodes["mean_num_sims"].append(
                        num_simulations_mean)

    time_taken_per_heuristic_per_num_nodes_df = pd.DataFrame(
        data=time_taken_per_heuristic_per_num_nodes)
    num_nodes_saved_per_heuristic_per_num_nodes_df = pd.DataFrame(
        data=num_nodes_saved_per_heuristic_per_num_nodes)
    num_simulations_per_heuristic_per_num_nodes_df = pd.DataFrame(
        data=num_simulations_per_heuristic_per_num_nodes)

    return time_taken_per_heuristic_per_num_nodes_df, num_nodes_saved_per_heuristic_per_num_nodes_df, num_simulations_per_heuristic_per_num_nodes_df

