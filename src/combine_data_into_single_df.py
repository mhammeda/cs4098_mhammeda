import os
import numpy as np
import pandas as pd


graph_name_in_focus = ""
fig_num = 0
seed_num = 0
num_sims_per_combination = 50


def main():
    # Need to change time_take to time_taken, when new data arrives
    summary_stats = ["avg_time_taken", "avg_num_saved", "avg_num_simulations"]

    for i in range(num_sims_per_combination):
        summary_stats.append("time_taken_run*" + str((i + 1)))
        summary_stats.append("num_saved_run*" + str((i + 1)))
        summary_stats.append("num_simulations_run*" + str((i + 1)))

    df = pd.DataFrame()
    df["graph_name"] = np.nan

    df['graph_name'] = df.graph_name.astype(str)

    file_names = os.listdir("./data_gen")

    for file_name in file_names:
        file_name_no_extension = file_name.replace('.pkl', '')
        meta_data = file_name_no_extension.split("*")
        graph_file_name = meta_data[0]
        seed_num = meta_data[1]
        strategy = meta_data[2]
        p_fire_initial = meta_data[3]
        p_def_per_round = meta_data[4]

        for summary_stat in summary_stats:

            df_to_extract_data = pd.read_pickle("./data_gen/" + file_name)

            column_to_add_to = "*".join([strategy, seed_num,
                                        p_fire_initial, p_def_per_round, summary_stat])

            if (not (np.any(df["graph_name"].str.contains(graph_file_name)))):
                df = df.append(
                    {
                        "graph_name": graph_file_name
                    },
                    ignore_index=True
                )

            if (not (column_to_add_to in df)):
                df[column_to_add_to] = np.nan

            data_to_add = df_to_extract_data.iloc[0][column_to_add_to]

            df.loc[df['graph_name'] == graph_file_name,
                   column_to_add_to] = data_to_add

    df.to_csv('./data/prelim_dataset.csv', index=False)


if __name__ == "__main__":
    main()
