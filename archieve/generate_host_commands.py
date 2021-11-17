import os


file_names = os.listdir("./gnu_arguments/gnu_arguments_broken_down_host")

with open('gnu_arguments/commands_host.sh', 'w') as rsh:
    for file_name in file_names:
        string_to_add = 'parallel -j8 --eta "python ./src/fire_fighter_simulation_host.py" :::: gnu_arguments/gnu_arguments_broken_down_host/' + file_name + '\n'
        rsh.write(string_to_add)


"""
import csv
with open('./gnu_arguments/commands.txt', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')
    file_names = os.listdir("./gnu_arguments/gnu_arguments_broken_down")

    for file_name in file_names:
        # Doesn't produce quotiations for the other arguments, since they don't have a space!
        writer.writerow(
            [
                'parallel',
                '-eta',
                "python ./src/fire_fighter_simulation.py",
                '::::',
                'gnu_arguments/gnu_arguments_broken_down/' + file_name
            ]
        )
"""
