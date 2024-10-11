import subprocess
import os
import shutil
import sys
import json
from copy import deepcopy
# List of programs to run

programs = []
# runkinds = ["highthing"]
runkinds = ["normal", "highthing", "lowthing", "highcontainer", "highgoalplace", "wheelchair"]
with open("dataset/test_env.json", "r") as file:
    total_test_env_data = json.load(file)
for keyword in runkinds:
    for runpoint in total_test_env_data:
        programs.append(f"python ./scene_generate_indoor.py {runpoint['scene']} {runpoint['layout'].replace('_', ' ')} {keyword}")

# Dictionary to keep track of reruns
rerun_counts = {program: 0 for program in programs}

# List to keep track of unrun programs
unrun_programs = []

# Function to run the programs
def run_programs():
    for program in programs:
        success = run_command(program)
        if not success:
            unrun_programs.append(program)
            
# Function to run a single command and handle reruns
def run_command(command):
    max_retries = 3
    while rerun_counts[command] < max_retries:
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            rerun_counts[command] += 1
        else:
            return True
    return False

# Run all programs
run_programs()

def execute_scene_scripts(l):
    for type in l:
        total_test_env_data_to_load = deepcopy(total_test_env_data)
        for i, test_env_data in enumerate(total_test_env_data):
            foldername = f"{type}"
            metadata_filename = f"dataset/{foldername}/{test_env_data['scene']}_{test_env_data['layout']}_metadata.json"
     

            files_to_copy = [
                "list.json",
                "name_map.json",
                "room_types.json",
                "object_scale.json"
            ]

            with open(metadata_filename, "r") as file:
                metadata_data = json.load(file)

            total_test_env_data_to_load[i]["task"] = metadata_data["task"]

            with open(f"dataset/{foldername}/test_env.json", "w") as file:
                json.dump(total_test_env_data_to_load, file, indent=4)

            for file in files_to_copy:
                source = f"dataset/{file}"
                destination = f"dataset/{foldername}/{file}"
                shutil.copy2(source, destination)


# Print the list of unrun programs
if unrun_programs:
    print("Unrun programs:")
    for program in unrun_programs:
        print(program)
else:
    execute_scene_scripts(runkinds)

