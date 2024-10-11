import os
import subprocess
import json

# Constants
BASE_PATH = "CHAIC"
DATA_PATH = os.path.join(BASE_PATH, "dataset/obstacles_debug")
TEMP_PATH = os.path.join(BASE_PATH, "debug")

def kill_process_on_port():
    cmd = "ps ux | grep port\ 3062 | awk {'print $2'} | xargs kill"
    subprocess.call(cmd, shell=True)

agent_name = 'lm_helper_agent'

def run_challenge(task_name):
    # Create a unique output directory for this task point
    output_dir = f"results/{agent_name}/{task_name}_debug"
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = f"""python3 tdw-gym/challenge.py \
    --output_dir {output_dir} \
    --run_id 2044 \
    --port 3062 \
    --agents lm_agent {agent_name} \
    --prompt_template_path LLM/prompt_com.csv \
    --max_tokens 256 \
    --cot \
    --lm_id gpt-3.5-turbo \
    --max_frames 4000 \
    --data_prefix dataset/debug/ \
    --debug"""
    subprocess.call(cmd, shell=True)


def main():
    # Create temp folder if it doesn't exist
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)
    
    # Copy common files
    common_files = ["list.json", "name_map.json", "object_scale.json", "room_types.json"]
    for file in common_files:
        subprocess.call(f"cp {os.path.join(DATA_PATH, file)} {TEMP_PATH}", shell=True)

    # Task points
    task_points = [name.split('_')[0] + '_' + name.split('_')[1] for name in os.listdir(DATA_PATH) if not name.endswith('metadata.json') and name.count('_') == 2]

    for task in task_points:
        # Copy task json and metadata
        subprocess.call(f"cp {os.path.join(DATA_PATH, task)}.json {TEMP_PATH}", shell=True)
        subprocess.call(f"cp {os.path.join(DATA_PATH, task)}_metadata.json {TEMP_PATH}", shell=True)
        
        # Generate test_env.json
        scene, layout = task.split('_')
        test_env_content = [{"scene": scene, "layout": layout, "seed": 2824}]
        with open(os.path.join(TEMP_PATH, "test_env.json"), "w") as outfile:
            json.dump(test_env_content, outfile)
        
        # Run challenge with the current task name as output directory
        kill_process_on_port()
        run_challenge(task)
        kill_process_on_port()

if __name__ == "__main__":
    main()
