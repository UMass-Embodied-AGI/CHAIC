import os
import json
import numpy as np
task = 'outdoor_furniture'
result_path_single = os.path.join('results', task + '_task_no_helper')
run_id_single = 'experiment_2'
result_path_random = os.path.join('results', task + '_task_random_helper')
run_id_random = 'experiment_2'
result_path_oracle = os.path.join('results', task + '_task_plan_helper')
run_id_oracle = 'experiment_2'
# result_path_LLM = os.path.join('results', task + '_task_llm_helper')
# run_id_LLM = 'experiment'

def get_avg_frame(result_path, task):
    if not os.path.exists(os.path.join(result_path, 'eval_result.json')):
        return 0, 0, 0
    
    with open(os.path.join(result_path, 'eval_result.json'), 'r') as f:
        result = json.load(f)
    if "avg_frames" in result.keys():
        total_episodes = len(result['episode_results'])
        return result["avg_frames"], total_episodes * result["avg_frames"], total_episodes
    
    total_frames = 0
    total_episodes = 0
    max_frames = 3000
    if "furniture" in task:
        max_frames = 1500

    for episode in result['episode_results']:
        img_folder = os.path.join(result_path, episode, 'teaser_image')
        if not os.path.exists(img_folder):
            img_folder = os.path.join(result_path, episode, 'top_down_image')
            if not os.path.exists(img_folder):
                assert False, "No image folder found"

        images = [img for img in os.listdir(img_folder) if (img.endswith(".png") or img.endswith(".jpg"))]
        total_frames += min(len(images), max_frames)
        total_episodes += 1

    return total_frames / total_episodes, total_frames, total_episodes


print('avg_frames of single: ', get_avg_frame(os.path.join(result_path_single, run_id_single), task))
print('avg_frames of random: ', get_avg_frame(os.path.join(result_path_random, run_id_random), task))
print('avg_frames of oracle: ', get_avg_frame(os.path.join(result_path_oracle, run_id_oracle), task))
