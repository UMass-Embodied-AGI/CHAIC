import os
import json
import numpy as np

tasks = ['normal', 'highthing', 'highcontainer', 'highgoalplace', 'lowthing', 'wheelchair', 'outdoor_shopping', 'outdoor_furniture']
for task in tasks:
    path = os.path.join('organized_result', task)
    baselines = os.listdir(path)

    for baseline in baselines:
        cur_path = os.path.join(path, baseline)
        
        total_count = 0
        total_helper_count = 0
        data = []
        for seed in os.listdir(cur_path):
            result_path = os.path.join(cur_path, seed)
            
            with open(os.path.join(result_path, 'eval_result.json'), 'r') as f:
                result = json.load(f)
            
            for episode in result['episode_results']:
                data.append(result['episode_results'][episode]["finish"] / result['episode_results'][episode]["total"])
        
        if len(data) == 0:
            print(f"{baseline} for {task} has no episode!")
        else:
            print(f"std TR for {baseline} for {task}: {np.std(data)}")