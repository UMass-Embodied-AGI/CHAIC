import os
import json
import numpy as np
task = 'lowthing'
path = os.path.join('organized_result', task)
baselines = os.listdir(path)

single_results = {}
single_num = {}
single_path = os.path.join(path, 'no_helper')
for seed in os.listdir(single_path):
    result_path = os.path.join(single_path, seed)
    with open(os.path.join(result_path, 'eval_result.json'), 'r') as f:
        result = json.load(f)
    for episode in result['episode_results']:
        finish = result['episode_results'][episode]["finish"]
        if episode not in single_results:
            single_results[episode] = 0
            single_num[episode] = 0
        single_results[episode] += finish
        single_num[episode] += 1

for baseline in baselines:
    if baseline == 'no_helper':
        continue
    cur_path = os.path.join(path, baseline)
    cur_results = {}
    cur_num = {}
    for seed in os.listdir(cur_path):
        result_path = os.path.join(cur_path, seed)
        with open(os.path.join(result_path, 'eval_result.json'), 'r') as f:
            result = json.load(f)
        for episode in result['episode_results']:
            finish = result['episode_results'][episode]["finish"]
            if episode not in cur_results:
                cur_results[episode] = 0
                cur_num[episode] = 0
            cur_results[episode] += finish
            cur_num[episode] += 1
    
    ei = []
    for episode in cur_results:
        if episode not in single_results:
            continue

        finish = cur_results[episode] / cur_num[episode]
        single_finish = single_results[episode] / single_num[episode]
        ei.append((finish - single_finish) / max(max(finish, single_finish), 1))
    
    if len(ei) == 0:
        print(f"{baseline} has no episode")
        continue

    print(f"EI for {baseline} for {task}: {np.mean(ei)}")
    print(f"Num of episodes for {baseline} for {task}: {len(ei)}")

