dataset_path = 'dataset_test/highgoalplace_debug'
json_path = 'dataset_test/highgoalplace_debug/test_env.json'
json_path_new = 'dataset_test/highgoalplace_debug/test_env.json'

import json
import os
with open(json_path, 'r') as f:
    data = json.load(f)
dataset_files = os.listdir(dataset_path)
dataset_files = [os.path.join(dataset_path, file) for file in dataset_files if file.endswith('metadata.json')]
for file in dataset_files:
    with open(file, 'r') as f:
        metadata = json.load(f)
    scene = file.split('/')[-1].split('_')[0]
    layout = file.split('/')[-1].split('_')[1] + '_' + file.split('/')[-1].split('_')[2]
    for i in range(len(data)):
        if data[i]['scene'] == scene and data[i]['layout'] == layout:
            data[i]['task'] = metadata['task']

with open(json_path_new, 'w') as f:
    json.dump(data, f, indent=4)