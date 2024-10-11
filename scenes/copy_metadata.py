dataset_path = 'dataset_train/normal_debug'

import json
import os

test_env = json.load(open(os.path.join(dataset_path, 'test_env_old.json'), "r"))

for test in test_env:
    file_name = test['scene'] + '_' + test['layout']
    single_test_env = json.load(open(os.path.join(dataset_path, file_name + '_metadata.json'), "r"))
    test['task'] = single_test_env['task']
    
with open(os.path.join(dataset_path, 'test_env.json'), 'w') as f:
    json.dump(test_env, f, indent=4)