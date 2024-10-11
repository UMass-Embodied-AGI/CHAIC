import os
import json
import numpy as np

def get_CR(result_path):
    if not os.path.exists(os.path.join(result_path, 'eval_result.json')):
        return 0, 0, 0
        
    with open(os.path.join(result_path, 'eval_result.json'), 'r') as f:
        result = json.load(f)
    CR = []
    total_count = 0
    total_helper_count = 0
    file = json.load(open(test_env_path, 'r'))
    name_map = json.load(open(name_map_path, 'r'))
    mass_map = json.load(open(mass_map_path, 'r'))

    for episode in result['episode_results']:
        action_file = os.path.join(result_path, episode, 'actions.json')
        with open(action_file, 'r') as f:
            raw_data = json.load(f)
        previous_action_list = raw_data['action']
        previous_status_list = raw_data['status']
        # assert len(previous_action_list) == 2, "there must be 2 agents when calculating CR"
        targets = dict()
        progress = dict()
        tests = file[int(episode)]["task"]["goal_task"]
        for item in tests:
            if item[0] in name_map.keys():
                targets[name_map[item[0]]] = item[1]
                progress[name_map[item[0]]] = 0
            else:
                targets[item[0]] = item[1]
                progress[item[0]] = 0

        success = (result['episode_results'][episode]["finish"] == result['episode_results'][episode]["total"])

        p0 = 0
        p1 = 0
        with_character_id = [[], []]
        with_character_name = [[], []]
        satisfied = {}
        count = 0
        tot = result['episode_results'][episode]["finish"]
        while p0 < len(previous_action_list['0']) and p1 < len(previous_action_list['1']):
            action = [previous_action_list['0'][p0], previous_action_list['1'][p1]]
            status = [previous_status_list['0'][p0], previous_status_list['1'][p1]]
            frame0 = int(action[0].split("at frame ")[-1])
            frame1 = int(action[1].split("at frame ")[-1])
            if frame0 < frame1:
                cur = 0
                p0 += 1
            else:
                cur = 1
                p1 += 1

            if 'pick up' in action[cur] and 'success' in status[cur]:
                with_character_name[cur].append(action[cur].split('<')[0][:-1].split(' ')[-1])
                with_character_id[cur].append(int(action[cur].split('<')[-1].split('>')[0]))
            if ('put the object in the right hand on' in action[cur] or 'put the object in the left hand on' in action[cur]) and ('success' in status[cur] or ("ongoing" in status[cur] and success)):
                for object_id, object_name in zip(with_character_id[cur], with_character_name[cur]):
                    if object_id in satisfied.keys():
                        continue
                    satisfied[object_id] = True
                    if object_name in targets.keys():
                        if progress[object_name] < targets[object_name]:
                            progress[object_name] += 1
                            if cur == 1:
                                if "furniture" in result_path:
                                    count += mass_map[object_name]
                                else:
                                    count += 1
                            elif object_id in with_character_id[1]:
                                if "furniture" in result_path:
                                    count += mass_map[object_name]
                                else:
                                    count += 1
                                with_character_id[1].remove(object_id)
                                with_character_name[1].remove(object_name)

                    with_character_id[cur] = []
                    with_character_name[cur] = []

        while p1 < len(previous_action_list['1']):
            action = previous_action_list['1'][p1]
            status = previous_status_list['1'][p1]
            if 'pick up' in action and 'success' in status:
                with_character_name[1].append(action.split('<')[0][:-1].split(' ')[-1])
                with_character_id[1].append(int(action.split('<')[-1].split('>')[0]))
            if ('put the object in the right hand on' in action or 'put the object in the left hand on' in action) and ('success' in status or ("ongoing" in status and success)):
                for object_id, object_name in zip(with_character_id[1], with_character_name[1]):
                    if object_id in satisfied.keys():
                        continue
                    satisfied[object_id] = True
                    if object_name in targets.keys():
                        if progress[object_name] < targets[object_name]:
                            progress[object_name] += 1
                            if "furniture" in result_path:
                                count += mass_map[object_name]
                            else:
                                count += 1

                    with_character_id[1] = []
                    with_character_name[1] = []
            p1 += 1

        # while p0 < len(previous_action_list['0']):
        #     action = previous_action_list['0'][p0]
        #     status = previous_status_list['0'][p0]
        #     if 'pick up' in action and 'success' in status:
        #         with_character_name[0].append(action.split('<')[0][:-1].split(' ')[-1])
        #         with_character_id[0].append(int(action.split('<')[-1].split('>')[0]))
        #     if ('put the object in the right hand on' in action or 'put the object in the left hand on' in action) and ('success' in status or ("ongoing" in status and success)):
        #         for object_id, object_name in zip(with_character_id[0], with_character_name[0]):
        #             if object_id in satisfied.keys():
        #                 continue
        #             satisfied[object_id] = True
        #             if object_name in targets.keys():
        #                 if progress[object_name] < targets[object_name]:
        #                     progress[object_name] += 1

        #             with_character_id[0] = []
        #             with_character_name[0] = []
        #     p0 += 1
        
        # sm = 0
        # for name in progress:
        #     sm += progress[name]
       
        if tot != 0:
            total_helper_count += count
            total_count += tot

    return total_helper_count, total_count

tasks = ['normal', 'highthing', 'highcontainer', 'highgoalplace', 'lowthing', 'wheelchair', 'outdoor_shopping', 'outdoor_furniture']
for task in tasks:
    path = os.path.join('organized_result', task)
    baselines = os.listdir(path)

    test_env_path = os.path.join('dataset/test_dataset', task, 'test_env.json')
    name_map_path = "dataset/name_map.json"
    mass_map_path = "dataset/test_dataset/outdoor_furniture/mass_map.json"

    for baseline in baselines:
        if baseline == 'no_helper':
            continue
        cur_path = os.path.join(path, baseline)
        
        total_count = 0
        total_helper_count = 0
        for seed in os.listdir(cur_path):
            result_path = os.path.join(cur_path, seed)
            
            helper_count, count = get_CR(result_path)
            total_helper_count += helper_count
            total_count += count

        if total_count == 0:
            print(f"Helper for {baseline} for {task} don't pickup anything!")
        else:
            print(f"CR for {baseline} for {task}: {total_helper_count / total_count}={total_helper_count}/{total_count}")