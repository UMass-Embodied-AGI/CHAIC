import argparse
import os
import json
import random
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="data/smart_help_data_revised")
parser.add_argument("--output_dir", type=str, default="data/smart_help_data_processed_revised")
parser.add_argument("--id_map_dir", type=str, default="id_map.json")
parser.add_argument("--name_map_dir", type=str, default="name_map.json")
parser.add_argument("--window_size", type=int, default=5)
parser.add_argument("--need_plan_state", action='store_true', default=False)

args = parser.parse_args()
dataset_dir = args.dataset_dir
output_dir = args.output_dir
window_size = args.window_size
observe_len = 60
with open (args.id_map_dir, "r") as f:
    id_map = json.load(f)
with open (args.name_map_dir, "r") as f:
    name_map = json.load(f)

def proc(obs):
    if obs["agent"] is not None:
        agent = obs["agent"]
        if agent["status"] is not None:
            if agent["status"] == "ActionStatus.ongoing":
                agent["status"] = 0
            elif agent["status"] == "ActionStatus.success":
                agent["status"] = 1
            else:
                agent["status"] = 2
        else: 
            agent["status"] = 3

        if agent["action"] is not None:
            action_sem = agent["action"].split(" at frame")[0]
            tp = action_sem.split(" ")[0]
            if tp == "ongoing":
                action = [1, 0, 0]
            elif tp == "moving":
                action = [2, 0, 0]
            elif tp == "pick":
                obj_name = action_sem.split("pick up ")[1].split(" <")[0]
                hand = action_sem.split("with ")[-1].split(" hand")[0]
                if "left" in hand:
                    hand = 0
                else:
                    hand = 1

                if obj_name in id_map:
                    action = [3, hand, id_map[obj_name]]
                else:
                    action = [3, hand, 0]

            elif tp == "put":
                if action_sem == "put the object in the container":
                    action = [4, 0, 0]
                else:
                    hand = action_sem.split("put the object in the ")[-1].split(" hand")[0]
                    if "left" in hand:
                        hand = 0
                    else:
                        hand = 1

                    place = action_sem.split("hand on ")[-1]
                    action = [5, hand, 0]
            elif tp == "wait":
                action = [6, 0, 0]
            else:
                raise NotImplementedError
            
            agent["action"] = action
        else:
            agent["action"] = [0, 0, 0]

        if agent["held_objects"] is not None:
            for i in range(2):
                name = agent["held_objects"][i]["name"]
                if name in id_map:
                    agent["held_objects"][i]["id"] = id_map[name]
                else:
                    agent["held_objects"][i]["id"] = 0
                
                agent["held_objects"][i]["contained_id"] = []
                for name2 in agent["held_objects"][i]["contained_name"]:
                    if name2 is None:
                        agent["held_objects"][i]["contained_id"].append(0)
                    elif name2 in id_map:
                        agent["held_objects"][i]["contained_id"].append(id_map[name2])
                    else:
                        agent["held_objects"][i]["contained_id"].append(0)

                if agent["held_objects"][i]["type"] == 1:
                    agent["held_objects"][i]["contained_id"] = agent["held_objects"][i]["contained_id"][:3]

                if agent["held_objects"][i]["type"] is None:
                    agent["held_objects"][i]["type"] = 0
                else:
                    agent["held_objects"][i]["type"] += 1
    else:
        agent = None

    objs = []
    for obj in obs["objects"]:
        if obj["name"] in id_map:
            obj["id"] = id_map[obj["name"]]
            objs.append(obj)

    return {"agent": agent, "objects": objs}

def balance(datapoints):
    count = dict()
    for i in range(len(datapoints)):
        goal = datapoints[i]["goal"]
        if goal[0] not in count:
            count[goal[0]] = 0
        count[goal[0]] += 1

    min_count = 10000000
    for k, v in count.items():
        if k != 0:
            min_count = min(min_count, v)

    print(count)
    counts = dict()
    new_datapoints = []
    for i in range(len(datapoints)):
        if datapoints[i]["goal"][0] not in counts:
            counts[datapoints[i]["goal"][0]] = 0

        if counts[datapoints[i]["goal"][0]] < min_count:
            counts[datapoints[i]["goal"][0]] += 1
            new_datapoints.append(datapoints[i])
    
    return new_datapoints

datapoints = []
max_len = 0
for task in os.listdir(dataset_dir):
    task_dir = os.path.join(dataset_dir, task)
    for file in os.listdir(task_dir):
        file_path = os.path.join(task_dir, file)
        print(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
        
        constaint = data["constraint"]
        observation = data["obs"]
        goal = data["goal"]
        plan_state = data["plan_state"]
        obs_history = [{"agent": None, "objects": []}] * window_size
        for num_frame in observation:
            if num_frame in goal and (not args.need_plan_state or plan_state[num_frame]):
                cur_goal = goal[num_frame]
                processed_goal = [0, 0]

                if cur_goal[0] == "explore":
                    processed_goal = [2, 0]
                elif cur_goal[0] == "wait":
                    processed_goal = [6, 0]
                elif cur_goal[0] == "pick":
                    if cur_goal[1] in name_map:
                        cur_name = name_map[cur_goal[1]]
                    else:
                        cur_name = cur_goal[1]
                    if cur_name in id_map:
                        processed_goal = [3, id_map[cur_name]]
                    else:
                        processed_goal = [3, 0]
                elif cur_goal[0] == "puton":
                    if cur_goal[1] in name_map:
                        cur_name = name_map[cur_goal[1]]
                    else:
                        cur_name = cur_goal[1]
                    if cur_name in id_map:
                        id1 = id_map[cur_name]
                    else:
                        id1 = 0

                    processed_goal = [5, id1]
                elif cur_goal[0] == "putin":
                    processed_goal = [4, 0]
                
                flag = True
                for obs in obs_history:
                    if obs["agent"] is not None:
                        flag = False
                        break
                
                if flag:
                    processed_goal = [0, 0]

                cur_agents = []
                cur_objs = []
                for i in range(len(obs_history)):
                    obs = obs_history[i]
                    # if obs["agent"] is None:
                    #     cur_agents.append([0] * 20)
                    # else:
                    #     cur_agents.append(obs["agent"]["position"] + obs["agent"]["forward"] + obs["agent"]["action"] + [obs["agent"]["status"]] + \
                    #                       [obs["agent"]["held_objects"][0]["type"]] + [obs["agent"]["held_objects"][0]["id"]] + obs["agent"]["held_objects"][0]["contained_id"][:3] + \
                    #                       [obs["agent"]["held_objects"][1]["type"]] + [obs["agent"]["held_objects"][1]["id"]]  + obs["agent"]["held_objects"][1]["contained_id"][:3])
                    
                    #     assert len(cur_agents[-1]) == 20
                    
                    if obs["agent"] is None:
                        cur_agents.append({"position": [0, 0, 0], "forward": [0, 0, 0], "action": [0, 0, 0], "status": 0, 
                                           "held_objects": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
                    else:
                        cur_agents.append(deepcopy(obs["agent"]))
                        held_objs = [obs["agent"]["held_objects"][0]["type"]] + [obs["agent"]["held_objects"][0]["id"]] + obs["agent"]["held_objects"][0]["contained_id"][:3] + \
                                                        [obs["agent"]["held_objects"][1]["type"]] + [obs["agent"]["held_objects"][1]["id"]]  + obs["agent"]["held_objects"][1]["contained_id"][:3]
                        cur_agents[-1]["held_objects"] = deepcopy(held_objs)

                    objs = {"id": [], "weight": [], "position": [], "height": []}
                    for obj in obs["objects"]:
                        # objs.append([obj["id"]] + [obj["weight"]] + obj["position"] + [obj["height"]])
                        # assert len(objs[-1]) == 6
                        objs["id"].append(obj["id"])
                        objs["weight"].append(obj["weight"])
                        objs["position"].append(obj["position"])
                        objs["height"].append(obj["height"])

                    while(len(objs["id"]) < observe_len):
                        objs["id"].append(0)
                        objs["weight"].append(0)
                        objs["position"].append([0, 0, 0])
                        objs["height"].append(0)

                    cur_objs.append(objs)
                    max_len = max(max_len, len(obs["objects"]))

                cur_datapoint = {"constraint": deepcopy(constaint), "goal": deepcopy(processed_goal), "agent": deepcopy(cur_agents), "objs": deepcopy(cur_objs)}

                datapoints.append(cur_datapoint) 

            obs = observation[num_frame]
            obs_history.pop(0)
            obs_history.append(proc(obs))

random.shuffle(datapoints)
datapoints = balance(datapoints)
random.shuffle(datapoints)
train_size = int(0.8 * len(datapoints))
train_datapoints = datapoints[:train_size]
test_datapoints = datapoints[train_size:]
print("Train size: ", len(train_datapoints))
print("Test size: ", len(test_datapoints))
print("Max len: ", max_len)
count = dict()
for i in range(len(train_datapoints)):
    goal = train_datapoints[i]["goal"]
    if goal[0] not in count:
        count[goal[0]] = 0
    count[goal[0]] += 1

print(count)
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_datapoints, f)

with open(os.path.join(output_dir, "test.json"), "w") as f:
    json.dump(test_datapoints, f)