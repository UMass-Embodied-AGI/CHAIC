import json
with_character_id = []
satisfied = {}
count = 0
log_path = "high_goalplace_task_plan_helper/vision_1/0/actions.json"
with open(log_path, 'r') as f:
    raw_data = json.load(f)
previous_action_list = raw_data['action']
previous_status_list = raw_data['status']
for action, status in zip(previous_action_list['1'], previous_status_list['1']):
    if 'pick up' in action and 'success' in status:
        with_character_id.append(int(action.split('<')[-1].split('>')[0]))
    if ('put the object in the right hand on' in action or 'put the object in the left hand on' in action) and 'success' in status:
        for object_id in with_character_id:
            if object_id in satisfied.keys():
                continue
            satisfied[object_id] = True
            if object_id in [12518061]:
                        count += 1
                    # controller.communicate({"$type": "destroy_object", "id": object_id})
            with_character_id = []
print(count)