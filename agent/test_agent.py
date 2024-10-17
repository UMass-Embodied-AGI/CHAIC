import random
import numpy as np
from detection import init_detection # please do not import this in your own agent!


class PlanAgent:
    def __init__(self, id, args):
        assert id == 1
        self.detection_model = init_detection()
        self.detection_threshold = 3
        self.ignore_ids = []
        
    def detect(self, rgb):
        detect_result = self.detection_model(rgb[..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((rgb.shape[0], rgb.shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.ignore_ids)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def reset(self, obs, info):
        self.env_api = info['env_api']

    def act(self, obs):
        # useful objects in obs:
        # "rgb": RGB image of the current agent's view
        # "depth": depth image of the current agent's view
        # "camera_matrix": the camera matrix of current agent's ego camera
        # "FOV": the field of view of current agent's ego camera
        # "agent": a list of length 6 that contains the position (x,y,z) and forward (fx,fy,fz) of the agent, formatted as [x, y, z, fx, fy, fz]. 
        # "held_obejcts": all the objects that current agent is holding. 
        # It is a list of length 2 that contains the information of the object that is held in the agent's two hands. 
        # Each object's information contains its name, type and a unique id. 
        # If it's a container, it also includes the information of the objects in it
        # "status": the status of current action, which is a number from 0 to 2. 0 for 'ongoing', 1 for 'failure', 2 for 'success'.
        # "current_frames": the number of frames passed
        # "valid": whether the last action of the agent is valid
        # "previous_action" & "previous_status": all previous actions of the agent and their corresponding status
        
        all_actions = [0, 1, 2, 7]
        held_objects = obs["held_objects"]
        visible_objects, seg_mask = self.detect(obs["rgb"])
        has_container = False
        if held_objects[0]['type'] == 1 or held_objects[1]['type'] == 1:
            has_container = True
        
        if held_objects[0]['id'] is None or held_objects[1]['id'] is None:
            flag = False
            for obj in visible_objects:
                if obj["type"] == 0 or (obj["type"] == 1 and not has_container):
                    flag = True
                    break
                
            if flag:
                all_actions.append(3)
        
        if (held_objects[0]['type'] == 0 and held_objects[1]['type'] == 1 and held_objects[1]['contained'][-1] is None) or \
            (held_objects[1]['type'] == 0 and held_objects[0]['type'] == 1 and held_objects[0]['contained'][-1] is None):
            all_actions.append(4)
            
        if held_objects[0]['id'] is not None or held_objects[1]['id'] is not None:
            flag = False
            for obj in visible_objects:
                if obj["type"] == 2:
                    flag = True
                    break
            
            if flag:
                all_actions.append(5)
                
        action_id = random.choice(all_actions)
        if action_id in [0, 1, 2, 4]:
            return {"type": action_id}
        elif action_id == 7:
            delay = random.randint(1, 50)
            return {"type": 7, "delay": delay}
        elif action_id == 3:
            if held_objects[0]["id"] is None:
                arm = "left"
            else:
                arm = "right"
                
            can_pick_ids = []
            for obj in visible_objects:
                if obj["type"] == 0 or (obj["type"] == 1 and not has_container):
                    can_pick_ids.append(obj["id"])
                    
            pick_id = random.choice(can_pick_ids)
            return {"type": 3, "arm": arm, "object": pick_id}
        elif action_id == 5:
            if held_objects[0]["id"] is not None:
                arm = "left"
            else:
                arm = "right"
                
            can_place_ids = []
            for obj in visible_objects:
                if obj["type"] == 2:
                    can_place_ids.append(obj["id"])
                    
            place_id = random.choice(can_place_ids)
            return {"type": 5, "arm": arm, "object": place_id}