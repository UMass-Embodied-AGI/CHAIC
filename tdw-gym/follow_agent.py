import numpy as np
import random
import copy
from PIL import Image
from agent_memory import AgentMemory
from enum import Enum
from logging import Logger
from LM_agent.LLM import LLM
import time
import os
import json


MOVE_TO_GOAL_PLACE_LIMIT = 750
MINIMUM_PATH_LENGTH = 25
MAX_EXPLORE_FRAMES = 250
MAX_MOVE_FRAMES = 1000

class PlanStatus(Enum):
    goto_target = 0
    goto_container = 1
    goto_drop_zone = 2
    # do not change 0, 1, 2
    put_in = 3
    explore = 4
    wait = 5
    remove_obstacle = 6
    follow_another_agent = 7

class FollowAgent:
    def __init__(self, agent_id: int, logger: Logger, max_frames: int, args, plan_mode: str = "default", output_dir = 'results', debug = False, gt_mask = False, no_save_img = False, task_data = None, gt_behavior = False):
        self.agent_names = ["Alice", "Bob"]
        self.max_frames = max_frames
        self.env_api = None
        self.agent_id = agent_id
        self.agent_type = 'follow_agent'
        self.logger = logger
        self.output_dir = output_dir
        self.last_action = None
        self.gt_behavior = gt_behavior
        
        self.task_kind = None
        if task_data is not None and "task" in task_data and "task_kind" in task_data["task"]:
            self.task_kind = task_data["task"]["task_kind"]
            
        if self.task_kind is not None and self.task_kind == "outdoor_shopping":
            self.map_size = (224, 108)
            self._scene_bounds = {
                "x_min": -3,
                "x_max": 25,
                "z_min": -6,
                "z_max": 7.5
            }
        elif self.task_kind is not None and self.task_kind == "outdoor_furniture":
            self.map_size = (192, 108)
            self._scene_bounds = {
                "x_min": -5,
                "x_max": 19,
                "z_min": -6,
                "z_max": 7.5
            }
        else:
            self.map_size = (240, 120)
            self._scene_bounds = {
                "x_min": -15,
                "x_max": 15,
                "z_min": -7.5,
                "z_max": 7.5
            }

        self.agent_memory = None
        self.invalid_count = None
        self.communication = False
        assert plan_mode == 'default', "No need to plan"
        self.object_list = None
        self.object_per_room = None
        self.debug = debug
        self.plan_mode = plan_mode  #["default", "random", "LLM"]
        self.plan_start_frame = None
        # Used in plan helper, recording which object the constraint agent is holding
        self.prefer_target = None
        self.gt_mask = gt_mask
        self.no_save_img = no_save_img
        self.info = None
        self.constraint_type = None

    def reset(self, obs, target_object_names = [], agent_color = [0, 0, 0], output_dir = None, env_api = None, rooms_name = None, obstacle_names = [], info = None, constraint_type = None, force_ignore = []):
        self.info = info
        self.constraint_type = constraint_type
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, env_api=env_api, constraint_type = constraint_type, map_size = self.map_size, scene_bounds = self._scene_bounds)
        self.target_object_names = target_object_names
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.rooms_name = rooms_name
        self.room_distance = 0
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.with_oppo = []
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rooms_explored = {}
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        self.object_list = {0: [], 1: [], 2: []}
        self.info_history = []
        # self.with_character = [self.agent_id]
        if self.agent_id == 1:
            self.prefer_target = []
        else:
            self.prefer_target = None
        self.obstacle_names = obstacle_names
        if self.plan_mode == 'LLM':
            self.LLM.reset(self.rooms_name, self.info, output_dir)
        self.target_position = None

    @property
    def holding_objects_id(self):
        return [x['id'] for x in self.obs['held_objects'] if x['id'] is not None]
    
    @property
    def num_frames(self):
        return self.obs['current_frames']
    
    @property
    def satisfied(self):
        return self.obs['satisfied']
    
    def have_target_with_character(self):
        for x in self.obs['held_objects']:
            if x['type'] == 0:
                return True
            elif x['type'] == 1:
                for y in x['contained']:
                    if y is None or y not in self.agent_memory.object_info:
                        continue
                    if self.agent_memory.object_info[y]['type'] == 0:
                        return True
        return False

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[-1] - g[-1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold = 1.5):
        if target_pos is None:
            # Ground
            return True
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        print("distance:", d, 'target_pos:', (gx, gz), 'current_pos:', (x, z))
        return d < threshold

    def gofollow(self):
        if self.agent_memory.oppo_pos is None and 'oppo_pos' not in self.obs:
            self.plan = PlanStatus.explore
            return self.goexplore()
        else:
            self.plan = PlanStatus.follow_another_agent
            if 'oppo_pos' in self.obs:
                self.target_position = self.obs['oppo_pos']
            else:
                self.target_position = self.agent_memory.oppo_pos
            return self.agent_memory.move_to_pos(self.target_position, explore=False, follow = True)[0]

    def gopickup(self, target_id = None, target_type = None):
        assert target_id is not None or target_type is not None
        if self.target_id is None:
            min_distance = 100000
            if target_id is None:
                if self.prefer_target is not None and len(self.prefer_target) > 0 and self.plan_mode == 'default':
                    perfer_id = []
                    for obj in self.object_list[target_type]:
                        if obj['name'] in self.prefer_target:
                            perfer_id.append(obj['id'])
                    if len(perfer_id) > 0:
                        for id in perfer_id:
                            distance = self.l2_distance(self.position, self.agent_memory.object_info[id]['position'])
                            if distance < min_distance:
                                min_distance = distance
                                self.target_id = id
                    else:
                        if self.plan_mode == 'default':
                            for obj in self.object_list[target_type]:
                                distance = self.l2_distance(self.position, obj['position'])
                                if distance < min_distance:
                                    min_distance = distance
                                    self.target_id = obj['id']
                        else:
                            self.target_id = self.object_list[target_type][random.randint(1, len(self.object_list[target_type])) - 1]['id']
                else:
                    if self.plan_mode == 'default':
                        for obj in self.object_list[target_type]:
                            distance = self.l2_distance(self.position, obj['position'])
                            if distance < min_distance:
                                min_distance = distance
                                self.target_id = obj['id']
                    else:
                        self.target_id = self.object_list[target_type][random.randint(1, len(self.object_list[target_type])) - 1]['id']
            else:
                self.target_id = target_id
        if self.target_id in self.agent_memory.ignore_ids:
            # The target object is not here any more
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if self.target_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
            self.plan = None
            return None

        if self.target_position is None:
            self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]['position'])

        if self.target_id not in self.agent_memory.object_info or self.target_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
            return None

        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if not self.reach_target_pos(self.target_position) or path_len > MINIMUM_PATH_LENGTH:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
                return None
            return action

        action = {"type": 3, "object": self.target_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action
    
    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action

    def goexplore(self):
        if self.target_position is None:
            self.target_position = self.agent_memory.explore()
        if self.num_frames - self.plan_start_frame > MAX_EXPLORE_FRAMES:
            self.plan = None
            return None
        action, _ = self.agent_memory.move_to_pos(self.target_position, explore=True)
        if action is None:
            # Reach the target position
            self.plan = None
            self.target_position = None
        return action
    
    def goputon(self, target_id = None, target_type = None):
        assert target_id is not None or target_type is not None
        if self.target_id is None:
            if target_id is None:
                self.target_id = self.object_list[target_type][random.randint(1, len(self.object_list[target_type])) - 1]['id']
            else:
                self.target_id = target_id

        if self.target_position is None:
            if self.target_id != -1:
                self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]['position'])
            else:
                self.target_position = None
        
        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if (not self.reach_target_pos(self.target_position, threshold = 2.0) or path_len > MINIMUM_PATH_LENGTH) and not self.drop_flag:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
                return None
            return action

        self.drop_flag = True
        if self.obs["held_objects"][0]['type'] is not None:
            return {"type": 5, "arm": "left", "object": self.target_id}
        elif self.obs["held_objects"][1]['type'] is not None:
            return {"type": 5, "arm": "right", "object": self.target_id}
        else:
            self.plan = None
            self.logger.info(f"successful put_on!")
            return None

    def remove_obstacle(self):
        if self.target_id is None:
            min_distance = 100000
            for obj in self.object_list[4]:
                distance = self.l2_distance(self.position, obj['position'])
                if distance < min_distance:
                    min_distance = distance
                    self.target_id = obj['id']
        if self.target_id in self.agent_memory.ignore_ids:
            # The target object is not here any more
            if self.debug:
                self.logger.debug(f"remove obstacle failed. object is not here any more!")
            self.plan = None
            return None
        
        if self.target_position is None:
            self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]['position'])

        if self.target_id not in self.agent_memory.object_info or self.target_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"remove obstacle failed. object is not here any more!")
            self.plan = None
            self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
            return None

        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if not self.reach_target_pos(self.target_position) or path_len > MINIMUM_PATH_LENGTH:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames = self.obs['current_frames'], ignore_ids = [self.target_id])
                return None
            return action

        action = {"type": 7, "object": self.target_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def update_perfer_target(self):
        #Update prefer_target
        return

    def act(self, obs):
        if obs['status'] == 0: 
            # ongoing
            return {'type': 'ongoing'}
        self.obs = obs
        ignore_obstacles = []
        ignore_ids = []
        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                ignore_ids.append(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 100, "invalid action for 100 times"
    
        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}")
        
        self.update_perfer_target()

        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'

        temp_with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs['held_objects']:
            temp_with_character.append(x['id'])
            if 'contained' in x:
                for y in x['contained']:
                    if y is not None:
                        temp_with_character.append(y)
        for x in self.obs['oppo_held_objects']:
            temp_with_oppo.append(x['id'])
            if 'contained' in x:
                for y in x['contained']:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = temp_with_character + ignore_obstacles
        ignore_ids = temp_with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        if not self.gt_mask:
            # use the detection model to get the mask
            self.obs['visible_objects'], self.obs['seg_mask'] = self.agent_memory.detect(self.obs['rgb'])

        self.agent_memory.update(obs, ignore_ids = ignore_ids, ignore_obstacles = ignore_obstacles, save_img = not self.no_save_img)

        info = {'satisfied': self.satisfied,
            #    'object_list': self.object_list,
                'current_room': self.current_room,
            #    'visible_objects': self.filtered(self.obs['visible_objects']),
                'frame': self.num_frames,
            #    'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
                'last_action_status': self.obs['status'],
                'oppo_in_view': self.agent_memory.oppo_this_step,
              }

        self.plan_start_frame = self.num_frames
        action = self.gofollow()
        info.update({"action": action, "plan": self.plan.name if self.plan is not None else None})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        self.info_history.append(info)
        os.makedirs(os.path.join(self.output_dir, 'info'), exist_ok=True)
        json.dump(self.info_history, open(os.path.join(self.output_dir, 'info', str(self.agent_id)) + ".json", 'w'))
        return action