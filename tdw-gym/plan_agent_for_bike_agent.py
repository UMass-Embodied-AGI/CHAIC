import numpy as np
import random
import copy
from PIL import Image
from agent_memory import AgentMemory
from enum import Enum
from logging import Logger
from LM_agent.LLM import LLM
import json
import os

MOVE_TO_GOAL_PLACE_LIMIT = 1000
MINIMUM_PATH_LENGTH = 25
MAX_EXPLORE_FRAMES = 250
MAX_MOVE_FRAMES = 1000

class PlanStatus(Enum):
    goto_target = 0
    goto_drop_zone = 2
    explore = 4
    wait = 5

class PlanAgentForBikeAgent:
    def __init__(self, agent_id: int, logger: Logger, max_frames: int, args, plan_mode: str = "default", output_dir = 'results', debug = False, gt_mask = False, gt_behavior = False, no_save_img = False, task_data = None):
        self.agent_names = ["Alice", "Bob"]
        self.max_frames = max_frames
        self.env_api = None
        self.agent_id = agent_id
        self.agent_type = 'plan_agent'
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
        if plan_mode == 'LLM':
            # self.source = args.source
            # self.lm_id = args.lm_id
            # self.prompt_template_path = args.prompt_template_path
            # self.communication = args.communication
            # self.cot = args.cot
            # self.args = args
            # self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id, self.output_dir)
            raise AssertionError
        
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

    def reset(self, obs, target_object_names: dict = {}, agent_color = [0, 0, 0], output_dir = None, env_api = None, rooms_name = None, obstacle_names = [], info = None, constraint_type = None, force_ignore = []):
        self.info = info
        self.constraint_type = constraint_type
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=self.gt_behavior, env_api=env_api, constraint_type = constraint_type, map_size = self.map_size, scene_bounds = self._scene_bounds)
        if type(target_object_names) == dict:
            self.target_object_names = list(target_object_names.keys())
        elif type(target_object_names) == list:
            self.target_object_names = target_object_names
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.rooms_name = rooms_name
        self.room_distance = 0
        self.oppo_pos = None
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.container_held = None
        # self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
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
        self.force_ignore = []
        # self.with_character = [self.agent_id]
        if self.agent_id == 1:
            self.prefer_target = []
        else:
            self.prefer_target = None
        self.obstacle_names = obstacle_names
        if self.plan_mode == 'LLM':
            # self.LLM.reset(self.rooms_name, self.info, output_dir)
            raise NotImplementedError
        # New setting, for task with target number
        self.picked_num = dict()
        self.required_num = target_object_names

    @property
    def holding_objects_id(self):
        return [x['id'] for x in self.obs['held_objects'] if x['id'] is not None]

    @property
    def oppo_holding_objects_id(self):
        return [x['id'] for x in self.obs['oppo_held_objects'] if x['id'] is not None]
    
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
            elif x['type'] == 5:
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

    def LLM_plan(self):
        raise NotImplementedError
        

    def get_object_list(self, target_object_names = []):
        #Extract object info from agent memory
        object_list = {0: [], 1: [], 2: [], 4: []}
        object_per_room = {room: {0: [], 1: [], 2: [], 4: []} for room in self.rooms_name}
        for object_type in [0, 1, 2, 4]:
            obj_map_indices = np.where(self.agent_memory.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.agent_memory.id_map[i, j]
                if object_type == 0 and (self.agent_memory.object_info[id]['name'] not in target_object_names):
                    if self.agent_id == 1:
                        print("ignore target:", self.agent_memory.object_info[id]['name'])
                    continue
                if id in self.satisfied or id in self.holding_objects_id or id in self.oppo_holding_objects_id or self.agent_memory.object_info[id] in object_list[object_type]:
                    continue
                object_list[object_type].append(self.agent_memory.object_info[id])
                room = self.env_api['belongs_to_which_room'](self.agent_memory.object_info[id]['position'])
                if room is None:
                    self.logger.warning(f"obj {self.agent_memory.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                object_per_room[room][object_type].append(self.agent_memory.object_info[id])
        return object_list, object_per_room

    def default_plan(self):
        if self.obs["held_objects"][0]['contained'][-1] is not None:
            sub_goal = PlanStatus.goto_drop_zone
        elif self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT and self.have_target_with_character():
            # Limited time, need to go to the drop zone
            sub_goal = PlanStatus.goto_drop_zone
        elif self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT * 1.5 and self.have_target_with_character() and len(self.object_list[2]) == 0:
            # Limited time, need to go to the drop zone, and we need to find the drop zone first.
            sub_goal = PlanStatus.goto_drop_zone
        else:
            # Grasp an object
            sub_goal = PlanStatus.goto_target
        # print(sub_goal, sub_goal.value, self.object_list[sub_goal.value])
        if sub_goal in [PlanStatus.goto_target, PlanStatus.goto_drop_zone] and len(self.object_list[sub_goal.value]) == 0:
            sub_goal = PlanStatus.explore
        return sub_goal, {}

    def get_valid_plan(self):
        valid_plan = [PlanStatus.explore, PlanStatus.goto_target]
        if self.have_target_with_character():
            valid_plan.append(PlanStatus.goto_drop_zone)

        return valid_plan

    def random_plan(self):
        valid_plan = self.get_valid_plan()
        sub_goal = random.choice(valid_plan)
        return sub_goal, {}
    
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
        if self.obs["held_objects"][0]['type'] is not None and self.obs["held_objects"][0]['contained'][0] is not None:
            return {"type": 5, "arm": "left", "object": self.target_id}
        else:
            self.plan = None
            self.logger.info(f"successful put_on!")
            return None

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def update_perfer_target(self):
        #Update prefer_target
        if self.agent_id == 1:
            return
            raise NotImplementedError
        elif self.agent_id == 0:
            # Main agent, know the ground truth task
            # Target list is the same as the prefer list
            self.target_object_names = []
            self.prefer_target = []
            for target in self.required_num.keys():
                if self.required_num[target] > self.picked_num.get(target, 0):
                    self.target_object_names.append(target)
                    self.prefer_target.append(target)

    def act(self, obs):
        if obs['status'] == 0: 
            #  Last action is still ongoing
            return {'type': 'ongoing'}

        if obs['status'] == 1:
            # Last action is successful
            # Add it to the picked_num, which is used to identify the remaining objects
            if self.last_action is not None and self.last_action['type'] == 3:
                self.picked_num[self.agent_memory.object_info[self.last_action['object']]['name']] = self.picked_num.get(self.last_action['object'], 0) + 1

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
        
        # need prefer target for agent 0
        self.update_perfer_target()
        
        # if self.agent_id == 1:
        #    print("self.obs: ", self.obs)
        # print("Agent:", self.agent_id, "perfer_target:", self.prefer_target, self.obs['held_objects'])
        # print("held_objects:", self.obs['held_objects'])
        # print("oppo_held_objects:", self.obs['oppo_held_objects'])
        # print("target_object_names:", self.target_object_names)
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'

        print(self.current_room)
        print(self.room_distance)
        print(self.rooms_explored)

        temp_with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs['held_objects']:
            if x['id'] is None: continue
            temp_with_character.append(x['id'])
            if 'contained' in x:
                for y in x['contained']:
                    if y is not None:
                        temp_with_character.append(y)

        for x in self.obs["visible_objects"]:
            if x["name"] is not None and (
                x["name"] == "tent" or x["name"] == "wood_shelving"
            ):
                if x["id"] is not None:
                    temp_with_character.append(x["id"])

        for x in self.obs['oppo_held_objects']:
            if x['id'] is None: continue
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

        self.object_list, self.object_per_room = self.get_object_list(self.target_object_names)
        print("object_list:", self.object_list)
        print("satisfied:", self.satisfied)

        info = {'satisfied': self.satisfied,
            #    'object_list': self.object_list,
                'current_room': self.current_room,
            #    'visible_objects': self.filtered(self.obs['visible_objects']),
                'frame': self.num_frames,
            #    'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
                'last_action_status': self.obs['status'],
                'oppo_in_view': self.agent_memory.oppo_this_step,
              }

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_id = None
                self.target_position = None
                self.drop_flag = False
                if lm_times > 0:
                    print(plan)
                    print("LLM failed!")
                    import time
                    time.sleep(1010)
                if lm_times > 3:
                    plan = random.choice(self.get_valid_plan())
                if self.plan_mode == 'default':
                    plan, a_info = self.default_plan()
                elif self.plan_mode == 'random':
                    plan, a_info = self.random_plan()
                elif self.plan_mode == 'LLM':
                    plan, self.target_id, a_info = self.LLM_plan()
                    print(plan, self.target_id, a_info)
                    if plan not in self.get_valid_plan():
                        plan = PlanStatus.explore
                        self.target_id = None
                    if plan in [PlanStatus.goto_target]:
                        if not (self.target_id in self.object_list[0]):
                            self.target_id = None
                    else:
                        self.target_id = None
                    if self.target_id:
                        info.update({"target_id": self.target_id})
                if plan is None:
                    # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    import time
                    time.sleep(1010)
                    plan = PlanStatus.wait
                self.plan = plan
                # self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
                self.plan_start_frame = copy.deepcopy(self.num_frames)
                print(self.plan, self.plan_start_frame)
            if self.plan == PlanStatus.goto_target:
                action = self.gopickup(target_type = 0)
            elif self.plan == PlanStatus.goto_drop_zone:
                action = self.goputon(target_type = 2)
            elif self.plan == PlanStatus.explore:
                action = self.goexplore()
            elif self.plan == PlanStatus.wait:
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")
            self.object_list, self.object_per_room = self.get_object_list(self.target_object_names)

        info.update({"action": str(action), "plan": self.plan.name if self.plan is not None else None})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        
        self.info_history.append(info)
        os.makedirs(os.path.join(self.output_dir, 'info'), exist_ok=True)
        json.dump(self.info_history, open(os.path.join(self.output_dir, 'info', str(self.agent_id)) + ".json", 'w'))
        return action