from LM_agent.LLM import LLM
from LM_agent.VLM import VLM

import numpy as np
import random
import copy
from agent_memory import AgentMemory
from enum import Enum
from logging import Logger
import json
import os

import torch
import torch.nn as nn
from smart_help_baseline.model import CustomModel

MOVE_TO_GOAL_PLACE_LIMIT = 750
MINIMUM_PATH_LENGTH = 25
MAX_EXPLORE_FRAMES = 250
MAX_FOLLOW_FRAMES = 250
MAX_FOLLOW_CHILD_FRAMES = 300
MAX_MOVE_FRAMES = 1000
AGENT_DIST_THRESHOLD = 3.5
AGENT_DIST_THRESHOLD_LARGE = 5.5



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
    follow_third_agent = 8
    turn_around = 9
    

plan_to_text_back = {
    "goto and pick up target": PlanStatus.goto_target,
    "goto and pick up container": PlanStatus.goto_container,
    "transport object in hand to goal space": PlanStatus.goto_drop_zone,
    "put the object in your one hand to the container in your other hand": PlanStatus.put_in,
    "explore": PlanStatus.explore,
    "remove obstacle": PlanStatus.remove_obstacle,
    "follow Alice": PlanStatus.follow_another_agent,
    "follow David": PlanStatus.follow_another_agent,
    "follow child": PlanStatus.follow_third_agent,
    "turn around": PlanStatus.turn_around,
}

plan_to_text = {
    PlanStatus.goto_target: "goto and pick up target, {target_id_you_want_to_go_to}",
    PlanStatus.goto_container: "goto and pick up container, {container_id_you_want_to_go_to}",
    PlanStatus.goto_drop_zone: "transport object in hand to goal space",
    PlanStatus.put_in: "put the object in your one hand to the container in your other hand",
    PlanStatus.explore: "explore",
    PlanStatus.remove_obstacle: "remove obstacle, {obstacle_id_you_want_to_go_to_and_remove}",
    PlanStatus.follow_another_agent: "follow another agent",
    PlanStatus.follow_third_agent: "follow child",
    PlanStatus.turn_around: "turn around",
}

    
class PlanAgent:
    def __init__(
        self,
        agent_id: int,
        logger: Logger,
        max_frames: int,
        args,
        plan_mode: str = "default",
        output_dir="results",
        debug=False,
        gt_mask=False,
        gt_behavior=False,
        no_save_img=False,
        task_data=None,
        number_of_agents=2,
        rm_behavior = False
    ):
        self.agent_names = ["Alice", "Bob"]
        self.max_frames = max_frames
        self.env_api = None
        self.agent_id = agent_id
        self.agent_type = 'plan_agent'
        self.logger = logger
        self.output_dir = output_dir
        self.last_action = None
        self.number_of_agents = number_of_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.invalid_count = None
        self.communication = False
        self.rm_behavior = rm_behavior
        if plan_mode == "LLM":
            self.source = args.source
            self.lm_id = args.lm_id
            self.prompt_template_path = args.prompt_template_path
            self.communication = args.communication
            self.cot = args.cot
            self.args = args
            self.LLM = LLM(
                self.source,
                self.lm_id,
                self.prompt_template_path,
                self.communication,
                self.cot,
                self.args,
                self.agent_id,
                self.output_dir,
                self.task_kind,
                self.rm_behavior
            )
            
        if plan_mode == "VLM":
            self.source = args.source
            self.vlm_id = args.vlm_id
            self.vlm_prompt_template_path = args.vlm_prompt_template_path
            self.communication = args.communication
            self.cot = args.cot
            self.args = args
            self.VLM = VLM(
                self.source,
                self.vlm_id,
                self.vlm_prompt_template_path,
                self.communication,
                self.cot,
                self.args,
                self.agent_id,
                self.output_dir,
                self.task_kind
            )
        
        if plan_mode == "RL":
            with open("dataset/name_map.json", "r") as f:
                name_map = json.load(f)
                self.name_list = list(name_map.keys())
            assert False, "RL is not implemented yet"
            self.RL = RL()

        if plan_mode == "smart_help":
            self.smart_help_oppo_model = nn.DataParallel(CustomModel()).to(self.device)
            model_path = 'smart_help_baseline/oppo_model.pth'
            ckpt = torch.load(model_path)
            # state_dict = {}
            # for k, v in ckpt.items():
            #     new_key = k.replace('module.', '')
            #     state_dict[new_key] = v

            self.smart_help_oppo_model.load_state_dict(ckpt)
            self.smart_help_oppo_model.eval()

        self.object_list = None
        self.object_per_room = None
        self.debug = debug
        self.plan_mode = plan_mode
        self.plan_start_frame = None
        self.force_ignore = []
        # Used in plan helper, recording which object the constraint agent is holding
        self.prefer_target = None
        self.gt_mask = gt_mask
        self.gt_behavior = gt_behavior
        self.no_save_img = no_save_img
        self.info = None
        self.constraint_type = None

    def reset(self, obs, target_object_names: dict = {}, agent_color = [0, 0, 0], output_dir = None, env_api = None, rooms_name = None, obstacle_names = [], info = None, constraint_type = None, force_ignore = []):
        self.info = info
        self.constraint_type = constraint_type
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=self.gt_behavior or self.plan_mode != "LLM", env_api=env_api, constraint_type = constraint_type, map_size = self.map_size, scene_bounds = self._scene_bounds)
        if type(target_object_names) == dict:
            self.target_object_names = list(target_object_names.keys())
        elif type(target_object_names) == list:
            self.target_object_names = target_object_names
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.rooms_name = rooms_name
        self.room_distance = 0
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        # self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rooms_explored = {}
        self.last_turn_around = 0
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        self.object_list = {0: [], 1: [], 2: []}
        self.info_history = []
        self.planned = False
        self.force_ignore = force_ignore
        if self.agent_id == 1:
            self.prefer_target = []
        else:
            self.prefer_target = None
        self.obstacle_names = obstacle_names
        if self.plan_mode == "LLM":
            self.LLM.reset(self.rooms_name, self.info, output_dir)
        if self.plan_mode == "VLM":
            self.VLM.reset(self.rooms_name, self.info, output_dir)
        if self.plan_mode == "smart_help":
            self.smart_help_obs = [{"agent_pos": [0] * 3, "agent_rot": [0] * 3, "agent_action": [0] * 3, "agent_status": [0], "agent_held": [0] * 10, 
                                   "obj_id": [[0]] * 60, "obj_weight": [[0]] * 60, "obj_pos": [[0] * 3] * 60, "obj_height": [[0]] * 60}] * 5

            with open ("smart_help_baseline/id_map.json", "r") as f:
                self.id_map = json.load(f)
            with open ("smart_help_baseline/name_map.json", "r") as f:
                self.name_map = json.load(f)

            with open("dataset/train_dataset/outdoor_furniture/mass_map.json", "r") as f:
                self.mass_map = json.load(f)


        # New setting, for task with target number
        self.picked_num = dict()
        self.required_num = target_object_names
        self.turning_around = False
        self.counter = 0

    @property
    def holding_objects_id(self):
        return [x["id"] for x in self.obs["held_objects"] if x["id"] is not None]

    @property
    def oppo_holding_objects_id(self):
        return [x["id"] for x in self.obs["oppo_held_objects"] if x["id"] is not None]

    @property
    def num_frames(self):
        return self.obs["current_frames"]

    @property
    def satisfied(self):
        return self.obs["satisfied"]

    def have_target_with_character(self):
        for x in self.obs["held_objects"]:
            if x["type"] == 0:
                return True
            elif x["type"] == 1:
                for y in x["contained"]:
                    if y is None or y not in self.agent_memory.object_info:
                        continue
                    if self.agent_memory.object_info[y]["type"] == 0:
                        return True
        return False

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[-1] - g[-1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold=1.5):
        if target_pos is None:
            # Ground
            return True
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        print("distance:", d, "target_pos:", (gx, gz), "current_pos:", (x, z))
        return d < threshold

    def RL_observation(self):
        held = self.with_character.copy()
        for x in self.force_ignore:
            held.remove(x)
        held.remove(self.agent_id)
        held_object_number = np.zeros(10)
        held_object_number[len(held) - 1] = 1
        obs = dict(
            rgb = self.obs["rgb"],
            senmatic_map = self.agent_memory.draw_map(previous_name = None, save = False),
            # The semantic map
            pos = self.position,
            # Agent pos
            forward = self.forward,
            # Agent forward
            have_container = [any([x["type"] == 1 for x in self.obs["held_objects"]])],
            # If agent have container
            held_object_number = held_object_number
            # Agent held objects
        )
        return obs
    
    def smart_help_observation(self):
        held = self.with_character.copy()
        for x in self.force_ignore:
            held.remove(x)
        held.remove(self.agent_id)
        held_object_number = np.zeros(10)
        held_object_number[len(held) - 1] = 1
        new_obs = self.get_smart_help_new_obs(self.obs)
        obs = copy.deepcopy(new_obs)
        obs["self_pos"] = self.position
        obs["self_forward"] = self.forward
        obs["have_container"] = [any([x["type"] == 1 for x in self.obs["held_objects"]])]
        obs["held_object_number"] = held_object_number
        goal_feature, tar_1_feature, constraint_feature = self.get_latent_vector()
        obs["oppo_model_feature"] = goal_feature + tar_1_feature + constraint_feature
        return obs

    def RL_plan(self, rl_act):
        # action_space: dim_0 = 8, dim_1 = all_objects (by reading name map)
        target_name = self.name_list[rl_act[1]]
        plan = None
        if rl_act[0] == 0: # goto_target
            plan = PlanStatus.goto_target
            target_id = self.agent_memory.nearest_object_id(target_name, 0)
        elif rl_act[0] == 1: # goto_container
            plan = PlanStatus.goto_container
            target_id = self.agent_memory.nearest_object_id(target_name, 1)
        elif rl_act[0] == 2: # goto_drop_zone
            plan = PlanStatus.goto_drop_zone
        elif rl_act[0] == 3: # put_in
            plan = PlanStatus.put_in
        elif rl_act[0] == 4:
            plan = PlanStatus.explore
        elif rl_act[0] == 5:
            plan = PlanStatus.wait
        elif rl_act[0] == 6:
            plan = PlanStatus.remove_obstacle
            target_id = self.agent_memory.nearest_object_id(target_name, 4)
        elif rl_act[0] == 7:
            plan = PlanStatus.follow_another_agent

        valid_plan = self.get_valid_plan()
        if plan not in valid_plan:
            plan = PlanStatus.explore
            target_id = None
        
        return plan, target_id, {"rl_result": rl_act}

    def LLM_plan(self):        
        valid_plan = self.get_valid_plan()
        valid_plan = [plan_to_text[x] for x in valid_plan]
        oppo_held_objects_history = self.agent_memory.oppo_held_objects_history
        # if any(obj['type'] is not None for obj in oppo_held_objects_latest):
            # breakpoint()  #! for debugging only
        if len(self.agent_memory.action_history_dict.keys()) > 1:
            print(f"\n\n>>>> oppo action history: {self.agent_memory.action_history_dict[1 - self.agent_id]}\n")
            print(f">>>> oppo status history: {self.agent_memory.status_history_dict[1 - self.agent_id]}\n\n")
            plan, plan_id, info = self.LLM.run(
                self.num_frames,
                self.obs["held_objects"],
                self.agent_memory.ignored_filter_object_info(),
                [self.agent_memory.object_info[x] for x in self.satisfied if x in self.agent_memory.object_info.keys()],
                self.object_list,
                self.agent_memory.action_history_dict[self.agent_id],
                self.agent_memory.status_history_dict[self.agent_id],
                self.agent_memory.action_history_dict[1 - self.agent_id],
                self.agent_memory.status_history_dict[1 - self.agent_id],
                oppo_held_objects_history,
                valid_plan,
                current_pos=self.position,
                oppo_pos=self.agent_memory.oppo_pos,
                child_pos=self.agent_memory.third_agent_pos
            )
        else:
            assert False    # you can remove this line; it's just for debugging
            plan, plan_id, info = self.LLM.run(
                self.num_frames,
                self.obs["held_objects"],
                self.agent_memory.ignored_filter_object_info(),
                [self.agent_memory.object_info[x] for x in self.satisfied if x in self.agent_memory.object_info.keys()],
                self.object_list,
                self.agent_memory.action_history_dict[self.agent_id],
                self.agent_memory.status_history_dict[self.agent_id],
                None,
                None,
                oppo_held_objects_latest,
                valid_plan,
                current_pos=self.position,
            )

        if plan is None:
            return PlanStatus.wait, info
        else:
            parsed_plan = "explore"
            for k in plan_to_text_back.keys():
                if k in plan:
                    parsed_plan = k
                    break

            return plan_to_text_back[parsed_plan], plan_id, info
        
    def VLM_plan(self):        
        valid_plan = self.get_valid_plan()
        valid_plan = [plan_to_text[x] for x in valid_plan]
        oppo_held_objects_history = self.agent_memory.oppo_held_objects_history
        # if any(obj['type'] is not None for obj in oppo_held_objects_latest):
            # breakpoint()  #! for debugging only
        if len(self.agent_memory.action_history_dict.keys()) > 1:
            print(f"\n\n>>>> oppo action history: {self.agent_memory.action_history_dict[1 - self.agent_id]}\n")
            print(f">>>> oppo status history: {self.agent_memory.status_history_dict[1 - self.agent_id]}\n\n")
            plan, plan_id, info = self.VLM.run(
                self.num_frames,
                self.obs["held_objects"],
                self.agent_memory.ignored_filter_object_info(),
                [self.agent_memory.object_info[x] for x in self.satisfied if x in self.agent_memory.object_info.keys()],
                self.object_list,
                self.agent_memory.action_history_dict[self.agent_id],
                self.agent_memory.status_history_dict[self.agent_id],
                self.agent_memory.action_history_dict[1 - self.agent_id],
                self.agent_memory.status_history_dict[1 - self.agent_id],
                oppo_held_objects_history,
                valid_plan,
                current_pos=self.position,
                oppo_pos=self.agent_memory.oppo_pos,
                child_pos=self.agent_memory.third_agent_pos
            )
        else:
            assert False    # you can remove this line; it's just for debugging
            plan, plan_id, info = self.VLM.run(
                self.num_frames,
                self.obs["held_objects"],
                self.agent_memory.ignored_filter_object_info(),
                [self.agent_memory.object_info[x] for x in self.satisfied if x in self.agent_memory.object_info.keys()],
                self.object_list,
                self.agent_memory.action_history_dict[self.agent_id],
                self.agent_memory.status_history_dict[self.agent_id],
                None,
                None,
                oppo_held_objects_latest,
                valid_plan,
                current_pos=self.position,
            )

        if plan is None:
            return PlanStatus.wait, info
        else:
            parsed_plan = "explore"
            for k in plan_to_text_back.keys():
                if k in plan:
                    parsed_plan = k
                    break

            return plan_to_text_back[parsed_plan], plan_id, info

    def get_object_list(self, target_object_names = []):
        # Extract object info from agent memory
        object_list = {0: [], 1: [], 2: [], 4: []}  # target item, container, target place, obstacle
        object_per_room = {room: {0: [], 1: [], 2: [], 4: []} for room in self.rooms_name}
        for object_type in [0, 1, 2, 4]:
            obj_map_indices = np.where(self.agent_memory.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.agent_memory.id_map[i, j]
                if object_type == 0 and self.agent_memory.object_info[id]["name"] not in target_object_names:
                    if self.agent_id == 1:
                        print("ignore target:", self.agent_memory.object_info[id]["name"])
                    continue
                if (
                    id in self.satisfied
                    or id in self.holding_objects_id
                    or id in self.oppo_holding_objects_id
                    or self.agent_memory.object_info[id] in object_list[object_type]
                ):
                    continue
                object_list[object_type].append(self.agent_memory.object_info[id])
                room = self.env_api["belongs_to_which_room"](self.agent_memory.object_info[id]["position"])
                if room is None:
                    self.logger.warning(f"obj {self.agent_memory.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                object_per_room[room][object_type].append(self.agent_memory.object_info[id])
        return object_list, object_per_room

    def default_plan(self):
        if self.number_of_agents == 3 and self.agent_memory.oppo_this_step and self.agent_memory.third_agent_this_step and \
           self.l2_distance(self.agent_memory.oppo_pos, self.agent_memory.third_agent_pos) > AGENT_DIST_THRESHOLD:
           sub_goal = PlanStatus.follow_third_agent
        elif self.number_of_agents == 3 and self.agent_memory.oppo_pos is not None and self.agent_memory.third_agent_pos is not None and \
           self.num_frames - self.agent_memory.oppo_frame < 50 and self.num_frames - self.agent_memory.third_agent_frame < 50 and \
               self.l2_distance(self.agent_memory.oppo_pos, self.agent_memory.third_agent_pos) > AGENT_DIST_THRESHOLD_LARGE:
           sub_goal = PlanStatus.follow_third_agent
        elif self.number_of_agents == 3 and (self.num_frames - self.agent_memory.oppo_frame > 200 or self.num_frames - self.agent_memory.third_agent_frame > 200) and self.num_frames - self.last_turn_around > 200:
           sub_goal = PlanStatus.turn_around
        elif self.obs["held_objects"][0]["id"] is not None and self.obs["held_objects"][1]["id"] is not None:
            # Need to go to the drop zone
            if self.obs["held_objects"][0]["type"] + self.obs["held_objects"][1]["type"] == 1:
                # One container and one target
                if (
                    self.obs["held_objects"][0]["contained"][-1] is None
                    and self.obs["held_objects"][1]["contained"][-1] is None
                ):
                    # Have space in container, so that we can put the target into the container
                    sub_goal = PlanStatus.put_in
                else:
                    # We need to go to the drop zone since the container is full
                    sub_goal = PlanStatus.goto_drop_zone
            else:
                sub_goal = PlanStatus.goto_drop_zone
        elif self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT and self.have_target_with_character():
            # Limited time, need to go to the drop zone
            sub_goal = PlanStatus.goto_drop_zone
        elif (
            self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT * 1.5
            and self.have_target_with_character()
            and len(self.object_list[2]) == 0
        ):
            # Limited time, need to go to the drop zone, and we need to find the drop zone first.
            sub_goal = PlanStatus.goto_drop_zone
        elif (
            (self.obs["held_objects"][0]["id"] is None or self.obs["held_objects"][1]["id"] is None)
            and len(self.object_list[4]) > 0
            and self.agent_id == 1
        ):
            # Remove an obstacle
            sub_goal = PlanStatus.remove_obstacle
        elif (
            (self.obs["held_objects"][0]["type"] != 1 and self.obs["held_objects"][1]["type"] != 1)
            and len(self.object_list[1]) > 0
            and self.num_frames < self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT * 2
        ):
            # Grasp a container if we have enough time
            sub_goal = PlanStatus.goto_container
        else:
            # Grasp an object
            sub_goal = PlanStatus.goto_target
        # print(sub_goal, sub_goal.value, self.object_list[sub_goal.value])
        if (
            sub_goal in [PlanStatus.goto_container, PlanStatus.goto_target, PlanStatus.goto_drop_zone]
            and len(self.object_list[sub_goal.value]) == 0
        ):
            sub_goal = PlanStatus.explore
        return sub_goal, {}

        # if self.obs["held_objects"][0]['id'] is not None or self.obs["held_objects"][1]['id'] is not None:
        #     # Need to go to the drop zone
        #     sub_goal = PlanStatus.goto_drop_zone
        # elif self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT and self.have_target_with_character():
        #     # Limited time, need to go to the drop zone
        #     sub_goal = PlanStatus.goto_drop_zone
        # elif self.num_frames > self.max_frames - MOVE_TO_GOAL_PLACE_LIMIT * 1.5 and self.have_target_with_character() and len(self.object_list[2]) == 0:
        #     # Limited time, need to go to the drop zone, and we need to find the drop zone first.
        #     sub_goal = PlanStatus.goto_drop_zone
        # elif (self.obs["held_objects"][0]['id'] is None or self.obs["held_objects"][1]['id'] is None) and len(self.object_list[4]) > 0 and self.agent_id == 1:
        #     # Remove an obstacle
        #     sub_goal = PlanStatus.remove_obstacle
        # else:
        #     sub_goal = PlanStatus.goto_target
        # # print(sub_goal, sub_goal.value, self.object_list[sub_goal.value])

        # if sub_goal in [PlanStatus.goto_container, PlanStatus.goto_target, PlanStatus.goto_drop_zone] and len(self.object_list[sub_goal.value]) == 0:
        #     sub_goal = PlanStatus.explore
        # return sub_goal, {}

    def get_valid_plan(self):
        if self.number_of_agents == 3:
            valid_plan = [PlanStatus.explore, PlanStatus.follow_another_agent, PlanStatus.follow_third_agent, PlanStatus.turn_around]
        else:
            valid_plan = [PlanStatus.explore, PlanStatus.follow_another_agent, PlanStatus.turn_around]         
        if len(self.object_list[0]) > 0 and (
            self.obs["held_objects"][0]["id"] is None or self.obs["held_objects"][1]["id"] is None
        ):
            valid_plan.append(PlanStatus.goto_target)
        if len(self.object_list[1]) > 0 and (
            self.obs["held_objects"][0]["id"] is None or self.obs["held_objects"][1]["id"] is None
        ):
            if not (self.obs["held_objects"][0]["type"] == 1 or self.obs["held_objects"][1]["type"] == 1):
                valid_plan.append(PlanStatus.goto_container)
        if len(self.object_list[2]) > 0 and (
            self.obs["held_objects"][0]["id"] is not None or self.obs["held_objects"][1]["id"] is not None
        ):
            valid_plan.append(PlanStatus.goto_drop_zone)
        if (
            self.obs["held_objects"][0]["id"] is not None
            and self.obs["held_objects"][1]["id"] is not None
            and self.obs["held_objects"][0]["type"] + self.obs["held_objects"][1]["type"] == 1
            and self.obs["held_objects"][0]["contained"][-1] is None
            and self.obs["held_objects"][1]["contained"][-1] is None
        ):
            valid_plan.append(PlanStatus.put_in)
        if len(self.object_list[4]) > 0 and self.agent_id == 1:
            valid_plan.append(PlanStatus.remove_obstacle)
        return valid_plan

    def random_plan(self):
        valid_plan = self.get_valid_plan()
        sub_goal = random.choice(valid_plan)
        return sub_goal, {}

    def gopickup(self, target_id=None, target_type=None):
        assert target_id is not None or target_type is not None
        if self.target_id is None:
            min_distance = 100000
            if target_id is None:
                if self.prefer_target is not None and len(self.prefer_target) > 0 and self.plan_mode == "default":
                    perfer_id = []
                    for obj in self.object_list[target_type]:
                        if obj["name"] in self.prefer_target:
                            perfer_id.append(obj["id"])
                    if len(perfer_id) > 0:
                        for id in perfer_id:
                            distance = self.l2_distance(self.position, self.agent_memory.object_info[id]["position"])
                            if distance < min_distance:
                                min_distance = distance
                                self.target_id = id
                    else:
                        if self.plan_mode == "default":
                            for obj in self.object_list[target_type]:
                                distance = self.l2_distance(self.position, obj["position"])
                                if distance < min_distance:
                                    min_distance = distance
                                    self.target_id = obj["id"]
                        else:
                            self.target_id = self.object_list[target_type][
                                random.randint(1, len(self.object_list[target_type])) - 1
                            ]["id"]
                else:
                    if self.plan_mode == "default":
                        for obj in self.object_list[target_type]:
                            distance = self.l2_distance(self.position, obj["position"])
                            if distance < min_distance:
                                min_distance = distance
                                self.target_id = obj["id"]
                    else:
                        self.target_id = self.object_list[target_type][
                            random.randint(1, len(self.object_list[target_type])) - 1
                        ]["id"]
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
            self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
            self.plan = None
            return None

        if self.target_position is None:
            self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]["position"])

        if self.target_id not in self.agent_memory.object_info or self.target_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
            return None

        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if not self.reach_target_pos(self.target_position) or path_len > MINIMUM_PATH_LENGTH:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
                return None
            return action

        action = {
            "type": 3,
            "object": self.target_id,
            "arm": "left" if self.obs["held_objects"][0]["id"] is None else "right",
        }
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
        action, _ = self.agent_memory.move_to_pos(self.target_position, explore=True, nav_step = self.num_frames - self.plan_start_frame)
        if action is None:
            # Reach the target position
            self.plan = None
        return action

    def goputon(self, target_id=None, target_type=None):
        assert target_id is not None or target_type is not None
        if self.target_id is None:
            if target_id is None:
                self.target_id = self.object_list[target_type][
                    random.randint(1, len(self.object_list[target_type])) - 1
                ]["id"]
            else:
                self.target_id = target_id

        if self.target_position is None or self.agent_memory.object_info[self.target_id]['type'] == 2:
            # target place is large, move to there may change the position in different views.
            if self.target_id != -1:
                self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]["position"])

        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if (
            not self.reach_target_pos(self.target_position, threshold=2.0) \
            and not (self.agent_memory.object_info[self.target_id]['type'] == 2 and self.agent_memory.dist_to_goalplace() <= 1.25 and self.reach_target_pos(self.target_position, threshold=3.0)) \
            or path_len > MINIMUM_PATH_LENGTH
        ) and not self.drop_flag:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
                return None
            return action

        self.drop_flag = True
        if self.obs["held_objects"][0]["type"] is not None and self.obs["held_objects"][0]["type"] != 5:
            return {"type": 5, "arm": "left", "object": self.target_id}
        elif self.obs["held_objects"][1]["type"] is not None:
            return {"type": 5, "arm": "right", "object": self.target_id}
        else:
            self.plan = None
            self.logger.info(f"successful put_on!")
            return None

    def follow_another_agent(self):
        if self.agent_memory.oppo_pos is not None:
            self.target_position = self.agent_memory.oppo_pos
            follow_tag = True
            explore_tag = False
        else:
            follow_tag = False
            explore_tag = True
        if self.target_position is None:
            self.target_position = self.agent_memory.explore()
        if self.num_frames - self.plan_start_frame > MAX_FOLLOW_FRAMES:
            self.plan = None
            return None
        action, _ = self.agent_memory.move_to_pos(self.target_position, explore=explore_tag, follow=follow_tag, nav_step = self.num_frames - self.plan_start_frame)
        if action is None:
            self.plan = None
        return action

    def follow_third_agent(self):
        if self.agent_memory.third_agent_pos is not None:
            self.target_position = self.agent_memory.third_agent_pos
            follow_tag = True
        else:
            follow_tag = False
            
        if self.target_position is None:
            self.target_position = self.agent_memory.explore()
        if self.num_frames - self.plan_start_frame > MAX_FOLLOW_CHILD_FRAMES:
            self.plan = None
            return None
        action, _ = self.agent_memory.move_to_pos(self.target_position, explore=False, follow_third_agent = follow_tag)
        if action is None:
            self.plan = None
        return action
    
    def turn_around(self):
        if not self.turning_around:
            self.turning_around = True
            self.counter = 0

        if self.counter >= 24 or (self.num_frames - self.agent_memory.oppo_frame < 50 and self.num_frames - self.agent_memory.third_agent_frame < 50):
            self.turning_around = False
            self.plan = None
            self.last_turn_around = self.num_frames
            return None
        else:
            self.counter += 1
            return {"type": 1}

    def remove_obstacle(self):
        if self.target_id is None:
            min_distance = 100000
            for obj in self.object_list[4]:
                distance = self.l2_distance(self.position, obj["position"])
                if distance < min_distance:
                    min_distance = distance
                    self.target_id = obj["id"]
        if self.target_id in self.agent_memory.ignore_ids:
            # The target object is not here any more
            if self.debug:
                self.logger.debug(f"remove obstacle failed. object is not here any more!")
            self.plan = None
            return None

        if self.target_position is None:
            self.target_position = copy.deepcopy(self.agent_memory.object_info[self.target_id]["position"])

        if self.target_id not in self.agent_memory.object_info or self.target_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"remove obstacle failed. object is not here any more!")
            self.plan = None
            self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
            return None

        action, path_len = self.agent_memory.move_to_pos(self.target_position)
        if not self.reach_target_pos(self.target_position) or path_len > MINIMUM_PATH_LENGTH:
            if self.num_frames - self.plan_start_frame > MAX_MOVE_FRAMES:
                self.plan = None
                self.agent_memory.ignore_logic(current_frames=self.obs["current_frames"], ignore_ids=[self.target_id])
                return None
            return action

        action = {
            "type": 7,
            "object": self.target_id,
            "arm": "left" if self.obs["held_objects"][0]["id"] is None else "right",
        }
        return action

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o["type"] is not None and o["type"] < 4:
                visible_obj.append(o)
        return visible_obj

    def update_perfer_target(self):
        # Update prefer_target
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

    def get_smart_help_new_obs(self, obs):
        new_obs = {"agent_pos": [0] * 3, "agent_rot": [0] * 3, "agent_action": [0] * 3, "agent_status": [0], "agent_held": [0] * 10, 
                    "obj_id": [[0]] * 60, "obj_weight": [[0]] * 60, "obj_pos": [[0] * 3] * 60, "obj_height": [[0]] * 60}
        
        obj_num = 0
        self.agent_memory.obs = obs 
        for obj in obs["visible_objects"]:
            if obj["id"] is None:
                continue
            if obj["type"] == 3 and obj["id"] == 0:
                new_obs["agent_pos"] = list(map(float, list(obs["oppo_pos"])))
                new_obs["agent_rot"] = list(map(float, list(obs["oppo_forward"])))
                if obs["previous_action"][0] is not None:
                    action_sem = obs["previous_action"][0].split(" at frame")[0]
                    tp = action_sem.split(" ")[0]
                    if tp == "ongoing":
                        new_obs["agent_action"] = [1, 0, 0]
                    elif tp == "moving":
                        new_obs["agent_action"] = [2, 0, 0]
                    elif tp == "pick":
                        obj_name = action_sem.split("pick up ")[1].split(" <")[0]
                        hand = action_sem.split("with ")[-1].split(" hand")[0]
                        if "left" in hand:
                            hand = 0
                        else:
                            hand = 1

                        if obj_name in self.id_map:
                            new_obs["agent_action"] = [3, hand, self.id_map[obj_name]]
                        else:
                            new_obs["agent_action"] = [3, hand, 0]

                    elif tp == "put":
                        if action_sem == "put the object in the container":
                            new_obs["agent_action"] = [4, 0, 0]
                        else:
                            hand = action_sem.split("put the object in the ")[-1].split(" hand")[0]
                            if "left" in hand:
                                hand = 0
                            else:
                                hand = 1

                            new_obs["agent_action"] = [5, hand, 0]
                    elif tp == "wait":
                        new_obs["agent_action"] = [6, 0, 0]
                else:
                    new_obs["agent_action"] = [0, 0, 0]

                if obs["previous_status"][0] is not None:
                    if obs["previous_status"][0] == "ActionStatus.ongoing":
                        new_obs["agent_status"] = [0]
                    elif obs["previous_status"][0] == "ActionStatus.success":
                        new_obs["agent_status"] = [1]
                    else:
                        new_obs["agent_status"] = [2]
                else: 
                    new_obs["agent_status"] = [3]

                new_obs["agent_held"] = []
                for i in range(2):
                    if obs["oppo_held_objects"][i]["type"] is None:
                        new_obs["agent_held"].append(0)
                    else:
                        new_obs["agent_held"].append(obs["oppo_held_objects"][i]["type"] + 1)

                    name = obs["oppo_held_objects"][i]["name"]
                    if name in self.id_map:
                        new_obs["agent_held"].append(self.id_map[name])
                    else:
                        new_obs["agent_held"].append(0)
                    
                    if obs["oppo_held_objects"][i]["type"] == 1:
                        for j in range(min(3, len(obs["oppo_held_objects"][i]["contained_name"]))):
                            name2 = obs["oppo_held_objects"][i]["contained_name"][j]
                            if name2 is None:
                                new_obs["agent_held"].append(0)
                            elif name2 in self.id_map:
                                new_obs["agent_held"].append(self.id_map[name2])
                            else:
                                new_obs["agent_held"].append(0)
                    else:
                        new_obs["agent_held"].extend([0, 0, 0])
                    
                 
            elif obj["type"] != 3:
                if obj["name"] is None or obj["name"] not in self.id_map:
                    continue

                position, pc = self.agent_memory.cal_object_position(obj)
                if position is None:
                    continue

                new_obs["obj_id"][obj_num] = [self.id_map[obj["name"]]]
                
                if obj["name"] in self.mass_map:
                    new_obs["obj_weight"][obj_num] = [self.mass_map[obj["name"]]]
                else:
                    new_obs["obj_weight"][obj_num] = [10]
                
                new_obs["obj_pos"][obj_num] = list(map(float, position))
                new_obs["obj_height"][obj_num] = [float(position[1])]
                obj_num += 1

        return new_obs
    
    def update_smart_help_obs(self, obs):
        self.smart_help_obs.pop(0)
        new_obs = self.get_smart_help_new_obs(obs)
        self.smart_help_obs.append(new_obs)

    def get_latent_vector(self):
        agent_pos = torch.tensor([[self.smart_help_obs[i]["agent_pos"] for i in range(5)]]).to(self.device)
        agent_rot = torch.tensor([[self.smart_help_obs[i]["agent_rot"] for i in range(5)]]).to(self.device)
        agent_action = torch.tensor([[self.smart_help_obs[i]["agent_action"] for i in range(5)]]).to(self.device)
        agent_status = torch.tensor([[self.smart_help_obs[i]["agent_status"] for i in range(5)]]).to(self.device)
        agent_held = torch.tensor([[self.smart_help_obs[i]["agent_held"] for i in range(5)]]).to(self.device)
        obj_id = torch.tensor([[self.smart_help_obs[i]["obj_id"] for i in range(5)]]).to(self.device)
        obj_weight = torch.tensor([[self.smart_help_obs[i]["obj_weight"] for i in range(5)]]).to(self.device)
        obj_pos = torch.tensor([[self.smart_help_obs[i]["obj_pos"] for i in range(5)]]).to(self.device)
        obj_height = torch.tensor([[self.smart_help_obs[i]["obj_height"] for i in range(5)]]).to(self.device)

        goal_feature, tar_1_feature, constraint_feature = self.smart_help_oppo_model(agent_pos, agent_rot, agent_action, agent_status, agent_held, obj_id, obj_weight, obj_pos, obj_height)
        goal_feature = list(goal_feature.view(-1).cpu().detach().numpy())
        tar_1_feature = list(tar_1_feature.view(-1).cpu().detach().numpy())
        constraint_feature = list(constraint_feature.view(-1).cpu().detach().numpy())
        return goal_feature, tar_1_feature, constraint_feature
    
    def get_prediction(self):
        agent_pos = torch.tensor([[self.smart_help_obs[i]["agent_pos"] for i in range(5)]]).to(self.device)
        agent_rot = torch.tensor([[self.smart_help_obs[i]["agent_rot"] for i in range(5)]]).to(self.device)
        agent_action = torch.tensor([[self.smart_help_obs[i]["agent_action"] for i in range(5)]]).to(self.device)
        agent_status = torch.tensor([[self.smart_help_obs[i]["agent_status"] for i in range(5)]]).to(self.device)
        agent_held = torch.tensor([[self.smart_help_obs[i]["agent_held"] for i in range(5)]]).to(self.device)
        obj_id = torch.tensor([[self.smart_help_obs[i]["obj_id"] for i in range(5)]]).to(self.device)
        obj_weight = torch.tensor([[self.smart_help_obs[i]["obj_weight"] for i in range(5)]]).to(self.device)
        obj_pos = torch.tensor([[self.smart_help_obs[i]["obj_pos"] for i in range(5)]]).to(self.device)
        obj_height = torch.tensor([[self.smart_help_obs[i]["obj_height"] for i in range(5)]]).to(self.device)

        goal_predict, tar_index_1_predict, constraint_predict = self.smart_help_oppo_model(agent_pos, agent_rot, agent_action, agent_status, agent_held, obj_id, obj_weight, obj_pos, obj_height, ignore_classifiers = False)
        return goal_predict, tar_index_1_predict, constraint_predict

    def act(self, obs, rl_act = None):
        assert obs['status'] != 0, "Last action is still ongoing"
        self.obs = obs
        if self.plan_mode == "smart_help":
            self.update_smart_help_obs(obs)
            # goal_predict, tar_index_1_predict, constraint_predict = self.get_prediction()
            # goal_predict = goal_predict[0].argmax().item()
            # tar_index_1_predict = tar_index_1_predict[0].argmax().item()
            # with open("file.txt", "a") as f:
            #     f.write(f"{goal_predict}, {tar_index_1_predict}, {constraint_predict[0]}\n")

        ignore_obstacles = []
        ignore_ids = []
        if obs["valid"] == False:
            if self.last_action is not None and "object" in self.last_action:
                ignore_ids.append(self.last_action["object"])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 100, "invalid action for 100 times"

        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}")
        
        # need prefer target for agent 0
        self.update_perfer_target()

        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api["belongs_to_which_room"](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api["get_room_distance"](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != "all":
            self.rooms_explored[self.current_room] = "part"

        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        # for x in self.obs["visible_objects"]:
        #     if x["name"] is not None and (
        #         x["name"] == "tent" or x["name"] == "wood_shelving"
        #     ):
        #         if x["id"] is not None:
        #             self.with_character.append(x["id"])
        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        if not self.gt_mask:
            # use the detection model to get the mask
            self.obs["visible_objects"], self.obs["seg_mask"] = self.agent_memory.detect(self.obs["rgb"])

        # print(f"==> previous action: {self.obs['previous_action']}")
        
        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img = not self.no_save_img
        )
        
        # if self.agent_id == 1:
        #     with open("file.txt", "a") as f:
        #         f.write(f"{self.num_frames}, {self.agent_memory.oppo_pos}, {self.agent_memory.third_agent_pos}\n")


        if obs['status'] == 1:
            # Last action is successful
            # Add it to the picked_num, which is used to identify the remaining objects
            if self.last_action is not None and self.last_action['type'] == 3:
                self.picked_num[self.agent_memory.object_info[self.last_action['object']]['name']] = self.picked_num.get(self.last_action['object'], 0) + 1

        # now all the information is updated
        if self.plan_mode == "RL":
            if rl_act is None:
                return self.RL_observation()
            
        elif self.plan_mode == "smart_help":
            if rl_act is None:
                return self.smart_help_observation()

        print(self.agent_id, self.target_object_names)

        info = {
            "satisfied": self.satisfied,
            #    'object_list': self.object_list,
            "current_room": self.current_room,
            #    'visible_objects': self.filtered(self.obs['visible_objects']),
            "frame": self.num_frames,
            #    'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
            "last_action_status": self.obs["status"],
            "oppo_in_view": self.agent_memory.oppo_this_step,
        }
        self.planned = False

        action = None
        lm_times = 0
        while action is None:
            self.object_list, self.object_per_room = self.get_object_list(self.target_object_names)
            if self.plan is None:
                self.target_id = None
                self.target_position = None
                self.drop_flag = False
                self.planned = True
                if lm_times > 0 and self.plan_mode in ["LLM", "VLM"]:
                    print(plan)
                    print(f"{self.plan_mode} failed!")
                    print(self.agent_id)
                    import time
                    time.sleep(1010)
                if lm_times > 4:
                    plan = random.choice(self.get_valid_plan())
                    a_info = {}
                elif self.plan_mode == "default":
                    plan, a_info = self.default_plan()
                elif self.plan_mode == "random":
                    plan, a_info = self.random_plan()
                elif self.plan_mode == "follow":
                    plan = PlanStatus.follow_another_agent
                    a_info = {}
                elif self.plan_mode == "LLM":
                    plan, self.target_id, a_info = self.LLM_plan()
                    # print(plan, self.target_id, a_info)
                    if plan not in self.get_valid_plan():
                        plan = PlanStatus.explore
                        self.target_id = None
                    if plan in [PlanStatus.goto_target]:
                        if not (self.target_id in [item['id'] for item in self.object_list[0]]):
                            self.target_id = None
                    elif plan in [PlanStatus.goto_container]:
                        if not (self.target_id in [item['id'] for item in self.object_list[1]]):
                            self.target_id = None
                    elif plan in [PlanStatus.remove_obstacle]:
                        if not (self.target_id in [item['id'] for item in self.object_list[4]]):
                            self.target_id = None
                    else:
                        self.target_id = None
                elif self.plan_mode == "VLM":
                    plan, self.target_id, a_info = self.VLM_plan()
                    # print(plan, self.target_id, a_info)
                    if plan not in self.get_valid_plan():
                        plan = PlanStatus.explore
                        self.target_id = None
                    if plan in [PlanStatus.goto_target]:
                        if not (self.target_id in [item['id'] for item in self.object_list[0]]):
                            self.target_id = None
                    elif plan in [PlanStatus.goto_container]:
                        if not (self.target_id in [item['id'] for item in self.object_list[1]]):
                            self.target_id = None
                    elif plan in [PlanStatus.remove_obstacle]:
                        if not (self.target_id in [item['id'] for item in self.object_list[4]]):
                            self.target_id = None
                    else:
                        self.target_id = None
                elif self.plan_mode == "RL":
                    plan, self.target_id, a_info = self.RL_plan(rl_act)
                elif self.plan_mode == "smart_help":
                    plan, self.target_id, a_info = self.RL_plan(rl_act)
                else:
                    raise ValueError(f"unavailable plan mode {self.plan_mode}")

                assert plan is not None, "No more things to do!"
                self.plan = plan
                # TODO: add local action history
                # self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")
                a_info.update({"Frames": self.num_frames})
                info.update({f"{self.plan_mode}": a_info})
                if self.target_id:
                    info.update({"target_id": self.target_id})
                lm_times += 1
                self.plan_start_frame = copy.deepcopy(self.num_frames)
                print(self.plan, self.target_id, self.plan_start_frame)

            if self.plan_mode == "default":
                if self.number_of_agents == 3 and self.agent_memory.oppo_this_step and self.agent_memory.third_agent_this_step and \
                    self.l2_distance(self.agent_memory.oppo_pos, self.agent_memory.third_agent_pos) > AGENT_DIST_THRESHOLD:
                    self.plan = PlanStatus.follow_third_agent
                    self.plan_start_frame = copy.deepcopy(self.num_frames)

                elif self.number_of_agents == 3 and self.agent_memory.oppo_pos is not None and self.agent_memory.third_agent_pos is not None and \
                    self.num_frames - self.agent_memory.oppo_frame < 50 and self.num_frames - self.agent_memory.third_agent_frame < 50 and \
                        self.l2_distance(self.agent_memory.oppo_pos, self.agent_memory.third_agent_pos) > AGENT_DIST_THRESHOLD_LARGE:
                    self.plan = PlanStatus.follow_third_agent
                    self.plan_start_frame = copy.deepcopy(self.num_frames)

                elif self.number_of_agents == 3 and (self.num_frames - self.agent_memory.oppo_frame > 200 or self.num_frames - self.agent_memory.third_agent_frame > 200) and self.plan != PlanStatus.turn_around and self.plan != PlanStatus.follow_third_agent and self.num_frames - self.last_turn_around > 200:
                    self.plan = PlanStatus.turn_around
                    self.plan_start_frame = copy.deepcopy(self.num_frames)

            print('agent id', self.agent_id, 'plan', self.plan)

            if self.plan == PlanStatus.goto_target:
                action = self.gopickup(target_type=0)
            elif self.plan == PlanStatus.goto_container:
                action = self.gopickup(target_type=1)
            elif self.plan == PlanStatus.goto_drop_zone:
                action = self.goputon(target_type=2)
            elif self.plan == PlanStatus.put_in:
                action = self.putin()
            elif self.plan == PlanStatus.explore:
                action = self.goexplore()
            elif self.plan == PlanStatus.remove_obstacle:
                action = self.remove_obstacle()
            elif self.plan == PlanStatus.follow_another_agent:
                action = self.follow_another_agent()
            elif self.plan == PlanStatus.follow_third_agent:
                action = self.follow_third_agent()
            elif self.plan == PlanStatus.turn_around:
                action = self.turn_around()
            elif self.plan == PlanStatus.wait:
                action = {"type": 8, "delay": 10}
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        info.update({"action": str(action), "plan": self.plan.name if self.plan is not None else None})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action

        self.info_history.append(info)
        os.makedirs(os.path.join(self.output_dir, "info"), exist_ok=True)
        json.dump(self.info_history, open(os.path.join(self.output_dir, "info", str(self.agent_id)) + ".json", "w"))
        return action