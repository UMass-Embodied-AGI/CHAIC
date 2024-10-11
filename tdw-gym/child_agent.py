from LM_agent.LLM import LLM

import numpy as np
import random
import copy
from PIL import Image
from agent_memory import AgentMemory
from enum import Enum
from logging import Logger
import json
import os

HELPER_THRESHOLD = 2.5
MAIN_AGENT_THRESHOLD = 2
MAX_FOLLOW_FRAMES = 100

class PlanStatus(Enum):
    follow_main_agent = 0
    run_away = 1

class Condition(Enum):
    follow = 0
    run_away = 1
    goto_main_agent = 2

class ChildAgent:
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
    ):
        self.agent_names = ["Alice", "Bob"]
        self.max_frames = max_frames
        self.env_api = None
        self.agent_id = agent_id
        self.agent_type = 'plan_agent'
        self.logger = logger
        self.output_dir = output_dir
        self.last_action = None

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

        self.object_list = None
        self.object_per_room = None
        self.debug = debug
        self.plan_mode = plan_mode  # ["default", "random", "LLM"]
        self.plan_start_frame = None
        # Used in plan helper, recording which object the constraint agent is holding
        self.prefer_target = None
        self.gt_mask = gt_mask
        self.gt_behavior = gt_behavior
        self.no_save_img = no_save_img
        self.info = None
        self.constraint_type = None

    def reset(self, obs, target_object_names: dict = {}, agent_color = [0, 0, 0], output_dir = None, env_api = None, rooms_name = None, obstacle_names = [], info = None, constraint_type = None, seed: int = 0, force_ignore = []):
        self.info = info
        self.constraint_type = constraint_type
        self.agent_memory = AgentMemory(agent_id = self.agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=self.gt_mask, gt_behavior=self.gt_behavior, env_api=env_api, constraint_type = constraint_type, map_size = self.map_size, scene_bounds = self._scene_bounds)
        self.rng = np.random.RandomState(seed)
        if type(target_object_names) == dict:
            self.target_object_names = list(target_object_names.keys())
        elif type(target_object_names) == list:
            self.target_object_names = target_object_names
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.rooms_name = rooms_name
        self.room_distance = 0
        self.condition = Condition.follow
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
        self.planned = False
        self.run_away_frames = 0
        self.last_proc_frame = 0
        if self.agent_id == 1:
            self.prefer_target = []
        else:
            self.prefer_target = None
        self.obstacle_names = obstacle_names
        # New setting, for task with target number
        self.picked_num = dict()
        self.required_num = target_object_names

    @property
    def holding_objects_id(self):
        return [x["id"] for x in self.obs["held_objects"] if x["id"] is not None]

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
        if self.condition == Condition.follow:
            rnd_num = self.rng.randint(1, 101)
                
            if rnd_num <= 20:
                self.condition = Condition.run_away
                self.plan = PlanStatus.run_away
                return self.plan, {"condition": "run_away"}
            else:
                self.plan = PlanStatus.follow_main_agent
                return self.plan, {"condition": "follow"}
        elif self.condition == Condition.run_away:
            helper_agent_pos = self.obs["helper_agent_pos"]
            if self.l2_distance(self.position, helper_agent_pos) < HELPER_THRESHOLD:
                self.condition = Condition.goto_main_agent
                self.plan = PlanStatus.follow_main_agent
                return self.plan, {"condition": "follow"}
            else:
                self.plan = PlanStatus.run_away
                return self.plan, {"condition": "run_away"}
        else:
            main_agent_pos = self.obs["main_agent_pos"]
            if self.l2_distance(self.position, main_agent_pos) < 3:
                self.condition = Condition.follow
            
            self.plan = PlanStatus.follow_main_agent
            return self.plan, {"condition": "follow"}
        

    def run_away(self):
        if self.target_position is None:
            self.target_position = self.agent_memory.explore(random_prob = 1, run_away = True, main_agent_pos = self.obs["main_agent_pos"])

        action, _ = self.agent_memory.move_to_pos(self.target_position)
        if action is None:
            # Reach the target position
            self.plan = None
        return action

    def follow_main_agent(self):
        self.target_position = self.obs["main_agent_pos"]

        if self.num_frames - self.plan_start_frame > MAX_FOLLOW_FRAMES:
            self.plan = None
            return None
        
        action, _ = self.agent_memory.move_to_pos(self.target_position, explore=False, follow=False, follow_main_agent=True)
        if action is None:
            self.plan = None
        return action

    def act(self, obs):
        self.obs = obs   
        if self.condition == Condition.run_away and self.l2_distance(self.obs["main_agent_pos"], self.obs["agent"][:3]) > 3.5:
            self.run_away_frames += (self.num_frames - self.last_proc_frame)
        
        self.last_proc_frame = self.num_frames
        if obs['status'] == 0: 
            #  Last action is still ongoing
            return {'type': 'ongoing'}

          
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        ignore_obstacles = []
        ignore_ids = []

        temp_with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x["id"] is None:
                continue
            temp_with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_character.append(y)

        # for x in self.obs["visible_objects"]:
        #     if x["name"] is not None and (
        #         x["name"] == "white_shopping_bag" or x["name"] == "b01_tent" or x["name"] == "4ft_wood_shelving"
        #     ):
        #         if x["id"] is not None:
        #             temp_with_character.append(x["id"])

        for x in self.obs["oppo_held_objects"]:
            if x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        bike_id = self.obs["oppo_held_objects"][0]["id"]

        ignore_obstacles = temp_with_character + ignore_obstacles + [bike_id]
        ignore_ids = temp_with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img=not self.no_save_img
        )

        self.object_list, self.object_per_room = self.get_object_list(self.target_object_names)
        # open("file.txt", "w").close()
        # with open("file.txt", "a") as f:
        #     f.write(f"{self.obs['current_frames']} object_list: " + str(self.object_list) + "\n")

        # print("satisfied:", self.satisfied)

        info = {
            "frame": self.num_frames,
            "last_action_status": self.obs["status"],
            "oppo_in_view": self.agent_memory.oppo_this_step,
        }
        self.planned = False

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_id = None
                self.target_position = None
                self.drop_flag = False
                self.planned = True
                if self.plan_mode == "default":
                    plan, a_info = self.default_plan()
                else:
                    raise NotImplementedError
                
                self.plan = plan
               
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                self.plan_start_frame = copy.deepcopy(self.num_frames)
                print(self.plan, self.target_id, self.plan_start_frame)

            helper_agent_pos = self.obs["helper_agent_pos"]
            if self.condition == Condition.run_away and self.l2_distance(self.position, helper_agent_pos) < HELPER_THRESHOLD:
                self.condition = Condition.goto_main_agent
                self.plan = PlanStatus.follow_main_agent
                self.target_position = None
                self.plan_start_frame = copy.deepcopy(self.num_frames)
                print("change to follow main agent")

            print('agent id', self.agent_id, 'plan', self.plan)
            if self.plan == PlanStatus.follow_main_agent:
                action = self.follow_main_agent()
            elif self.plan == PlanStatus.run_away:
                action = self.run_away()
            else:
                raise ValueError(f"unavailable plan {self.plan}")
            self.object_list, self.object_per_room = self.get_object_list(self.target_object_names)

        info.update({"action": str(action), "plan": self.plan.name if self.plan is not None else None})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action

        self.info_history.append(info)
        os.makedirs(os.path.join(self.output_dir, "info"), exist_ok=True)
        json.dump(self.info_history, open(os.path.join(self.output_dir, "info", str(self.agent_id)) + ".json", "w"))
        return action