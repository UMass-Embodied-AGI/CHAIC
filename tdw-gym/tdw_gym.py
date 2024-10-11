import string
from typing import Optional
from functools import partial

import gym
from gym.core import Env
import numpy as np
import os
import time
import copy

from tdw.replicant.arm import Arm
from tdw.tdw_utils import TDWUtils

from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from transport_challenge_multi_agent.outdoor_transport_challenge import OutdoorTransportChallenge
from transport_challenge_multi_agent.agent_ability_info import RiderAbility
import transport_challenge_multi_agent.utils as utils
from collections import Counter
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from tdw.container_data.container_tag import ContainerTag
import json

MAX_OBJECT_IN_A_FRAME = 50

def quaternion_to_yaw(quaternion):
    y = 2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])
    x = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
    yaw = np.arctan2(y, x)
    
    return np.rad2deg(yaw) % 360

class TDW(Env):
    def __init__(self, port = 1071, number_of_agents = 1, rank=0, train=False, screen_size = 256, gt_mask = True, \
                exp = False, launch_build=True, gt_occupancy = False, gt_behavior = True, enable_collision_detection = False, \
                save_dir = 'results', max_frames = 2000, data_prefix = 'dataset/nips_dataset/', no_save_img = False, behaviour_data_gen = False, only_save_rgb = False, oracle = False):
        self.data_prefix = data_prefix
        self.oracle = oracle
        self.replicant_colors = None
        self.replicant_ids = None
        self.names_mapping = None
        self.rooms_name = None
        self.action_buffer = None
        self.goal_description = None
        self.occupancy_map = None
        self.satisfied = None
        self.count = 0
        self.constraint_agent_id = 0 # Notice! Must be 0! notice when writing params
        self.number_of_agents = number_of_agents
        self.seed = None
        self.num_step = 0
        self.reward = 0
        self.done = False
        self.exp = exp
        self.success = False
        self.num_frames = 0
        self.data_id = rank
        self.train = train
        self.port = port
        self.gt_occupancy = gt_occupancy
        self.screen_size = screen_size
        self.launch_build = launch_build
        self.enable_collision_detection = enable_collision_detection
        self.gt_mask = gt_mask or self.oracle
        self.gt_behavior = gt_behavior or self.oracle
        self.controller = None
        self.message_per_frame = 500
        self.record_step = None
        self.who_complete = {}
        self.max_frame = max_frames
        self.action_list = []
        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.fov = 90
        self.force_ignore = []
        self.save_dir = save_dir
        self.f = open(os.path.join(self.save_dir, "actions.log"), 'w')
        self.valid_action = None
        self.no_save_img = no_save_img
        self.only_save_rgb = only_save_rgb
        self.room_type_path = './dataset/room_types.json'
        self.names_mapping_path = './dataset/name_map.json'
        self.object_property_path = './dataset/object_property.json'
        self.behaviour_data_gen = behaviour_data_gen
        with open(self.room_type_path, 'r') as f: self.room_types = json.load(f)
        with open(self.names_mapping_path, 'r') as f: self.names_mapping = json.load(f)
        with open(self.object_property_path, 'r') as f: self.object_property = json.load(f)
        self.translate_constraint = {
            'high': 'a short child', \
            'low': 'a wheelchaired human', \
            'wheelchair': "a wheelchaired human", \
            'weight': 'a woman', \
            'riding': 'walking a bike with his left hand while accompanying his child', 
            None: 'a normal person'}

        print("Controller connected ...")
    
    def get_object_type(self, id):
        if id in self.controller.state.possible_target_object_ids:
            return 0
        elif id in self.controller.state.container_ids:
            return 1
        elif id == self.goal_position_id or (type(self.goal_position_id) == list and id in self.goal_position_id):
            return 2
        elif id in self.controller.state.obstacle_ids:
            return 4
        else:
            return 5

    def obs_filter(self, obs):
        new_obs = copy.deepcopy(obs)
        for agent in new_obs:
            if int(agent) == 0 or int(agent) == 2: continue
            if 'seg_mask' not in new_obs[agent]: continue
            if not self.gt_mask:
                # if no gt mask
                # only save agent mask here
                no_filter = np.zeros_like(new_obs[agent]['seg_mask'][:, :, 0])
                for color in self.replicant_colors.values():
                    no_filter = np.logical_or(no_filter, (obs[agent]['seg_mask'] == color).all(axis=2))

                new_obs[agent]['seg_mask'] = np.zeros_like(obs[agent]['seg_mask'])
                new_obs[agent]['seg_mask'][no_filter] = obs[agent]['seg_mask'][no_filter]
                new_obs[agent]['visible_objects'] = []
                for i in range(len(obs[agent]['visible_objects'])):
                    if obs[agent]['visible_objects'][i]['type'] == 3:
                        new_obs[agent]['visible_objects'].append(obs[agent]['visible_objects'][i])

                while len(new_obs[agent]['visible_objects']) < MAX_OBJECT_IN_A_FRAME:
                    new_obs[agent]['visible_objects'].append({
                        'id': None,
                        'type': None,
                        'seg_color': None,
                        'name': None,
                    })
            if not self.gt_behavior and agent != '2':
                new_obs[agent]['previous_action'][1 - int(agent)] = None
                new_obs[agent]['previous_status'][1 - int(agent)] = None
        return new_obs

    def get_id_from_mask(self, agent_id, mask, name = None):
        r'''
        Get the object id from the mask
        '''
        seg_with_mask = (self.obs[str(agent_id)]['seg_mask'] * np.expand_dims(mask, axis = -1)).reshape(-1, 3)
        seg_with_mask = [tuple(x) for x in seg_with_mask]
        seg_counter = Counter(seg_with_mask)
        
        for seg in seg_counter:
            if seg == (0, 0, 0): continue
            if seg_counter[seg] / np.sum(mask) > 0.5:
                for i in range(len(self.obs[str(agent_id)]['visible_objects'])):
                    if self.obs[str(agent_id)]['visible_objects'][i]['seg_color'] == seg:
                        return self.obs[str(agent_id)]['visible_objects'][i]
        return {
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                }

    def get_with_character_mask(self, agent_id, character_object_ids):
        color_set = [self.segmentation_colors[id] for id in character_object_ids if id in self.segmentation_colors] + [self.replicant_colors[id] for id in character_object_ids if id in self.replicant_colors]
        curr_with_seg = np.zeros_like(self.obs[str(agent_id)]['seg_mask'])
        curr_seg_flag = np.zeros((self.screen_size, self.screen_size), dtype = bool)
        for i in range(len(color_set)):
            color_pos = (self.obs[str(agent_id)]['seg_mask'] == np.array(color_set[i])).all(axis=2)
            curr_seg_flag = np.logical_or(curr_seg_flag, color_pos)
            curr_with_seg[color_pos] = color_set[i]
        return curr_with_seg, curr_seg_flag

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        reset the environment
        input:
            data_id: reset based on the data_id
        """
        # close the previous controller, reset the seed and counters
        if self.controller is not None:
            self.controller.communicate({"$type": "terminate"})
            self.controller.socket.close()
        self.seed = np.random.RandomState(seed)
        super().reset(seed=seed)
        self.success = False
        self.done = False
        self.valid_action = [True for _ in range(self.number_of_agents)]
        self.messages = [None for _ in range(self.number_of_agents)]
        self.reward = 0
        self.num_step = 0
        self.num_frames = 0
        self.containment_all = {}
        self.goal_position_id = None
        self.action_buffer = [[] for _ in range(self.number_of_agents)]
        self.previous_action_list = {_: [] for _ in range(self.number_of_agents)}
        self.previous_status_list = {_: [] for _ in range(self.number_of_agents)}
        self.last_satisfied = {str(_): [] for _ in range(self.number_of_agents)}
        self.delay_frame_count = [0 for _ in range(self.number_of_agents)]
        self.bike_contained = []
        self.bike_contained_name = []
        self.bike_agent_put_on = False

        # start a new controller
        scene_info = options
        self.satisfied = {}
        if output_dir is not None: self.save_dir = output_dir
        self.f = open(os.path.join(self.save_dir, "actions.log"), 'w')
        self.replicants_name = ['replicant_0', 'replicant_0']
        self.replicants_ability = ['helper', 'helper']
        self.can_grasp_low_place = [True, True]
        self.can_touch_high_place = [True, True]
        self.task = None

        if scene_info is not None:
            scene = scene_info['scene']
            layout = scene_info['layout']
            self.scene_info = scene_info
            if 'task' in scene_info:
                self.task = scene_info['task']
                task = scene_info['task']
                self.constraint_type = task['constraint_type']
                if self.constraint_type == "wheelchair":
                    self.replicants_ability[self.constraint_agent_id] = "wheelchair"
                    self.replicants_name[self.constraint_agent_id] = 'fireman'
                elif self.constraint_type == "low":
                    self.replicants_ability[self.constraint_agent_id] = "wheelchair"
                    self.replicants_name[self.constraint_agent_id] = 'fireman'
                elif self.constraint_type == "high":
                    self.replicants_ability[self.constraint_agent_id] = "girl"
                    self.replicants_name[self.constraint_agent_id] = 'girl_casual'
                elif self.constraint_type == "weight":
                    # self.replicants_ability[self.constraint_agent_id] = "woman"
                    # self.replicants_name[self.constraint_agent_id] = 'woman_casual'
                    if self.number_of_agents == 2:
                        self.replicants_name = ['woman_casual', 'man_casual']
                        self.replicants_ability = ['woman', 'helper']
                    else:
                        self.replicants_name = ['woman_casual', 'man_casual', 'replicant_0']
                        self.replicants_ability = ['woman', 'helper', 'helper']
                elif self.constraint_type == "riding":
                    # self.replicants_ability[self.constraint_agent_id] = "rider"
                    # self.replicants_name[1] = 'man_casual'
                    self.replicants_name = ['replicant_0', 'man_casual', 'girl_casual']
                    self.replicants_ability = ['rider', 'helper', 'girl']
                elif self.constraint_type == None:
                    pass
                else:
                    raise ValueError("Constraint type not supported!")
            else:
                raise ValueError("No task info assigned!")
        else: raise ValueError("No scene info assigned!")
        #self.reach_threshold = [1 if i == "girl_casual" else 2 for i in self.replicants_name]
        #currently it is done in the replicant init function.

        self.target_object_count = dict()
        # self.possible_target_object_count = dict()
        self.target_object_count_total = 0
        for it in self.task['goal_task']:
            self.target_object_count[self.names_mapping[it[0]]] = it[1]
            self.target_object_count_total += it[1]
        # for one_task in self.task['possible_task']:
        #     for it in one_task:
        #         self.possible_target_object_count[self.names_mapping[it[0]]] = max(self.possible_target_object_count.get(self.names_mapping[it[0]], 0), it[1])
        
        self.task["possible_object_names"] = []
        # self.possible_target_object_count = dict()
        for name in scene_info['possible_target_object']:
            assert name in self.names_mapping, f"{name} not in names_mapping"
            self.task["possible_object_names"].append(name)


        if self.task is not None and "outdoor" in self.task['task_kind']:
            self.controller = OutdoorTransportChallenge(port=self.port, check_version=True, launch_build=self.launch_build, 
                            screen_width=self.screen_size, screen_height=self.screen_size, \
                            image_frequency= ImageFrequency.always, png=True, image_passes=None, \
                            enable_collision_detection = self.enable_collision_detection, \
                            logger_dir = output_dir, replicants_name=self.replicants_name, replicants_ability = self.replicants_ability)
        
            self.controller._start_trial(scene=scene, layout=layout, replicants=self.number_of_agents,
                                   random_seed=seed, task_meta = task, data_prefix = self.data_prefix)

        else:
            self.controller = TransportChallenge(port=self.port, check_version=True, launch_build=self.launch_build, 
                            screen_width=self.screen_size, screen_height=self.screen_size, \
                            image_frequency= ImageFrequency.always, png=True, image_passes=None, \
                            enable_collision_detection = self.enable_collision_detection, \
                            logger_dir = output_dir, replicants_name=self.replicants_name, replicants_ability = self.replicants_ability)
            
            self.controller.start_floorplan_trial(scene=scene, layout=layout, replicants=self.number_of_agents,
                                   random_seed=seed, task_meta = task, data_prefix = self.data_prefix, object_property = self.object_property)
        
        for i in range(len(self.task["possible_object_names"])):
            self.task["possible_object_names"][i] = self.names_mapping[self.task["possible_object_names"][i]]

        if self.task is not None and self.task['task_kind'] == "outdoor_shopping" and self.number_of_agents >= 2:
            self.force_ignore = []
            for object_id in self.controller.object_manager.objects_static.keys():
                if self.controller.object_manager.objects_static[object_id].name == "4ft_wood_shelving" or self.controller.object_manager.objects_static[object_id].name == "b01_tent":
                    self.force_ignore.append(object_id)

        if self.task is not None and "outdoor_furniture" in self.task['task_kind']:
            self.target_object_weight_total = 0
            for k in self.controller.mass_map:
                self.target_object_weight_total += self.controller.mass_map[k]
        # Set the field of view of the agent.
        for replicant_id in self.controller.replicants:
            self.controller.communicate({"$type": "set_field_of_view",
                          "avatar_id" : str(replicant_id), "field_of_view" : self.fov})

        data = self.controller.communicate({"$type": "send_segmentation_colors",
                          "show": False,
                          "frequency": "once"})

        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.replicant_ids = [self.controller.replicants[i].static.replicant_id for i in range(self.number_of_agents)]
        
        object_kinematic_commend = []
        for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            if r_id == "segm":
                segm = SegmentationColors(data[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    self.segmentation_colors[object_id] = segm.get_object_color(j)
                    self.object_names[object_id] = segm.get_object_name(j).lower()
                    if self.object_names[object_id] in self.names_mapping:
                        self.object_names[object_id] = self.names_mapping[self.object_names[object_id]]

                    if object_id not in self.controller.state.possible_target_object_ids + self.controller.state.container_ids + self.controller.state.obstacle_ids and "outdoor" not in self.task['task_kind']:
                        object_kinematic_commend.append({"$type": "set_kinematic_state",
                                    "id": object_id,
                                    "is_kinematic": True,
                                    "use_gravity": True})

                    self.controller.state.object_drop_frame[object_id] = -1e9
                    self.object_categories[object_id] = segm.get_object_category(j)
                    if self.object_categories[object_id] in task['goal_position_names']:
                        if 'door' not in self.object_names[object_id] and not (task['task_kind'] == 'highgoalplace' and self.gety(object_id) < 1.5):
                            if self.goal_position_id is None:
                                self.goal_position_id = object_id
                            else:
                                if type(self.goal_position_id) == int:
                                    self.goal_position_id = [self.goal_position_id]
                                self.goal_position_id += [object_id]
        
        self.controller.communicate(object_kinematic_commend)

        #print self.object_names to files
        with open(os.path.join(self.save_dir, 'object_names.txt'), 'w') as f:
            for key in self.object_names.keys():
                f.write("%s, %s\n" % (key, self.object_names[key]))
        #print self.object_categories to files
        with open(os.path.join(self.save_dir, 'object_categories.txt'), 'w') as f:
            for key in self.object_categories.keys():
                f.write("%s, %s\n" % (key, self.object_categories[key]))
        
        self.replicant_colors ={i: self.controller.replicants[i].static.segmentation_color for i in range(self.number_of_agents)}
        # check colors are different:
        for x in self.segmentation_colors.keys():
            for y in self.segmentation_colors.keys():
                if x != y: assert (self.segmentation_colors[x] != self.segmentation_colors[y]).any()

        self.goal_description = dict(Counter([self.object_names[i] for i in self.controller.state.target_object_ids]))

        if self.task is not None and "outdoor" in self.task['task_kind']:
            self.rooms_name = {0: 'outdoor'}
            self.controller.rooms_name = self.rooms_name
            self.all_rooms = [self.rooms_name[i] for i in range(len(self.rooms_name)) if self.rooms_name[i] is not None]
        else:
            self.rooms_name = {}
            #<room_type> (id) for each room.
            if type(layout) == str: now_layout = int(layout[0])
            else: now_layout = layout
            for i, rooms_name in enumerate(self.room_types[scene[0]][now_layout]):
                if rooms_name not in ['Kitchen', 'Livingroom', 'Bedroom', 'Office']:
                    the_name = None
                else:
                    the_name = f'<{rooms_name}> ({1000 * (i + 1)})'
                self.rooms_name[i] = the_name
            self.controller.rooms_name = self.rooms_name
            self.all_rooms = [self.rooms_name[i] for i in range(len(self.rooms_name)) if self.rooms_name[i] is not None]

        if type(self.goal_position_id) == int:
            goal_position_name = self.object_names[self.goal_position_id]
            goal_place_id = self.goal_position_id
        else:
            goal_position_name = self.object_names[self.goal_position_id[0]]
            goal_place_id = self.goal_position_id[0]

        # if self.task["task_kind"] == "outdoor_shopping":
        #     for object_id in self.controller.object_manager.objects_static.keys():
        #         if self.controller.object_manager.objects_static[object_id].name == "white_shopping_bag":
        #             self.bike_contained.append(object_id)
        #             self.bike_contained_name.append("white_shopping_bag")

        bound = self.controller.object_manager.bounds[goal_place_id]
        self.extra_distance = 2 + max(np.linalg.norm(bound.right[[0, 2]] - bound.left[[0, 2]]), np.linalg.norm(bound.front[[0, 2]] - bound.back[[0, 2]])) / 2

        self.meta_info = {
            'target_object_names': self.target_object_count,
            # 'possible_task': self.task['possible_task'],
            'possible_target_object_names': self.task['possible_object_names'],
            'goal_position_names': self.task['goal_position_names'],
            'names_mapping': self.names_mapping,
            'goal_description': self.goal_description,
            'rooms_name': self.all_rooms,
            'agent_colors': self.replicant_colors,
            'goal_position_id': self.goal_position_id,
            'goal_position_name': goal_position_name,
            'constraint': self.translate_constraint[self.constraint_type],
            'obstacle_names': self.task.get('obstacle_names', []),
            # for new task of target number
            'target_object_count': self.target_object_count,
            #'possible_target_object_count': self.possible_target_object_count,
        }

        # print("mata info:", self.meta_info)
        # print("goal position id:", self.goal_position_id)
        # print("goal position", self.controller.object_manager.transforms[self.goal_position_id].position)
        self.env_api = [{
            'belongs_to_which_room': partial(utils.belongs_to_which_room, controller = self.controller),
            'center_of_room': partial(utils.center_of_room, controller = self.controller),
            'check_pos_in_room': partial(utils.check_pos_in_room, controller = self.controller),
            'get_room_distance': partial(utils.get_room_distance, controller = self.controller),
            'get_id_from_mask': partial(self.get_id_from_mask, agent_id=i),
            'get_with_character_mask': partial(self.get_with_character_mask, agent_id=i),
        } for i in range(self.number_of_agents)]

        # print("target object ids: ", self.controller.state.target_object_ids)
        # print("goal position id: ", self.goal_position_id)
        # print("container ids: ", self.controller.state.container_ids)
        # print("obstacle ids: ", self.controller.state.obstacle_ids)

        self.last_action = dict()
        for replicant_id in self.controller.replicants:
            self.last_action[replicant_id] = None

        self.obs = self.get_obs()
        return self.obs_filter(self.obs), self.meta_info, self.env_api
    
    def get_agent_api(self, agents):
        self.agents = agents
        if self.task is not None and "outdoor_furniture" in self.task['task_kind']:
            for i in range(len(self.agents)):
                self.agents[i].picking_furniture_together = False

    def check_grasp(self, object_id: int) -> bool:
        '''
            Check whether the object is grasped by the agent, 
            or in some container, and the container is grasped by the agent.
        '''
        if self.task['task_kind'] == "outdoor_shopping" and object_id in self.bike_contained:
            return True
        
        for replicant_id in self.controller.replicants:
            held_objects = list(self.controller.state.replicants[replicant_id].values())
            if object_id in held_objects:
                return True
            for container_id in held_objects:
                if container_id is None:
                    continue
                if container_id in self.containment_all.keys():
                    contained_obj = [x for x in self.containment_all[container_id] if x not in held_objects and x in self.controller.state.target_object_ids]
                    if object_id in contained_obj:
                        return True
        return False

    def check_goal(self):
        r'''
            Check whether the goal is achieved
            return: 
                count: the total number of sshoppinghieved
        '''
        if type(self.goal_position_id) == int:
            place_poses = [self.controller.object_manager.bounds[self.goal_position_id].top]
            goal_place_id = self.goal_position_id
        else:
            # all the pos are ok
            place_poses = [self.controller.object_manager.bounds[i].top for i in self.goal_position_id]
            goal_place_id = self.goal_position_id[0]

        count_all = 0
        count_individial = dict()
        for object_id in self.controller.state.possible_target_object_ids + self.controller.state.container_ids:
            if object_id in self.satisfied.keys():
                pass
            else:
                pos = self.controller.object_manager.transforms[object_id].position
                #Satisfied: if the object and the bound of the goal position is close enough, and the object is above the goal position.
                for place_pos in place_poses:
                    if utils.get_2d_distance(pos, place_pos) < self.extra_distance \
                    and (pos[1] > 1.5 or self.task['task_kind'] != 'highgoalplace') \
                    and (not self.check_grasp(object_id)) \
                    and self.num_frames - self.controller.state.object_drop_frame[object_id] <= 200:
                        # from drop to 200 frames is the time for the object to be regarded as satisfied.
                        self.satisfied[object_id] = True
                        # Make the object static.
                        self.controller.communicate({"$type": "set_kinematic_state",
                                    "id": object_id,
                                    "is_kinematic": True,
                                    "use_gravity": False})
                        break

            if object_id in self.satisfied.keys() and object_id in self.controller.state.target_object_ids:
                count_individial[self.object_names[object_id]] = count_individial.get(self.object_names[object_id], 0) + 1
                if self.target_object_count[self.object_names[object_id]] >= count_individial[self.object_names[object_id]]:
                    if self.task is not None and "outdoor_furniture" in self.task['task_kind']:
                        count_all += self.controller.mass_map[str(object_id)]
                    else:
                        count_all += 1

        for object_id in self.controller.state.obstacle_ids:
            if object_id in self.satisfied.keys():
                continue
            if self.check_grasp(object_id):
                self.satisfied[object_id] = True
                #self.controller.communicate({"$type": "destroy_object", "id": object_id})

        #WARNING: Hard code for high goal place since the detection may not be accurate
        for i in range(min(self.number_of_agents, 2)):
            with_character_id = []
            for action, status in zip(self.previous_action_list[i], self.previous_status_list[i]):
                if 'pick up' in action and 'success' in status:
                    with_character_id.append(int(action.split('<')[-1].split('>')[0]))
                if ('put the object in the right hand on' in action or 'put the object in the left hand on' in action) and 'success' in status:
                    for object_id in with_character_id:
                        if object_id in self.satisfied.keys() or object_id in self._with_agent(i):
                            # Already satisfied, or not put on yet.
                            continue
                        self.satisfied[object_id] = True
                        count_individial[self.object_names[object_id]] = count_individial.get(self.object_names[object_id], 0) + 1
                        if self.object_names[object_id] in self.target_object_count and self.target_object_count[self.object_names[object_id]] >= count_individial[self.object_names[object_id]]:
                            count_all += 1
                        # self.controller.communicate({"$type": "destroy_object", "id": object_id})
                        self.controller.communicate({"$type": "set_kinematic_state",
                                    "id": object_id,
                                    "is_kinematic": True,
                                    "use_gravity": False})
                    with_character_id = []

        if self.task is not None and "outdoor_furniture" in self.task['task_kind']:
            return count_all, self.target_object_weight_total, count_all == self.target_object_weight_total
        else:
            return count_all, self.target_object_count_total, count_all == self.target_object_count_total
    
    def get_env_semantic_description(self):
        l = []
        for container_shape_id in self.controller.container_manager.events:
            event = self.controller.container_manager.events[container_shape_id]
            l.append((event.container_id, event.object_ids, event.tag))
        return l
    
    def debug_get_env_semantic_text_description(self, target_list):
        def get_object_text(object_id):
            return "<" + self.object_categories[object_id] + "> (" + str(object_id) + ")"
        env_description = self.get_env_semantic_description()
        description = []
        ttl = [i for i in target_list if i in self.object_names.keys()]
        for ijk in env_description:
            container_id, object_ids, tag = ijk
            object_ids = [object_id for object_id in object_ids if object_id in ttl]
            if len(object_ids) == 0:
                continue
            if self.controller.container_manager.container_shapes[container_id] not in ttl:
                continue
            tag_description = {ContainerTag.inside: "inside", ContainerTag.on: "on", ContainerTag.enclosed: "inside"}[tag]
            description.append(f"{','.join([get_object_text(object_id) for object_id in object_ids])} is {tag_description} {get_object_text(self.controller.container_manager.container_shapes[container_id])}.")
        description = " ".join(description)
        return description

    def get_env_semantic_text_description(self):
        def get_object_text(object_id):
            return "<" + self.object_categories[object_id] + "> (" + str(object_id) + ")"
        env_description = self.get_env_semantic_description()
        description = []
        for ijk in env_description:
            container_id, object_ids, tag = ijk
            object_ids = [object_id for object_id in object_ids if object_id in self.object_names.keys()]
            if len(object_ids) == 0:
                continue
            if self.controller.container_manager.container_shapes[container_id] not in self.object_names.keys():
                continue
            tag_description = {ContainerTag.inside: "inside", ContainerTag.on: "on", ContainerTag.enclosed: "inside"}[tag]
            description.append(f"{','.join([get_object_text(object_id) for object_id in object_ids])} is {tag_description} {get_object_text(self.controller.container_manager.container_shapes[container_id])}.")
        description = " ".join(description)
        description += "\nAnd here is all of the object in the room and their location [x, y, z]. y means the height of the top of the object (meters): "
        description += ", ".join([get_object_text(i) for i in self.object_names.keys()])
        # description += "\nAnd here is all of the object in the room and their location [x, y, z]. y means the height of the top of the object (meters): "
        # description += ", ".join([get_object_text(i) + ' ' + np.array2string(self.controller.object_manager.bounds[i].top, precision=2, separator=',') for i in self.object_names.keys()])
        return description

    def action_mappping(self, action: dict):
        r'''
            change actions into text description
        '''
        if action['type'] == 'ongoing':
            return "ongoing"
        elif action['type'] in [0, 1, 2]:
            return "moving"
        elif action['type'] == 3:
            return f"pick up {self.object_names[action['object']]} <{action['object']}> with {action['arm']} hand"
        elif action['type'] == 4:
            return f"put the object in the container"
        elif action['type'] == 5:
            if action['object'] in [-1, 'ground']:
                return f"put the object in the {action['arm']} hand on ground"
            else:
                return f"put the object in the {action['arm']} hand on {self.object_names[action['object']]} <{action['object']}>"
        elif action['type'] == 6:
            return "send message"
        elif action['type'] == 7:
            return f"remove obstacle {self.object_names[action['object']]} <{action['object']}> with {action['arm']} hand"
        elif action['type'] == 8:
            return f"wait {action['delay']} frames"
        else:
            raise NotImplementedError

    def _with_agent(self, agent_id):
        held_objects = list(self.controller.state.replicants[agent_id].values())
        object_list = []
        for hand in range(2):
            if held_objects[hand] is None: continue
            object_list.append(held_objects[hand])
            if held_objects[hand] in self.containment_all.keys():
                contained_obj = [x for x in self.containment_all[held_objects[hand]] if x not in held_objects and x in self.controller.state.possible_target_object_ids and x not in self.satisfied]
                object_list += contained_obj
        if self.bike_contained is not None:
            object_list += self.bike_contained
        return object_list

    def get_obs(self):
        # Update containment:
        for x in self.controller.state.containment.keys():
            if x not in self.containment_all.keys():
                self.containment_all[x] = []
            for y in self.controller.state.containment[x]:
                if y not in self.containment_all[x]:
                    if self.number_of_agents == 1 or not (y in self._with_agent(0) and y in self._with_agent(1)):
                        # when two agent collides, the containment will be updated twice, so we need to check whether the object is in both agent's hand
                        self.containment_all[x].append(y)
        obs = {str(i): {} for i in range(self.number_of_agents)}
        containment_info_get = {str(i): [str(i)] for i in range(self.number_of_agents)}
        for replicant_id in self.controller.replicants:
            id = str(replicant_id)
            # If ongoing, do not get new observation, and only return status.
            goal_place_visible_or_nearby = False
            oppo_visible = False
            obs[id]['status'] = utils.map_status(self.controller.replicants[replicant_id].action.status, len(self.action_buffer[replicant_id]))
            if obs[id]['status'] == ActionStatus.ongoing:
                continue
            obs[id]['semantic_text'] = self.get_env_semantic_text_description()
            obs[id]['visible_objects'] = []

            # Visible objects
            if 'img' in self.controller.replicants[replicant_id].dynamic.images.keys():
                obs[id]['rgb'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('img'))
                obs[id]['seg_mask'] = np.array(self.controller.replicants[replicant_id].dynamic.get_pil_image('id'))
                colors = Counter(self.controller.replicants[replicant_id].dynamic.get_pil_image('id').getdata())
                for object_id in self.segmentation_colors:
                    segmentation_color = tuple(self.segmentation_colors[object_id])
                    object_name = self.object_names[object_id]
                    if segmentation_color in colors:
                        obs[id]['visible_objects'].append({
                            'id': object_id,
                            'type': self.get_object_type(object_id),
                            'seg_color': segmentation_color,
                            'name': object_name,
                        })
                        if self.get_object_type(object_id) == 2:
                            goal_place_visible_or_nearby = True

                for agent_id in self.replicant_colors:
                    if tuple(self.replicant_colors[agent_id]) in colors:
                        obs[id]['visible_objects'].append({
                            'id': agent_id,
                            'type': 3,
                            'seg_color': tuple(self.replicant_colors[agent_id]),
                            'name': 'agent',
                        })
                        if agent_id != replicant_id:
                            oppo_visible = True
                        if str(agent_id) not in containment_info_get[id]: containment_info_get[id].append(str(agent_id))
                        
                obs[id]['depth'] = np.array(TDWUtils.get_depth_values(self.controller.replicants[replicant_id].dynamic.get_pil_image('depth'),
                        width = self.screen_size,
                        height = self.screen_size))
                obs[id]['camera_matrix'] = np.array(self.controller.replicants[replicant_id].dynamic.camera_matrix).reshape((4, 4))
            else:
                assert -1, "No image received"
            while len(obs[id]['visible_objects']) < 50:
                obs[id]['visible_objects'].append({
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                })

            # Update goal progress
            if type(self.goal_position_id) == int:
                goal_place_poses = [self.controller.object_manager.bounds[self.goal_position_id].top]
            else:
                # all the pos are ok
                goal_place_poses = [self.controller.object_manager.bounds[i].top for i in self.goal_position_id]

            for pos in goal_place_poses:
                if utils.get_2d_distance(pos, self.controller.replicants[replicant_id].dynamic.transform.position) < self.extra_distance:
                    goal_place_visible_or_nearby = True

            if goal_place_visible_or_nearby or (self.oracle and replicant_id == 1):
                obs[id]['satisfied'] = list(self.satisfied.keys())
                self.last_satisfied[id] = list(self.satisfied.keys())
            else:
                obs[id]['satisfied'] = self.last_satisfied[id]
                # agent always knows the obstacle status
                for object_id in self.satisfied.keys():
                    if self.get_object_type(object_id) == 4:
                        obs[id]['satisfied'].append(object_id)
                obs[id]['satisfied'] = list(set(obs[id]['satisfied']))
            
            # Position of the agent
            x, y, z = self.controller.replicants[replicant_id].dynamic.transform.position
            fx, fy, fz = self.controller.replicants[replicant_id].dynamic.transform.forward
            obs[id]['agent'] = [x, y, z, fx, fy, fz]

            # 'oppo_pos' is for action recognition
            # 'oppo pos' and 'oppo forward' is for smart-help
            # TODO: the 'oppo_pos' and 'oppo_forward' should be removed when oppo is not visible
            # TODO: can we remove all of them?
            if self.number_of_agents >= 2 and replicant_id <= 1:
                obs[id]['oppo_pos'] = self.controller.replicants[1 - replicant_id].dynamic.transform.position
                obs[id]['oppo_forward'] = self.controller.replicants[1 - replicant_id].dynamic.transform.forward
                if self.number_of_agents == 3:
                    obs[id]['third_agent_pos'] = self.controller.replicants[2].dynamic.transform.position
                    obs[id]['third_agent_forward'] = self.controller.replicants[2].dynamic.transform.forward
            else:
                obs[id]['oppo_pos'] = None
                obs[id]['oppo_forward'] = None
                obs[id]['third_agent_pos'] = None
                obs[id]['third_agent_forward'] = None

            if id == '2':
                obs[id]['main_agent_pos'] = self.controller.replicants[0].dynamic.transform.position
                obs[id]['helper_agent_pos'] = self.controller.replicants[1].dynamic.transform.position
            
            # Update the character's object information
            held_objects = list(self.controller.state.replicants[replicant_id].values())
            obs[id]['held_objects'] = []
            for hand in range(2):
                if held_objects[hand] is None:
                    obs[id]['held_objects'].append({
                        'id': None,
                        'type': None,
                        'name': None,
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                elif self.get_object_type(held_objects[hand]) == 0:
                    obs[id]['held_objects'].append({
                        'id': held_objects[hand],
                        'type': 0,
                        'name': self.object_names[held_objects[hand]],
                        'contained': [None, None, None],
                        'contained_name': [None, None, None],
                    })
                else:
                    if held_objects[hand] in self.containment_all.keys():
                        contained_obj = [x for x in self.containment_all[held_objects[hand]] if x not in held_objects and x in self.controller.state.possible_target_object_ids and x not in self.satisfied]
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': contained_obj + [None] * (3 - len(contained_obj)),
                            'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                        })
                    else:
                        obs[id]['held_objects'].append({
                            'id': held_objects[hand],
                            'type': 1,
                            'name': self.object_names[held_objects[hand]],
                            'contained': [None] * 3,
                            'contained_name': [None] * 3,
                        })
            
            if id == '0' and self.task['task_kind'] == "outdoor_shopping":
                obs[id]['held_objects'][0] = {
                    'id': self.controller.bike_id,
                    'type': 5,
                    'name': self.object_names[self.controller.bike_id],
                    'contained': self.bike_contained + [None] * (3 - len(self.bike_contained)),
                    'contained_name': self.bike_contained_name + [None] * (3 - len(self.bike_contained_name)),
                }

            # Update the character's object information of the other agent
            # Only if the other agent is visible can the agent know the information
            if id != '2':
                if oppo_visible or (self.oracle and replicant_id == 1):
                    oppo_held_objects = list(self.controller.state.replicants[1 - replicant_id].values())
                else:
                    oppo_held_objects = [None, None]
                obs[id]['oppo_held_objects'] = []
                for hand in range(2):
                    if oppo_held_objects[hand] is None:
                        obs[id]['oppo_held_objects'].append({
                            'id': None,
                            'type': None,
                            'name': None,
                            'contained': [None, None, None],
                            'contained_name': [None, None, None]
                        })
                    elif self.get_object_type(oppo_held_objects[hand]) == 0:
                        obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[hand],
                            'type': 0,
                            'name': self.object_names[oppo_held_objects[hand]],
                            'contained': [None, None, None],
                            'contained_name': [None, None, None],
                        })
                    else:
                        if oppo_held_objects[hand] in self.containment_all.keys(): 
                            contained_obj = [x for x in self.containment_all[oppo_held_objects[hand]] if x not in oppo_held_objects and x in self.controller.state.possible_target_object_ids and x not in self.satisfied]
                            obs[id]['oppo_held_objects'].append({
                                'id': oppo_held_objects[hand],
                                'type': 1,
                                'name': self.object_names[oppo_held_objects[hand]],
                                'contained': contained_obj + [None] * (3 - len(contained_obj)),
                                'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                            })
                        else:
                            obs[id]['oppo_held_objects'].append({
                                'id': oppo_held_objects[hand],
                                'type': 1,
                                'name': self.object_names[oppo_held_objects[hand]],
                                'contained': [None] * 3,
                                'contained_name': [None] * 3,
                            })

                if id == '1' and self.task['task_kind'] == "outdoor_shopping":
                    obs[id]['oppo_held_objects'][0] = {
                        'id': self.controller.bike_id,
                        'type': 5,
                        'name': self.object_names[self.controller.bike_id],
                        'contained': self.bike_contained + [None] * (3 - len(self.bike_contained)),
                        'contained_name': self.bike_contained_name + [None] * (3 - len(self.bike_contained_name)),
                    }
            
            else:
                oppo_held_objects = list(self.controller.state.replicants[0].values())
                oppo_held_objects += list(self.controller.state.replicants[1].values())
                obs[id]['oppo_held_objects'] = []
                for i in range(4):
                    if oppo_held_objects[i] is None:
                        obs[id]['oppo_held_objects'].append({
                            'id': None,
                            'type': None,
                            'name': None,
                            'contained': [None, None, None],
                            'contained_name': [None, None, None]
                        })
                    elif self.get_object_type(oppo_held_objects[i]) == 0:
                        obs[id]['oppo_held_objects'].append({
                            'id': oppo_held_objects[i],
                            'type': 0,
                            'name': self.object_names[oppo_held_objects[i]],
                            'contained': [None, None, None],
                            'contained_name': [None, None, None],
                        })
                    else:
                        if oppo_held_objects[i] in self.containment_all.keys(): 
                            contained_obj = [x for x in self.containment_all[oppo_held_objects[i]] if x not in oppo_held_objects and x in self.controller.state.possible_target_object_ids and x not in self.satisfied]
                            obs[id]['oppo_held_objects'].append({
                                'id': oppo_held_objects[i],
                                'type': 1,
                                'name': self.object_names[oppo_held_objects[i]],
                                'contained': contained_obj + [None] * (3 - len(contained_obj)),
                                'contained_name': [self.object_names[object_id] for object_id in contained_obj] + [None] * (3 - len(contained_obj)),
                            })
                        else:
                            obs[id]['oppo_held_objects'].append({
                                'id': oppo_held_objects[i],
                                'type': 1,
                                'name': self.object_names[oppo_held_objects[i]],
                                'contained': [None] * 3,
                                'contained_name': [None] * 3,
                            })

                if self.task['task_kind'] == "outdoor_shopping":
                    obs[id]['oppo_held_objects'][0] = {
                        'id': self.controller.bike_id,
                        'type': 5,
                        'name': self.object_names[self.controller.bike_id],
                        'contained': self.bike_contained + [None] * (3 - len(self.bike_contained)),
                        'contained_name': self.bike_contained_name + [None] * (3 - len(self.bike_contained_name)),
                    }

            obs[id]['FOV'] = self.fov
            # The agent cannot communicate with the other agent
            # obs[id]['messages'] = [None, None]
            obs[id]['valid'] = self.valid_action[int(id)]
            obs[id]['current_frames'] = self.num_frames
            # In the newset setting, the agent does not know each other's previous actions and status history
            # In gt.behavior mode, the agent knows the previous actions and status of the other agent (no history)
            if id != '2':
                # id == 2 means the child agent
                obs[id]['previous_action'] = dict()
                obs[id]['previous_status'] = dict()
                for agent_action_id in self.previous_action_list:
                    if (agent_action_id != replicant_id and not oppo_visible) or len(self.previous_action_list[agent_action_id]) == 0:
                        assert self.num_frames < 10 or not oppo_visible
                        obs[id]['previous_action'][agent_action_id] = None
                        obs[id]['previous_status'][agent_action_id] = None
                    else:
                        index = -1
                        while ("moving" in self.previous_action_list[agent_action_id][index] or "wait" in self.previous_action_list[agent_action_id][index]) and index > -len(self.previous_action_list[agent_action_id]):
                            index -= 1
                        obs[id]['previous_action'][agent_action_id] = self.previous_action_list[agent_action_id][index]
                        obs[id]['previous_status'][agent_action_id] = self.previous_status_list[agent_action_id][index]
                    # print("previous action: ", obs[id]['previous_action'][agent_action_id])
                    # print("previous status: ", obs[id]['previous_status'][agent_action_id])
        return obs
    
    def get_info(self):
        # Add info needed
        return {}

    def add_name(self, inst):
        if type(inst) == int and inst in self.object_names:
            return f'{inst}_{self.object_names[inst]}'
        else:
            if type(inst) == dict:
                return {self.add_name(key): self.add_name(value) for key, value in inst.items()}
            elif type(inst) == list:
                return [self.add_name(item) for item in inst]
            else: raise NotImplementedError
    
    def add_name_and_empty(self, inst):
        for x in self.controller.state.container_ids:
            if x not in inst:
                inst[x] = []
        return self.add_name(inst)

    def gety(self, object_id):
        return self.controller.object_manager.bounds[object_id].top[1]
    
    def is_obstacle_between(self, room1, room2):
        #TODO: rewrite this func, all auxilary functions are in utils.py
        # this could be very dirty since one object could be near the wall, but not between rooms. So it's still better to use occ map?

        # check all objects. If there is a object its position is near both room, return True
        for object_id in self.controller.object_manager.bounds:
            # if object is in one room and distance to the other room is less than 1.5m, return True
            pos = self.controller.object_manager.bounds[object_id].bottom
            if self.belongs_to_which_room(pos) == room1:
                if self.get_room_distance_certain(pos, room2) < 0.3 and pos[1] < 0.2:
                    # breakpoint()
                    with open('logobstacles.txt', 'a') as f:
                        f.write(f'object {self.object_names[object_id]} is near both room {room1} and room {room2}\n')
                    if self.object_names[object_id] != 'bag':
                        # for debug
                        breakpoint()
                    return True
            elif self.belongs_to_which_room(pos) == room2:
                if self.get_room_distance_certain(pos, room1) < 0.3 and pos[1] < 0.2:
                    # breakpoint()
                    with open('logobstacles.txt', 'a') as f:
                        f.write(f'object {self.object_names[object_id]} is near both room {room1} and room {room2}\n')
                    if self.object_names[object_id] != 'bag':
                        # for debug
                        breakpoint()
                    return True
        return False
    
    def do_actions(self, sync = False):
        finish = False
        num_frames = 0
        while not finish:
            # continue until any agent's action finishes
            finish = True
            # print("0 status: ", self.controller.replicants[0].action.status)
            # print("1 status: ", self.controller.replicants[1].action.status)
            for replicant_id in self.controller.replicants:    
                if self.controller.replicants[replicant_id].action != None and self.controller.replicants[replicant_id].action.status == ActionStatus.ongoing:
                    finish = False

        #    for replicant_id in self.controller.replicants:
        #        if self.controller.replicants[replicant_id].action != None:
        #            print(replicant_id, self.controller.replicants[replicant_id].action.status)
            
            if finish: break
            if sync:
                data = self.controller.replicant_furniture_sychronize()
            else:
                data = self.controller.communicate([])

            if self.no_save_img == False:
                for i in range(len(data) - 1):
                    r_id = OutputData.get_data_type_id(data[i])
                    if r_id == 'imag' and self.only_save_rgb == False:
                        images = Images(data[i])
                        if images.get_avatar_id() == "a" and (self.num_frames + num_frames) % 1 == 0:
                            TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
                        if images.get_avatar_id() == "teaser" and (self.num_frames + num_frames) % 1 == 0:
                            TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'teaser_image'))
                    
                utils.save_images(os.path.join(self.save_dir, "Images"), controller=self.controller, screen_size=self.screen_size, step = num_frames + self.num_frames, only_save_rgb = self.only_save_rgb)
                num_frames += 1

        self.num_frames += num_frames
        return num_frames
    
    def navigate_both(self, furniture_id):
        for replicant_id in self.controller.replicants:
            # self.controller.replicants[replicant_id].collision_detection.exclude_objects.append(furniture_id)
            self.controller.replicants[replicant_id].collision_detection.exclude_objects.append(1 - replicant_id)

        target_yaw = [270, 90]
        num_frames = 0
        while True:
            actions = [None, None]
            obs = self.get_obs()
            for i in range(2):
                self.agents[i].obs = obs[str(i)]
                self.agents[i].agent_memory.update(obs[str(i)])
                if i == 0:
                    target_pos = self.controller.object_manager.bounds[furniture_id].left
                else:
                    target_pos = self.controller.object_manager.bounds[furniture_id].right

                action, path_len = self.agents[i].agent_memory.move_to_pos(target_pos)
                if not self.agents[i].reach_target_pos(target_pos, threshold = 0.4):
                    actions[i] = action
                    
                # else:
                #     yaw = quaternion_to_yaw(self.controller.replicants[i].dynamic.transform.rotation)
                #     while yaw > target_yaw[i]:
                #         yaw -= 360.0
                #     while yaw < target_yaw[i]:
                #         yaw += 360.0
                #     if yaw - target_yaw[i] > 20 and yaw - target_yaw[i] < 340:
                #         if yaw - target_yaw[i] > 180:
                #             actions[i] = {"type": 2}
                #         else:
                #             actions[i] = {"type": 1}
                        
                #         print(i, yaw - target_yaw[i])
            
            if actions[0] is None and actions[1] is None:
                break

            for i in range(2):
                if actions[i] is None:
                    continue

                if actions[i]["type"] == 0:
                    self.controller.replicants[i].move_forward()
                elif actions[i]["type"] == 1:
                    self.controller.replicants[i].turn_by(angle = -15)
                elif actions[i]["type"] == 2:
                    self.controller.replicants[i].turn_by(angle = 15)

            finish = False
            while not finish:
                finish = True
                for i in range(2):
                    if self.controller.replicants[i].action.status == ActionStatus.ongoing:
                        finish = False
                        break
                
                if finish:
                    break
                    
                data = self.controller.communicate([])

                if self.no_save_img == False:
                    for i in range(len(data) - 1):
                        r_id = OutputData.get_data_type_id(data[i])
                        if r_id == 'imag' and self.only_save_rgb == False:
                            images = Images(data[i])
                            if images.get_avatar_id() == "a" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
                            if images.get_avatar_id() == "teaser" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'teaser_image'))
                        
                    utils.save_images(os.path.join(self.save_dir, "Images"), controller=self.controller, screen_size=self.screen_size, step = num_frames + self.num_frames, only_save_rgb = self.only_save_rgb)
                    num_frames += 1
        
        self.num_frames += num_frames
        return num_frames
    
    def turn_to_put(self):
        if self.task['task_kind'] == "outdoor_furniture":
            target_yaw = 0
        else:
            target_yaw = 90

        num_frames = 0
        while True:
            yaw = quaternion_to_yaw(self.controller.replicants[0].dynamic.transform.rotation)
            while yaw > target_yaw:
                yaw -= 360.0
            while yaw < target_yaw:
                yaw += 360.0
            if yaw - target_yaw > 20 and yaw - target_yaw < 340:
                if yaw - target_yaw > 180:
                    self.controller.replicants[0].turn_by(angle = 15)
                else:
                    self.controller.replicants[0].turn_by(angle = -15)
            else:
                break
          
            while self.controller.replicants[0].action.status == ActionStatus.ongoing:
                data = self.controller.replicant_furniture_sychronize()

                if self.no_save_img == False:
                    for i in range(len(data) - 1):
                        r_id = OutputData.get_data_type_id(data[i])
                        if r_id == 'imag' and self.only_save_rgb == False:
                            images = Images(data[i])
                            if images.get_avatar_id() == "a" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
                            if images.get_avatar_id() == "teaser" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'teaser_image'))
                        
                    utils.save_images(os.path.join(self.save_dir, "Images"), controller=self.controller, screen_size=self.screen_size, step = num_frames + self.num_frames, only_save_rgb = self.only_save_rgb)
                    num_frames += 1
        
        self.num_frames += num_frames
        return num_frames

    def pick_furniture_together(self, furniture_id):
        print("pick furniture together start!")
        num_frames = 0
        self.controller.picking_furniture_together = True
        for replicant_id in self.controller.replicants:
            self.controller.replicants[replicant_id].reset_arms()
        
        num_frames += self.do_actions()
        print("reset arm finished!")

        # num_frames += self.navigate_both(furniture_id = furniture_id)
        object_name = self.controller.object_manager.objects_static[furniture_id].name
        arrived_at = self.controller.reach_for_dict[object_name]
        for replicant_id in self.controller.replicants:
            self.controller.replicants[replicant_id].move_to_object(int(furniture_id), arrived_at = arrived_at)
        
        num_frames += self.do_actions()
        
        print("move to position finished!")

        for replicant_id in self.controller.replicants: 
            if replicant_id == 0:
                self.controller.replicants[replicant_id].pick_up(furniture_id, Arm.right, object_weight = self.controller.mass_map[str(furniture_id)], no_weight_check = True, lift_pos = {"x": 0, "y": 0.9, "z": 0.4})
            else:
                self.controller.replicants[replicant_id].pick_up(furniture_id, Arm.right, object_weight = self.controller.mass_map[str(furniture_id)], no_weight_check = True, no_grasp = True, lift_pos = {"x": 0, "y": 0.9, "z": 0.4})
        
        num_frames += self.do_actions()

        print("pick up finished!")

        self.controller.getting_sync_information(furniture_id)
        return num_frames
    
    def puton_furniture_together(self, furniture_id, position):
        num_frames = 0
        self.controller.replicants[0].move_to_object(self.controller.replicants[0].truck_id)
        
        num_frames += self.do_actions(sync = True)
        num_frames += self.turn_to_put()

        for replicant_id in self.controller.replicants: 
            if replicant_id == 0:
                self.controller.replicants[replicant_id].put_on(self.controller.truck_id, arm=Arm.right, position = position)
            else:
                self.controller.replicants[replicant_id].put_on(self.controller.truck_id, arm=Arm.right, position = position, no_drop = True)

        num_frames += self.do_actions()
        return num_frames

    def step(self, actions):
        r'''
            Run one step of the environment's dynamics
            Until any agent's action finishes or the max_frame is reached
        '''

        def json_serial(obj):
            """JSON serializer for objects not serializable by default json code."""
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return obj
            raise TypeError("Type not serializable")

        # for i in range(self.number_of_agents):
        #     print(f"Start Agent {i} action: {actions[str(i)]}")

        start = time.time()
        #print(self.bike_contained_name)
        # Receive actions
        pick_furniture_together = False
        puton_furniture_together = False
        furniture_id = None
        for replicant_id in self.controller.replicants:
            action = actions[str(replicant_id)]
            if "outdoor_furniture" in self.task["task_kind"] and self.controller.picking_furniture_together and replicant_id == 1:
                self.delay_frame_count[replicant_id] = 500
                continue
                
            print(action)
            if action is None or action['type'] == 'ongoing': continue
            if not(len(self.previous_action_list[replicant_id]) > 0 and self.action_mappping(action) in self.previous_action_list[replicant_id][-1]):
                # Record previous actions and their status
                self.previous_action_list[replicant_id].append(self.action_mappping(action) + " at frame " + str(self.num_frames))
                self.previous_status_list[replicant_id].append(str(ActionStatus.ongoing))
                if len(self.previous_action_list[replicant_id]) > 1:
                    self.previous_status_list[replicant_id][-2] = str(self.controller.replicants[replicant_id].action.status)
            # otherwise we start an action directly
            self.action_buffer[replicant_id] = []
            self.last_action[replicant_id] = action
            if "arm" in action:
                action['arm'] = utils.map_arms(action['arm'])
            if action["type"] == 0:
                # move forward 0.5m
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'move_forward'})
            elif action["type"] == 1:
                # turn left by 15 degree
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'turn_left'})
            elif action["type"] == 2:
                # turn right by 15 degree
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'turn_right'})
            elif action["type"] == 3:
                # go to and pick up object with arm
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'reach_for'})
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'pick_up', 'reached': True})
                with open(os.path.join(self.save_dir, "loginvalid.txt"), 'a') as f:
                    f.write(f"Info gym Frame: {self.num_frames} performing type 3 action (grasp) with replicant_id {replicant_id} and to held objects {int(action['object'])}\n")
                
                if "outdoor_furniture" in self.task["task_kind"] and "object" in action and self.number_of_agents >= 2 and not self.controller.picking_furniture_together:
                    if self.last_action[1 - replicant_id] is not None and self.last_action[1 - replicant_id]["type"] == 3 and "object" in self.last_action[1 - replicant_id] and action["object"] == self.last_action[1 - replicant_id]["object"] and \
                        self.controller.replicants[1 - replicant_id]._state.replicants[1 - replicant_id][Arm.right] != action["object"] and \
                        self.controller.replicants[1 - replicant_id]._state.replicants[1 - replicant_id][Arm.left] != action["object"]:
                            pick_furniture_together = True
                            furniture_id = int(action["object"])
                    
                    elif actions[str(1 - replicant_id)] is not None and actions[str(1 - replicant_id)]["type"] == 3 and "object" in actions[str(1 - replicant_id)] and action["object"] == actions[str(1 - replicant_id)]["object"] and \
                        self.controller.replicants[1 - replicant_id]._state.replicants[1 - replicant_id][Arm.right] != action["object"] and \
                        self.controller.replicants[1 - replicant_id]._state.replicants[1 - replicant_id][Arm.left] != action["object"]:
                            pick_furniture_together = True
                            furniture_id = int(action["object"])
            
            elif action["type"] == 4:
                # put in container
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'put_in'})
            elif action["type"] == 5:
                # put object on some surface
                object_list = self._with_agent(replicant_id)
                for object_id in object_list:
                    self.controller.state.object_drop_frame[object_id] = self.num_frames
                if "outdoor_furniture" in self.task["task_kind"] and self.controller.picking_furniture_together:
                    puton_furniture_together = True
                elif self.task["task_kind"] == "outdoor_shopping" and replicant_id == 0:
                    self.bike_agent_put_on = True
                    self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'reach_for'})
                    self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'put_on'})
                else:
                    self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'reach_for'})
                    self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'put_on'})

            elif action["type"] == 6:
                # send message, but will never be used
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'send_message'})
            elif action["type"] == 7:
                # remove obstacle
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'reach_for'})
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'pick_up'})
                self.action_buffer[replicant_id].append({**copy.deepcopy(action), 'type': 'put_on'})
            elif action["type"] == 8:    
                # noop
                self.delay_frame_count[replicant_id] = action["delay"]
            else:
                assert False, "Invalid action type"

        num_frames = 0

        if pick_furniture_together:
            for replicant_id in self.controller.replicants:
                self.action_buffer[replicant_id] = []
            
            self.controller.replicants[0].action = None
            self.controller.replicants[1].action = None

            num_frames = self.pick_furniture_together(furniture_id)
            for i in range(len(self.agents)):
                self.agents[i].picking_furniture_together = True
        elif puton_furniture_together:
            for replicant_id in self.controller.replicants:
                self.action_buffer[replicant_id] = []

            self.delay_frame_count[1] = 0
            furniture_id = self.controller.holding_furniture_id
            position = self.controller.truck_occ_map.get_place_point()
            num_frames = self.puton_furniture_together(furniture_id, position)
            self.controller.picking_furniture_together = False
            for i in range(len(self.agents)):
                self.agents[i].picking_furniture_together = False
        else:
            # Do action here
            self.valid_action = [True for _ in range(self.number_of_agents)]
            finish = False
            # with open("file.txt", "a") as f:
            #     f.write(f"{self.bike_agent_put_on}\n")
            while not finish:
                self.check_goal()
                # continue until any agent's action finishes
                for replicant_id in self.controller.replicants:
                    if self.delay_frame_count[replicant_id] > 0:
                        #The agent do nothing in the step since the last action takes several steps.
                        self.delay_frame_count[replicant_id] -= 1
                        continue

                    #print(replicant_id, self.controller.replicants[replicant_id].action.status)
                    if self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing and len(self.action_buffer[replicant_id]) == 0:
                        finish = True
                        if utils.map_status(self.controller.replicants[replicant_id].action.status) == 2:
                            self.valid_action[replicant_id] = False

                        if replicant_id == 0 and self.bike_agent_put_on:
                            # with open("file.txt", "a") as f:
                            #     f.write(f"delete\n")
                                
                            self.bike_agent_put_on = False
                            self.bike_contained = []
                            self.bike_contained_name = []

                    elif self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing:
                        curr_action = self.action_buffer[replicant_id].pop(0)
                        if curr_action['type'] == 'move_forward':       # move forward 0.5m
                            self.controller.replicants[replicant_id].move_forward()
                        elif curr_action['type'] == 'turn_left':     # turn left by 15 degree
                            self.controller.replicants[replicant_id].turn_by(angle = -15)
                        elif curr_action['type'] == 'turn_right':     # turn right by 15 degree
                            self.controller.replicants[replicant_id].turn_by(angle = 15)
                        elif curr_action['type'] == 'reach_for':     # go to and grasp object with arm
                            if isinstance(self.controller.replicants[replicant_id].ability, RiderAbility):
                                self.controller.replicants[replicant_id].move_to_object(int(curr_action["object"]), reset_arms = False)
                            elif "outdoor_furniture" in self.task["task_kind"] and int(curr_action["object"]) != self.controller.truck_id:
                                object_id = int(curr_action["object"])
                                object_name = self.controller.object_manager.objects_static[object_id].name
                                arrived_at = self.controller.reach_for_dict[object_name]
                                self.controller.replicants[replicant_id].move_to_object(int(curr_action["object"]), arrived_at = arrived_at)
                            else: 
                                self.controller.replicants[replicant_id].move_to_object(int(curr_action["object"]))
                        elif curr_action['type'] == 'pick_up':      # grasp object with arm
                            
                            if replicant_id == 0 and self.task["task_kind"] == "outdoor_shopping":
                                self.controller.replicants[replicant_id].pick_up(int(curr_action["object"]), self.controller.object_manager)
                                self.bike_contained.append(int(curr_action["object"]))
                                self.bike_contained_name.append(self.object_names[int(curr_action["object"])])
                            elif "outdoor_furniture" in self.task["task_kind"]:
                                name = self.controller.object_manager.objects_static[int(curr_action["object"])].name
                                if name == "kettle_2":
                                    lift_pos = {"x": 0, "y": 0.9, "z": 0.4}
                                else:
                                    lift_pos = {"x": 0, "y": 0.9, "z": 0}
                                self.controller.replicants[replicant_id].pick_up(int(curr_action["object"]), curr_action["arm"], object_weight = self.controller.mass_map[str(curr_action["object"])], lift_pos = lift_pos, behaviour_data_gen = self.behaviour_data_gen)
                            else:
                                self.controller.replicants[replicant_id].pick_up(int(curr_action["object"]), curr_action["arm"])
                        
                        elif curr_action["type"] == 'put_in':      # put in container
                            self.controller.replicants[replicant_id].put_in()
                            held_objects = list(self.controller.state.replicants[replicant_id].values())
                            if held_objects[0] is not None and held_objects[1] is not None:
                                container, target = None, None
                                if self.get_object_type(held_objects[0]) == 1:
                                    container = held_objects[0]
                                else:
                                    target = held_objects[0]
                                if self.get_object_type(held_objects[1]) == 1:
                                    container = held_objects[1]
                                else:
                                    target = held_objects[1]
                                if container is not None and target is not None:
                                    if container in self.containment_all:
                                        if target not in self.containment_all[container]:
                                            self.containment_all[container].append(target)
                                    else:
                                        self.containment_all[container] = [target]
                        elif curr_action["type"] == 'send_message':      # send message
                            self.messages[replicant_id] = copy.deepcopy(curr_action['message'])
                            self.delay_frame_count[replicant_id] = max((len(self.messages[replicant_id]) - 1) // self.message_per_frame, 0)
                        elif curr_action["type"] == 'put_on':      # put on
                            if "outdoor_furniture" in self.task["task_kind"]:
                                self.controller.replicants[replicant_id].put_on(curr_action['object'], arm=curr_action["arm"], position = self.controller.truck_occ_map.get_place_point())
                            elif self.task["task_kind"] == "outdoor_shopping":
                                if replicant_id == 0:
                                    self.controller.replicants[replicant_id].put_on(curr_action['object'], arm=curr_action["arm"], bike_id = self.controller.bike_id, bike_contained = self.bike_contained, object_manager = self.controller.object_manager)
                                else:
                                    self.controller.replicants[replicant_id].put_on(curr_action['object'], arm=curr_action["arm"])
                            else:
                                self.controller.replicants[replicant_id].put_on(curr_action['object'], arm=curr_action["arm"])
                
                if finish: break
                if self.task['task_kind'] == "outdoor_shopping":
                    data = self.controller.replicant_bike_sychronize()
                elif "outdoor_furniture" in self.task['task_kind']:
                    data = self.controller.replicant_furniture_sychronize()
                else:
                    data = self.controller.communicate([])

                if self.no_save_img == False:
                    for i in range(len(data) - 1):
                        r_id = OutputData.get_data_type_id(data[i])
                        if r_id == 'imag' and self.only_save_rgb == False:
                            images = Images(data[i])
                            if images.get_avatar_id() == "a" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
                            if images.get_avatar_id() == "teaser" and (self.num_frames + num_frames) % 1 == 0:
                                TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'teaser_image'))
                    
                    utils.save_images(os.path.join(self.save_dir, "Images"), controller=self.controller, screen_size=self.screen_size, step = num_frames + self.num_frames, only_save_rgb = self.only_save_rgb)
                num_frames += 1

                # print("agent arms: ", self.num_frames + num_frames, self.controller.replicants[1]._state.replicants[1][Arm.left], self.controller.replicants[1]._state.replicants[1][Arm.right])


            self.num_frames += num_frames

        # if self.task['task_kind'] == "outdoor_shopping":
        #     self.controller.replicant_bike_sychronize()
        # elif self.task['task_kind'] == "outdoor_furniture":
        #     self.controller.replicant_furniture_sychronize()    
            
        self.action_list.append(actions)
        goal_put, goal_total, self.success = self.check_goal()
        reward = 0
        for replicant_id in self.controller.replicants:
            action = actions[str(replicant_id)]
            task_status = self.controller.replicants[replicant_id].action.status
            self.f.write('step: {}, action: {}, time: {}, status: {}\n'
                    .format(self.num_step, "None" if action is None else action["type"],
                    time.time() - start,
                    task_status))
            container_info = self.add_name_and_empty(copy.deepcopy(self.controller.state.containment))
            self.f.write('position: {}, forward: {}, containment: {}, goal: {}, container: {}\n'.format(
                    self.controller.replicants[replicant_id].dynamic.transform.position,
                    self.controller.replicants[replicant_id].dynamic.transform.forward,
                    container_info, self.add_name(self.controller.state.target_object_ids), self.add_name(self.controller.state.container_ids)))
            self.f.flush()
            if task_status != ActionStatus.success and task_status != ActionStatus.ongoing:
                reward -= 0.1
                with open(os.path.join(self.save_dir, "loginvalid.txt"), 'a') as f:
                    f.write(f"gym Frame: {self.num_frames} noreason invalid with replicant_id {replicant_id} with action {action}\n")

            if self.controller.replicants[replicant_id].action.status != ActionStatus.ongoing and len(self.action_buffer[replicant_id]) == 0:
                self.previous_status_list[replicant_id][-1] = str(self.controller.replicants[replicant_id].action.status)

        self.num_step += 1        
        self.reward += reward
        done = False
        if self.num_frames >= self.max_frame or self.success:
            done = True
            self.done = True
            save_action_status = {
                'success': self.success,
                'action': self.previous_action_list,
                'status': self.previous_status_list
            }
            with open(os.path.join(self.save_dir, 'actions.json'), 'a') as f:
                json.dump(save_action_status, f, indent=4)
        
        self.obs = self.get_obs()
        obs = self.obs_filter(self.obs)
        # add messages to obs
        if self.number_of_agents >= 2:
            for replicant_id in self.controller.replicants:
                # Send messages
                obs[str(replicant_id)]['messages'] = copy.deepcopy(self.messages)
            self.messages = [None for _ in range(self.number_of_agents)]

        if "outdoor_furniture" in self.task["task_kind"]:
            self.controller.truck_occ_map.generate()

        info = self.get_info()
        info['done'] = done
        info['num_frames_for_step'] = num_frames
        info['num_step'] = self.num_step
        if done:
            info['reward'] = self.reward
        
        """
        # for id in range(len(self.agents)):
        #     agent = self.agents[id]
        #     file_path = os.path.join(self.save_dir, f"agent_{id}_obs_{self.num_frames}.txt")
        #     with open(file_path, "w") as f:
        #         f.write(f"{agent.obs['visible_objects']}\n")
        #         f.write(f"{agent.obs['seg_mask']}\n")
        """
        
        # with open("file.txt", "a") as f:
        #     for i in range(self.number_of_agents):
        #         # print(f"Agent {i} action: {actions[str(i)]}, valid: {self.valid_action[i]}, status: {self.controller.replicants[i].action.status}")
        #         f.write(f"Agent {i} action: {actions[str(i)]}, valid: {self.valid_action[i]}, status: {self.controller.replicants[i].action.status}\n")
        
        # with open("file.txt", "a") as f:
        #     f.write(str(self.num_frames))
        #     f.write("\n")
        #     x, y = self.controller.truck_occ_map.grid.shape
        #     for i in range(x):
        #         for j in range(y):
        #             f.write(f"{self.controller.truck_occ_map.grid[i, j]} ")
        #         f.write("\n")
            
        #     f.write("\n\n\n\n")
        
        return obs, reward, done, info
     
    def render(self):
        return None

    def close(self):
        print('close environment ...')
    #    with open(os.path.join(self.save_dir, 'action.pkl'), 'wb') as f:
    #        d = {'scene_info': self.scene_info, 'actions': self.action_list}
    #        pickle.dump(d, f)
