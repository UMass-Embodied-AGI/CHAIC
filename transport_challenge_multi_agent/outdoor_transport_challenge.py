from csv import DictReader
from typing import List, Dict, Union, Tuple, Optional
from subprocess import run, PIPE
from pathlib import Path
import numpy as np
import cv2
from tdw.replicant.arm import Arm
from scipy.spatial import cKDTree
from tdw.version import __version__
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.replicant import Replicant
from tdw.librarian import HumanoidLibrarian
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.nav_mesh import NavMesh
from tdw.add_ons.container_manager import ContainerManager
from tdw.replicant.image_frequency import ImageFrequency
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from tdw.replicant.action_status import ActionStatus
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.replicant_outdoor_shopping_transport_challenge import ReplicantOutdoorShoppingTransportChallenge
from transport_challenge_multi_agent.replicant_outdoor_furniture_transport_challenge import ReplicantOutdoorFurnitureTransportChallenge
from transport_challenge_multi_agent.paths import CONTAINERS_PATH, TARGET_OBJECTS_PATH, TARGET_OBJECT_MATERIALS_PATH
from transport_challenge_multi_agent.globals import Globals
from transport_challenge_multi_agent.asset_cached_controller import AssetCachedController
from transport_challenge_multi_agent.agent_ability_info import ability_mapping
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from tdw.add_ons.logger import Logger
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.replicant.actions.reach_for import ReachFor
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.actions.grasp import Grasp
from tdw.output_data import OutputData, Raycast
from tdw.add_ons.add_on import AddOn
import os
import json

def quaternion_to_yaw(quaternion):
    y = 2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])
    x = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
    yaw = np.arctan2(y, x)
    
    return np.rad2deg(yaw) % 360


class TruckOccupancyMap(AddOn):

    def __init__(self, grid_size: float, x_min: float, x_max: float, z_min: float, z_max: float, floor_height: float):
        super().__init__()
        self.grid_size = grid_size
        self.floor_height = floor_height
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.x_num = int((x_max - x_min) / grid_size)
        self.z_num = int((z_max - z_min) / grid_size)
        self.grid = np.zeros((self.x_num, self.z_num), dtype = int)
        self.initialized = True
    
    def get_initialization_commands(self) -> List[dict]:
        return []
    
    def on_send(self, resp: List[bytes]) -> None:
        self.grid = np.zeros((self.x_num, self.z_num), dtype=int)
        ht = np.zeros(self.grid.shape, dtype = np.float32)
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "rayc":
                rayc = Raycast(resp[i])
                idx = rayc.get_raycast_id()
                if idx >= 142857 and idx < 142857 + self.x_num * self.z_num:
                    idx -= 142857
                    if rayc.get_hit():
                        hit_y = rayc.get_point()[1]
                        ht[idx // self.z_num][idx % self.z_num] = max(ht[idx // self.z_num][idx % self.z_num], hit_y)
                        # print("hit point=", rayc.get_point(), "i, j=", idx // self.num_grid[1], idx % self.num_grid[1])
                        if hit_y > self.floor_height + 0.01:
                            self.grid[idx // self.z_num][idx % self.z_num] = 1
                    else:
                        self.grid[idx // self.z_num][idx % self.z_num] = 100
    
        # with open("file.txt", "a") as f:
        #     x, y = ht.shape
        #     for i in range(x):
        #         for j in range(y):
        #             f.write(str(ht[i][j]) + " ")
        #         f.write("\n")

        #     f.write("\n\n\n")


    def grid_to_real(self, position):
        return [position[0] * self.grid_size + self.x_min, self.floor_height, position[1] * self.grid_size + self.z_min]

    def real_to_grid(self, position):
        if len(position) > 2:
            position = [position[0], position[2]]
        return [int((position[0] - self.x_min + 0.01) / self.grid_size), int((position[1] - self.z_min + 0.01) / self.grid_size)]

    def generate(self):
        for i in range(self.x_num):
            for j in range(self.z_num):
                # print(self.grid_to_real([i, j]))
                start = np.array(self.grid_to_real([i, j])) - [0, 2, 0]
                end = start + [0, 4, 0]
                self.commands.append({"$type": "send_boxcast",
                                      "half_extents": {"x": self.grid_size / 2, "y": 0, "z": self. grid_size / 2},
                                      "origin": TDWUtils.array_to_vector3(end),
                                      "destination": TDWUtils.array_to_vector3(start),
                                      "id": i * self.z_num + j + 142857})
    
    def get_place_point(self):
        # with open("file.txt", "a") as f:
        #     f.write("now")
        #     f.write("\n")
        #     x, y = self.grid.shape
        #     for i in range(x):
        #         for j in range(y):
        #             f.write(f"{self.grid[i, j]} ")
        #         f.write("\n")

        # x, y = self.grid.shape
        # for i in range(x):
        #     for j in range(y):
        #         print(f"{self.grid[i, j]} ", end = "")
        #     print()

        dist = np.zeros((self.x_num, self.z_num), dtype = int)
        for i in range(self.x_num):
            for j in range(self.z_num):
                for k in range(0, 100):
                    if i < k or i + k >= self.x_num or j < k or j + k >= self.z_num or self.grid[i - k : i + k + 1, j - k : j + k + 1].sum() > 0:
                        dist[i][j] = k
                        break

        position = np.argmax(dist)
        position = np.unravel_index(position, dist.shape)
        # with open("file.txt", "a") as f:
        #     f.write(f"{position[0]} {position[1]}\n\n\n\n\n")

        return self.grid_to_real(position)


class OutdoorTransportChallenge(TransportChallenge):
    """
    A subclass of `TransportChallenge` for the Outdoor Transport Challenge.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reach_for_dict = {
            "4ft_wood_shelving": 0.9,
            "appliance-ge-profile-microwave3": 0.7,
            "arflex_strips_sofa": 1.0,
            "b03_grandpiano2014": 1.0,
            "b04_backpack": 0.7,
            "cabinet_24_door_drawer_wood_beach_honey": 0.7,
            "dining_room_table": 0.8,
            "dishwasher_4": 0.7,
            "emeco_navy_chair": 0.7,
            "hp_printer": 0.7,
            "kettle_2": 0.6,
            "sm_tv": 0.7,
        }

    def _start_trial(self, scene, layout, replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2, 
                    random_seed: int = None, task_meta = None, data_prefix = 'dataset/dataset_train') -> None:
        
        self.scene = scene
        self.layout = layout
        self.data_prefix = data_prefix
        # self.communicate({"$type": "set_floorplan_roof", "show": False})
        load_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}.json")
        with open(load_path, "r") as f: scene = json.load(f)
        load_agent_position_avatar_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_metadata.json")
        with open(load_agent_position_avatar_path, "r") as f: metadata = json.load(f)
        self.communicate(scene)
        self.communicate({"$type": "add_hdri_skybox", "name": "sky_white",
                                 "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white",
                                 "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90,
                                 "sun_initial_angle": 0, "sun_intensity": 1.25})
        self.state = ChallengeState()
        self.add_ons.clear() # Clear the add-ons.
        if self.logger is not None:
            self.add_ons.append(self.logger)
        self.replicants.clear()
        # Add an occupancy map.
        self.add_ons.append(self.occupancy_map)
        replicant_positions: List[Dict[str, float]] = list()
        
        # Add the Replicants. If the position is fixed, the position is the same as the last time.
        for i in range(replicants):
            replicant_positions.append(np.array(metadata["agent"][str(i)]))

        print(f"Replicant positions: {replicant_positions}")

        self.mass_map = metadata["mass_map"]
        
        for i, replicant_position in enumerate(replicant_positions):
            if task_meta is not None and task_meta["task_kind"] == "outdoor_shopping":
                if i == 0: # bike agent
                    replicant = ReplicantOutdoorShoppingTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    rotation={"x": 0, "y": 270, "z": 0},
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection,
                                                    name=self.replicants_name[i],
                                                    ability=self.replicants_ability[i]
                                                    )
                elif i == 2:
                    replicant = ReplicantTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    rotation={"x": 0, "y": 225, "z": 0},
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection,
                                                    name=self.replicants_name[i],
                                                    ability=self.replicants_ability[i]
                                                    )
                else:
                    replicant = ReplicantTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection,
                                                    name=self.replicants_name[i],
                                                    ability=self.replicants_ability[i]
                                                    )
            else:
                replicant = ReplicantOutdoorFurnitureTransportChallenge(replicant_id=i,
                                                    state=self.state,
                                                    position=replicant_position,
                                                    image_frequency=self._image_frequency,
                                                    target_framerate=self._target_framerate,
                                                    enable_collision_detection=self.enable_collision_detection,
                                                    name=self.replicants_name[i],
                                                    ability=self.replicants_ability[i]
                                                    )
            # print(f"Replicant {i} is {replicant._record.name} with ability {replicant.ability.__class__.__name__}")
            self.replicants[replicant.replicant_id] = replicant
            self.add_ons.append(replicant)

        # Set the pass masks.
        # Add a challenge state and object manager.
        self.object_manager.reset()
        self.add_ons.extend([self.state, self.object_manager])
        self.communicate([])
        commands = []
        # since the fridge is easy to drop, I want to make it kinematic.
     
        if task_meta is not None:
            assert(task_meta["task_kind"] == "outdoor_furniture" or task_meta["task_kind"] == "outdoor_shopping" or task_meta["task_kind"] == "outdoor_furniture_crossway")
            if 'goal_object_names' not in task_meta.keys():
                # For indoor scenes: not goal object name, but possible tasks
                task_meta['goal_object_names'] = []
                # task_meta['possible_object_names'] = []
                for it in task_meta['goal_task']:
                    # name & number
                    task_meta['goal_object_names'].append(it[0])
                # for task in task_meta['possible_task']:
                #     for it in task:
                #         if it[0] not in task_meta['possible_object_names']:
                #             task_meta['possible_object_names'].append(it[0])
            for object_id in self.object_manager.objects_static.keys():
                if self.object_manager.objects_static[object_id].name in task_meta['goal_object_names']:
                    self.state.target_object_ids.append(object_id)
                if self.object_manager.objects_static[object_id].name in task_meta['possible_object_names']:
                    self.state.possible_target_object_ids.append(object_id)
                if self.object_manager.objects_static[object_id].name in task_meta['container_names']:
                    self.state.container_ids.append(object_id)
                if 'obstacle_names' in task_meta.keys():
                    if self.object_manager.objects_static[object_id].name in task_meta['obstacle_names']:
                        self.state.obstacle_ids.append(object_id)

        for replicant_id in self.replicants:
            # Set pass masks.
            commands.append({"$type": "set_pass_masks",
                             "pass_masks": self._image_passes,
                             "avatar_id": self.replicants[replicant_id].static.avatar_id})
            # Ignore collisions with target objects.
            self.replicants[replicant_id].collision_detection.exclude_objects.extend(self.state.target_object_ids)
        # Add a NavMesh.
        nav_mesh_exclude_objects = list(self.replicants.keys())
        nav_mesh_exclude_objects.extend(self.state.target_object_ids)
        nav_mesh = NavMesh(exclude_objects=nav_mesh_exclude_objects)
        self.add_ons.append(nav_mesh)
        self.container_manager = ContainerManager()
        self.add_ons.append(self.container_manager)
        # Send the commands.
        # self.communicate(commands)
        # Reset the heads.
        for replicant_id in self.replicants:
            self.replicants[replicant_id].reset_head(scale_duration=Globals.SCALE_IK_DURATION)
        reset_heads = False
        while not reset_heads:
            self.communicate([])
            reset_heads = True
            for replicant_id in self.replicants:
                if self.replicants[replicant_id].action.status == ActionStatus.ongoing:
                    reset_heads = False
                    break
                
        resp = self.communicate([{"$type": "send_scene_regions"}])
        self.scene_bounds = SceneBounds(resp=resp)
        for replicant_id in self.replicants:
            self.replicants[replicant_id].scene_bounds = self.scene_bounds

        if task_meta["task_kind"] == "outdoor_shopping":
            for object_id in self.object_manager.objects_static.keys():
                if self.object_manager.objects_static[object_id].name == "b03_fire_hydrant":
                    for replicant_id in self.replicants:
                        self.replicants[replicant_id].collision_detection.exclude_objects.append(object_id)

            self.bike_id = metadata["bike_id"]
            self.replicants[0].holding_bike_id = self.bike_id
            
            for object_id in self.object_manager.objects_static.keys():
                if self.object_manager.objects_static[object_id].name == "b01_tent" or \
                    self.object_manager.objects_static[object_id].name == "4ft_wood_shelving" or \
                    self.object_manager.objects_static[object_id].name == "huffy_nel_lusso_womens_cruiser_bike_2011vray" or \
                    self.object_manager.objects_static[object_id].name == "white_shopping_bag":
                    for replicant_id in self.replicants:
                        self.replicants[replicant_id].collision_detection.exclude_objects.append(object_id)

            # self.bike_human_diff = np.array([-0.5, 0, -0.1])
            # reach_for_position = np.array(position) + np.array([-0.35, 1.45, 0.25])
            # self.replicants[0].action = ReachForTransportChallenge(target = reach_for_position,
            #                                                     arm = Arm.left,
            #                                                     dynamic = self.replicants[0].dynamic,
            #                                                     max_distance = 1.5,
            #                                                     absolute = True)

            self.replicants[0].action = ReachFor(targets = [self.bike_id, self.bike_id],
                                   arms = [Arm.right, Arm.left],
                                   absolute = True,
                                   dynamic = self.replicants[0].dynamic,
                                   collision_detection = self.replicants[0].collision_detection,
                                   offhand_follows = False,
                                   arrived_at = 0.01,
                                   previous = self.replicants[0]._previous_action,
                                   duration = 0.25,
                                   scale_duration = True,
                                   max_distance = 1.5,
                                   from_held = False,
                                   held_point = "bottom")
            # bike_position = self.object_manager.transforms[self.bike_id].position
            # print(bike_position)
            # left_arm_position = np.array(bike_position) + np.array([-0.45, 0.9, -0.3])
            # right_arm_position = np.array(bike_position) + np.array([-0.45, 0.9, 0.3])
            # self.replicants[0].action = ReachForTransportChallenge(target = TDWUtils.array_to_vector3(left_arm_position),
            #                                           arm = Arm.left,
            #                                           dynamic = self.replicants[0].dynamic,
            #                                           max_distance = 1.5,
            #                                           absolute = True)

            # while self.replicants[0].action.status == ActionStatus.ongoing:
            #     self.communicate([])

            # self.replicants[0].action = ReachForTransportChallenge(target = TDWUtils.array_to_vector3(right_arm_position),
            #                                           arm = Arm.right,
            #                                           dynamic = self.replicants[0].dynamic,
            #                                           max_distance = 1.5,
            #                                           absolute = True)

            while self.replicants[0].action.status == ActionStatus.ongoing:
                self.communicate([])

            # commands = [{"$type": "set_kinematic_state", "id": self.bike_id, "is_kinematic": False}]
            # self.communicate(commands)
            # self.replicants[0].action = Grasp(target = self.bike_id,
            #                                 arm = Arm.left,
            #                                 dynamic = self.replicants[0].dynamic,
            #                                 angle = 0,
            #                                 axis = "yaw",
            #                                 relative_to_hand = False,
            #                                 offset = 0,
            #                                 kinematic_objects = self.replicants[0]._kinematic_objects)

            # while self.replicants[0].action.status == ActionStatus.ongoing:
            #     self.communicate([])
            #     print("now2: ",  self.replicants[0].action.status)

            # teleport_position = np.array(position) + self.bike_human_diff
            # commands = [{"$type": "teleport_object", "position": TDWUtils.array_to_vector3(teleport_position), "id": self.bike_id}]
            # self.communicate(commands)
            position = self.replicants[0].dynamic.transform.position
            bike_position = self.object_manager.transforms[self.bike_id].position
            self.replicants[0].bike_human_diff = bike_position - np.array(position)
            self.replicants[0].bike_basket_position_diff = np.array([-0.5, 0.9, 0])
            self.replicants[0].human_left_hand_position = self.replicants[0].dynamic.body_parts[self.replicants[0].static.hands[Arm.left]].position - np.array(bike_position)
            self.replicants[0].human_right_hand_position = self.replicants[0].dynamic.body_parts[self.replicants[0].static.hands[Arm.right]].position - np.array(bike_position)
            self.replicants[0].bike_drop_diff = np.array([-0.5, 0.9, 0.5])
            
        elif "outdoor_furniture" in task_meta["task_kind"]:
            for replicant_id in self.replicants:
                self.replicants[replicant_id].task_kind = task_meta["task_kind"]

            for object_id in self.object_manager.objects_static.keys():
                if self.object_manager.objects_static[object_id].name == "b06_dump":
                    self.truck_id = object_id
                    for replicant_id in self.replicants:
                        self.replicants[replicant_id].truck_id = object_id
                        self.replicants[replicant_id].collision_detection.exclude_objects.append(object_id)
            if task_meta["task_kind"] == "outdoor_furniture":  
                self.truck_occ_map = TruckOccupancyMap(grid_size = 0.125, x_min = 10.5, x_max = 14.875, z_min = -0.75, z_max = 1, floor_height = 0.95)
            else:
                self.truck_occ_map = TruckOccupancyMap(grid_size = 0.125, x_min = 57.25, x_max = 59, z_min = -22.75, z_max = -18.5, floor_height = 0.95)
            self.truck_occ_map.generate()
            self.add_ons.append(self.truck_occ_map)
            self.picking_furniture_together = False

            # for replicant_id in self.replicants:
            #     self.replicants[replicant_id].weight_constraint = metadata["agent_weight_constraint"][str(replicant_id)]

    def replicant_bike_sychronize(self):
        position_rep = self.replicants[0].dynamic.transform.position
        rotation_rep = quaternion_to_yaw(self.replicants[0].dynamic.transform.rotation) - 270.00
        if rotation_rep < 0:
            rotation_rep += 360
        rotation_bike = quaternion_to_yaw(self.object_manager.transforms[self.bike_id].rotation)
        commands = []
        rotation_diff = abs(rotation_rep - rotation_bike)
        if rotation_diff > 1 and rotation_diff < 359:
            commands = [{"$type": "rotate_object_by", "angle": rotation_rep - rotation_bike, "id": self.bike_id, "axis": "yaw"}]
        
        dx = self.replicants[0].bike_human_diff[0] * np.cos(np.deg2rad(rotation_rep)) + self.replicants[0].bike_human_diff[2] * np.sin(np.deg2rad(rotation_rep))
        dz = self.replicants[0].bike_human_diff[2] * np.cos(np.deg2rad(rotation_rep)) - self.replicants[0].bike_human_diff[0] * np.sin(np.deg2rad(rotation_rep))
        teleport_position = np.array(position_rep) + np.array([dx, 0, dz])
        commands.extend([{"$type": "teleport_object", "position": TDWUtils.array_to_vector3(teleport_position), "id": self.bike_id}])
        data = self.communicate(commands)
        return data
    
    def getting_sync_information(self, furniture_id):
        assert len(self.replicants) >= 2 and self.picking_furniture_together == True
        self.holding_furniture_id = furniture_id
        self.replicant0_rot = quaternion_to_yaw(self.replicants[0].dynamic.transform.rotation)
        self.replicant1_rot = quaternion_to_yaw(self.replicants[1].dynamic.transform.rotation)
        self.furniture_rot = quaternion_to_yaw(self.object_manager.transforms[furniture_id].rotation)
        self.furniture_human_diff = self.object_manager.transforms[furniture_id].position - np.array(self.replicants[0].dynamic.transform.position)
        self.human_human_diff = np.array(self.replicants[1].dynamic.transform.position) - np.array(self.replicants[0].dynamic.transform.position)
    
    def replicant_furniture_sychronize(self):
        commands = []
        if self.picking_furniture_together:
            position_rep0 = self.replicants[0].dynamic.transform.position
            rotation_rep0 = quaternion_to_yaw(self.replicants[0].dynamic.transform.rotation)
            original_rotation_rep0 = self.replicant0_rot
            rotation_rep1 = quaternion_to_yaw(self.replicants[1].dynamic.transform.rotation)
            original_rotation_rep1 = self.replicant1_rot
            furniture_id = self.holding_furniture_id
            rotation_furniture = quaternion_to_yaw(self.object_manager.transforms[furniture_id].rotation)
            original_rotation_furniture = self.furniture_rot
            tot_rotation_furniture = original_rotation_furniture + rotation_rep0 - original_rotation_rep0 - rotation_furniture
            while tot_rotation_furniture < 0:
                tot_rotation_furniture += 360
            while tot_rotation_furniture > 360:
                tot_rotation_furniture -= 360

            if tot_rotation_furniture > 1 and tot_rotation_furniture < 359:
                commands.extend([{"$type": "rotate_object_by", "angle": tot_rotation_furniture, "id": furniture_id, "axis": "yaw"}])

            tot_rotation_rep1 = original_rotation_rep1 + rotation_rep0 - original_rotation_rep0 - rotation_rep1
            while tot_rotation_rep1 < 0:
                tot_rotation_rep1 += 360
            while tot_rotation_rep1 > 360:
                tot_rotation_rep1 -= 360

            if tot_rotation_rep1 > 1 and tot_rotation_rep1 < 359:
                commands.extend([{"$type": "rotate_object_by", "angle": tot_rotation_rep1, "id": 1, "axis": "yaw"}])

            rotation_rep0 -= original_rotation_rep0
            furniture_human_diff = self.furniture_human_diff
            dx = furniture_human_diff[0] * np.cos(np.deg2rad(rotation_rep0)) + furniture_human_diff[2] * np.sin(np.deg2rad(rotation_rep0))
            dz = furniture_human_diff[2] * np.cos(np.deg2rad(rotation_rep0)) - furniture_human_diff[0] * np.sin(np.deg2rad(rotation_rep0))
            dy = furniture_human_diff[1]
            teleport_position = np.array(position_rep0) + np.array([dx, dy, dz])
            commands.extend([{"$type": "teleport_object", "position": TDWUtils.array_to_vector3(teleport_position), "id": furniture_id}])

            human_human_diff = self.human_human_diff
            dx = human_human_diff[0] * np.cos(np.deg2rad(rotation_rep0)) + human_human_diff[2] * np.sin(np.deg2rad(rotation_rep0))
            dz = human_human_diff[2] * np.cos(np.deg2rad(rotation_rep0)) - human_human_diff[0] * np.sin(np.deg2rad(rotation_rep0))
            dy = human_human_diff[1]
            teleport_position = np.array(position_rep0) + np.array([dx, dy, dz])
            commands.extend([{"$type": "teleport_object", "position": TDWUtils.array_to_vector3(teleport_position), "id": 1}])
        else:
            for replicant_id in self.replicants:
                if self.replicants[replicant_id].holding_furniture == False:
                    continue

                position_rep = self.replicants[replicant_id].dynamic.transform.position
                rotation_rep = quaternion_to_yaw(self.replicants[replicant_id].dynamic.transform.rotation)
                original_rotation_rep = self.replicants[replicant_id].human_rotation
                furniture_id = self.replicants[replicant_id].holding_furniture_id
                rotation_furniture = quaternion_to_yaw(self.object_manager.transforms[furniture_id].rotation)
                original_rotation_furniture = self.replicants[replicant_id].furniture_rotation
                tot_rotation = original_rotation_furniture + rotation_rep - original_rotation_rep - rotation_furniture
                while tot_rotation < 0:
                    tot_rotation += 360
                while tot_rotation > 360:
                    tot_rotation -= 360

                if tot_rotation > 1 and tot_rotation < 359:
                    commands.extend([{"$type": "rotate_object_by", "angle": tot_rotation, "id": furniture_id, "axis": "yaw"}])
                
                rotation_rep -= original_rotation_rep
                furniture_human_diff = self.replicants[replicant_id].furniture_human_diff
                dx = furniture_human_diff[0] * np.cos(np.deg2rad(rotation_rep)) + furniture_human_diff[2] * np.sin(np.deg2rad(rotation_rep))
                dz = furniture_human_diff[2] * np.cos(np.deg2rad(rotation_rep)) - furniture_human_diff[0] * np.sin(np.deg2rad(rotation_rep))
                dy = furniture_human_diff[1]
                teleport_position = np.array(position_rep) + np.array([dx, dy, dz])
                commands.extend([{"$type": "teleport_object", "position": TDWUtils.array_to_vector3(teleport_position), "id": furniture_id}])

        data = self.communicate(commands)
        return data