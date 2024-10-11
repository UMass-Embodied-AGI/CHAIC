from tdw.add_ons.logger import Logger
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Raycast
from tdw.add_ons.log_playback import LogPlayback
from tdw.librarian import ModelLibrarian
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from typing import Union
import random
import numpy as np
import os
from typing import List, Optional
import shutil

import json
import copy
import argparse

"""
0: unoccupied
1: occupied
100: no floor
"""

# class OutputCommandContoller(Controller):
#     def __init__(self, **kwargs):
#         self.commands = []
#         super().__init__(**kwargs)
        
#     def communicate(self, commands: Union[dict, List[dict]], write = True) -> list:
#         if not isinstance(commands, list):
#             commands = [commands]
            
#         if write:
#             self.commands.extend(commands)

#         return super().communicate(commands=commands)
PATH = os.path.dirname(os.path.abspath(__file__))

def get_add_object(model_name, position, rotation, object_id, library):

    if library == "":
        library = "models_core.json"
    if library not in Controller.MODEL_LIBRARIANS:
        Controller.MODEL_LIBRARIANS[library] = ModelLibrarian(library)
    record = Controller.MODEL_LIBRARIANS[library].get_record(model_name)

    asset_path = os.path.join("local_asset/Linux", model_name)
    if os.path.exists(asset_path):
        url = "file://" + asset_path.replace("\\", "/")
    else:
        url = record.get_url()

    return {"$type": "add_object",
            "name": model_name,
            "url": url,
            "scale_factor": record.scale_factor,
            "position": position if position is not None else {"x": 0, "y": 0, "z": 0},
            "rotation": rotation if rotation is not None else {"x": 0, "y": 0, "z": 0},
            "category": record.wcategory,
            "id": object_id,
            "affordance_points": record.affordance_points}

def l2_dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

class BoxCastOccupancyMap(AddOn):
    def __init__(self):
        super().__init__()
        self.grid: Optional[np.ndarray] = None
        self.origin: Optional[np.ndarray] = None
        self.grid_size: Optional[np.ndarray] = None
        self.num_grid: Optional[List[int]] = None
        self.initialized = True
        self.floor_height: Optional[float] = None
    
    def get_initialization_commands(self) -> List[dict]:
        return []
    
    def on_send(self, resp: List[bytes]) -> None:
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "rayc":
                rayc = Raycast(resp[i])
                idx = rayc.get_raycast_id()
                if idx >= 142857 and idx < 142857 + self.num_grid[0] * self.num_grid[1]:
                    idx -= 142857
                    if rayc.get_hit():
                        hit_y = rayc.get_point()[1]
                        # print("hit point=", rayc.get_point(), "i, j=", idx // self.num_grid[1], idx % self.num_grid[1])
                        if hit_y > self.floor_height + 0.01:
                            self.grid[idx // self.num_grid[1]][idx % self.num_grid[1]] = 1
                    else:
                        self.grid[idx // self.num_grid[1]][idx % self.num_grid[1]] = 100
    
    def grid_to_real(self, position):
        if not isinstance(position, list):
            position = position.tolist()
        return [position[0] * self.grid_size - self.origin[0] * self.grid_size, self.floor_height, position[1] * self.grid_size - self.origin[1] * self.grid_size]

    def real_to_grid(self, position):
        if not isinstance(position, list):
            position = position.tolist()
        if len(position) > 2:
            position = [position[0], position[2]]
        return [int((position[0] + self.origin[0] * self.grid_size + 0.01) / self.grid_size), int((position[1] + self.origin[1] * self.grid_size + 0.01) / self.grid_size)]

    def generate(self, grid_size: float = 0.25, boundX = [-6, 6], boundZ = [-6, 6], floor_height = 0.0) -> None:
        self.grid_size = grid_size
        self.num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
        self.origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]
        self.floor_height = floor_height

        self.grid = np.zeros(self.num_grid, dtype=int)
        for i in range(self.num_grid[0]):
            for j in range(self.num_grid[1]):
                start = np.array(self.grid_to_real([i, j])) - [0, 20, 0]
                end = start + [0, 40, 0]
                # print(start, end, i, j)
                self.commands.append({"$type": "send_boxcast",
                                      "half_extents": {"x": grid_size / 2, "y": 0, "z": grid_size / 2},
                                      "origin": TDWUtils.array_to_vector3(end),
                                      "destination": TDWUtils.array_to_vector3(start),
                                      "id": i * self.num_grid[1] + j + 142857})
    def find_free(self, r):
        candidates = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                s = self.grid[max(i-r, 0):min(i+r+1, self.grid.shape[0]), max(j-r, 0):min(j+r+1, self.grid.shape[1])].sum()
                if s == 0:
                    candidates.append([i, j])
        if len(candidates) == 0:
            return None
        pos = random.choice(candidates)
        return self.grid_to_real(pos)
    
    def get_border(self, grid_size: float = 0.25, boundX = [-6, 6], boundZ = [-6, 6], floor_height = 0.0):
        self.grid_size = grid_size
        self.num_grid = [int((boundX[1] - boundX[0]) / grid_size) + 5, int((boundZ[1] - boundZ[0]) / grid_size) + 5]
        self.origin = [int(-boundX[0] / grid_size) + 2, int(-boundZ[0] / grid_size) + 2]
        self.floor_height = floor_height
        ret = []
        for i in range(self.num_grid[0]):
            for j in range(self.num_grid[1]):
                if i == 0 or i == self.num_grid[0] - 1 or j == 0 or j == self.num_grid[1] - 1:
                    ret.append(self.grid_to_real([i, j]))
        
        return ret
        
global container_list, object_list, possible_tasks
def load_config():
    global container_list, object_list, possible_tasks
    container_json = os.path.join(PATH, "configs/container_list.json")
    object_json = os.path.join(PATH, "configs/object_list.json")
    possible_task_config = os.path.join(PATH, "configs/possible_tasks.json")
    with open(container_json, "r") as f:
        container_list = json.load(f)
    with open(object_json, "r") as f:
        object_list = json.load(f)
    with open(possible_task_config, "r") as f:
        possible_tasks = json.load(f)

    possible_tasks = possible_tasks["possible_task"]


# go to parent directory until the folder name is HAZARD
# while os.path.basename(PATH) != "HAZARD":
#     PATH = os.path.dirname(PATH)

available_scenes = ["suburb_scene_2023"]

num_objects = 10
num_containers = 3


"""
55 0
-58 -9
58 -51
-61 55"""
position_config = [
    #[-1, -2],
    [58, -9],
    [55, 0],
    [-58, -9],
    [58, -51],
    [-61, 55],
]

tent_pos = [
    [10, -2.5], 
    [10, 4],
    [15, -2.5],
    [15, 4],
    [20, -2.5],
    [20, 4]
]

stores = [
    {"apple": 2, "b04_orange_00": 3, "b03_banan_001": 1},
    {"apple": 1, "b04_red_grapes": 3, "b03_banan_001": 2},
    {"b03_loafbread": 2, "b01_croissant": 2, "b03_burger": 2},
    {"b03_loafbread": 1, "b01_croissant": 1, "b03_pink_donuts_mesh": 2},
    {"102_pepsi_can_12_fl_oz_vray": 2, "104_sprite_can_12_fl_oz_vray": 1, "fanta_orange_can_12_fl_oz_vray": 2},
    {"b03_cocacola_can_cage": 3, "104_sprite_can_12_fl_oz_vray": 2, "fanta_orange_can_12_fl_oz_vray": 1},
]

categories = [["apple", "b04_orange_00", "b04_red_grapes", "b03_banan_001"],
              ["b03_loafbread", "b01_croissant", "b03_burger", "b03_pink_donuts_mesh"],
              ["102_pepsi_can_12_fl_oz_vray", "b03_cocacola_can_cage", "104_sprite_can_12_fl_oz_vray", "fanta_orange_can_12_fl_oz_vray"]]

def generate(scene_name, p, times, file_dir, img_dir):
    num = len(possible_tasks) * times
    if "train" in file_dir:
        np.random.seed(12345678)
    elif "test" in file_dir:
        np.random.seed(87654321)
    else:
        np.random.seed(12345678)

    seeds = [0 for i in range(num)]
    for i in range(num):
        seeds[i] = np.random.randint(0, 10000000)
    
    test_env = []
    print(possible_tasks)

    for scene_id in range(num):
        seed = seeds[scene_id]
        np.random.seed(seed)
        random.seed(seed)
        floor_height = 0.02
        origin = p
        task = possible_tasks[scene_id % len(possible_tasks)]

        c = Controller(launch_build=True, port=8888)
        logger = Logger(path=os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}", "log.txt"))
        c.add_ons.append(logger)

        commands = [c.get_add_scene(scene_name=scene_name), {"$type": "set_screen_size", "width": 512, "height": 512}]

        # camera = ThirdPersonCamera(avatar_id="teaser", position={"x": 66, "y": 10, "z": -10}, look_at={"x": 40, "y": -10, "z": 10})
        avatar_id = "teaser"
        # avatar_pos = {"x": -50, "y": 20, "z": 0}
        # avatar_lookat = {"x": -50, "y": 0, "z": 0}
        avatar_pos = {"x": 27, "y": 5, "z": 0.7}
        avatar_lookat = {"x": 0, "y": 0, "z": 0.7}
        camera = ThirdPersonCamera(avatar_id=avatar_id, position=avatar_pos, look_at=avatar_lookat)
        
        c.add_ons.append(camera)
        c.communicate(commands)
        c.communicate({"$type": "set_field_of_view", "avatar_id": "teaser", "field_of_view" : 80})
        c.communicate([{"$type": "set_pass_masks",
                  "pass_masks": ["_img"],
                  "avatar_id": "teaser"},
                 {"$type": "send_images",
                  "frequency": "always",
                  "ids": ["teaser"]}])
        
        shelf_pos = []
        slot_pos = []
        
        for ps in tent_pos:
            pos = [ps[0], 0, ps[1]]
            idx = c.get_unique_id()
            object_name = "b01_tent"
            commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 90, 0]), object_id=idx, library="models_full.json")]
            commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
            commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([0.4, 1, 0.8])})
            c.communicate(commands)

            idx = c.get_unique_id()
            object_name = "4ft_wood_shelving"
            if ps[1] > 0:
                pos = [ps[0] - 0.7, 0, ps[1] - 0.5]
            else:
                pos = [ps[0] - 0.7, 0, ps[1] + 0.5]

            shelf_pos.append([])
            shelf_pos[-1].append(pos)
            commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 90, 0]), object_id=idx, library="models_full.json")]
            commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
            commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([1, 0.8, 1.2])})
            c.communicate(commands)

            idx = c.get_unique_id()
            object_name = "4ft_wood_shelving"
            if ps[1] > 0:
                pos = [ps[0] + 0.7, 0, ps[1] - 0.5]
            else:
                pos = [ps[0] + 0.7, 0, ps[1] + 0.5]

            commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 90, 0]), object_id=idx, library="models_full.json")]
            commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
            commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([1, 0.8, 1.2])})
            shelf_pos[-1].append(pos)
            c.communicate(commands)

        mass_map = {}
        store_idx = [i for i in range(len(shelf_pos))]
        have_box = random.sample(store_idx, num_containers)
        container_names = []
        for i in range(len(have_box)):
            while True:
                container = random.choice(container_list["box"])   
                container_name = container["name"]
                container_mass = container["mass"]
                if container_name not in container_names:
                    break

            pos = copy.copy(shelf_pos[have_box[i]][0])
            pos[1] += 0.99

            container_names.append(container_name)

            idx = c.get_unique_id()
            if "scale" not in container:
                scale = 1
            else:
                scale = container["scale"]

            commands = [get_add_object(model_name=container_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
            
            commands.append({"$type": "set_mass", "id": idx, "mass": container_mass})
            commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
            mass_map[idx] = container_mass

            c.communicate(commands)


        for i in range(len(shelf_pos)):
            slot_pos.append([])
            for j in range(len(shelf_pos[i])):
                pos = shelf_pos[i][j]
                if pos[2] < 0:
                    slot_pos[-1].append([pos[0], pos[1] + 0.33, pos[2]])
                    slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.33, pos[2]])  
                    slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.33, pos[2]])  
                    slot_pos[-1].append([pos[0], pos[1] + 0.66, pos[2]])
                    slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.66, pos[2]])  
                    slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.66, pos[2]]) 
                    if j != 0 or i not in have_box:
                        slot_pos[-1].append([pos[0], pos[1] + 0.99, pos[2]])
                        slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.99, pos[2]])  
                        slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.99, pos[2]])
                else: 
                    slot_pos[-1].append([pos[0], pos[1] + 0.33, pos[2]])
                    slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.33, pos[2]])  
                    slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.33, pos[2]])  
                    slot_pos[-1].append([pos[0], pos[1] + 0.66, pos[2]])
                    slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.66, pos[2]])  
                    slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.66, pos[2]]) 
                    if j != 0 or i not in have_box:
                        slot_pos[-1].append([pos[0], pos[1] + 0.99, pos[2]])
                        slot_pos[-1].append([pos[0] - 0.4, pos[1] + 0.99, pos[2]])  
                        slot_pos[-1].append([pos[0] + 0.4, pos[1] + 0.99, pos[2]])

        print(slot_pos)
        object_dict = {}        
        for i in range(len(slot_pos)):
            slots = slot_pos[i]
            random.shuffle(slots)
            print(slots)
            for name, num in stores[i].items():
                for j in range(num):
                    pos = slots.pop()
                    idx = c.get_unique_id()
                    for obj in object_list["object"]:
                        if obj["name"] == name:
                            object_description = obj
                            break

                    object_name = object_description["name"]
                    object_mass = object_description["mass"]
                    print(object_name, object_mass, pos)

                    commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
                    if "scale" not in object_description:
                        scale = 1
                    else:
                        scale = object_description["scale"]

                    commands.append({"$type": "set_mass", "id": idx, "mass": object_mass})
                    commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
                    mass_map[idx] = object_mass
                    c.communicate(commands)

                    if object_name not in object_dict:
                        object_dict[object_name] = 0
                    
                    object_dict[object_name] += 1

        idx = scene_id % 3
        task = []
        for name in categories[idx]:
            task.append([name, object_dict[name]])
        
        other_names = []
        for i in range(3):
            if i != idx:
                for name in categories[i]:
                    other_names.append(name)
        
        other_name = random.choice(other_names)
        task.append([other_name, object_dict[other_name]])

        # put containers 
        # container_names = []
        # for _ in range(num_containers):
        #     while True:
        #         container = random.choice(container_list["box"])   
        #         container_name = container["name"]
        #         container_mass = container["mass"]
        #         if container_name not in container_names:
        #             break

        #     pos = occ.find_free(6)
        #     if pos is None:
        #         continue

        #     container_names.append(container_name)

        #     idx = c.get_unique_id()
        #     if "scale" not in container:
        #         scale = 1
        #     else:
        #         scale = container["scale"]

        #     commands = [get_add_object(model_name=container_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
            
        #     commands.append({"$type": "set_mass", "id": idx, "mass": container_mass})
        #     commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
        #     mass_map[idx] = container_mass

        #     occ.generate(floor_height=floor_height, boundX=[-3 + origin[0], 3 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
        #     c.communicate(commands)

        #     for i in range(occ.grid.shape[0]):
        #         for j in range(occ.grid.shape[1]):
        #             if occ.grid[i][j] == 0:
        #                 print(".", end = "")
        #             elif occ.grid[i][j] == 1:
        #                 print("X", end = "")
        #             elif occ.grid[i][j] == 100:
        #                 print(" ", end = "")
        #         print()

        #     print()
        #     print()
        #     print()

        # all_objects = []    
        # names = []
        
        # for item in task:
        #     item_num = item[1]
        #     item_name = item[0]
        #     names.append(item_name)
        #     for obj in object_list["object"]:
        #         if obj["name"] == item_name:
        #             for _ in range(item_num):
        #                 all_objects.append(obj)

        # objects_left = [obj for obj in object_list["object"] if obj["name"] not in names]
        # num_left = num_objects - len(all_objects)
        # for _ in range(num_left):
        #     object_description = random.choice(objects_left)
        #     all_objects.append(object_description)
        
        # random.shuffle(all_objects)

        # for i in range(num_objects):
        #     object_description = all_objects[i]
        #     object_name = object_description["name"]
        #     object_mass = object_description["mass"]

        #     pos = occ.find_free(5)
                
        #     if pos is None:
        #         continue
        #     idx = c.get_unique_id()
        #     pos[1] += 0.3
        #     commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        #     if "scale" not in object_description:
        #         scale = 1
        #     else:
        #         scale = object_description["scale"]

        #     commands.append({"$type": "set_mass", "id": idx, "mass": object_mass})
        #     commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})

        #     mass_map[idx] = object_mass
        #     occ.generate(floor_height=floor_height, boundX=[-3 + origin[0], 3 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
        #     c.communicate(commands)

        #     for i in range(occ.grid.shape[0]):
        #         for j in range(occ.grid.shape[1]):
        #             if occ.grid[i][j] == 0:
        #                 print(".", end = "")
        #             elif occ.grid[i][j] == 1:
        #                 print("X", end = "")
        #             elif occ.grid[i][j] == 100:
        #                 print(" ", end = "")
        #         print()

        #     print()
        #     print()
        #     print()

        # border_pos = occ.get_border(floor_height=floor_height, boundX=[-3 + origin[0], 3 + origin[0]], boundZ=[-8 + origin[1], 8 + origin[1]])
        # for pos in border_pos:
        #     idx = c.get_unique_id()
        #     object_name = "b04_fireextinguisher"
        #     commands = [c.get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        #     scale = 2.5
        #     commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        #     commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
        #     c.communicate(commands)
        
        # agent
        lx = 10
        rx = 20
        lz = -1
        rz = 2.5
        while True:
            agent0_pos = [random.uniform(lx, rx), 0.1, random.uniform(lz, rz)]
            agent1_pos = [random.uniform(lx, rx), 0.1, random.uniform(lz, rz)]
            if l2_dist(agent0_pos, agent1_pos) > 5:
                break

        if agent0_pos is None or agent1_pos is None:
            print("Bad!")
            c.communicate({"$type": "terminate"})
            c.socket.close()
            
            
        pos = [2, 0, 5.5]
        object_name = "b03_fire_hydrant"
        idx = c.get_unique_id()
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 90, 0]), object_id=idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        # commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([1, 1, 1])})
        c.communicate(commands)

        pos = [agent0_pos[0] - 0.1, agent0_pos[1], agent0_pos[2] - 0.3]
        bike_idx = c.get_unique_id()

        object_name = "huffy_nel_lusso_womens_cruiser_bike_2011vray"
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 0, 0]), object_id=bike_idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": bike_idx, "is_kinematic": True})
        commands.append({"$type": "scale_object", "id": bike_idx, "scale_factor": TDWUtils.array_to_vector3([1.2, 1.2, 1.2])})
        c.communicate(commands)
        
        bag_idx = c.get_unique_id()
        object_name = "white_shopping_bag"
        bag_pos = copy.copy(pos)
        bag_pos[0] -= 0.3
        bag_pos[1] += 0.6
        bag_pos[2] -= 0.15
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(bag_pos), rotation=TDWUtils.array_to_vector3([0, 0, 0]), object_id=bag_idx, library="models_full.json")]
        # commands.append({"$type": "scale_object", "id": bag_idx, "scale_factor": TDWUtils.array_to_vector3([2, 2, 2])})
        commands.append({"$type": "parent_object_to_object", "parent_id": bike_idx, "id": bag_idx})
        commands.append({"$type": "set_kinematic_state", "id": bag_idx, "is_kinematic": True, "use_gravity": False})
        c.communicate(commands)

        bag_idx = c.get_unique_id()
        object_name = "white_shopping_bag"
        bag_pos = copy.copy(pos)
        bag_pos[0] -= 0.3
        bag_pos[1] += 0.6
        bag_pos[2] -= 0.225
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(bag_pos), rotation=TDWUtils.array_to_vector3([0, 0, 0]), object_id=bag_idx, library="models_full.json")]
        # commands.append({"$type": "scale_object", "id": bag_idx, "scale_factor": TDWUtils.array_to_vector3([2, 2, 2])})
        commands.append({"$type": "parent_object_to_object", "parent_id": bike_idx, "id": bag_idx})
        commands.append({"$type": "set_kinematic_state", "id": bag_idx, "is_kinematic": True, "use_gravity": False})
        c.communicate(commands)
        
        # grasp_pos = copy.copy(pos)
        # grasp_pos[0] -= 0.77
        # grasp_pos[1] += 1.5
        # grasp_pos[2] -= 0.3
        
        
        config = dict()
        config["agent"] = {"0": agent0_pos, "1": agent1_pos}
        config["avatar"] = {"id": avatar_id, "position": avatar_pos, "look_at": avatar_lookat}
        config["mass_map"] = mass_map
        config["scene_id"] = scene_id
        cur_task = dict()
        cur_task["goal_position_names"] = "fire hydrant"
        cur_task["goal_task"] = task
        cur_task["container_names"] = container_names
        cur_task["task_kind"] = "outdoor_shopping"
        cur_task["constraint_type"] = "riding"
        config["task"] = cur_task

        env = dict()
        env["scene"] = "2a"
        env["layout"] = f"0_{scene_id}"
        env["seed"] = seed
        env["task"] = cur_task
        test_env.append(env)
        config["bike_id"] = bike_idx

        data = c.communicate({"$type": "step_physics", "frames": 50})
        print("Done!")
        print(len(data))
        for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            print(r_id)
            if r_id == 'imag':
                images = Images(data[i])
                print(images.get_avatar_id())
                if images.get_avatar_id() == "teaser":
                    TDWUtils.save_images(images=images, filename= f"{scene_id}", output_directory = img_dir)
        
        c.communicate({"$type": "terminate"})
        c.socket.close()
        # print(c.commands)
        # with open(path, "w") as f:
        #     json.dump(c.commands, f, indent = 4)
        # while True:
        #     c.communicate({"$type": "step_physics", "frames": 1}, write = False)
        # c.communicate({"$type": "terminate"})
        # c.socket.close()
        
        # # clean up the log file
        playback = LogPlayback()
        playback.load(os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}", "log.txt"))
        
        logger.reset(path=os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}", "log.txt"))
        c = Controller(launch_build=True, port=8888)
        c.add_ons.append(logger)
        all_commands = []
        flag = False
        for a in playback.playback:
            commands = []
            for cc in a:
                tp = cc["$type"]
                if tp.find("send") != -1 or tp.find("terminate") != -1 and tp != "set_pass_masks" and tp != "send_images":
                    continue

                if tp == "look_at_position":
                    if flag:
                        continue
                
                    flag = True

                commands.append(cc)
            c.communicate(commands)
            all_commands.extend(commands)
            # print(commands)

        commands_json = os.path.join(file_dir, f"2a_0_{scene_id}.json")
        metadata_json = os.path.join(file_dir, f"2a_0_{scene_id}_metadata.json")
        with open(commands_json, "w") as f:
            json.dump(all_commands, f, indent = 4)

        with open(metadata_json, "w") as f:
            json.dump(config, f, indent = 4)

        c.communicate({"$type": "terminate"})
        c.socket.close()
    
    with open(os.path.join(file_dir, "test_env.json"), "w") as f:
        json.dump(test_env, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="train")
    args = parser.parse_args()
    load_config()

    scene_name = available_scenes[0]
    p = position_config[0]
    if args.type == "train":
        file_dir = os.path.join(PATH, "train")
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)

        os.mkdir(file_dir)
        img_dir = os.path.join(file_dir, "images")
        os.mkdir(img_dir)
        generate(scene_name, p, 2, file_dir, img_dir)
    elif args.type == "test":
        file_dir = os.path.join(PATH, "test")
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)

        os.mkdir(file_dir)
        img_dir = os.path.join(file_dir, "images")
        os.mkdir(img_dir)
        generate(scene_name, p, 2, file_dir, img_dir)
    else:
        file_dir = os.path.join(PATH, "debug")
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)

        os.mkdir(file_dir)
        img_dir = os.path.join(file_dir, "images")
        os.mkdir(img_dir)
        generate(scene_name, p, 1, file_dir, img_dir)