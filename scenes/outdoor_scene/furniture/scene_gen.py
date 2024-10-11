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
        
global object_list, possible_tasks
def load_config():
    global object_list, possible_tasks
    object_json = os.path.join(PATH, "configs/object_list.json")
    possible_task_config = os.path.join(PATH, "configs/possible_tasks.json")
    with open(object_json, "r") as f:
        object_list = json.load(f)
    with open(possible_task_config, "r") as f:
        possible_tasks = json.load(f)

    possible_tasks = possible_tasks["possible_task"]


# go to parent directory until the folder name is HAZARD
# while os.path.basename(PATH) != "HAZARD":
#     PATH = os.path.dirname(PATH)

available_scenes = ["suburb_scene_2023"]


"""
55 0
-58 -9
58 -51
-61 55"""
position_config = [
    [58, -6.5],
    [2, 0.7],
    [55, 0],
    [-58, -9],
    [58, -51],
    [-61, 55],
]

def generate(scene_name, p, times, file_dir, img_dir):
    num = len(possible_tasks) * times
    if "train" in file_dir:
        np.random.seed(39842892)
    elif "test" in file_dir:
        np.random.seed(30923211)
    else:
        num = 1
        np.random.seed(39842892)

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
        if "train" not in file_dir and "test" not in file_dir:
            task = [["arflex_strips_sofa", 1]]

        print(task)

        c = Controller(launch_build=True, port=8888)
        logger = Logger(path=os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}", "log.txt"))
        occ = BoxCastOccupancyMap()
        occ.generate(floor_height=floor_height, boundX=[-1.5 + origin[0], 1.5 + origin[0]], boundZ=[-3 + origin[1], 3 + origin[1]])
        c.add_ons.append(logger)
        c.add_ons.append(occ)

        commands = [c.get_add_scene(scene_name=scene_name), {"$type": "set_screen_size", "width": 512, "height": 512}]

        # camera = ThirdPersonCamera(avatar_id="teaser", position={"x": 66, "y": 10, "z": -10}, look_at={"x": 40, "y": -10, "z": 10})
        avatar_id = "teaser"
        # avatar_pos = {"x": -3, "y": 8, "z": -3}
        # avatar_lookat = {"x": 15, "y": -8, "z": 10}
        avatar_pos = {"x": 63, "y": 6, "z": -17}
        avatar_lookat = {"x": 53, "y": -6, "z": 3}
        # avatar_pos = {"x": 50, "y": 50, "z": 0}
        # avatar_lookat = {"x": 50, "y": -6, "z": 0}
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
        # occ.generate(floor_height=floor_height)
        # c.communicate([c.get_add_object(model_name="b04_shoppping_cart", position=TDWUtils.array_to_vector3([0, 3, 0]), object_id=1, library="models_full.json")])

        # fout = open("grid.txt", "w")
        # for i in range(occ.grid.shape[0]):
        #     for j in range(occ.grid.shape[1]):
        #         if occ.grid[i][j] == 0:
        #         elif occ.grid[i][j] == 1:
        #             fout.write("X")
        #         elif occ.grid[i][j] == 100:
        #             fout.write(" ")
        #     fout.write("\n")
        # fout.close()
        
        # input()
        # c.communicate({"$type": "terminate"})
        # c.socket.close()
        # return

        # put containers
        mass_map = {}
        all_objects = []    
        names = []
        
        for item in task:
            item_num = item[1]
            item_name = item[0]
            names.append(item_name)
            for obj in object_list["object"]:
                if obj["name"] == item_name:
                    for _ in range(item_num):
                        all_objects.append(obj)

        random.shuffle(all_objects)
        print(all_objects)

        for i in range(len(all_objects)):
            object_description = all_objects[i]
            object_name = object_description["name"]
            object_mass = object_description["mass"]

            pos = occ.find_free(5)
            print(pos)    
            if pos is None:
                continue
            idx = c.get_unique_id()
            pos[1] += 0.3
            commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
            if "scale" not in object_description:
                scale = 1
            else:
                scale = object_description["scale"]

            commands.append({"$type": "set_mass", "id": idx, "mass": object_mass})
            commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})

            mass_map[idx] = object_mass
            occ.generate(floor_height=floor_height, boundX=[-1.5 + origin[0], 1.5 + origin[0]], boundZ=[-3 + origin[1], 3 + origin[1]])
            c.communicate(commands)

            # for i in range(occ.grid.shape[0]):
            #     for j in range(occ.grid.shape[1]):
            #         if occ.grid[i][j] == 0:
            #             print(".", end = "")
            #         elif occ.grid[i][j] == 1:
            #             print("X", end = "")
            #         elif occ.grid[i][j] == 100:
            #             print(" ", end = "")
            #     print()

            # print()
            # print()
            # print()

        pos = [58, 0, -17]
        idx = c.get_unique_id()
        object_name = "b06_dump"
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 180, 0]), object_id=idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([0.9, 0.5, 1])})
        c.communicate(commands)

        pos = [60, 0, 6]
        idx = c.get_unique_id()
        object_name = "traffic_lights"
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 180, 0]), object_id=idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        c.communicate(commands)
        
        pos = [55, 0, -3]
        idx = c.get_unique_id()
        object_name = "traffic_lights"
        commands = [get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, 0, 0]), object_id=idx, library="models_full.json")]
        commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        c.communicate(commands)

        # border_pos = occ.get_border(floor_height=floor_height, boundX=[55, 61], boundZ=[-14, -4])
        # for pos in border_pos:
        #     idx = c.get_unique_id()
        #     object_name = "b04_fireextinguisher"
        #     commands = [c.get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        #     scale = 2.5
        #     commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        #     commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
        #     c.communicate(commands)
        
        # agent
        # idx = c.get_unique_id()
        # object_name = "b04_fireextinguisher"
        # pos = [8.2, 0, 0]
        # commands = [c.get_add_object(model_name=object_name, position=TDWUtils.array_to_vector3(pos), rotation=TDWUtils.array_to_vector3([0, random.random() * 360, 0]), object_id=idx, library="models_full.json")]
        # scale = 1
        # commands.append({"$type": "set_kinematic_state", "id": idx, "is_kinematic": True})
        # commands.append({"$type": "scale_object", "id": idx, "scale_factor": TDWUtils.array_to_vector3([scale, scale, scale])})
        # c.communicate(commands)

        agent0_pos = occ.find_free(1)
        agent1_pos = occ.find_free(1)
        agent2_pos = occ.find_free(1)
        agent0_pos[1] = 0.1
        agent1_pos[1] = 0.1
        agent2_pos[1] = 0.1

        if agent0_pos is None or agent1_pos is None:
            print("Bad!")
            c.communicate({"$type": "terminate"})
            c.socket.close()
        
        
        config = dict()
        config["agent"] = {"0": agent0_pos, "1": agent1_pos, "2": agent2_pos}
        config["avatar"] = {"id": avatar_id, "position": avatar_pos, "look_at": avatar_lookat}
        config["mass_map"] = mass_map
        config["scene_id"] = scene_id
        config["agent_weight_constraint"] = {"0": 100, "1": 600} 
        cur_task = dict()
        cur_task["goal_position_names"] = "truck"
        cur_task["goal_task"] = task
        cur_task["container_names"] = []
        cur_task["task_kind"] = "outdoor_furniture_crossway"
        cur_task["constraint_type"] = "weight"
        config["task"] = cur_task

        env = dict()
        env["scene"] = "2a"
        env["layout"] = f"0_{scene_id}"
        env["seed"] = seed
        env["task"] = cur_task
        test_env.append(env)
        # config["containers"] = containers
        # rad = random.random() * 2 * np.pi
        # magnitude = random.random() * 3 + 3
        # config["wind"] = [np.cos(rad) * magnitude, 0, np.sin(rad) * magnitude]
        # config["wind_resistence"] = wind_resistence
        # config["task"] = "wind"
        # config["containers"] = containers
        # config["agent"] = agent_pos
        # config["targets"] = list(wind_resistence.keys())
        # rad = random.random() * 2 * np.pi
        # magnitude = random.random() * 5 + 5
        # config["other"] = dict(
        #     wind_resistence=wind_resistence,
        #     wind=[np.cos(rad) * magnitude, 0, np.sin(rad) * magnitude]
        # )

        # info_path = os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}")
        # os.makedirs(info_path, exist_ok=True)
        # info = open(os.path.join(PATH, "data", "suburb_scene", f"{scene_name}-{p}-{seed}", "info.json"), "w")
        # json.dump(config, info)
        # info.close()
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
        generate(scene_name, p, 1, file_dir, img_dir)
    elif args.type == "test":
        file_dir = os.path.join(PATH, "test")
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)

        os.mkdir(file_dir)
        img_dir = os.path.join(file_dir, "images")
        os.mkdir(img_dir)
        generate(scene_name, p, 1, file_dir, img_dir)
    else:
        file_dir = os.path.join(PATH, "debug")
        if os.path.exists(file_dir):
            shutil.rmtree(file_dir)

        os.mkdir(file_dir)
        img_dir = os.path.join(file_dir, "images")
        os.mkdir(img_dir)
        generate(scene_name, p, 1, file_dir, img_dir)