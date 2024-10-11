from tdw.add_ons.logger import Logger
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.add_on import AddOn
from tdw.output_data import OutputData, Raycast
from tdw.add_ons.log_playback import LogPlayback
from typing import Union
import random
import numpy as np
import os
from typing import List, Optional
import cv2

import json
    
if __name__ == "__main__":
    c = Controller(launch_build=True, port=8888)
    commands = [c.get_add_scene(scene_name="suburb_scene_2023"), {"$type": "set_screen_size", "width": 1024, "height": 1024}]
    c.communicate(commands)
    cell_size = 0.5
    x_min = -100
    x_max = 100
    z_min = -50
    z_max = 50
    x_len = int((x_max - x_min) / cell_size)
    z_len = int((z_max - z_min) / cell_size)
    grid = np.zeros((x_len, z_len, 3), dtype=int)
    commands = []
    for i in range(x_len):
        for j in range(z_len):
            start = np.array([x_min + i * cell_size, 20, z_min + j * cell_size])
            end = np.array([x_min + i * cell_size, -20, z_min + j * cell_size])
            commands.append({"$type": "send_boxcast",
                                      "half_extents": {"x": cell_size / 2, "y": 0, "z": cell_size / 2},
                                      "origin": TDWUtils.array_to_vector3(start),
                                      "destination": TDWUtils.array_to_vector3(end),
                                      "id": i * z_len + j + 142857})
    
    resp = c.communicate(commands)
    height = np.zeros((x_len, z_len))
    for i in range(len(resp) - 1):
        r_id = OutputData.get_data_type_id(resp[i])
        if r_id == "rayc":
            rayc = Raycast(resp[i])
            idx = rayc.get_raycast_id()
            if idx >= 142857 and idx < 142857 + x_len * z_len:
                idx -= 142857
                if rayc.get_hit():
                    hit_x = rayc.get_point()[0]
                    hit_y = rayc.get_point()[1]
                    hit_z = rayc.get_point()[2]
                    height[idx // z_len][idx % z_len] = max(height[idx // z_len][idx % z_len], hit_y)
                else:
                    height[idx // z_len][idx % z_len] = 1

    for i in range(x_len):
        for j in range(z_len):
            if height[i][j] < 0:
                height[i][j] = 0
            if height[i][j] > 0.3:
                height[i][j] = 0.3
            color = int(255 * height[i][j] / 0.3)
            grid[i][j] = [color, color, color]
    
    grid = np.rot90(grid, 1)
    cv2.imwrite("map.png", grid)
    with open("height.txt", "w") as f:
        for i in range(x_len):
            for j in range(z_len):
                f.write(str(height[i][j]))
                f.write(" ")
            f.write("\n")

    c.communicate({"$type": "terminate"})
    c.socket.close()

