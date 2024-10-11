import numpy as np
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.arm import Arm
from PIL import Image
import os
from tdw.tdw_utils import TDWUtils
def pos_to_2d_box_distance(px, py, rx1, ry1, rx2, ry2):
    if px < rx1:
        if py < ry1:
            return ((px - rx1) ** 2 + (py - ry1) ** 2) ** 0.5
        elif py > ry2:
            return ((px - rx1) ** 2 + (py - ry2) ** 2) ** 0.5
        else:
            return rx1 - px
    elif px > rx2:
        if py < ry1:
            return ((px - rx2) ** 2 + (py - ry1) ** 2) ** 0.5
        elif py > ry2:
            return ((px - rx2) ** 2 + (py - ry2) ** 2) ** 0.5
        else:
            return px - rx2
    else:
        if py < ry1:
            return ry1 - py
        elif py > ry2:
            return py - ry2
        else:
            return 0
    
def belongs_to_which_room(pos, controller: TransportChallenge):
    min_dis = 100000
    room = None
    for i, region in enumerate(controller.scene_bounds.regions):
        distance = pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
        if distance < min_dis and controller.rooms_name[i] is not None:
            min_dis = distance
            room = controller.rooms_name[i]
    return room
    
def get_room_distance(pos, controller: TransportChallenge):
    min_dis = 100000
    room = None
    for i, region in enumerate(controller.scene_bounds.regions):
        distance = pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
        if distance < min_dis and controller.rooms_name[i] is not None:
            min_dis = distance
            room = controller.rooms_name[i]
    return min_dis

def get_room_distance_certain(pos, room, controller: TransportChallenge):
    for i, region in enumerate(controller.scene_bounds.regions):
        if controller.rooms_name[i] == room:
            distance = pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
    return distance
    
def center_of_room(self, room, controller: TransportChallenge):
    assert type(room) == str
    for index, name in controller.rooms_name.items():
        if name == room:
            room = index
    return controller.scene_bounds.regions[room].center
    
def check_pos_in_room(pos, controller: TransportChallenge):
    if len(pos) == 3:
        for region in controller.scene_bounds.regions:
            if region.is_inside(pos[0], pos[2]):
                return True
    elif len(pos) == 2:
        for region in controller.scene_bounds.regions:
            if region.is_inside(pos[0], pos[1]):
                return True
    return False

def map_status(status, buffer_len = 0):
    if status == ActionStatus.ongoing or buffer_len > 0:
        return 0
    elif status == ActionStatus.success or status == ActionStatus.still_dropping or status == ActionStatus.detected_obstacle or status == ActionStatus.collision:
        return 1
    else: return 2
    
def map_arms(arm: str):
    if arm == 'left':
        return Arm.left
    elif arm == 'right':
        return Arm.right
    else:
        raise ValueError("Arm type not supported!")

def get_2d_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1[[0, 2]]) - np.array(pos2[[0, 2]]))

def get_3d_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def save_images(save_dir='./Images', controller: TransportChallenge = None, screen_size = 256, step = 0, only_save_rgb = False):
    '''
    save images of current step
    '''
    os.makedirs(save_dir, exist_ok=True)
    for replicant_id in controller.replicants:
        if only_save_rgb and replicant_id != 1:
            continue
        save_path = os.path.join(save_dir, str(replicant_id))
        os.makedirs(save_path, exist_ok=True)
        img = controller.replicants[replicant_id].dynamic.get_pil_image('img')
        img.save(os.path.join(save_path, f'{step:04}.png'))
        if only_save_rgb:
            continue
        depth = np.array(TDWUtils.get_depth_values(controller.replicants[replicant_id].dynamic.get_pil_image('depth'), width = screen_size, height = screen_size), dtype = np.float32)
        depth_img = Image.fromarray(100 / depth).convert('RGB')
        depth_img.save(os.path.join(save_path, f'{step:04}_depth.png'))
        seg = controller.replicants[replicant_id].dynamic.get_pil_image('id')
        seg.save(os.path.join(save_path, f'{step:04}_seg.png'))