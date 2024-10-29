from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.logger import Logger
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.agent_ability_info import HelperAbility, GirlAbility, WheelchairAbility
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.replicant.image_frequency import ImageFrequency
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
import transport_challenge_multi_agent.utils as utils
import os
import numpy as np
from tdw.output_data import OutputData, Transforms

import time
print("start", time.time())
controller = TransportChallenge(port=1077, check_version=True, launch_build=True, 
                            screen_width=1024, screen_height=1024, \
                            image_frequency= ImageFrequency.always, png=True, image_passes=None, \
                            enable_collision_detection = False, \
                            logger_dir = "demo/floorplan", replicants_name=['girl_casual'], replicants_ability = ['girl'])

scene_info = {
    "scene": "2a",
    "layout": "0_0",
    "seed": 0,
    "task_meta": {
        'goal_position_names': ['bed'], 
        'goal_task': [['b05_executive_pen', 2], ['mouse_02_vray', 2], ['vase_05', 2], ['b04_banana', 1], ['b05_calculator', 2], ['pencil_all', 2]], 
        'container_names': ['b04_bowl_smooth', 'plate06', 'teatray', 'basket_18inx18inx12iin_plastic_lattice', 'basket_18inx18inx12iin_wicker', 'basket_18inx18inx12iin_wood_mesh'], 
        'task_kind': 'highthing', 
        'constraint_type': 'high', 
        'possible_object_names': ['bread', 'b03_burger', 'b03_loafbread', 'apple', 'b04_banana', 'b04_orange_00', 'b05_calculator', 'mouse_02_vray', 'b05_executive_pen', 'b04_lighter', 'small_purse', 'f10_apple_iphone_4', 'apple_ipod_touch_yellow_vray', 'pencil_all', 'key_brass'], 
        'goal_object_names': ['b05_executive_pen', 'mouse_02_vray', 'vase_05', 'b04_banana', 'b05_calculator', 'pencil_all']
    },
    "data_prefix": "dataset/test_dataset/highthing",
    "object_property": {'vase_05': 'target'}

}

# increase x: right
# increase z: up
# 90 degree: turn right
controller.start_floorplan_trial(scene=scene_info['scene'], layout=scene_info['layout'], replicants= 1,
                                  random_seed=scene_info['seed'], task_meta = scene_info['task_meta'], 
                                  data_prefix = scene_info['data_prefix'], object_property = scene_info['object_property'],
                                  replicant_init_position = [{"x": 4.5, "y": 0, "z": -2}], replicant_init_rotation = [{"x": 0, "y": 60, "z": 0}])


object_id = 123456789
data = controller.communicate(Controller.get_add_physics_object(model_name="b04_orange_00",
                                                  position={"x": 6.5, "y": 1.75, "z": -0.60},
                                                  rotation={"x": 0,
                                                    "y": 234,
                                                    "z": 0},
                                                  scale_factor={"x": 1,
                                                        "y": 1,
                                                        "z": 1},
                                                  object_id=123456789,
                                                  kinematic=False))

camera = ThirdPersonCamera(position={
            "x": 5,
            "y": 8,
            "z": -7
        }, avatar_id="demo", look_at={
            "x": 5,
            "y": -8,
            "z": 11
        })
controller.add_ons.extend([camera])

# need to reset the object manager to get the object position after adding the object
controller.object_manager.reset()
data = controller.communicate([])

for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            if r_id == "tran":
                transforms = Transforms(data[i])
                for j in range(transforms.get_num()):
                    if transforms.get_id(j) == object_id:
                        print("object position:", transforms.get_position(j))

print("object_id:", object_id)
for replicant_id in controller.replicants:
    controller.communicate({"$type": "set_field_of_view", "avatar_id": str(replicant_id), "field_of_view": 90})

action_buffer = []
action_buffer.append({'type': 'reach_for', 'object': object_id})
action_buffer.append({'type': 'pick_up', 'arm': Arm.left, 'object': object_id})

save_dir = "demo/floorplan"
os.makedirs(save_dir, exist_ok=True)
num_frames = 0
finish = False

while not finish: # continue until any agent's action finishes
    replicant = controller.replicants[0]
    if replicant.action.status != ActionStatus.ongoing and len(action_buffer) == 0:
        finish = True
    elif replicant.action.status != ActionStatus.ongoing:
        print("num_frames:", num_frames, replicant.action.status)
        curr_action = action_buffer.pop(0)
        print("curr_action:", curr_action)
        if curr_action['type'] == 'move_forward':
            # move forward 0.5m
            replicant.move_forward()
        elif curr_action['type'] == 'turn_left':
            # turn left by 15 degree
            replicant.turn_by(angle = -15)
        elif curr_action['type'] == 'turn_right':
            # turn right by 15 degree
            replicant.turn_by(angle = 15)
        elif curr_action['type'] == 'reach_for':
            # go to and pick_up object with arm
            replicant.move_to_object(int(curr_action["object"]))
        elif curr_action['type'] == 'pick_up':
            replicant.pick_up(int(curr_action["object"]), curr_action["arm"])
        elif curr_action["type"] == 'put_in':
            # put in container
            replicant.put_in()
        elif curr_action["type"] == 'put_on':
            replicant.put_on(target = curr_action['target'], arm = curr_action['arm'])
    if finish: break
    time.sleep(0.1)
    data = controller.communicate([])
    for i in range(len(data) - 1):
        r_id = OutputData.get_data_type_id(data[i])
        if r_id == 'imag':
            images = Images(data[i])
            if images.get_avatar_id() == "a":
                TDWUtils.save_images(images=images, filename=str(num_frames), output_directory=os.path.join(save_dir, 'top_down_image'))
            if images.get_avatar_id() == "demo":
                TDWUtils.save_images(images=images, filename=f"demo_{num_frames}", output_directory=os.path.join(save_dir, 'top_down_image'))
    utils.save_images(os.path.join(save_dir, "Images"), controller=controller, screen_size=1024, step=num_frames, only_save_rgb=False)
    num_frames += 1
print("num_frames:", num_frames, replicant.action.status)

controller.communicate({"$type": "terminate"})
controller.socket.close()