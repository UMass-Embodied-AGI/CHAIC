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

import time
print("start", time.time())
controller = TransportChallenge(port=1077, check_version=True, launch_build=True, 
                            screen_width=512, screen_height=512, \
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


controller.start_floorplan_trial(scene=scene_info['scene'], layout=scene_info['layout'], replicants= 1,
                                      random_seed=scene_info['seed'], task_meta = scene_info['task_meta'], data_prefix = scene_info['data_prefix'], object_property = scene_info['object_property'])

controller.communicate({"$type": "terminate"})
controller.socket.close()