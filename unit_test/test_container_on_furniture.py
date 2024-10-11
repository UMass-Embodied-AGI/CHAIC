from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.logger import Logger
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.agent_ability_info import HelperAbility, GirlAbility
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from os import chdir
from subprocess import call
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.container_manager import ContainerManager
from tdw.add_ons.object_manager import ObjectManager
import os
import shutil
import time
print("start", time.time())
c = Controller(port=1077)
#logger = Logger('log.txt')
#c.add_ons.extend([logger])
camera = ThirdPersonCamera(position={"x": 0, "y": 6, "z": 0}, avatar_id="a", look_at={"x": 0, "y": -6, "z": 0}, field_of_view=90)
c.add_ons.extend([camera])
state = ChallengeState()
commands = [TDWUtils.create_empty_room(12, 12),
            {"$type": "set_screen_size",
             "width": 512,
             "height": 512}]
object_ids = [100, 200, 300]

commands.extend(Controller.get_add_physics_object(model_name="vase_05",
                                                  position={"x": -1, "y": 0.15, "z": -0.5},
                                                  scale_factor={"x": 1,
                                                        "y": 1,
                                                        "z": 1},
                                                  object_id=object_ids[0],
                                                  kinematic=False))

commands.extend(Controller.get_add_physics_object(model_name="de_castelli_placas_table_low",
                                                  position={
                                                    "x": 2,
                                                    "y": 0,
                                                    "z": 2
                                                },
                                                  scale_factor={"x": 0.7,
                                                        "y": 5.5,
                                                        "z": 0.7},
                                                  object_id=12345,
                                                  kinematic=True))

commands.extend(Controller.get_add_physics_object(model_name="plate06",
                                                  position={"x": 2, "y": 2, "z": 2},
                                                  scale_factor={"x": 1,
                                                        "y": 1,
                                                        "z": 1},
                                                  object_id=object_ids[1],
                                                  kinematic=False))

commands.extend(Controller.get_add_physics_object(model_name="vase_05",
                                                  position={"x": -1, "y": 0, "z": -1},
                                                  scale_factor={"x": 1,
                                                        "y": 1,
                                                        "z": 1},
                                                  object_id=object_ids[2],
                                                  kinematic=False))

state.target_object_ids += object_ids
state.possible_target_object_ids += object_ids
state.vase_ids = object_ids[0: 3]
container_id = Controller.get_unique_id()
commands.extend(Controller.get_add_physics_object(model_name="plate06",
                                                  position={"x": 0, "y": 0.2, "z": 0.5},
                                                  scale_factor={"x": 1.25,
                                                        "y": 1.25,
                                                        "z": 1.25},
                                                  object_id=container_id,
                                                  kinematic=False))
state.container_ids.append(container_id)
replicant = ReplicantTransportChallenge(replicant_id=0,
                                            state=state,
                                            position={"x": 1, "y": 0, "z": 0},
                                            enable_collision_detection=False,
                                            image_frequency=ImageFrequency.always)
for x in object_ids:
    replicant.collision_detection.exclude_objects.append(x)
for x in state.container_ids:
    replicant.collision_detection.exclude_objects.append(x)
c.add_ons.extend([replicant])
c.communicate(commands)
c.communicate([])
c.communicate({"$type": "set_field_of_view",
                          "avatar_id" : '0', "field_of_view" : 90})
action_buffer = []
action_buffer.append({'type': 'reach_for', 'object': object_ids[1]})
action_buffer.append({'type': 'pick_up', 'arm': Arm.right, 'object': object_ids[1]})
action_buffer.append({'type': 'reach_for', 'object': object_ids[2]})
num_frames = 0
finish = False

folder_name = "demo_container_on_furniture"

os.makedirs(folder_name, exist_ok=True)
shutil.rmtree(folder_name)
os.makedirs(folder_name, exist_ok=True)

container_manager = ContainerManager()
c.add_ons.append(container_manager)

while not finish: # continue until any agent's action finishes
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
            replicant.move_to_object(int(curr_action["object"]), set_max_distance = 10)
        elif curr_action['type'] == 'pick_up':
            replicant.pick_up(int(curr_action["object"]), curr_action["arm"])
        elif curr_action["type"] == 'put_in':
            # put in container
            replicant.put_in()
        elif curr_action["type"] == 'put_on':
            replicant.put_on(target = curr_action['target'], arm = curr_action['arm'])
    if finish: break
    time.sleep(0.1)
    data = c.communicate([])
    for i in range(len(data) - 1):
        r_id = OutputData.get_data_type_id(data[i])
        if r_id == 'imag':
            images = Images(data[i])
            TDWUtils.save_images(images=images, filename= f"{num_frames:04}", output_directory = f'{folder_name}/images_{images.get_avatar_id()}')
    num_frames += 1
print("num_frames:", num_frames, replicant.action.status)
c.communicate({"$type": "terminate"})