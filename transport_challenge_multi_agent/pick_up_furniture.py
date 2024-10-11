from enum import Enum
from typing import List
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.turn_to import TurnTo
from tdw.replicant.actions.grasp import Grasp
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction
from tdw.tdw_utils import TDWUtils
from tdw.replicant.actions.reach_for import ReachFor
from tdw.replicant.collision_detection import CollisionDetection
from tdw.output_data import OutputData, Transforms

def quaternion_to_yaw(quaternion):
    y = 2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])
    x = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
    yaw = np.arctan2(y, x)
    
    return np.rad2deg(yaw) % 360

class _PickUpState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """

    turning_to = 0
    reaching_for = 1
    grasping = 2
    parenting = 3
    lifting = 4


class PickUp(MultiAction):
    """
    A combination of `TurnTo` + `ReachFor` + `Grasp` + [`ResetArms`](reset_arms.md).
    """

    def __init__(self, target: int, reach_for_max_distance: float, state: ChallengeState, lift_pos: dict, replicant_name: str, highest_weight: int, object_weight: int = 10, replicant = None, no_weight_check: bool = False, no_grasp: bool = False, behaviour_data_gen: bool = False) -> None:
        """
        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) picking up the object.
        :param target: The object ID.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        self._target: int = target
        self.reach_for_max_distance = reach_for_max_distance
        self._pick_up_state: _PickUpState = _PickUpState.turning_to
        self._end_status: ActionStatus = ActionStatus.success
        self._replicant_name = replicant_name
        self._highest_weight = highest_weight
        self._object_weight = object_weight
        self._replicant = replicant
        self._no_weight_check = no_weight_check
        self._no_grasp = no_grasp
        self._lift_pos = lift_pos
        self._behaviour_data_gen = behaviour_data_gen

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """
        # with open("file.txt", "a") as f:
        #     f.write(f"turning to the object {self._target}\n")
            
        print("turning to the object", self._target)
        self._sub_action = TurnTo(target=self._target)
        # Is a Replicant already holding this object?
        if not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            print("cannot pick up object", self._target, "because it is too far or grasped by another agent")
            self._end_status = ActionStatus.cannot_grasp

        if Arm.left in dynamic.held_objects or Arm.right in dynamic.held_objects:
            print("cannot pick up object", self._target, "because it is already held")
            self._end_status = ActionStatus.cannot_grasp

        # Turn to face the object.
        self._image_frequency = image_frequency
        cmds = self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                image_frequency=image_frequency)
        
        # with open("file.txt", "a") as f:
        #     f.write(f"{cmds}\n")
        
        return cmds

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """
        # Check if another Replicant is holding the object.
        if self._pick_up_state != _PickUpState.lifting and self._pick_up_state != _PickUpState.parenting and \
            not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            self._end_status = ActionStatus.cannot_grasp
            cmds = self._reset(resp=resp, static=static, dynamic=dynamic)
            # with open("file.txt", "a") as f:
            #     f.write(f"{cmds}\n")
        
            return cmds
        # Continue an ongoing sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
            cmds = self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
            # with open("file.txt", "a") as f:
            #     f.write(f"{cmds}\n")
        
            return cmds
        elif self._sub_action.status == ActionStatus.success:
            # Reach for the object.
            if self._pick_up_state == _PickUpState.turning_to:
                # with open("file.txt", "a") as f:
                #     f.write(f"reach for object or position {self._target} with both arms\n")

                print("reach for object or position", self._target, "with both arms")
                cmds = self._reach_for(resp=resp, static=static, dynamic=dynamic)
                # with open("file.txt", "a") as f:
                #     f.write(f"{cmds}\n")
        
                return cmds
            # Grasp the object.
            elif self._pick_up_state == _PickUpState.reaching_for:
                if not self._no_weight_check:
                    if not self._behaviour_data_gen:
                        prob = np.exp(min(0, (self._highest_weight - self._object_weight) / 50.00))
                        print(self._highest_weight, self._object_weight, prob)
                    else:
                        prob = 0.5

                    st = np.random.choice([True, False], p = [prob, 1 - prob])
                    if not st:
                        # with open("file.txt", "a") as f:
                        #     f.write(f"cannot grasp object {self._target} because it is too heavy\n")
                        print("cannot grasp object", self._target, "because it is too heavy")
                        self._end_status = ActionStatus.cannot_grasp
                        cmds = self._reset(resp=resp, static=static, dynamic=dynamic)
                        # with open("file.txt", "a") as f:
                        #     f.write(f"{cmds}\n")
        
                        return cmds
                
                # with open("file.txt", "a") as f:
                #     f.write(f"grasp object or position {self._target} with both arms\n")

                print("grasp object or position", self._target, "with both arms")
                if not self._no_grasp:
                    cmds = self._grasp(resp=resp, static=static, dynamic=dynamic)
                    # with open("file.txt", "a") as f:
                    #     f.write(f"{cmds}\n")
        
                    return cmds
                else:
                    self._pick_up_state = _PickUpState.grasping
                    return []
                
            # Lift the arm.
            elif self._pick_up_state == _PickUpState.grasping:
                if not self._no_grasp:
                    # with open("file.txt", "a") as f:
                    #     f.write(f"parent the object\n")
                    print("parent the object")
                    cmds = self._parent(resp=resp, static=static, dynamic=dynamic)
                    # with open("file.txt", "a") as f:
                    #     f.write(f"{cmds}\n")
        
                    return cmds
                else:
                    self._pick_up_state = _PickUpState.parenting
                    self._replicant.collision_detection.exclude_objects.append(self._target)
                    return []
            elif self._pick_up_state == _PickUpState.parenting:
                # with open("file.txt", "a") as f:
                #     f.write(f"lift the object\n")
                print("lift the object")
                cmds = self._lift(resp=resp, static=static, dynamic=dynamic)
                # with open("file.txt", "a") as f:
                #     f.write(f"{cmds}\n")
        
                return cmds
            # We're done!
            elif self._pick_up_state == _PickUpState.lifting:
                if self._end_status == ActionStatus.success:
                    self._replicant.holding_furniture = True
                    self._replicant.holding_furniture_id = self._target
                    self._replicant.furniture_human_diff = np.array(self._get_object_position(self._target, resp)) - np.array(dynamic.transform.position)
                    self._replicant.human_rotation = quaternion_to_yaw(dynamic.transform.rotation)
                    self._replicant.furniture_rotation = quaternion_to_yaw(self._get_object_rotation(self._target, resp))
                
                self.status = self._end_status
                return []
            else:
                raise Exception(self._pick_up_state)
        # We failed.
        else:
            # Remember the fail status.
            # with open("file.txt", "a") as f:
            #     f.write(f"failed to lift the target {self._target} \n")
            self._end_status = self._sub_action.status
            cmds = self._reset(resp=resp, static=static, dynamic=dynamic)
            # with open("file.txt", "a") as f:
            #     f.write(f"{cmds}\n")
        
            return cmds
        
    def _get_object_rotation(self, object_id: int, resp: List[bytes]) -> np.ndarray:
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            if r_id == "tran":
                transforms = Transforms(resp[i])
                for j in range(transforms.get_num()):
                    if transforms.get_id(j) == object_id:
                        return np.array(transforms.get_rotation(j))
        
        return None

    def _can_pick_up(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> bool:
        """
        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: True if we can pick up this object. An object can't be picked up if it is already grasped or contained.
        """

        if type(self._target) == np.ndarray:
            object_position = np.array([self._target[0], 0, self._target[2]])
            if np.linalg.norm(dynamic.transform.position - object_position) > 3:
                return False
            else:
                return True

        
        if not self._no_grasp:
            # Is the object already grasped by other agents?
            for replicant_id in self._state.replicants:
                if replicant_id == static.replicant_id:
                    continue
                for arm in [Arm.left, Arm.right]:
                    if self._target == self._state.replicants[replicant_id][arm]:
                        return False
                    
        # Is the object too far away?
        bound = self._get_object_bounds(object_id=self._target, resp=resp)
        object_position = np.array([bound['center'][0], 0, bound['center'][2]])
        if np.linalg.norm(dynamic.transform.position - object_position) > 3:
            return False
        else:
            return True

    def _reach_for(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reach for the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._pick_up_state = _PickUpState.reaching_for
        # self._sub_action = ReachForTransportChallenge(target=self._target,
        #                                               arm=self._arm,
        #                                               dynamic=dynamic,
        #                                               max_distance=self.reach_for_max_distance,
        #                                               absolute=True)
        self._sub_action = ReachFor(targets = [self._target, self._target],
                                   arms = [Arm.left, Arm.right],
                                   absolute = False,
                                   dynamic = dynamic,
                                   collision_detection = CollisionDetection(objects=False, held=False),
                                   offhand_follows = False,
                                   arrived_at = 0.09,
                                   previous = None,
                                   duration = 0.25,
                                   scale_duration = True,
                                   max_distance = 1.5,
                                   from_held = False,
                                   held_point = "bottom")
        if self._end_status == ActionStatus.cannot_grasp:
            # only reach animation, no grasp animation
            self._pick_up_state = _PickUpState.grasping
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _grasp(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to grasp the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = Grasp(target=self._target,
                                 arm=Arm.right,
                                 dynamic=dynamic,
                                 angle=None,
                                 axis="yaw", #pitch",
                                 offset=0,
                                 relative_to_hand=False,
                                 kinematic_objects=[])#True)
        self._pick_up_state = _PickUpState.grasping
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _lift(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reset the arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ReachFor(targets = [self._lift_pos],
                                   arms = [Arm.right],
                                   absolute = False,
                                   dynamic = dynamic,
                                   collision_detection = CollisionDetection(objects=False, held=False),
                                   offhand_follows = True,
                                   arrived_at = 0.09,
                                   previous = None,
                                   duration = 0.25,
                                   scale_duration = True,
                                   max_distance = 1.5,
                                   from_held = False,
                                   held_point = "bottom")
        self._pick_up_state = _PickUpState.lifting
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
    
    def _reset(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reset the arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ResetArms(reach_for_max_distance = self.reach_for_max_distance, state=self._state, replicant_name = self._replicant_name)
        self._pick_up_state = _PickUpState.lifting
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
    
    def _parent(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        commands = []
        commands.extend([{"$type": "parent_object_to_object",
                          "parent_id": self._replicant.replicant_id,
                          "id": self._target}])
        self._pick_up_state = _PickUpState.parenting
        return commands
