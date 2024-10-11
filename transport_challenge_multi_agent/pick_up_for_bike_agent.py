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
from tdw.replicant.actions.reset_arm import ResetArm
from transport_challenge_multi_agent.multi_action import MultiAction
from tdw.tdw_utils import TDWUtils
from tdw.replicant.collision_detection import CollisionDetection
from transport_challenge_multi_agent.globals import Globals
from tdw.replicant.actions.drop import Drop


class _PickUpState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """
    stopping_bike = 0
    turning_to = 1
    reaching_for = 2
    grasping = 3
    move_to_basket = 4 
    dropping = 5
    resetting = 6

def quaternion_to_yaw(quaternion):
    y = 2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])
    x = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
    yaw = np.arctan2(y, x)
    
    return np.rad2deg(yaw) % 360

class PickUp(MultiAction):
    """
    A combination of `TurnTo` + `ReachFor` + `Grasp` + [`ResetArms`](reset_arms.md).
    """

    def __init__(self, target: int, reach_for_max_distance: float, state: ChallengeState, replicant_name: str, holding_bike_id: int, human_left_hand_position: np.array, human_right_hand_position: np.array, bike_basket_position_diff: np.array, object_manager):
        """
        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) picking up the object.
        :param target: The object ID.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        self._arm: Arm = Arm.right
        self._target: int = target
        self.reach_for_max_distance = reach_for_max_distance
        self._pick_up_state: _PickUpState = _PickUpState.stopping_bike
        self._end_status: ActionStatus = ActionStatus.success
        self._replicant_name = replicant_name
        self._holding_bike_id = holding_bike_id
        self._human_left_hand_position = human_left_hand_position
        self._human_right_hand_position = human_right_hand_position
        self._bike_basket_position_diff = bike_basket_position_diff
        self._object_manager = object_manager

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        self._sub_action = None
        self._counter = 50
        # Is a Replicant already holding this object?
        if not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            print("cannot pick up object", self._target, "with arm", self._arm, "because it is too far or grasped by another agent")
            self._end_status = ActionStatus.cannot_grasp

        self._image_frequency = image_frequency
        return []

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """
        # Check if another Replicant is holding the object.
        if self._pick_up_state != _PickUpState.resetting and self._pick_up_state != _PickUpState.dropping and  \
            self._pick_up_state != _PickUpState.move_to_basket and not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            self._end_status = ActionStatus.cannot_grasp
            return self._reset(resp=resp, static=static, dynamic=dynamic)
        # Continue an ongoing sub-action.
        if self._sub_action is not None and self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action is None or self._sub_action.status == ActionStatus.success:
            if self._pick_up_state == _PickUpState.stopping_bike:
                self._counter -= 1
                if self._counter <= 0:
                    # Turn to the object.
                    print("Turn to the object")
                    return self._turn_to(resp=resp, static=static, dynamic=dynamic)
                else:
                    return []
            # Reach for the object.
            elif self._pick_up_state == _PickUpState.turning_to:
                print("reach for object or position", self._target, "with arm", self._arm)
                return self._reach_for(resp=resp, static=static, dynamic=dynamic)
            # Grasp the object.
            elif self._pick_up_state == _PickUpState.reaching_for:
                print("grasp object or position", self._target, "with arm", self._arm)
                return self._grasp(resp=resp, static=static, dynamic=dynamic)
            # Reset the arm.
            elif self._pick_up_state == _PickUpState.grasping:
                print("move the hand to the basket")
                return self._move_to_basket(resp=resp, static=static, dynamic=dynamic)
            # We're done!
            elif self._pick_up_state == _PickUpState.move_to_basket:
                print("dropping the object into the basket")
                return self._drop(resp=resp, static=static, dynamic=dynamic)
            elif self._pick_up_state == _PickUpState.dropping:
                print("resetting the right arm to the bike")
                return self._reset(resp=resp, static=static, dynamic=dynamic)
            elif self._pick_up_state == _PickUpState.resetting:
                self.status = self._end_status
                return []
            else:               
                raise Exception(self._pick_up_state)
        # We failed.
        else:
            # Remember the fail status.
            self._end_status = self._sub_action.status
            return self._reset(resp=resp, static=static, dynamic=dynamic)

    def _turn_to(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:

        self._sub_action = TurnTo(target=self._target)
        self._pick_up_state = _PickUpState.turning_to
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)

    def _can_pick_up(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> bool:
        """
        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: True if we can pick up this object. An object can't be picked up if it is already grasped or contained.
        """

        if type(self._target) == np.ndarray:
            object_position = np.array([self._target[0], 0, self._target[2]])
            if np.linalg.norm(dynamic.transform.position - object_position) > 2:
                return False
            else:
                return True

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
        if np.linalg.norm(dynamic.transform.position - object_position) > 2:
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
        self._sub_action = ReachForTransportChallenge(target=self._target,
                                                      arm=self._arm,
                                                      dynamic=dynamic,
                                                      max_distance=self.reach_for_max_distance,
                                                      absolute=True)
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
                                 arm=self._arm,
                                 dynamic=dynamic,
                                 angle=0,
                                 axis="yaw", #pitch",
                                 offset=0,
                                 relative_to_hand=False,
                                 kinematic_objects=[])#True)
        self._pick_up_state = _PickUpState.grasping
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
    
    def _move_to_basket(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        bike_rotation = quaternion_to_yaw(self._object_manager.transforms[self._holding_bike_id].rotation)
        dx = self._bike_basket_position_diff[0] * np.cos(np.deg2rad(bike_rotation)) + self._bike_basket_position_diff[2] * np.sin(np.deg2rad(bike_rotation))
        dz = self._bike_basket_position_diff[2] * np.cos(np.deg2rad(bike_rotation)) - self._bike_basket_position_diff[0] * np.sin(np.deg2rad(bike_rotation))
        bike_basket_position = self._get_object_position(resp=resp, object_id=self._holding_bike_id) + np.array([dx, self._bike_basket_position_diff[1], dz])
        target_position = TDWUtils.array_to_vector3(bike_basket_position)
        target_position["y"] += 0.4
        self._sub_action = ReachForTransportChallenge(target = target_position,
                                                      arm = Arm.right,
                                                      dynamic = dynamic,
                                                      max_distance = self.reach_for_max_distance,
                                                      absolute = True)
        self._pick_up_state = _PickUpState.move_to_basket
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
        
    def _drop(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        bike_rotation = quaternion_to_yaw(self._object_manager.transforms[self._holding_bike_id].rotation)
        dx = self._bike_basket_position_diff[0] * np.cos(np.deg2rad(bike_rotation)) + self._bike_basket_position_diff[2] * np.sin(np.deg2rad(bike_rotation))
        dz = self._bike_basket_position_diff[2] * np.cos(np.deg2rad(bike_rotation)) - self._bike_basket_position_diff[0] * np.sin(np.deg2rad(bike_rotation))
        bike_basket_position = self._get_object_position(resp=resp, object_id=self._holding_bike_id) + np.array([dx, self._bike_basket_position_diff[1], dz])
        self._sub_action = Drop(arm = Arm.right,
                                dynamic = dynamic,
                                max_num_frames = 25,
                                offset = 0.1)
        self._pick_up_state = _PickUpState.dropping
        commands = self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
        commands.append({"$type": "teleport_object",
                        "id": self._target,
                        "position": TDWUtils.array_to_vector3(bike_basket_position)})
        # Parent the target object and make it kinematic.
        commands.extend([{"$type": "parent_object_to_object",
                          "parent_id": self._holding_bike_id,
                          "id": self._target},
                         {"$type": "set_kinematic_state",
                          "id": self._target,
                          "is_kinematic": True,
                          "use_gravity": False}])
        self._sub_action.status = ActionStatus.success
        return commands
    
    def _parenting_object_to_bike(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        self._pick_up_state = _PickUpState.parenting_object_to_bike
        commands = []
        bike_rotation = quaternion_to_yaw(self._object_manager.transforms[self._holding_bike_id].rotation)
        dx = self._bike_basket_position_diff[0] * np.cos(np.deg2rad(bike_rotation)) + self._bike_basket_position_diff[2] * np.sin(np.deg2rad(bike_rotation))
        dz = self._bike_basket_position_diff[2] * np.cos(np.deg2rad(bike_rotation)) - self._bike_basket_position_diff[0] * np.sin(np.deg2rad(bike_rotation))
        bike_basket_position = self._get_object_position(resp=resp, object_id=self._holding_bike_id) + np.array([dx, self._bike_basket_position_diff[1], dz])
        commands.append({"$type": "teleport_object",
                        "id": self._target,
                        "position": TDWUtils.array_to_vector3(bike_basket_position)})
        # Parent the target object and make it kinematic.
        commands.extend([{"$type": "parent_object_to_object",
                          "parent_id": self._holding_bike_id,
                          "id": self._target},
                         {"$type": "set_kinematic_state",
                          "id": self._target,
                          "is_kinematic": True,
                          "use_gravity": False}])
        return commands

    def _reset(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reset the arm.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """
        bike_rotation = quaternion_to_yaw(self._object_manager.transforms[self._holding_bike_id].rotation)
        dx = self._human_right_hand_position[0] * np.cos(np.deg2rad(bike_rotation)) + self._human_right_hand_position[2] * np.sin(np.deg2rad(bike_rotation))
        dz = self._human_right_hand_position[2] * np.cos(np.deg2rad(bike_rotation)) - self._human_right_hand_position[0] * np.sin(np.deg2rad(bike_rotation))
        
        bike_hold_position = self._get_object_position(resp=resp, object_id=self._holding_bike_id) + np.array([dx, self._human_right_hand_position[1], dz])
        self._sub_action = ReachForTransportChallenge(target = bike_hold_position,
                                                      arm = Arm.right,
                                                      dynamic = dynamic,
                                                      max_distance = self.reach_for_max_distance,
                                                      absolute = True)
        self._pick_up_state = _PickUpState.resetting
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
