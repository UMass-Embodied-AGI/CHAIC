from enum import Enum
from typing import List, Optional
import numpy as np
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.arm import Arm
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.drop import Drop
from typing import Dict
from tdw.tdw_utils import TDWUtils
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.reach_for_with_plan_transport_challenge import ReachForWithPlanTransportChallenge
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.replicant_target_position import ReplicantTargetPosition
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction
from tdw.replicant.ik_plans.ik_plan_type import IkPlanType
from tdw.replicant.actions.turn_to import TurnTo
from tdw.replicant.actions.grasp import Grasp


class _PutOnState(Enum):
    """
    Enum state values for the `PutOn` action.
    """

    reaching_for = 0
    unparentting = 1
    grasping = 2
    moving_out_of_basket = 3
    dropping_object = 4
    resetting = 5

def quaternion_to_yaw(quaternion):
    y = 2 * (quaternion[0] * quaternion[2] + quaternion[1] * quaternion[3])
    x = 1 - 2 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
    yaw = np.arctan2(y, x)
    
    return np.rad2deg(yaw) % 360

class PutOnFromBike(MultiAction):
    """
    Put an object in a surface.

    The Replicant must already be holding the object in one hand.

    Internally, this action calls several "sub-actions", each a Replicant `Action`:

    #maybe should add turn to surface like the pick up action. Should test
    1. `ReachFor` to move the hand holding the object to be above the surface.
    2. `Drop` to drop the object into the surface.
    3. [`ResetArms`](reset_arms.md) to reset both arms.

    If the object lands in the surface, the action ends in success and the object is then made kinematic and parented to the surface.
    """

    def __init__(self, target, arm: Arm, dynamic: ReplicantDynamic, reach_for_max_distance: float, lowest_height: float, highest_height: float, state: ChallengeState, replicant_name: str, bike_contained: list, holding_bike_id: int, human_left_hand_position: np.array, human_right_hand_position: np.array, bike_drop_diff: np.array, object_manager):
        """
        :param target: the place to put the object..either a surface object id or a position or 'ground'
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        self._target = target
        self._replicant_name = replicant_name
        self._state = state
        self._reach_for_max_distance = reach_for_max_distance
        self._bike_contained = bike_contained
        self._bike_drop_diff = bike_drop_diff
        self._object_manager = object_manager
        self._holding_bike_id = holding_bike_id
        self._human_left_hand_position = human_left_hand_position
        self._human_right_hand_position = human_right_hand_position
        self._end_status: ActionStatus = ActionStatus.success

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """
        self.num_objects = len(self._bike_contained)
        if self.num_objects == 0:
            self.status = ActionStatus.success
            return []
        
        self.cur_object = self._bike_contained[0]
        self.cur_idx = 0
        self._image_frequency = image_frequency
        self._put_on_state = _PutOnState.reaching_for
        self._sub_action = ReachForTransportChallenge(target=self.cur_object,
                                                      arm=Arm.right,
                                                      dynamic=dynamic,
                                                      max_distance=self._reach_for_max_distance,
                                                      absolute=True)
        
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
        

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """
        if self._sub_action is not None and self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action is None or self._sub_action.status == ActionStatus.success:
            if self._put_on_state == _PutOnState.reaching_for:
                self._put_on_state = _PutOnState.unparentting
                print("unparentting the object")
                commands = [
                    {"$type": "unparent_object", "id": self.cur_object},
                    {"$type": "set_kinematic_state", "id": self.cur_object, "is_kinematic": False}
                ]
                return commands
            elif self._put_on_state == _PutOnState.unparentting:
                print("grasp the object")
                return self._grasp(resp=resp, static=static, dynamic=dynamic)
            elif self._put_on_state == _PutOnState.grasping:
                print("moving the hand out of the basket")
                return self._move_out_of_basket(resp=resp, static=static, dynamic=dynamic)
            elif self._put_on_state == _PutOnState.moving_out_of_basket:
                print("dropping the object")
                return self._drop(resp=resp, static=static, dynamic=dynamic)
            elif self._put_on_state == _PutOnState.dropping_object:
                self.cur_idx += 1
                if self.cur_idx < self.num_objects:
                    self.cur_object = self._bike_contained[self.cur_idx]
                    self._put_on_state = _PutOnState.reaching_for
                    return self._reach_for(resp=resp, static=static, dynamic=dynamic)
                else:
                    print("resetting the right arm to the bike")
                    return self._reset(resp=resp, static=static, dynamic=dynamic)
            elif self._put_on_state == _PutOnState.resetting:
                self.status = self._end_status
                return []
            else:               
                raise Exception(self._put_on_state)
        # We failed.
        else:
            # Remember the fail status.
            self._end_status = self._sub_action.status
            return self._reset(resp=resp, static=static, dynamic=dynamic)
    
    def _reach_for(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reach for the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._put_on_state = _PutOnState.reaching_for
        self._sub_action = ReachForTransportChallenge(target=self.cur_object,
                                                      arm=Arm.right,
                                                      dynamic=dynamic,
                                                      max_distance=self._reach_for_max_distance,
                                                      absolute=True)

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

        self._sub_action = Grasp(target=self.cur_object,
                                 arm=Arm.right,
                                 dynamic=dynamic,
                                 angle=0,
                                 axis="yaw", #pitch",
                                 offset=0,
                                 relative_to_hand=False,
                                 kinematic_objects=[])#True)
        self._put_on_state = _PutOnState.grasping
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
    
    def _move_out_of_basket(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        bike_rotation = quaternion_to_yaw(self._object_manager.transforms[self._holding_bike_id].rotation)
        dx = self._bike_drop_diff[0] * np.cos(np.deg2rad(bike_rotation)) + self._bike_drop_diff[2] * np.sin(np.deg2rad(bike_rotation))
        dz = self._bike_drop_diff[2] * np.cos(np.deg2rad(bike_rotation)) - self._bike_drop_diff[0] * np.sin(np.deg2rad(bike_rotation))
        drop_position = self._get_object_position(resp=resp, object_id=self._holding_bike_id) + np.array([dx, self._bike_drop_diff[1], dz])
        target_position = TDWUtils.array_to_vector3(drop_position)
        target_position["y"] += 0.4
        self._sub_action = ReachForTransportChallenge(target = target_position,
                                                      arm = Arm.right,
                                                      dynamic = dynamic,
                                                      max_distance = self._reach_for_max_distance,
                                                      absolute = True)
        self._put_on_state = _PutOnState.moving_out_of_basket
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
    
    def _drop(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        self._sub_action = Drop(arm = Arm.right,
                                dynamic = dynamic,
                                max_num_frames = 25,
                                offset = 0.1)
        self._put_on_state = _PutOnState.dropping_object
        position = self._get_object_position(resp=resp, object_id=self.cur_object)
        position[1] = 0.2
        commands = self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
        commands.append({"$type": "teleport_object",
                        "id": self.cur_object,
                        "position": TDWUtils.array_to_vector3(position)})
        # Parent the target object and make it kinematic.
        commands.extend([{"$type": "set_kinematic_state",
                          "id": self.cur_object,
                          "is_kinematic": True,
                          "use_gravity": False}])
        self._sub_action.status = ActionStatus.success
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
                                                      max_distance = self._reach_for_max_distance,
                                                      absolute = True)
        self._put_on_state = _PutOnState.resetting
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
