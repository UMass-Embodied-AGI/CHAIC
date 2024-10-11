from enum import Enum
from typing import List
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.turn_to import TurnTo
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction
from transport_challenge_multi_agent.grasp import TransportChallengeGrasp
from tdw.tdw_utils import TDWUtils


class _PickUpState(Enum):
    """
    Enum values describing the state of a `PickUp` action.
    """

    turning_to = 0
    reaching_for = 1
    grasping = 2
    resetting = 3


class PickUp(MultiAction):
    """
    A combination of `TurnTo` + `ReachFor` + `Grasp` + [`ResetArms`](reset_arms.md).
    """

    def __init__(self, arm: Arm, target: int, reach_for_max_distance: float, lowest_height: float, highest_height: float, state: ChallengeState, replicant_name: str, highest_weight: int, object_weight: int = 10):
        """
        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) picking up the object.
        :param target: The object ID.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        self.highest_height = highest_height
        self.lowest_height = lowest_height
        self._arm: Arm = arm
        self._target: int = target
        self._target_id = target
        self.reach_for_max_distance = reach_for_max_distance
        self._pick_up_state: _PickUpState = _PickUpState.turning_to
        self._end_status: ActionStatus = ActionStatus.success
        self._replicant_name = replicant_name
        self._highest_weight = highest_weight
        self._object_weight = object_weight
        self.broken_tag = False

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        self._sub_action = TurnTo(target=self._target)
        # Is a Replicant already holding this object?
        if not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            print("cannot pick up object", self._target, "with arm", self._arm, "because it is too far or grasped by another agent")
            self._end_status = ActionStatus.cannot_grasp

        if self._arm in dynamic.held_objects:
            print("cannot pick up object", self._target, "with arm", self._arm, "because it is already held")
            self._end_status = ActionStatus.cannot_grasp

        # Check the height constraints.
        bound = self._get_object_bounds(object_id=self._target, resp=resp)
        self._surface_position = TDWUtils.array_to_vector3(np.array([bound['center'][0], bound['top'][1], bound['center'][2]]))
        if self._replicant_name == 'girl_casual' and self._target in self._state.vase_ids:
            # Have some danger to broken
            if self._surface_position["y"] > self.highest_height:
                broken_rate = 0.5
            else:
                broken_rate = 0.25
            st = np.random.choice([True, False], p = [broken_rate, 1 - broken_rate])
            print("broken rate:", broken_rate)
            if st: # broken
                print("target object vase is broken!")
                self.broken_tag = True

        if self._surface_position["y"] < self.lowest_height or self._surface_position["y"] > self.highest_height or self.broken_tag:
            delta = max(self.lowest_height - self._surface_position["y"], self._surface_position["y"] - self.highest_height)
            prob = np.exp(-delta) / 2
            print("pick up prob:", prob, "delta:", delta, "highest:", self.highest_height, "lowest:", self.lowest_height)
            st = np.random.choice([True, False], p = [prob, 1 - prob])
            if not st or self.broken_tag:
                print("cannot pick up object", self._target, "with arm", self._arm, "because it is too high or too low or broken")
                if self._surface_position["y"] < self.lowest_height:
                    target_position = np.array([bound['center'][0], self.lowest_height, bound['center'][2]])
                else:
                    target_position = np.array([bound['center'][0], self.highest_height, bound['center'][2]])
                self._target = target_position
                self._end_status = ActionStatus.cannot_grasp

    #    elif self._object_weight > self._highest_weight:
    #        print("cannot pick up object", self._target, "with arm", self._arm, "because it is too heavy")
    #        self._end_status = ActionStatus.cannot_grasp

        # Turn to face the object.
        self._image_frequency = image_frequency
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                image_frequency=image_frequency)

    def get_ongoing_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Evaluate an action per-frame to determine whether it's done.

        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.

        :return: A list of commands to send to the build to continue the action.
        """
        # Check if another Replicant is holding the object.
        if self._pick_up_state != _PickUpState.resetting and not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            self._end_status = ActionStatus.cannot_grasp
            return self._reset(resp=resp, static=static, dynamic=dynamic)
        # Continue an ongoing sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success:
            # Reach for the object.
            if self._pick_up_state == _PickUpState.turning_to:
                print("reach for object or position", self._target, "with arm", self._arm)
                return self._reach_for(resp=resp, static=static, dynamic=dynamic)
            # Grasp the object.
            elif self._pick_up_state == _PickUpState.reaching_for:
                print("grasp object or position", self._target, "with arm", self._arm)
                return self._grasp(resp=resp, static=static, dynamic=dynamic)
            # Reset the arm.
            elif self._pick_up_state == _PickUpState.grasping:
                print("reset arm", self._arm)
                return self._reset(resp=resp, static=static, dynamic=dynamic)
            # We're done!
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
        extra_commands = []
        self._pick_up_state = _PickUpState.reaching_for
        self._sub_action = ReachForTransportChallenge(target=self._target,
                                                      arm=self._arm,
                                                      dynamic=dynamic,
                                                      max_distance=self.reach_for_max_distance,
                                                      absolute=True)
        if self._end_status == ActionStatus.cannot_grasp:
            # only reach animation, no grasp animation
            self._pick_up_state = _PickUpState.grasping
        extra_commands += self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
        return extra_commands

    def _grasp(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to grasp the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = TransportChallengeGrasp(target=self._target,
                                 arm=self._arm,
                                 dynamic=dynamic,
                                 angle=0,
                                 axis="yaw", #pitch",
                                 offset=0,
                                 relative_to_hand=False,
                                 kinematic_objects=[],
                                 state = self._state)#True)
        self._pick_up_state = _PickUpState.grasping
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
        extra_commands = []
        if self.broken_tag:
            extra_commands = [{"$type": "teleport_object",
                            "id": self._target_id,
                            "position": {
                                "x": 0,
                                "y": -100,
                                "z": 0,
                            }}]
            extra_commands.extend([{'$type': 'add_object', 
                                    'name': 'cgaxis_models_66_18_vray', 
                                    'url': 'file://local_asset/Linux/cgaxis_models_66_18_vray', 
                                    'scale_factor': 0.6, 
                                    'position': {'x': float(dynamic.transform.position[0] - dynamic.transform.forward[2] / 2), 'y': 0.5, 'z': float(dynamic.transform.position[2] + dynamic.transform.forward[0] / 2)}, 
                                    'category': 'garden plant', 
                                    'id': np.random.randint(0, 16777215), 
                                    'affordance_points': []}])

        self._sub_action = ResetArms(reach_for_max_distance = self.reach_for_max_distance, state=self._state, replicant_name = self._replicant_name)
        self._pick_up_state = _PickUpState.resetting
        return extra_commands + self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
