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


class _PutOnState(Enum):
    """
    Enum state values for the `PutOn` action.
    """

    turning_to = 0
    reach_for = 1
    reaching_for = 2
    moving_object_over_surface = 3
    dropping_object = 4
    resetting = 5

class PutOn(MultiAction):
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

    def __init__(self, target, arm: Arm, dynamic: ReplicantDynamic, reach_for_max_distance: float, lowest_height: float, highest_height: float, state: ChallengeState, replicant_name: str):
        """
        :param target: the place to put the object..either a surface object id or a position or 'ground'
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """

        super().__init__(state=state)
        # Track the state of the action.
        self._put_on_state: _PutOnState = _PutOnState.turning_to
        # Make sure we are holding both objects, and that one object is a surface and the other is a target object.
        self.reach_for_max_distance = reach_for_max_distance
        # print("held_objects:", dynamic.held_objects, "arms:", arm)
        self._object_arm = arm
        if self._object_arm is None or not (self._object_arm in dynamic.held_objects):
            self.status = ActionStatus.not_holding
            self._object_id: int = -1
        else:
            self._object_id = int(dynamic.held_objects[self._object_arm])
        self._target = target
        self.lowest_height = lowest_height
        self.highest_height = highest_height
        self._replicant_name = replicant_name
        self.final_status = None

    def len_scale(self, vec, length):
        return vec / np.linalg.norm(vec) * length

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """

        if self._target == 'ground' or self._target == -1:
            self._target = self._get_object_bounds(object_id=dynamic.held_objects[self._object_arm], resp=resp)['center']
            self._target[1] = 0

        # Remember the image frequency.
        self._image_frequency = image_frequency
        self._sub_action = TurnTo(target=self._target)

        if isinstance(self._target, int):
            bound = self._get_object_bounds(object_id=self._target, resp=resp)
            self._surface_position = TDWUtils.array_to_vector3(np.array([bound['center'][0], bound['top'][1], bound['center'][2]]))
        elif isinstance(self._target, np.ndarray):
            self._surface_position = TDWUtils.array_to_vector3(self._target)
        elif isinstance(self._target, dict):
            self._surface_position = {k: v for k, v in self._target.items()}
        else:
            raise Exception(f"Invalid target: {self._target}")
        
        if self._surface_position['y'] < self.lowest_height or self._surface_position['y'] > self.highest_height:
            delta = max(self.lowest_height - self._surface_position["y"], self._surface_position["y"] - self.highest_height)
            prob = np.exp(-delta)
            print("put on success prob:", prob)
            st = np.random.choice([True, False], p = [prob, 1 - prob])
            if not st:
                print("cannot reach surface because of height")
                self.final_status = ActionStatus.cannot_reach
                self._target = dynamic.transform.position[[0, 1, 2]] + self.len_scale(dynamic.transform.position[[0, 1, 2]] - self._get_object_bounds(object_id=dynamic.held_objects[self._object_arm], resp=resp)['center'], 2.5)
                # turn back and put it to ground.
                self._target[1] = 0
                self._surface_position = TDWUtils.array_to_vector3(self._target)

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
        # Continue an ongoing sub-action.
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success or ActionStatus.still_dropping:
            if self._put_on_state == _PutOnState.turning_to:
                print("reach for object or position", self._target, "with arm", self._object_arm)
                return self._reach_for(resp=resp, static=static, dynamic=dynamic)
            # Grasp the object.
            elif self._put_on_state == _PutOnState.reaching_for:
                print("grasp object or position", self._target, "with arm", self._object_arm)
                return self._move_object_over_surface(resp=resp, static=static, dynamic=dynamic)
            if self._put_on_state == _PutOnState.moving_object_over_surface:
                # print("dropping object")
                return self._drop(resp=resp, dynamic=dynamic, static=static)
            # Reset the arms.
            elif self._put_on_state == _PutOnState.dropping_object:
                # print("resetting arms")
                return self._reset_arms(resp=resp, dynamic=dynamic, static=static)
            # We're done!
            elif self._put_on_state == _PutOnState.resetting:
                self.status = ActionStatus.success
                if self.final_status is not None:
                    self.status = self.final_status
                return []
            else:
                raise Exception(self._put_on_state)
        # The sub-action failed.
        else:
            self.status = self._sub_action.status
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)

    def _move_object_over_surface(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to move the target object to be above the surface.

        Set `self._put_on_state` to `moving_object_over_surface`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        #Get a position over the surface.
        # print(self._surface_position)
        target_position = self._surface_position
        target_position['y'] += 0.2
        self._sub_action = ReachForWithPlanTransportChallenge(target=self._surface_position,
                                                      arm=self._object_arm,
                                                      dynamic=dynamic,
                                                      max_distance=self.reach_for_max_distance,
                                                      absolute=True,
                                                      plan = IkPlanType.vertical_horizontal)
        self._put_on_state = _PutOnState.moving_object_over_surface
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)

    def _drop(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to drop the target object into the surface.

        Set `self._put_on_state` to `dropping_object`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """
        self._sub_action = Drop(arm=self._object_arm,
                                dynamic=dynamic,
                                max_num_frames=25,
                                offset=0.1)
        self._put_on_state = _PutOnState.dropping_object
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)

    def _reset_arms(self, resp: List[bytes], dynamic: ReplicantDynamic, static: ReplicantStatic) -> List[dict]:
        """
        Start to reset the arms.

        Set `self._put_on_state` to `resetting`.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._sub_action = ResetArms(reach_for_max_distance = self.reach_for_max_distance, state=self._state, replicant_name=self._replicant_name)
        self._put_on_state = _PutOnState.resetting
        return self._sub_action.get_initialization_commands(resp=resp, dynamic=dynamic, static=static,
                                                            image_frequency=self._image_frequency)
    
    def _get_offset(self, arm: Arm, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> Dict[str, float]:
        if arm in dynamic.held_objects:
            bounds = self._get_object_bounds(object_id=dynamic.held_objects[arm], resp=resp)
            object_position = bounds['center']
            hand_position = dynamic.body_parts[static.hands[arm]].position
            return TDWUtils.array_to_vector3(hand_position - object_position)
        else:
            return {"x": 0, "y": 0, "z": 0}

    def _reach_for(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic) -> List[dict]:
        """
        Start to reach for the object.

        :param resp: The response from the build.
        :param dynamic: The dynamic Replicant data.
        :param static: The static Replicant data.

        :return: A list of commands.
        """

        self._put_on_state = _PutOnState.reaching_for
        self._sub_action = ReachForTransportChallenge(target=self._target,
                                                      arm=self._object_arm,
                                                      dynamic=dynamic,
                                                      max_distance=self.reach_for_max_distance,
                                                      absolute=True)
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)