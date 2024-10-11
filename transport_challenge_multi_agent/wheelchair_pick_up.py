from enum import Enum
from typing import List, Optional, Dict
import numpy as np
from tdw.replicant.arm import Arm
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.replicant_static import ReplicantStatic
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.actions.action import Action
from tdw.wheelchair_replicant.actions.turn_to import TurnTo
from tdw.wheelchair_replicant.actions.move_by import MoveBy
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.actions.grasp import Grasp
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.wheelchair_reach_for_transport_challenge import ReachForTransportChallenge
from transport_challenge_multi_agent.wheelchair_reset_arms import ResetArms
from transport_challenge_multi_agent.multi_action import MultiAction
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

    def __init__(self, arm: Arm, target: int, reach_for_max_distance: float, lowest_height: float, highest_height: float, state: ChallengeState, replicant_name: str, 
                 collision_detection: CollisionDetection, previous: Optional[Action],
                 reset_arms: bool, reset_arms_duration: float, scale_reset_arms_duration: bool, arrived_at: float,
                 bounds_position: str, collision_avoidance_distance: float,
                 collision_avoidance_y: float,
                 collision_avoidance_half_extents: Dict[str, float]):
        """
        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        :param collision_detection: The [`CollisionDetection`](../collision_detection.md) rules.
        :param previous: The previous action, if any.
        :param reset_arms: If True, reset the arms to their neutral positions while beginning the walk cycle.
        :param reset_arms_duration: The speed at which the arms are reset in seconds.
        :param reset_arms_duration: The speed at which the arms are reset in seconds.
        :param arrived_at: If at any point during the action the difference between the target distance and distance traversed is less than this, then the action is successful.
        :param bounds_position: If `target` is an integer object ID, move towards this bounds point of the object. Options: `"center"`, `"top`", `"bottom"`, `"left"`, `"right"`, `"front"`, `"back"`.
        :param collision_avoidance_distance: If `collision_detection.avoid == True`, an overlap will be cast at this distance from the Wheelchair Replicant to detect obstacles.
        :param collision_avoidance_half_extents: If `collision_detection.avoid == True`, an overlap will be cast with these half extents to detect obstacles.
        """

        super().__init__(state=state)

        """:field
        The [`CollisionDetection`](../collision_detection.md) rules.
        """
        self.collision_detection: CollisionDetection = collision_detection
        """:field
        If True, reset the arms to their neutral positions while beginning the walk cycle.
        """
        self.reset_arms: bool = reset_arms
        """:field
        The speed at which the arms are reset in seconds.
        """
        self.reset_arms_duration: float = reset_arms_duration
        """:field
        If True, `reset_arms_duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds.
        """
        self.scale_reset_arms_duration: bool = scale_reset_arms_duration
        """:field
        If at any point during the action the difference between the target distance and distance traversed is less than this, then the action is successful.
        """
        self.arrived_at: float = arrived_at
        """:field
        If `target` is an integer object ID, move towards this bounds point of the object. Options: `"center"`, `"top`", `"bottom"`, `"left"`, `"right"`, `"front"`, `"back"`.
        """
        self.bounds_position: str = bounds_position
        """:field
        If `collision_detection.avoid == True`, an overlap will be cast at this distance from the Wheelchair Replicant to detect obstacles.
        """
        self.collision_avoidance_distance: float = collision_avoidance_distance
        self.collision_avoidance_y: float = collision_avoidance_y
        """:field
        If `collision_detection.avoid == True`, an overlap will be cast with these half extents to detect obstacles.
        """
        self.collision_avoidance_half_extents: Dict[str, float] = collision_avoidance_half_extents
        self._turning: bool = True
        self._image_frequency: ImageFrequency = ImageFrequency.once
        self._move_by: Optional[MoveBy] = None
        self._previous_action: Optional[Action] = previous



        self.highest_height = highest_height
        self.lowest_height = lowest_height
        self._arm: Arm = arm
        self._target: int = target
        self.reach_for_max_distance = reach_for_max_distance
        self._pick_up_state: _PickUpState = _PickUpState.turning_to
        self._end_status: ActionStatus = ActionStatus.success
        self._replicant_name = replicant_name

    def get_initialization_commands(self, resp: List[bytes], static: ReplicantStatic, dynamic: ReplicantDynamic,
                                    image_frequency: ImageFrequency) -> List[dict]:
        """
        :param resp: The response from the build.
        :param static: The [`ReplicantStatic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_static.md) data that doesn't change after the Replicant is initialized.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value describing how often image data will be captured.

        :return: A list of commands to initialize this action.
        """
        #self.arrived_at
        self.error_record = "TurnTo"
        # Warning arrived_at=1 is a magic num. I adjust it to make it just run.
        self._sub_action = TurnTo(target=self._target, wheel_values = None, dynamic=dynamic, collision_detection=self.collision_detection, previous=self._previous_action, reset_arms=self.reset_arms, reset_arms_duration=self.reset_arms_duration, 
                            scale_reset_arms_duration=self.scale_reset_arms_duration, arrived_at=0.1, collision_avoidance_distance=self.collision_avoidance_distance, collision_avoidance_half_extents=self.collision_avoidance_half_extents)
        # Is a Replicant already holding this object?
        if not self._can_pick_up(resp=resp, static=static, dynamic=dynamic):
            print("cannot pick up object", self._target, "with arm", self._arm, "because it is too far or grasped by another agent")
            self._end_status = ActionStatus.cannot_grasp

        if  self._arm in dynamic.held_objects:
            print("cannot pick up object", self._target, "with arm", self._arm, "because it is already held")
            self._end_status = ActionStatus.cannot_grasp

        # Check the height constraints.
        bound = self._get_object_bounds(object_id=self._target, resp=resp)
        self._surface_position = TDWUtils.array_to_vector3(np.array([bound['center'][0], bound['top'][1], bound['center'][2]]))
        if self._surface_position["y"] < self.lowest_height or self._surface_position["y"] > self.highest_height:
            delta = max(self.lowest_height - self._surface_position["y"], self._surface_position["y"] - self.highest_height)
            prob = np.exp(-delta * 2)
            st = np.random.choice([True, False], p = [prob, 1 - prob])
            if not st:
                print("cannot pick up object", self._target, "with arm", self._arm, "because it is too high or too low")
                if self._surface_position["y"] < self.lowest_height:
                    target_position = np.array([bound['center'][0], self.lowest_height, bound['center'][2]])
                else:
                    target_position = np.array([bound['center'][0], self.highest_height, bound['center'][2]])
                self._target = target_position
                self._end_status = ActionStatus.cannot_grasp
        
        # Turn to face the object.
        self._image_frequency = image_frequency
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                                image_frequency=image_frequency)
        # return self._reach_for(resp=resp, static=static, dynamic=dynamic)

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
    #    print(self._sub_action.status, self._pick_up_state)
        if self._sub_action.status == ActionStatus.ongoing:
            return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        elif self._sub_action.status == ActionStatus.success or ActionStatus.failed_to_turn:
            # Reach for the object.
            # Fall to turn maybe means detect obstacles
            if self._pick_up_state == _PickUpState.turning_to:
                # breakpoint()
                self.error_record = "ReachFor"
                return self._reach_for(resp=resp, static=static, dynamic=dynamic)
            # Grasp the object.
            elif self._pick_up_state == _PickUpState.reaching_for:
                # breakpoint()
                self.error_record = "Grasp"
                return self._grasp(resp=resp, static=static, dynamic=dynamic)
            # Reset the arm.
            elif self._pick_up_state == _PickUpState.grasping:
                # breakpoint()
                self.error_record = "Reset"
                return self._reset(resp=resp, static=static, dynamic=dynamic)
            # We're done!
            elif self._pick_up_state == _PickUpState.resetting:
                # breakpoint()
                self.status = self._end_status
                return []
            else:
                raise Exception(self._pick_up_state)
        # We failed.
        else:
            # Remember the fail status.
            print(self.error_record, "failed")
            breakpoint()
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
        # return []
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
        self._pick_up_state = _PickUpState.resetting
        # return []
        return self._sub_action.get_initialization_commands(resp=resp, static=static, dynamic=dynamic,
                                                            image_frequency=self._image_frequency)
