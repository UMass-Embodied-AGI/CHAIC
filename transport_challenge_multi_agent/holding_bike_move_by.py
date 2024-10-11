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
from tdw.replicant.actions.move_by import MoveBy
from transport_challenge_multi_agent.multi_action import MultiAction
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from tdw.tdw_utils import TDWUtils


class HoldingBikeMoveBy(MultiAction):
    """
    A combination of `TurnTo` + `ReachFor` + `Grasp` + [`ResetArms`](reset_arms.md).
    """

    def __init__(self, distance: float, reset_arms: bool = True, replicant: ReplicantTransportChallenge = None, reset_arms_duration: float = 0.25,
                scale_reset_arms_duration: bool = True, arrived_at: float = 0.1, animation: str = "walking_2",
                library: str = "humanoid_animations.json"):
        """
        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) picking up the object.
        :param target: The object ID.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        """
        super().__init__(state = replicant._state)
        self._distance = distance
        self._reset_arms = reset_arms
        self._reset_arms_duration = reset_arms_duration
        self._scale_reset_arms_duration = scale_reset_arms_duration
        self._arrived_at = arrived_at
        self._animation = animation
        self._library = library
        self._replicant = replicant
        self.wait = False
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

        self._sub_action = MoveBy(distance=self._distance,
                             dynamic=self._replicant.dynamic,
                             collision_detection=self._replicant.collision_detection,
                             previous=self._replicant._previous_action,
                             reset_arms=self._reset_arms,
                             reset_arms_duration=self._reset_arms_duration,
                             scale_reset_arms_duration=self._scale_reset_arms_duration,
                             arrived_at=self._arrived_at,
                             animation=self._animation,
                             library=self._library,
                             collision_avoidance_distance=self._replicant._record.collision_avoidance_distance,
                             collision_avoidance_half_extents=self._replicant._record.collision_avoidance_half_extents,
                             collision_avoidance_y=self._replicant._record.collision_avoidance_y)

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
            if self.wait:
                self.wait = False
                return []
            else:
                self.wait = True
                return self._sub_action.get_ongoing_commands(resp=resp, static=static, dynamic=dynamic)
        else:
            self.status = self._end_status
            return []