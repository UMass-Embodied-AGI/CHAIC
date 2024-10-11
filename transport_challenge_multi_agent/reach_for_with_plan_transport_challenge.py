from typing import Dict, Union
import numpy as np
from tdw.replicant.replicant_dynamic import ReplicantDynamic
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.arm import Arm
from tdw.replicant.actions.reach_for_with_plan import ReachForWithPlan
from transport_challenge_multi_agent.globals import Globals
from tdw.replicant.ik_plans.ik_plan_type import IkPlanType


class ReachForWithPlanTransportChallenge(ReachForWithPlan):
    """
    A `ReachFor` action with default parameters.
    """

    def __init__(self, target: Union[int, Arm, np.ndarray, Dict[str,  float]], arm: Arm, absolute: bool,
                 dynamic: ReplicantDynamic, max_distance: float = 1.5, duration: float = 0.25, offhand_follows: bool = False, plan: IkPlanType = IkPlanType.vertical_horizontal):
        """
        :param target: The target. If int: An object ID. If `Arm`: A position in front of one of the sides of the Replicant. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        :param absolute: If True, the target position is in world space coordinates. If False, the target position is relative to the Replicant. Ignored if `target` is an int.
        :param dynamic: The [`ReplicantDynamic`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/replicant_dynamic.md) data that changes per `communicate()` call.
        :param duration: The duration of the motion in seconds.
        :param offhand_follows: If True, the offhand will follow the primary hand, meaning that it will maintain the same relative position. Ignored if `len(arms) > 1` or if `target` is an object ID.
        """

        super().__init__(targets=[target],
                        #  offhand_follows=offhand_follows,
                         arrived_at=0.09,
                         max_distance=max_distance,
                         duration=duration,
                         scale_duration=Globals.SCALE_IK_DURATION,
                         arms=[arm],
                         dynamic=dynamic,
                         collision_detection=CollisionDetection(objects=False, held=False),
                         previous=None,
                         absolute=absolute,
                         from_held=True,
                         held_point="center",
                         plan = plan)
