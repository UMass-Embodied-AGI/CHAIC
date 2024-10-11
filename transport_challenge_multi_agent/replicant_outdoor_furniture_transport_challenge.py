from typing import Dict, Union
import numpy as np
from tdw.add_ons.replicant import Replicant
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.arm import Arm
from transport_challenge_multi_agent.pick_up_furniture import PickUp
from transport_challenge_multi_agent.move_to import MoveTo
from transport_challenge_multi_agent.put_in import PutIn
from transport_challenge_multi_agent.put_on import PutOn
from transport_challenge_multi_agent.put_onto_truck import PutOntoTruck
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.navigate_to import NavigateTo
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.globals import Globals
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.agent_ability_info import BaseAbility, HelperAbility, GirlAbility
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.holding_bike_move_by import HoldingBikeMoveBy


class ReplicantOutdoorFurnitureTransportChallenge(ReplicantTransportChallenge):
    """
    A wrapper class for `Replicant` for the Outdoor Transport Challenge.

    This class is a subclass of `ReplicantTransportChallenge`. It includes the entire `ReplicantTransportChallenge` API plus specialized Outdoor Transport Challenge actions. 
    
    Only the Outdoor Transport Challenge actions are documented here. 
    """

    def __init__(self, replicant_id: int, state: ChallengeState,
                 position: Union[Dict[str, float], np.ndarray] = None,
                 rotation: Union[Dict[str, float], np.ndarray] = None,
                 image_frequency: ImageFrequency = ImageFrequency.once,
                 target_framerate: int = 250,
                 enable_collision_detection: bool = False,
                 name: str = "replicant_0",
                 ability: BaseAbility = BaseAbility()
                 ):
        """
        :param replicant_id: The ID of the Replicant.
        :param state: The [`ChallengeState`](challenge_state.md) data.
        :param position: The position of the Replicant as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param rotation: The rotation of the Replicant in Euler angles (degrees) as an x, y, z dictionary or numpy array. If None, defaults to `{"x": 0, "y": 0, "z": 0}`.
        :param image_frequency: An [`ImageFrequency`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/image_frequency.md) value that sets how often images are captured.
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        """
        super().__init__(replicant_id=replicant_id, state=state, position=position, rotation=rotation,
                         image_frequency=image_frequency, target_framerate=target_framerate, name=name, ability=ability)
    
        self.holding_furniture = False
        self.holding_furniture_id = None
        self.furniture_human_diff = None
        self.human_rotation = None
        self.furniture_rotation = None
        self.truck_id = None
        self.task_kind = None

    def pick_up(self, target: int, arm: Arm, lift_pos: dict, object_weight: int = 10, behaviour_data_gen: bool = False, no_weight_check: bool = False, no_grasp: bool = False) -> None:
        """
        Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

        The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

        See: [`PickUp`](pick_up.md)

        :param target: The object ID.
        """

        self.action = PickUp(target=target, \
                             reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                             state=self._state, \
                             lift_pos = lift_pos,
                             replicant_name=self._replicant_name,
                             highest_weight=self.ability.HIGHEST_PICKUP_MASS,
                             object_weight=object_weight,
                             replicant = self,
                             no_weight_check = no_weight_check,
                             no_grasp = no_grasp,
                             behaviour_data_gen = behaviour_data_gen)
        
    def move_to_object(self, target: int, arrived_at: int = 0.7, ability_constraint = True, reset_arms = True) -> None:
        """
        Move to an object. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        max_distance = 100
        if target == self.truck_id:
            if self.task_kind == "outdoor_furniture":
                target = np.array([8.5, 0, 0])
            else:
                target = np.array([58, 0, -16.5])
            self.action = MoveTo(target=target,
                                collision_detection=self.collision_detection,
                                previous=self._previous_action,
                                reset_arms=reset_arms,
                                reset_arms_duration=0.25,
                                scale_reset_arms_duration=True,
                                arrived_at=arrived_at,
                                bounds_position="center",
                                animation="walking_2",
                                library="humanoid_animations.json",
                                collision_avoidance_distance=self._record.collision_avoidance_distance,
                                collision_avoidance_half_extents=self._record.collision_avoidance_half_extents,
                                collision_avoidance_y=self._record.collision_avoidance_y,
                                max_distance=max_distance,
                                state = self._state,
                                replicant_name = self._replicant_name)
        else:
            self.action = MoveTo(target=target,
                                collision_detection=self.collision_detection,
                                previous=self._previous_action,
                                reset_arms=reset_arms,
                                reset_arms_duration=0.25,
                                scale_reset_arms_duration=True,
                                arrived_at=arrived_at,
                                bounds_position="center",
                                animation="walking_2",
                                library="humanoid_animations.json",
                                collision_avoidance_distance=self._record.collision_avoidance_distance,
                                collision_avoidance_half_extents=self._record.collision_avoidance_half_extents,
                                collision_avoidance_y=self._record.collision_avoidance_y,
                                max_distance=max_distance,
                                state = self._state,
                                replicant_name = self._replicant_name)
    
    def put_on(self, target: int, arm: Arm, position: list, no_drop: bool = False) -> None:
        if target != self.truck_id:
            self.action = PutOn(target=target, arm=arm,  dynamic=self.dynamic, \
                        reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                        lowest_height = self.ability.LOWEST_PUT_ON_HEIGHT, \
                        highest_height = self.ability.HIGHEST_PUT_ON_HEIGHT, \
                        state=self._state,
                        replicant_name=self._replicant_name)
        else:
            if self.task_kind == "outdoor_furniture":
                surface_position = {"x": 8.9, "y": 1, "z": 0}
            else:
                surface_position = {"x": 58, "y": 1, "z": -16.9}

            self.action = PutOntoTruck(target=target, arm=arm,  dynamic=self.dynamic, \
                        reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                        lowest_height = self.ability.LOWEST_PUT_ON_HEIGHT, \
                        highest_height = self.ability.HIGHEST_PUT_ON_HEIGHT, \
                        state=self._state,
                        replicant_name=self._replicant_name,
                        replicant = self,
                        put_on_position = position,
                        no_drop = no_drop,
                        surface_position = surface_position)