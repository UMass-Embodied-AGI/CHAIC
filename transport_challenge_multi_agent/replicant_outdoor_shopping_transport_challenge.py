from typing import Dict, Union
import numpy as np
from tdw.add_ons.replicant import Replicant
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.arm import Arm
from transport_challenge_multi_agent.pick_up_for_bike_agent import PickUp
from transport_challenge_multi_agent.move_to import MoveTo
from transport_challenge_multi_agent.put_in import PutIn
from transport_challenge_multi_agent.put_on import PutOn
from transport_challenge_multi_agent.put_on_from_bike import PutOnFromBike
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.navigate_to import NavigateTo
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.globals import Globals
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.agent_ability_info import BaseAbility, HelperAbility, GirlAbility
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.holding_bike_move_by import HoldingBikeMoveBy


class ReplicantOutdoorShoppingTransportChallenge(ReplicantTransportChallenge):
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

        self.holding_bike_id = None
        self.bike_human_diff = None
        self.human_left_hand_position = None
        self.human_right_hand_position = None
        self.bike_basket_position_diff = None
        self.bike_drop_diff = None

    def move_forward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance forward. This calls `self.move_by(distance)`.

        :param distance: The distance.
        """
        if self.ability.WHEELCHAIRED:
            pass
            #TODO: wheelchair move forward constraint
        
        if self.ability.RIDING:
            self.holding_bike_move_by(distance=abs(distance), reset_arms=False)
            # self.move_by(distance=abs(distance), reset_arms=False)
        else:
            self.move_by(distance=abs(distance), reset_arms=False)

    def move_backward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance backward. This calls `self.move_by(-distance)`.

        :param distance: The distance.
        """
        print("WARNING: move_backward is not used in GYM environment")
        if self.ability.RIDING:
            self.holding_bike_move_by(distance=-abs(distance), reset_arms=False)
            # self.move_by(distance=abs(distance), reset_arms=False)
        else:
            self.move_by(distance=-abs(distance), reset_arms=False)

    def holding_bike_move_by(self, distance: float, reset_arms: bool = True) -> None:
        """
        move by when holding a bike
        """

        self.action = HoldingBikeMoveBy(distance=distance, reset_arms=reset_arms, replicant = self)

    def pick_up(self, target: int, object_manager) -> None:
        """
        Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

        The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

        See: [`PickUp`](pick_up.md)

        :param target: The object ID.
        """

        self.action = PickUp(target=target,
                             reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD,
                             state=self._state,
                             replicant_name=self._replicant_name,
                             holding_bike_id = self.holding_bike_id,
                             human_left_hand_position = self.human_left_hand_position,
                             human_right_hand_position = self.human_right_hand_position,
                             bike_basket_position_diff = self.bike_basket_position_diff,
                             object_manager = object_manager
                             )
        
    def put_on(self, target: int, arm: Arm, bike_id: int, bike_contained: list, object_manager) -> None:
        """
        Put an object on a container.

        The Replicant must already be holding the object in the one hand.

        See: [`PutOn`](put_on.md)
        """

        self.action = PutOnFromBike(target=target, arm=arm,  dynamic=self.dynamic, \
                        reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                        lowest_height = self.ability.LOWEST_PUT_ON_HEIGHT, \
                        highest_height = self.ability.HIGHEST_PUT_ON_HEIGHT, \
                        state=self._state,
                        replicant_name=self._replicant_name,
                        bike_contained = bike_contained,
                        holding_bike_id = self.holding_bike_id,
                        human_left_hand_position = self.human_left_hand_position,
                        human_right_hand_position = self.human_right_hand_position,
                        bike_drop_diff = self.bike_drop_diff,
                        object_manager = object_manager)