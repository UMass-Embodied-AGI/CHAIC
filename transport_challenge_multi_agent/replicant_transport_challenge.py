from typing import Dict, Union
import numpy as np
from tdw.add_ons.replicant import Replicant
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.arm import Arm
from transport_challenge_multi_agent.pick_up import PickUp
from transport_challenge_multi_agent.move_to import MoveTo
from transport_challenge_multi_agent.put_in import PutIn
from transport_challenge_multi_agent.put_on import PutOn
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.navigate_to import NavigateTo
from transport_challenge_multi_agent.reset_arms import ResetArms
from transport_challenge_multi_agent.globals import Globals
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.agent_ability_info import BaseAbility, HelperAbility, GirlAbility
from typing import List, Dict, Optional, Union


class ReplicantTransportChallenge(Replicant):
    """
    A wrapper class for `Replicant` for the Transport Challenge.

    This class is a subclass of `Replicant`. It includes the entire `Replicant` API plus specialized Transport Challenge actions. 
    
    Only the Transport Challenge actions are documented here. For the full Replicant documentation, [read this.](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/replicants/overview.md)

    ![](images/action_space.jpg)
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

        super().__init__(replicant_id=replicant_id, position=position, rotation=rotation,
                         image_frequency=image_frequency, target_framerate=target_framerate, name = name)
        self._state: ChallengeState = state
        self.collision_detection.held = False
        self.collision_detection.previous_was_same = False
        self.ability = ability
        self.scene_bounds : SceneBounds = None
        self._replicant_name = name

    def turn_by(self, angle: float) -> None:
        """
        Turn the Replicant by an angle.

        :param angle: The target angle in degrees. Positive value = clockwise turn.
        """

        super().turn_by(angle=angle)

    def turn_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Turn the Replicant to face a target object or position.

        :param target: The target. If int: An object ID. If dict: A position as an x, y, z dictionary. If numpy array: A position as an [x, y, z] numpy array.
        """

        super().turn_to(target=target)

    def drop(self, arm: Arm, max_num_frames: int = 100, offset: float = 0.1) -> None:
        """
        Drop a held target object.

        The action ends when the object stops moving or the number of consecutive `communicate()` calls since dropping the object exceeds `self.max_num_frames`.

        When an object is dropped, it is made non-kinematic. Any objects contained by the object are parented to it and also made non-kinematic. For more information regarding containment in TDW, [read this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/semantic_states/containment.md).

        :param arm: The [`Arm`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/replicant/arm.md) holding the object.
        :param max_num_frames: Wait this number of `communicate()` calls maximum for the object to stop moving before ending the action.
        :param offset: Prior to being dropped, set the object's positional offset. This can be a float (a distance along the object's forward directional vector). Or it can be a dictionary or numpy array (a world space position).
        """

        super().drop(arm=arm, max_num_frames=max_num_frames, offset=offset)

    def move_forward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance forward. This calls `self.move_by(distance)`.

        :param distance: The distance.
        """
        if self.ability.WHEELCHAIRED:
            super().move_by(distance=abs(distance), 
                            reset_arms=False, 
                            animation="limping",
                            library="humanoid_animations.json")
        else:
            super().move_by(distance=abs(distance), reset_arms=False)

    def move_backward(self, distance: float = 0.5) -> None:
        """
        Walk a given distance backward. This calls `self.move_by(-distance)`.

        :param distance: The distance.
        """
        print("WARNING: move_backward is not allowed in GYM environment")
        super().move_by(distance=-abs(distance), reset_arms=False)

    def move_to_object(self, target: int, arrived_at: int = 0.7, ability_constraint = True, reset_arms = True, set_max_distance = None) -> None:
        """
        Move to an object. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        if ability_constraint:
            max_distance = self.ability.REACH_FOR_THRESHOLD
        else:
            max_distance = 100
        if set_max_distance is not None:
            max_distance = set_max_distance
        if self.ability.WHEELCHAIRED:
            self.action = MoveTo(target=target,
                             collision_detection=self.collision_detection,
                             previous=self._previous_action,
                             reset_arms=reset_arms,
                             reset_arms_duration=0.25,
                             scale_reset_arms_duration=True,
                             arrived_at=arrived_at,
                             bounds_position="center",
                             animation="limping",
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
    
    def move_to_position(self, target: np.ndarray, arrived_at: int = 1.2) -> None:
        """
        Move to an position. This calls `self.move_to(target)`.

        :param target: The object ID.
        """
        assert isinstance(target, np.ndarray) and target.shape == (3,), f"target must be a 3D numpy array. Got {target}"
        target[1] = 0
        self.move_to(target=target, reset_arms=False, arrived_at = arrived_at)
        
    def pick_up(self, target: int, arm: Arm, object_weight: int = 10) -> None:
        """
        Reach for an object, grasp it, and bring the arm + the held object to a neutral holding position in from the Replicant.

        The Replicant will opt for picking up the object with its right hand. If its right hand is already holding an object, it will try to pick up the object with its left hand.

        See: [`PickUp`](pick_up.md)

        :param target: The object ID.
        """

        self.action = PickUp(arm=arm, \
                             target=target, \
                             reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                             lowest_height = self.ability.LOWEST_PICKUP_HEIGHT, \
                             highest_height = self.ability.HIGHEST_PICKUP_HEIGHT, \
                             state=self._state, \
                             replicant_name=self._replicant_name,
                             highest_weight=self.ability.HIGHEST_PICKUP_MASS,
                             object_weight=object_weight)

    def put_in(self) -> None:
        """
        Put an object in a container.

        The Replicant must already be holding the container in one hand and the object in the other hand.

        See: [`PutIn`](put_in.md)
        """

        self.action = PutIn(dynamic=self.dynamic, reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, state=self._state, replicant_name=self._replicant_name)

    def put_on(self, target: int, arm: Arm) -> None:
        """
        Put an object on a container.

        The Replicant must already be holding the object in the one hand.

        See: [`PutOn`](put_on.md)
        """

        self.action = PutOn(target=target, arm=arm,  dynamic=self.dynamic, \
                        reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, \
                        lowest_height = self.ability.LOWEST_PUT_ON_HEIGHT, \
                        highest_height = self.ability.HIGHEST_PUT_ON_HEIGHT, \
                        state=self._state,
                        replicant_name=self._replicant_name)

    def reset_arms(self) -> None:
        """
        Reset both arms, one after the other.

        If an arm is holding an object, it resets with to a position holding the object in front of the Replicant.

        If the arm isn't holding an object, it resets to its neutral position.
        """

        self.action = ResetArms(reach_for_max_distance = self.ability.REACH_FOR_THRESHOLD, state=self._state, replicant_name=self._replicant_name)

    def navigate_to(self, target: Union[int, Dict[str, float], np.ndarray]) -> None:
        """
        Navigate along a path to a target.

        See: [`NavigateTo`](navigate_to.md)

        :param target: The target object or position.
        """
        print("WARNING: navigate_to is not used in GYM environment")
        self.action = NavigateTo(target=target, collision_detection=self.collision_detection, state=self._state, \
                                 collision_avoidance_distance = self._record.collision_avoidance_distance, collision_avoidance_half_extents = self._record.collision_avoidance_half_extents, \
                                 collision_avoidance_y = self._record.collision_avoidance_y, replicant_name = self._replicant_name)

    def look_up(self, angle: float = 15) -> None:
        """
        Look upward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """
        print("WARNING: look_up is not used in GYM environment")
        self.rotate_head(axis="pitch", angle=-abs(angle), scale_duration=Globals.SCALE_IK_DURATION)

    def look_down(self, angle: float = 15) -> None:
        """
        Look downward by an angle.

        The head will continuously move over multiple `communicate()` calls until it is looking at the target.

        :param angle: The angle in degrees.
        """
        print("WARNING: look_down is not used in GYM environment")
        self.rotate_head(axis="pitch", angle=abs(angle), scale_duration=Globals.SCALE_IK_DURATION)

    def reset_head(self, duration: float = 0.1, scale_duration: bool = True) -> None:
        """
        Reset the head to its neutral rotation.

        The head will continuously move over multiple `communicate()` calls until it is at its neutral rotation.

        :param duration: The duration of the motion in seconds.
        :param scale_duration: If True, `duration` will be multiplied by `framerate / 60)`, ensuring smoother motions at faster-than-life simulation speeds.
        """

        super().reset_head(duration=duration, scale_duration=scale_duration)
    
    def pos_to_2d_box_distance(self, px, py, rx1, ry1, rx2, ry2):
        if px < rx1:
            if py < ry1:
                return ((px - rx1) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx1) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return rx1 - px
        elif px > rx2:
            if py < ry1:
                return ((px - rx2) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx2) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return px - rx2
        else:
            if py < ry1:
                return ry1 - py
            elif py > ry2:
                return py - ry2
            else:
                return 0
    
    def belongs_to_which_room(self, pos):
        if self.scene_bounds is None:
            return 0
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis:
                min_dis = distance
                room = i
        return room
    
    def on_send(self, resp: List[bytes]) -> None:
        super().on_send(resp)
        for i in range(len(self.commands)):
            if "name" in self.commands[i] and self.commands[i]["name"] == 'limping':
                self.commands[i]["framerate"] = 10
            # ad-hoc fix for the bug in the library