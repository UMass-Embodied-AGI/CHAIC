from enum import Enum
from typing import Dict


class ReplicantTargetPosition(Enum):
    """
    Enum values describing a target position for a Replicant's hand.
    """

    pick_up_end_left = 0  # During a `PickUp` left-handed action, reset the left hand to this position.
    pick_up_end_right = 1  # During a `PickUp` right-handed action, reset the right hand to this position.
    put_in_move_away_left = 2  # During a `PutIn` action, if the left hand is moving the target object, move the left hand to this position.
    put_in_move_away_right = 3  # During a `PutIn` action, if the right hand is moving the target object, move the right hand to this position.
    put_in_container_in_front = 4  # During a `PutIn` action, move the container to this position.
    reset_left = 5 # Reset the arm to its rest position.
    reset_right = 6 # Reset the arm to its rest position.


def get_positions(replicant_name) -> Dict[ReplicantTargetPosition, Dict[str, float]]:
    """
    :return: A dictionary. Key = `ReplicantTargetPosition`. Value = The relative position.
    """
    if replicant_name in ['replicant_0', 'man_casual', 'fireman', 'woman_casual']:
        positions: Dict[ReplicantTargetPosition, Dict[str, float]] = dict()
        for x, arm in zip([-0.2, 0.2], [ReplicantTargetPosition.pick_up_end_left, ReplicantTargetPosition.pick_up_end_right]):
            positions[arm] = {"x": x, "y": 1, "z": 0.55}
        for x, arm in zip([-0.45, 0.45], [ReplicantTargetPosition.put_in_move_away_left, ReplicantTargetPosition.put_in_move_away_right]):
            positions[arm] = {"x": x, "y": 1, "z": 0.55}
        positions[ReplicantTargetPosition.put_in_container_in_front] = {"x": 0, "y": 1.15, "z": 0.55}
        for x, arm in zip([-0.25, 0.25], [ReplicantTargetPosition.reset_left, ReplicantTargetPosition.reset_right]):
            positions[arm] = {"x": x, "y": 0.75, "z": 0}
    elif replicant_name == 'girl_casual':
        positions: Dict[ReplicantTargetPosition, Dict[str, float]] = dict()
        for x, arm in zip([-0.2, 0.2], [ReplicantTargetPosition.pick_up_end_left, ReplicantTargetPosition.pick_up_end_right]):
            positions[arm] = {"x": x, "y": 0.4, "z": 0.25}
        for x, arm in zip([-0.35, 0.35], [ReplicantTargetPosition.put_in_move_away_left, ReplicantTargetPosition.put_in_move_away_right]):
            positions[arm] = {"x": x, "y": 0.4, "z": 0.25}
        positions[ReplicantTargetPosition.put_in_container_in_front] = {"x": 0, "y": 0.25, "z": 0.25}
        for x, arm in zip([-0.15, 0.15], [ReplicantTargetPosition.reset_left, ReplicantTargetPosition.reset_right]):
            positions[arm] = {"x": x, "y": 0.4, "z": 0.0}
    else:
        raise Exception("Invalid replicant name: " + replicant_name)
    return positions
