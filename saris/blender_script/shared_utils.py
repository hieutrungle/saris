import math
from typing import Dict, List, Tuple, Sequence


def get_lead_follow_dict(
    num_rows: int,
    num_cols: int,
    follow_range: int,
):
    center_distance = follow_range * 2 + 1
    lead_follow_dict = {}
    # Find lead_idx
    for i in range(follow_range, num_cols, center_distance):
        for j in range(follow_range, num_rows, center_distance):
            lead_idx = i * num_rows + j

            # Find follow_idxs
            follow_idxs = []
            for n in range(i - follow_range, i + follow_range + 1):
                for m in range(j - follow_range, j + follow_range + 1):
                    follow_idx = n * num_rows + m
                    if follow_idx != lead_idx:
                        follow_idxs.append(follow_idx)

            lead_follow_dict.update({lead_idx: follow_idxs})
    return lead_follow_dict


def set_up_reflector() -> Tuple[Dict[int, List[int]], Tuple[float], Tuple[float]]:
    """
    Set up reflector.


    Returns:
    lead_follow_dict: Dict[int, List[int]]
        A dictionary, where the key is the index of the lead, and the value is the list of the index of the follow.
    init_angles: Tuple[float] in degrees
        Initial angles for the reflector.
    angle_deltas: Tuple[float] in degrees
        The maximum and minimum angle deltas from the init angles for the reflector.
    """

    max_delta = 30.0
    min_delta = -30.0
    angle_deltas = (min_delta, max_delta)

    init_theta = 90.0
    init_phi = -45.0
    init_angles = (init_theta, init_phi)

    num_rows = 12
    num_cols = 6
    follow_range = 1
    lead_follow_dict = get_lead_follow_dict(num_rows, num_cols, follow_range)

    return lead_follow_dict, init_angles, angle_deltas


def constraint_angle(angle: float, angle_delta: Sequence[float]) -> float:
    """
    Constrain the angle within the angle_delta.

    Args:
    angle: float
        The angle to be constrained. Unit: degree.
    angle_delta: Sequence[float]
        The maximum and minimum angle deltas. Unit: degree.
    """
    min_angle = angle_delta[0]
    max_angle = angle_delta[1]
    if angle < min_angle:
        angle = min_angle
    if angle > max_angle:
        angle = max_angle
    return angle
