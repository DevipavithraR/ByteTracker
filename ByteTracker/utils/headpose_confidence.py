"""headpose_confidence.py
~~~~~~~~~~~~~~~~~~~~~~~~
Utility functions for computing a confidence score that an estimated head‑pose
(yaw, pitch, roll) is within acceptable angular thresholds.
"""

from typing import Iterable, List
import functools

__all__ = [
    "generate_pose_conf",
    "_generate_angle_conf",
    "generate_model_conf",
]


def _generate_angle_conf(angle: float, angle_th: float) -> float:
    """Return confidence for a single angle.

    Parameters
    ----------
    angle : float
        The absolute yaw/pitch/roll angle predicted by the model, in degrees.
    angle_th : float
        The maximum permitted angle for the corresponding axis, in degrees.

    Returns
    -------
    float
        1.0 if angle is 0°, decreasing linearly to 0.0 at threshold. Beyond
        threshold → 0.0.
    """
    return 0.0 if abs(angle) > angle_th else 1.0 - (abs(angle) / angle_th)


def generate_model_conf(*confs: float | Iterable[float]) -> float:
    """Multiply individual axis confidences to form a single value.

    Parameters
    ----------
    confs : float or Iterable[float]
        Confidence scores for yaw, pitch, and roll.

    Returns
    -------
    float
        Overall confidence in [0.0, 1.0]
    """
    if len(confs) == 1 and isinstance(confs[0], Iterable):
        confs = tuple(confs[0])
    return functools.reduce(lambda a, b: a * b, confs, 1.0)


def generate_pose_conf(
    head_pose: List[float] | tuple[float, float, float],
    head_pose_th: List[float] | tuple[float, float, float],
) -> float:
    """Compute overall head‑pose confidence.

    Parameters
    ----------
    head_pose : list or tuple of float
        Predicted yaw, pitch, roll angles.
    head_pose_th : list or tuple of float
        Thresholds for each angle.

    Returns
    -------
    float
        Confidence in [0.0, 1.0]
    """
    if len(head_pose) != 3 or len(head_pose_th) != 3:
        return 0.0

    angle_confs = (
        _generate_angle_conf(angle, threshold)
        for angle, threshold in zip(head_pose, head_pose_th)
    )
    return generate_model_conf(angle_confs)

