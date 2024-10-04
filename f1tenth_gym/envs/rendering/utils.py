import numpy as np


def to_pygame(coords: np.ndarray, height: int) -> np.ndarray:
    """Convert coordinates into pygame coordinates (lower-left => top left)."""
    if coords.ndim == 1:
        return np.array([coords[0], height - coords[1]])
    return np.array([coords[:, 0], height - coords[:, 1]]).T