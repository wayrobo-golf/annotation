import numpy as np
from scipy.spatial.transform import Rotation as R


XTREME_EULER_SEQUENCE = "XYZ"


def matrix_to_xtreme_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(rotation_matrix).as_euler(XTREME_EULER_SEQUENCE)


def xtreme_euler_to_matrix(rotation3d: dict) -> np.ndarray:
    return R.from_euler(
        XTREME_EULER_SEQUENCE,
        [rotation3d["x"], rotation3d["y"], rotation3d["z"]],
    ).as_matrix()
