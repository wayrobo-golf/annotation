from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


_REQUIRED_FIELD_ALIASES = {
    "quat_cam_to_lidar": ("quat_cam_to_lidar",),
    "trans_cam_to_lidar": ("trans_cam_to_lidar",),
    "quat_lidar_to_imu": ("quat_lidar_to_imu",),
    "trans_lidar_to_imu": ("trans_lidar_to_imu",),
    "quat_imu_to_ins": ("quat_imu_to_ins",),
    "trans_imu_to_ins": ("trans_imu2ins", "trans_imu_to_ins"),
}


def build_default_tf_overrides(
    extrinsics_path: Path,
    *,
    quaternion_order: str = "wxyz",
) -> dict[str, list[float]]:
    """Build automatic_annotation default TF params from raw extrinsics YAML."""
    payload = yaml.safe_load(Path(extrinsics_path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Extrinsics YAML is empty: {extrinsics_path}")

    fields = _extract_required_fields(payload)
    cam_to_lidar_rot = _rotation_from_quaternion(
        fields["quat_cam_to_lidar"], quaternion_order
    )
    cam_to_lidar_trans = _vector3(fields["trans_cam_to_lidar"], "trans_cam_to_lidar")

    lidar_to_imu_rot = _rotation_from_quaternion(
        fields["quat_lidar_to_imu"], quaternion_order
    )
    lidar_to_imu_trans = _vector3(
        fields["trans_lidar_to_imu"], "trans_lidar_to_imu"
    )
    imu_to_ins_rot = _rotation_from_quaternion(
        fields["quat_imu_to_ins"], quaternion_order
    )
    imu_to_ins_trans = _vector3(fields["trans_imu_to_ins"], "trans_imu_to_ins")

    lidar_to_ins_rot = imu_to_ins_rot * lidar_to_imu_rot
    lidar_to_ins_trans = imu_to_ins_rot.apply(lidar_to_imu_trans) + imu_to_ins_trans

    return {
        "default_tf_lcam_to_lidar": _xyz_rpy(cam_to_lidar_trans, cam_to_lidar_rot),
        "default_tf_lidar_to_ins": _xyz_rpy(lidar_to_ins_trans, lidar_to_ins_rot),
    }


def _extract_required_fields(payload: Any) -> dict[str, Any]:
    result = {}
    for semantic_name, aliases in _REQUIRED_FIELD_ALIASES.items():
        matches = list(_find_key_values(payload, set(aliases), ()))
        if not matches:
            raise ValueError(
                f"Missing required extrinsics field: {' or '.join(aliases)}"
            )
        if len(matches) > 1:
            paths = ", ".join(path for path, _value in matches)
            raise ValueError(
                f"Found multiple values for extrinsics field {semantic_name}: {paths}"
            )
        result[semantic_name] = matches[0][1]
    return result


def _find_key_values(
    value: Any,
    wanted_keys: set[str],
    path: tuple[str, ...],
):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            if key in wanted_keys:
                yield (".".join(child_path), child)
            yield from _find_key_values(child, wanted_keys, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _find_key_values(child, wanted_keys, path + (f"[{index}]",))


def _rotation_from_quaternion(quaternion: Any, quaternion_order: str) -> R:
    quat = _numeric_array(quaternion, 4, "quaternion")
    if quaternion_order == "wxyz":
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
    elif quaternion_order == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError(
            f"Unsupported quaternion_order: {quaternion_order}. "
            "Expected 'wxyz' or 'xyzw'."
        )
    return R.from_quat(quat_xyzw)


def _vector3(value: Any, field_name: str) -> np.ndarray:
    return np.array(_numeric_array(value, 3, field_name), dtype=np.float64)


def _numeric_array(value: Any, expected_size: int, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_size:
        raise ValueError(
            f"Extrinsics field {field_name} must be a list of {expected_size} numbers."
        )
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Extrinsics field {field_name} must be a list of {expected_size} numbers."
        ) from exc


def _xyz_rpy(translation: np.ndarray, rotation: R) -> list[float]:
    yaw, pitch, roll = rotation.as_euler("ZYX", degrees=True)
    values = [
        translation[0],
        translation[1],
        translation[2],
        roll,
        pitch,
        yaw,
    ]
    return [float(value) for value in values]
