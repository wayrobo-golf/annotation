import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from my_package.workflow.xtreme_rotation import (
    matrix_to_xtreme_euler,
    xtreme_euler_to_matrix,
)


def load_replay_module(monkeypatch):
    io_modules = types.ModuleType("my_package.io_modules")
    nusc_exporter = types.ModuleType("my_package.io_modules.nusc_json_exporter")
    nusc_exporter.generate_nuscenes_metadata = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "my_package.io_modules", io_modules)
    monkeypatch.setitem(
        sys.modules, "my_package.io_modules.nusc_json_exporter", nusc_exporter
    )

    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "replay_rosbag_main.py"
    )
    spec = importlib.util.spec_from_file_location("replay_rosbag_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matrix_to_xtreme_euler_uses_intrinsic_xyz_for_golf_cart_pose():
    desired_matrix = R.from_euler(
        "xyz",
        np.deg2rad([-1.37110717, -9.53044021, -155.316584]),
    ).as_matrix()

    xtreme_euler = matrix_to_xtreme_euler(desired_matrix)

    assert np.rad2deg(xtreme_euler) == pytest.approx(
        [-2.751, 9.23, -155.209],
        abs=0.01,
    )
    np.testing.assert_allclose(
        R.from_euler("XYZ", xtreme_euler).as_matrix(),
        desired_matrix,
        atol=1e-9,
    )


def test_xtreme_euler_to_matrix_is_inverse_of_matrix_to_xtreme_euler():
    desired_matrix = R.from_euler(
        "xyz",
        np.deg2rad([-0.78, 8.389, 8.349]),
    ).as_matrix()

    xtreme_euler = matrix_to_xtreme_euler(desired_matrix)
    actual_matrix = xtreme_euler_to_matrix(
        {"x": xtreme_euler[0], "y": xtreme_euler[1], "z": xtreme_euler[2]}
    )

    np.testing.assert_allclose(actual_matrix, desired_matrix, atol=1e-9)


def test_convert_kitti_to_xtreme1_json_exports_intrinsic_xyz_rotation(
    tmp_path: Path, monkeypatch
):
    module = load_replay_module(monkeypatch)
    config_path = tmp_path / "camera_config.json"
    label_path = tmp_path / "label.txt"
    out_json_path = tmp_path / "result.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "camera_internal": {
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 960.0,
                        "cy": 540.0,
                    },
                    "width": 1920,
                    "height": 1080,
                    "camera_external": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    label_path.write_text(
        "Golf_Cart 0.00 0 0.00 0.00 0.00 10.00 10.00 "
        "1.9500 2.0500 5.0300 1.0 2.0 3.0 "
        "0.9785 0.7929 0.6579 static_Golf_Cart_000012 12\n",
        encoding="utf-8",
    )

    module.convert_kitti_to_xtreme1_json(label_path, config_path, out_json_path)

    payload = json.loads(out_json_path.read_text(encoding="utf-8"))
    rotation3d = payload["objects"][0]["contour"]["rotation3D"]
    actual = np.array([rotation3d["x"], rotation3d["y"], rotation3d["z"]])
    rot_cam = R.from_euler("xyz", [0.7929, 0.9785, 0.6579]).as_matrix()
    kitti_to_xtreme_local = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    expected = matrix_to_xtreme_euler(rot_cam @ kitti_to_xtreme_local)

    np.testing.assert_allclose(actual, expected, atol=1e-9)
