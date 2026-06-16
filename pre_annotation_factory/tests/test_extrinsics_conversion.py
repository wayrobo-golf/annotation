from pathlib import Path

import pytest

from my_package.workflow.extrinsics import build_default_tf_overrides


def test_build_default_tf_overrides_from_raw_extrinsics_yaml(tmp_path: Path):
    extrinsics_path = tmp_path / "extrinsics.yaml"
    extrinsics_path.write_text(
        "\n".join(
            [
                "calibration:",
                "  quat_cam_to_lidar: [0.7071067811865476, 0, 0, 0.7071067811865476]",
                "  trans_cam_to_lidar: [1.0, 2.0, 3.0]",
                "  lidar_to_imu:",
                "    quat_lidar_to_imu: [1.0, 0, 0, 0]",
                "    trans_lidar_to_imu: [0.5, 0.25, 1.0]",
                "  imu_to_ins:",
                "    quat_imu_to_ins: [0.7071067811865476, 0, 0, 0.7071067811865476]",
                "    trans_imu2ins: [10.0, 20.0, 30.0]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    overrides = build_default_tf_overrides(extrinsics_path)

    assert overrides["default_tf_lcam_to_lidar"] == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, 90.0], abs=1e-6
    )
    assert overrides["default_tf_lidar_to_ins"] == pytest.approx(
        [9.75, 20.5, 31.0, 0.0, 0.0, 90.0], abs=1e-6
    )


def test_build_default_tf_overrides_rejects_missing_required_field(tmp_path: Path):
    extrinsics_path = tmp_path / "extrinsics.yaml"
    extrinsics_path.write_text(
        "\n".join(
            [
                "quat_cam_to_lidar: [1.0, 0, 0, 0]",
                "trans_cam_to_lidar: [0, 0, 0]",
                "quat_lidar_to_imu: [1.0, 0, 0, 0]",
                "trans_lidar_to_imu: [0, 0, 0]",
                "quat_imu_to_ins: [1.0, 0, 0, 0]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="trans_imu"):
        build_default_tf_overrides(extrinsics_path)


def test_build_default_tf_overrides_rejects_duplicate_semantic_field(
    tmp_path: Path,
):
    extrinsics_path = tmp_path / "extrinsics.yaml"
    extrinsics_path.write_text(
        "\n".join(
            [
                "quat_cam_to_lidar: [1.0, 0, 0, 0]",
                "trans_cam_to_lidar: [0, 0, 0]",
                "quat_lidar_to_imu: [1.0, 0, 0, 0]",
                "trans_lidar_to_imu: [0, 0, 0]",
                "quat_imu_to_ins: [1.0, 0, 0, 0]",
                "trans_imu2ins: [0, 0, 0]",
                "nested:",
                "  trans_imu_to_ins: [1, 2, 3]",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple values"):
        build_default_tf_overrides(extrinsics_path)
