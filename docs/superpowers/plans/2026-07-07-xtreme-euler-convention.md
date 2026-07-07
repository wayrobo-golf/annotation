# Xtreme Euler Convention Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Xtreme `rotation3D.x/y/z` export/import so boxes displayed in Xtreme use the same 6DoF pose matrix computed by the map-to-lidar pipeline.

**Architecture:** Keep the existing map/XML -> KITTI -> Xtreme matrix pipeline. Add one shared helper that defines Xtreme Euler as intrinsic `XYZ`, then update both export and import scripts to use that helper. Tests cover the Golf_Cart-style high-yaw case where extrinsic `xyz` causes one bottom corner to sink and the opposite corner to lift.

**Tech Stack:** Python, SciPy `scipy.spatial.transform.Rotation`, pytest.

---

## File Structure

- Create: `pre_annotation_factory/my_package/workflow/xtreme_rotation.py`
  - Owns the Xtreme Euler convention in one place.
  - Exposes `matrix_to_xtreme_euler(rotation_matrix)` and `xtreme_euler_to_matrix(rotation3d)`.
- Create: `pre_annotation_factory/tests/test_xtreme_rotation.py`
  - Unit tests for intrinsic `XYZ` round trip and the Scene_191 Golf_Cart regression.
- Modify: `pre_annotation_factory/scripts/replay_rosbag_main.py`
  - Use `matrix_to_xtreme_euler()` when writing `contour.rotation3D`.
- Modify: `pre_annotation_factory/scripts/MergeXtremeDynamic.py`
  - Use `xtreme_euler_to_matrix()` when reading `contour.rotation3D`.
- Modify: `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`
  - Add a regression test proving `parse_xtreme_to_kitti_lines()` interprets Xtreme rotations with the same convention used by Xtreme UI.

## Task 1: Red Stage - Add Xtreme Euler Helper Tests

**Files:**
- Create: `pre_annotation_factory/tests/test_xtreme_rotation.py`

- [ ] **Step 1: Write failing tests**

Create `pre_annotation_factory/tests/test_xtreme_rotation.py` with:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

from my_package.workflow.xtreme_rotation import (
    matrix_to_xtreme_euler,
    xtreme_euler_to_matrix,
)


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
```

Also add the missing import:

```python
import pytest
```

- [ ] **Step 2: Run test and confirm red**

Run:

```bash
pytest pre_annotation_factory/tests/test_xtreme_rotation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'my_package.workflow.xtreme_rotation'`.

- [ ] **Step 3: Commit red stage**

Commit message:

```bash
test: init cases for xtreme euler convention
```

## Task 2: Green Stage - Implement Shared Xtreme Rotation Helper

**Files:**
- Create: `pre_annotation_factory/my_package/workflow/xtreme_rotation.py`

- [ ] **Step 1: Implement minimal helper**

Create `pre_annotation_factory/my_package/workflow/xtreme_rotation.py`:

```python
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
```

- [ ] **Step 2: Run helper tests**

Run:

```bash
pytest pre_annotation_factory/tests/test_xtreme_rotation.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit green helper**

Commit message:

```bash
feat: implement xtreme euler convention and pass tests
```

## Task 3: Green Stage - Use Helper When Exporting Xtreme JSON

**Files:**
- Modify: `pre_annotation_factory/scripts/replay_rosbag_main.py`

- [ ] **Step 1: Import helper**

Near existing imports in `pre_annotation_factory/scripts/replay_rosbag_main.py`, add:

```python
from my_package.workflow.xtreme_rotation import matrix_to_xtreme_euler
```

- [ ] **Step 2: Replace exported Euler conversion**

Replace:

```python
rx_l, ry_l, rz_l = R.from_matrix(R_lidar).as_euler('xyz')
```

with:

```python
rx_l, ry_l, rz_l = matrix_to_xtreme_euler(R_lidar)
```

- [ ] **Step 3: Add an export regression test**

Create a focused test in `pre_annotation_factory/tests/test_xtreme_rotation.py`:

```python
def test_golf_cart_pose_would_tilt_if_exported_with_extrinsic_xyz():
    desired_matrix = R.from_euler(
        "xyz",
        np.deg2rad([-1.37110717, -9.53044021, -155.316584]),
    ).as_matrix()
    wrong_euler = R.from_matrix(desired_matrix).as_euler("xyz")
    correct_euler = matrix_to_xtreme_euler(desired_matrix)

    wrong_ui_matrix = R.from_euler("XYZ", wrong_euler).as_matrix()
    correct_ui_matrix = R.from_euler("XYZ", correct_euler).as_matrix()

    assert np.rad2deg(R.from_matrix(desired_matrix.T @ wrong_ui_matrix).magnitude()) > 18.0
    assert np.rad2deg(R.from_matrix(desired_matrix.T @ correct_ui_matrix).magnitude()) < 1e-6
```

- [ ] **Step 4: Run rotation tests**

Run:

```bash
pytest pre_annotation_factory/tests/test_xtreme_rotation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit export fix**

Commit message:

```bash
feat: export xtreme rotations with intrinsic XYZ and pass tests
```

## Task 4: Green Stage - Use Helper When Importing Xtreme JSON

**Files:**
- Modify: `pre_annotation_factory/scripts/MergeXtremeDynamic.py`
- Modify: `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`

- [ ] **Step 1: Import helper**

Near existing imports in `pre_annotation_factory/scripts/MergeXtremeDynamic.py`, add:

```python
from my_package.workflow.xtreme_rotation import xtreme_euler_to_matrix
```

- [ ] **Step 2: Replace imported Euler conversion**

Replace:

```python
R_lidar_to_box = R.from_euler('xyz', [rot["x"], rot["y"], rot["z"]]).as_matrix()
```

with:

```python
R_lidar_to_box = xtreme_euler_to_matrix(rot)
```

- [ ] **Step 3: Add parse regression test**

Append to `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`:

```python
def test_parse_xtreme_to_kitti_lines_reads_intrinsic_xyz_rotation(tmp_path: Path):
    module = load_merge_module()
    config_path = tmp_path / "camera_config.json"
    xtreme_json_path = tmp_path / "frame.json"
    write_camera_config(config_path)

    desired_lidar_matrix = module.R.from_euler(
        "xyz",
        [-0.023930334527414, -0.16633756081770867, -2.7107857737142087],
    ).as_matrix()
    xtreme_euler = module.R.from_matrix(desired_lidar_matrix).as_euler("XYZ")
    xtreme_json_path.write_text(
        json.dumps(
            [
                {
                    "objects": [
                        {
                            "type": "3D_BOX",
                            "className": "Golf_Cart",
                            "contour": {
                                "size3D": {"x": 5.03, "y": 2.05, "z": 1.95},
                                "center3D": {"x": 1.904, "y": -2.737, "z": -1.051},
                                "rotation3D": {
                                    "x": xtreme_euler[0],
                                    "y": xtreme_euler[1],
                                    "z": xtreme_euler[2],
                                },
                            },
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    lines = module.parse_xtreme_to_kitti_lines(xtreme_json_path, config_path)

    parts = lines[0].split()
    ry = float(parts[14])
    rx = float(parts[15])
    rz = float(parts[16])
    parsed_matrix = module.R.from_euler("xyz", [rx, ry, rz]).as_matrix()
    expected_matrix = desired_lidar_matrix @ module.R_xtreme2kitti
    np.testing.assert_allclose(parsed_matrix, expected_matrix, atol=1e-4)
```

Also add the missing import near the top of `test_merge_xtreme_dynamic.py`:

```python
import numpy as np
```

- [ ] **Step 4: Run merge test**

Run:

```bash
pytest pre_annotation_factory/tests/test_merge_xtreme_dynamic.py::test_parse_xtreme_to_kitti_lines_reads_intrinsic_xyz_rotation -v
```

Expected: PASS.

- [ ] **Step 5: Run related test file**

Run:

```bash
pytest pre_annotation_factory/tests/test_xtreme_rotation.py pre_annotation_factory/tests/test_merge_xtreme_dynamic.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit import fix**

Commit message:

```bash
feat: import xtreme rotations with intrinsic XYZ and pass tests
```

## Task 5: Verification With Scene_191 Data

**Files:**
- No source files changed.

- [ ] **Step 1: Run a one-off Scene_191 numerical verification**

Run:

```bash
python3 - <<'PY'
import json, pathlib, numpy as np, math
from scipy.spatial.transform import Rotation as R
from pre_annotation_factory.my_package.workflow.xtreme_rotation import matrix_to_xtreme_euler

root = pathlib.Path('/home/keyaoli/Data/AutoAnnotation/Server Log/20260707-AnnotationAnalysisOnUnloadStation')
frame = '1782450090900391000'
conf = json.load(open(root / 'Scene_191_Origin/camera_config' / f'{frame}.json'))[0]
T = np.array(conf['tf_lidar_to_map']).reshape(4, 4, order='F')
map_up = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
objs = json.load(open(root / 'Scene_191_XtremeOutput/result' / f'{frame}.json'))[0]['objects']

for obj in objs:
    cls = obj.get('modelClass') or obj.get('className')
    if cls not in {'Unloading_Station', 'Golf_Cart'}:
        continue
    rot = obj['contour']['rotation3D']
    exported = np.array([rot['x'], rot['y'], rot['z']])
    desired_matrix = R.from_euler('xyz', exported).as_matrix()
    fixed_euler = matrix_to_xtreme_euler(desired_matrix)
    ui_matrix = R.from_euler('XYZ', fixed_euler).as_matrix()
    up = ui_matrix[:, 2]
    if up[2] < 0:
        up = -up
    angle = math.degrees(math.acos(np.clip(up.dot(map_up), -1.0, 1.0)))
    print(cls, np.round(np.rad2deg(fixed_euler), 3), 'up-map angle deg:', round(angle, 3))
PY
```

Expected output includes:

```text
Unloading_Station ... up-map angle deg: about 1.3
Golf_Cart ... up-map angle deg: about 0.05
```

- [ ] **Step 2: Regenerate a small Xtreme sample**

Use the existing replay/export flow for a small sample containing frame `1782450090900391000`. Confirm the generated `Scene_191/result/1782450090900391000.json` has Golf_Cart rotation near:

```text
x ≈ -2.751°
y ≈  9.230°
z ≈ -155.209°
```

- [ ] **Step 3: Visual check in Xtreme**

Load the regenerated frame in Xtreme. Expected:

```text
Golf_Cart bottom plane no longer has one corner buried and opposite corner lifted.
Unloading_Station remains close to ground, with only small residual height/box-size mismatch.
```

## Task 6: Refactor Stage - Keep Convention Centralized

**Files:**
- Modify only if needed: `pre_annotation_factory/my_package/workflow/xtreme_rotation.py`

- [ ] **Step 1: Scan for remaining direct Xtreme Euler conversions**

Run:

```bash
rg -n "as_euler\\('xyz'\\)|from_euler\\('xyz'|rotation3D|R_lidar_to_box|rx_l, ry_l, rz_l" pre_annotation_factory
```

Expected: direct `rotation3D` matrix conversion should go through `xtreme_rotation.py`. Other KITTI/internal `xyz` conversions may remain.

- [ ] **Step 2: Run final focused tests**

Run:

```bash
pytest pre_annotation_factory/tests/test_xtreme_rotation.py pre_annotation_factory/tests/test_merge_xtreme_dynamic.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit refactor stage if any cleanup was needed**

Commit message:

```bash
refactor: optimize xtreme euler convention logic
```

## Self-Review

- Spec coverage: The plan covers export, import, helper centralization, unit tests, and Scene_191 numerical verification.
- Placeholder scan: No placeholder implementation steps are present.
- Type consistency: Helper functions use `np.ndarray` matrices and Xtreme `rotation3D` dictionaries consistently across tasks.
