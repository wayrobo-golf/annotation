# Static Instance Track ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve stable Xtreme `trackId` / `trackName` for static XML instances across frames and Scenes, while keeping final KITTI labels free of track fields.

**Architecture:** Bind track identity once while parsing XML into `BoundingBox`, append it only to intermediate C++ KITTI labels, and let Python transfer those fields into Xtreme `result/*.json`. Finalize keeps track fields in temporary Xtreme result JSON but emits clean 17-field KITTI labels for downstream compatibility.

**Tech Stack:** ROS2 Humble, C++17, TinyXML2, PCL, OpenCV, Python 3, pytest, GoogleTest, Xtreme1 JSON import format

---

## File Structure

- Modify: `automatic_annotation/include/automatic_annotation/gen_prompt_point.hpp`
  - Add `track_id` and `track_name` to `BoundingBox`.
  - Add private helper declarations for stable track formatting.
- Modify: `automatic_annotation/src/gen_prompt_point.cpp`
  - Generate `track_id` / `track_name` in `LoadLabelsFromXML()`.
  - Append track fields in `GenerateKITTILabel()` intermediate labels.
  - Implement helper functions for class normalization and zero padding.
- Modify: `pre_annotation_factory/scripts/replay_rosbag_main.py`
  - Parse optional track fields in `convert_kitti_to_xtreme1_json()`.
  - Write `trackId`, `trackName`, and `modelClass` to Xtreme objects when available.
- Modify: `pre_annotation_factory/my_package/workflow/xtreme_gateway.py`
  - Preserve `trackId` / `trackName` when rebuilding Xtreme export result JSON.
- Modify: `pre_annotation_factory/scripts/MergeXtremeDynamic.py`
  - Accept result objects containing `trackId` / `trackName`.
  - Keep final KITTI output clean: do not append track fields.
- Modify: `pre_annotation_factory/tests/test_annotationctl_pipeline.py`
  - Add conversion tests for tracked and untracked intermediate KITTI labels.
- Modify: `pre_annotation_factory/tests/test_xtreme_gateway.py`
  - Add export rebuild test that preserves track fields.
- Modify: `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`
  - Add final KITTI output test proving track fields are not emitted.
- Modify or create: `automatic_annotation/test/test_static_track_id.cpp`
  - Add C++ helper tests for stable track formatting if helper is made visible in a small header; otherwise add tests to an existing C++ test file.
- Modify: `automatic_annotation/CMakeLists.txt`
  - Register the C++ helper test if a new test file is created.

## Task 1: Add Python Tests For KITTI -> Xtreme Track Transfer

**Files:**
- Modify: `pre_annotation_factory/tests/test_annotationctl_pipeline.py`
- Modify later: `pre_annotation_factory/scripts/replay_rosbag_main.py`

- [ ] **Step 1: Write the failing test for tracked intermediate labels**

Add this test near existing `replay_rosbag_main.py` tests:

```python
def test_convert_kitti_to_xtreme1_json_preserves_static_track_fields(tmp_path):
    module = load_replay_module()
    config_path = tmp_path / "camera_config.json"
    label_path = tmp_path / "label.txt"
    output_path = tmp_path / "result.json"

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
        "Distance_Marker 0.00 0 0.00 10.00 20.00 30.00 40.00 "
        "1.7400 1.7800 0.6900 1.0000 2.0000 10.0000 "
        "0.1000 0.0200 0.0300 static_Distance_Marker_000017 17\n",
        encoding="utf-8",
    )

    module.convert_kitti_to_xtreme1_json(label_path, config_path, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["objects"]) == 1
    obj = payload["objects"][0]
    assert obj["type"] == "3D_BOX"
    assert obj["className"] == "Distance_Marker"
    assert obj["modelClass"] == "Distance_Marker"
    assert obj["trackId"] == "static_Distance_Marker_000017"
    assert obj["trackName"] == "17"
    assert obj["id"] != obj["trackId"]
```

- [ ] **Step 2: Write the compatibility test for old 17-field labels**

Add:

```python
def test_convert_kitti_to_xtreme1_json_keeps_old_labels_untracked(tmp_path):
    module = load_replay_module()
    config_path = tmp_path / "camera_config.json"
    label_path = tmp_path / "label.txt"
    output_path = tmp_path / "result.json"

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
        "Bucket 0.00 0 0.00 10.00 20.00 30.00 40.00 "
        "1.0000 1.0000 1.0000 1.0000 2.0000 10.0000 "
        "0.1000 0.0200 0.0300\n",
        encoding="utf-8",
    )

    module.convert_kitti_to_xtreme1_json(label_path, config_path, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    obj = payload["objects"][0]
    assert obj["className"] == "Bucket"
    assert obj["modelClass"] == "Bucket"
    assert "trackId" not in obj
    assert "trackName" not in obj
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py::test_convert_kitti_to_xtreme1_json_preserves_static_track_fields \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py::test_convert_kitti_to_xtreme1_json_keeps_old_labels_untracked \
  -q
```

Expected: first test fails because `modelClass`, `trackId`, and `trackName` are not written by `convert_kitti_to_xtreme1_json()`.

- [ ] **Step 4: Implement minimal Python conversion support**

In `pre_annotation_factory/scripts/replay_rosbag_main.py`, inside `convert_kitti_to_xtreme1_json()`, after reading `rx` and `rz`, add optional field parsing:

```python
track_id = parts[17] if len(parts) >= 19 else None
track_name = parts[18] if len(parts) >= 19 else None
```

Then build the object in two steps:

```python
obj = {
    "id": str(uuid.uuid4()),
    "type": "3D_BOX",
    "className": class_name,
    "modelClass": class_name,
    "contour": {
        "size3D": {"x": l, "y": w, "z": h},
        "center3D": {
            "x": center_lidar[0],
            "y": center_lidar[1],
            "z": center_lidar[2],
        },
        "rotation3D": {"x": rx_l, "y": ry_l, "z": rz_l},
    },
}
if track_id and track_name:
    obj["trackId"] = track_id
    obj["trackName"] = track_name
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py::test_convert_kitti_to_xtreme1_json_preserves_static_track_fields \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py::test_convert_kitti_to_xtreme1_json_keeps_old_labels_untracked \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add pre_annotation_factory/scripts/replay_rosbag_main.py \
        pre_annotation_factory/tests/test_annotationctl_pipeline.py
git commit -m "test: init cases for static instance track ids"
```

## Task 2: Add C++ Static Track Fields To XML Labels

**Files:**
- Modify: `automatic_annotation/include/automatic_annotation/gen_prompt_point.hpp`
- Modify: `automatic_annotation/src/gen_prompt_point.cpp`
- Modify or create: `automatic_annotation/test/test_static_track_id.cpp`
- Modify if new test file is created: `automatic_annotation/CMakeLists.txt`

- [ ] **Step 1: Add a failing C++ helper test**

Create `automatic_annotation/test/test_static_track_id.cpp`:

```cpp
#include <gtest/gtest.h>

#include "automatic_annotation/gen_prompt_point.hpp"

namespace automatic_annotation {
namespace {

TEST(StaticTrackIdTest, FormatsTrackNameAsPlainGlobalIndex) {
  EXPECT_EQ(FormatStaticTrackNameForTest(17), "17");
}

TEST(StaticTrackIdTest, FormatsTrackIdWithClassAndZeroPaddedIndex) {
  EXPECT_EQ(FormatStaticTrackIdForTest("Distance_Marker", 17),
            "static_Distance_Marker_000017");
}

TEST(StaticTrackIdTest, NormalizesSpacesInClassName) {
  EXPECT_EQ(FormatStaticTrackIdForTest("Distance Marker", 5),
            "static_Distance_Marker_000005");
}

}  // namespace
}  // namespace automatic_annotation
```

Add to `automatic_annotation/CMakeLists.txt` inside `if(BUILD_TESTING)`:

```cmake
ament_add_gtest(
  test_static_track_id
  test/test_static_track_id.cpp
  src/gen_prompt_point.cpp
  src/image_matching_utils.cpp
)
if(TARGET test_static_track_id)
  target_include_directories(test_static_track_id PRIVATE include)
  ament_target_dependencies(test_static_track_id
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    common_utils
    pcl_conversions
    Eigen3
    cv_bridge_with_opencv411
    tf2
    tf2_ros
    tf2_eigen
  )
  target_include_directories(test_static_track_id PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TINYXML2_INCLUDE_DIRS}
  )
  target_link_libraries(test_static_track_id
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${TINYXML2_LIBRARIES}
  )
endif()
```

If linking `gen_prompt_point.cpp` into this helper test is too heavy in the local build, replace this with a small header-only helper `automatic_annotation/include/automatic_annotation/static_track_id_utils.hpp` and test only that helper. Keep the public behavior identical to the snippets below.

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
colcon test --packages-select automatic_annotation --event-handlers console_direct+ --ctest-args -R test_static_track_id
```

Expected: FAIL because `FormatStaticTrackNameForTest` and `FormatStaticTrackIdForTest` do not exist.

- [ ] **Step 3: Add fields and formatting helpers**

In `BoundingBox` in `gen_prompt_point.hpp`, add:

```cpp
std::string track_id;
std::string track_name;
```

Add helper declarations. If keeping helpers in `gen_prompt_point.hpp` for tests:

```cpp
std::string FormatStaticTrackNameForTest(size_t instance_index);
std::string FormatStaticTrackIdForTest(const std::string& class_name,
                                       size_t instance_index);
```

In `gen_prompt_point.cpp`, implement:

```cpp
std::string NormalizeStaticTrackClassName(std::string class_name) {
  for (char& ch : class_name) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      ch = '_';
    }
  }
  return class_name;
}

std::string FormatStaticTrackName(size_t instance_index) {
  return std::to_string(instance_index);
}

std::string FormatStaticTrackId(const std::string& class_name,
                                size_t instance_index) {
  std::ostringstream oss;
  oss << "static_" << NormalizeStaticTrackClassName(class_name) << "_"
      << std::setw(6) << std::setfill('0') << instance_index;
  return oss.str();
}

std::string FormatStaticTrackNameForTest(size_t instance_index) {
  return FormatStaticTrackName(instance_index);
}

std::string FormatStaticTrackIdForTest(const std::string& class_name,
                                       size_t instance_index) {
  return FormatStaticTrackId(class_name, instance_index);
}
```

- [ ] **Step 4: Bind track identity during XML parsing**

In `LoadLabelsFromXML()`, after a class is accepted and before `annotation_status_.current_labels.push_back(box);`, assign:

```cpp
const size_t static_instance_index =
    annotation_status_.current_labels.size() + 1;
box.track_name = FormatStaticTrackName(static_instance_index);
box.track_id = FormatStaticTrackId(box.object_type, static_instance_index);
```

Do not increment a separate frame-local counter in `GenerateKITTILabel()`.

- [ ] **Step 5: Append track fields to intermediate C++ labels**

In `GenerateKITTILabel()`, extend the `label_ofs` line after `rz`:

```cpp
label_ofs << std::fixed << std::setprecision(4) << class_name << " "
          << truncation << " " << occlusion << " " << alpha_str << " "
          << bbox_str << " " << std::setprecision(4) << box.h << " "
          << box.w << " " << box.l << " " << bottom_cam.x() << " "
          << bottom_cam.y() << " " << bottom_cam.z() << " " << ry << " "
          << rx << " " << rz;
if (!box.track_id.empty() && !box.track_name.empty()) {
  label_ofs << " " << box.track_id << " " << box.track_name;
}
label_ofs << "\n";
```

- [ ] **Step 6: Run C++ tests**

Run:

```bash
colcon test --packages-select automatic_annotation --event-handlers console_direct+
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add automatic_annotation/include/automatic_annotation/gen_prompt_point.hpp \
        automatic_annotation/src/gen_prompt_point.cpp \
        automatic_annotation/test/test_static_track_id.cpp \
        automatic_annotation/CMakeLists.txt
git commit -m "feat: implement static instance track ids and pass tests"
```

## Task 3: Preserve Track Fields Through Xtreme Export Rebuild

**Files:**
- Modify: `pre_annotation_factory/my_package/workflow/xtreme_gateway.py`
- Modify: `pre_annotation_factory/tests/test_xtreme_gateway.py`

- [ ] **Step 1: Write failing export rebuild test**

Add to `pre_annotation_factory/tests/test_xtreme_gateway.py`:

```python
def test_rebuild_result_tree_preserves_track_fields(tmp_path: Path):
    from my_package.workflow.xtreme_gateway import rebuild_result_tree_from_export_data

    export_root = tmp_path / "export"
    data_dir = export_root / "Scene_01" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "frame_001.json").write_text(
        json.dumps({"dataId": 1001, "name": "frame_001"}),
        encoding="utf-8",
    )

    def fake_fetch_annotations(data_ids):
        assert data_ids == ["1001"]
        return [
            {
                "dataId": 1001,
                "objects": [
                    {
                        "className": "Distance_Marker",
                        "classAttributes": {
                            "type": "3D_BOX",
                            "modelClass": "Distance_Marker",
                            "trackId": "static_Distance_Marker_000017",
                            "trackName": "17",
                            "contour": {
                                "size3D": {"x": 1.0, "y": 1.0, "z": 1.0},
                                "center3D": {"x": 1.0, "y": 2.0, "z": 3.0},
                                "rotation3D": {"x": 0.0, "y": 0.0, "z": 0.0},
                            },
                        },
                    }
                ],
            }
        ]

    rebuild_result_tree_from_export_data(export_root, fake_fetch_annotations)

    result = json.loads(
        (export_root / "Scene_01" / "result" / "frame_001.json").read_text(
            encoding="utf-8"
        )
    )
    obj = result[0]["objects"][0]
    assert obj["trackId"] == "static_Distance_Marker_000017"
    assert obj["trackName"] == "17"
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_xtreme_gateway.py::test_rebuild_result_tree_preserves_track_fields \
  -q
```

Expected: FAIL because `rebuild_result_tree_from_export_data()` drops `trackId` and `trackName`.

- [ ] **Step 3: Preserve fields in rebuild**

In `xtreme_gateway.py`, inside the object append block, change from a literal-only dict to:

```python
rebuilt_object = {
    "type": class_attributes.get("type"),
    "className": obj.get("className"),
    "modelClass": class_attributes.get("modelClass"),
    "contour": class_attributes.get("contour"),
}
if class_attributes.get("trackId") is not None:
    rebuilt_object["trackId"] = class_attributes.get("trackId")
if class_attributes.get("trackName") is not None:
    rebuilt_object["trackName"] = class_attributes.get("trackName")
objects.append(rebuilt_object)
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_xtreme_gateway.py::test_rebuild_result_tree_preserves_track_fields \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add pre_annotation_factory/my_package/workflow/xtreme_gateway.py \
        pre_annotation_factory/tests/test_xtreme_gateway.py
git commit -m "feat: preserve xtreme track ids during export rebuild"
```

## Task 4: Prove Final KITTI Output Stays Clean

**Files:**
- Modify: `pre_annotation_factory/scripts/MergeXtremeDynamic.py`
- Modify: `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`

- [ ] **Step 1: Write failing or guarding test**

Add to `pre_annotation_factory/tests/test_merge_xtreme_dynamic.py`:

```python
def test_parse_xtreme_to_kitti_lines_does_not_emit_track_fields(tmp_path):
    module = load_merge_module()
    config_path = tmp_path / "camera_config.json"
    xtreme_json_path = tmp_path / "result.json"

    write_camera_config(config_path, include_tf_lidar_to_map=True)
    xtreme_json_path.write_text(
        json.dumps(
            [
                {
                    "objects": [
                        {
                            "type": "3D_BOX",
                            "className": "Distance_Marker",
                            "modelClass": "Distance_Marker",
                            "trackId": "static_Distance_Marker_000017",
                            "trackName": "17",
                            "contour": {
                                "size3D": {"x": 1.0, "y": 1.0, "z": 1.0},
                                "center3D": {"x": 1.0, "y": 2.0, "z": 10.0},
                                "rotation3D": {"x": 0.0, "y": 0.0, "z": 0.0},
                            },
                        }
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    lines = module.parse_xtreme_to_kitti_lines(xtreme_json_path, config_path)

    assert len(lines) == 1
    parts = lines[0].strip().split()
    assert len(parts) == 17
    assert "static_Distance_Marker_000017" not in parts
    assert "17" not in parts[17:]
```

- [ ] **Step 2: Run test**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_merge_xtreme_dynamic.py::test_parse_xtreme_to_kitti_lines_does_not_emit_track_fields \
  -q
```

Expected: PASS if current output is already clean. If it fails because parsing expects only a subset of object keys, adjust parsing to ignore `trackId` / `trackName`.

- [ ] **Step 3: Keep parse logic tolerant**

If needed, ensure `parse_xtreme_to_kitti_lines()` only reads:

```python
s = obj["contour"]["size3D"]
c = obj["contour"]["center3D"]
rot = obj["contour"]["rotation3D"]
```

Do not append track fields to:

```python
line = (f"{class_name} {trunc} {occ} {alpha_val} {bbox_str} "
        f"{h:.4f} {w:.4f} {l:.4f} "
        f"{bottom_cam[0]:.4f} {bottom_cam[1]:.4f} {bottom_cam[2]:.4f} "
        f"{ry:.4f} {rx:.4f} {rz:.4f}\n")
```

- [ ] **Step 4: Commit Task 4**

```bash
git add pre_annotation_factory/scripts/MergeXtremeDynamic.py \
        pre_annotation_factory/tests/test_merge_xtreme_dynamic.py
git commit -m "refactor: keep final kitti labels free of track fields"
```

## Task 5: Add Real Xtreme Upload Shape Regression Test

**Files:**
- Modify: `pre_annotation_factory/tests/test_annotationctl_pipeline.py`

- [ ] **Step 1: Add test for Scene-shaped result JSON conversion**

Add a unit-level test that does not call Xtreme but proves the generated `result/*.json` object shape contains track fields:

```python
def test_xtreme_object_shape_matches_scene_import_track_fields(tmp_path):
    module = load_replay_module()
    config_path = tmp_path / "Scene_01" / "camera_config" / "frame_001.json"
    label_path = tmp_path / "Scene_01" / "label_2" / "frame_001.txt"
    output_path = tmp_path / "Scene_01" / "result" / "frame_001.json"
    config_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)

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
        "Distance_Marker 0.00 0 0.00 10.00 20.00 30.00 40.00 "
        "1.7400 1.7800 0.6900 1.0000 2.0000 10.0000 "
        "0.1000 0.0200 0.0300 static_Distance_Marker_000017 17\n",
        encoding="utf-8",
    )

    module.convert_kitti_to_xtreme1_json(label_path, config_path, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "objects": [
            {
                "id": payload["objects"][0]["id"],
                "type": "3D_BOX",
                "className": "Distance_Marker",
                "modelClass": "Distance_Marker",
                "trackId": "static_Distance_Marker_000017",
                "trackName": "17",
                "contour": payload["objects"][0]["contour"],
            }
        ]
    }
```

- [ ] **Step 2: Run test**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py::test_xtreme_object_shape_matches_scene_import_track_fields \
  -q
```

Expected: PASS after Task 1.

- [ ] **Step 3: Commit Task 5**

```bash
git add pre_annotation_factory/tests/test_annotationctl_pipeline.py
git commit -m "test: cover scene-shaped xtreme track output"
```

## Task 6: Full Verification

**Files:**
- No required file changes unless tests reveal issues.

- [ ] **Step 1: Run targeted Python tests**

Run:

```bash
PYTHONPATH=pre_annotation_factory pytest \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py \
  pre_annotation_factory/tests/test_xtreme_gateway.py \
  pre_annotation_factory/tests/test_merge_xtreme_dynamic.py \
  pre_annotation_factory/tests/test_extrinsics_conversion.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run targeted C++ tests**

Run:

```bash
colcon test --packages-select automatic_annotation --event-handlers console_direct+
```

Expected: PASS.

- [ ] **Step 3: Run a real-package smoke verification against local Xtreme**

Use a copied subset of:

```text
/home/keyaoli/Data/AutoAnnotation/auto_work_flow/YinXiu/jobs/4bcf7749a399/artifacts/xtreme_upload/Xtreme1_Upload_20260421142503_YinXiu20260414
```

Create a temporary package with `Scene_01` and `Scene_02`, at least two tracked frames per Scene, upload to local Xtreme, then query:

```bash
curl -H "Authorization: Bearer ${XTREME1_TOKEN}" \
  "http://127.0.0.1:8190/api/data/findByPage?datasetId=<dataset_id>&parentId=<scene_data_id>&pageNo=1&pageSize=50"
```

Then:

```bash
curl -H "Authorization: Bearer ${XTREME1_TOKEN}" \
  "http://127.0.0.1:8190/api/annotate/data/listByDataIds?dataIds=<comma_separated_frame_ids>"
```

Expected: frames from both Scenes that contain the same static XML instance have identical `classAttributes.trackId` and `classAttributes.trackName`.

- [ ] **Step 4: Confirm final KITTI labels are clean**

Run a small finalize fixture or inspect the unit test output. For any final label line:

```bash
awk '{print NF}' path/to/final_dataset/training/label_2/*.txt | sort -u
```

Expected: `17` for labels containing the current 6-DoF extension, never `19`.

- [ ] **Step 5: Final commit if any verification-only fixes were needed**

```bash
git add <changed_files>
git commit -m "refactor: optimize static instance track id logic"
```

## Self-Review Checklist

- Spec coverage:
  - Stable XML-bound track identity: Task 2.
  - Xtreme pre-annotation track transfer: Task 1 and Task 5.
  - Export rebuild preserving tracks: Task 3.
  - Final KITTI without track fields: Task 4 and Task 6.
  - Real Scene/cross-Scene validation: Task 6.
- Placeholder scan: no planned implementation step uses undefined behavior or leaves a field unspecified.
- Type consistency:
  - C++ uses `track_id` / `track_name`.
  - Xtreme JSON uses `trackId` / `trackName`.
  - Intermediate label order is `track_id track_name` after the existing 17 fields.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-27-static-instance-track-id.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Choose the execution mode before implementation starts.
