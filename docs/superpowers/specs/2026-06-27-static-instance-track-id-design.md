# Static Instance Track ID 设计方案

## 1. 背景

自动预标注流水线会从全局 XML 静态标注生成中间 KITTI label，再由 Python 转成 Xtreme1 上传包中的 `result/*.json`。XML 中的静态障碍物天然是实例级对象：每个 `<tracklets>/<item>` 表示地图中的一个固定物体，具有稳定类别、尺寸、地图坐标和姿态。

旧流程只保留几何信息，没有保留“这是同一个静态实例”的身份：

- C++ `BoundingBox` 只有类别、尺寸、位姿和状态。
- `GenerateKITTILabel()` 只输出类别和几何字段。
- `convert_kitti_to_xtreme1_json()` 为每个 Xtreme object 生成随机 `id`。
- Xtreme UI 右侧 Instances 中，同一静态物体跨帧或跨 Scene 会显示成不同编号。

Xtreme 本地验证确认：导入 JSON 中的 `trackId` / `trackName` 会被服务端保留；同一物体跨帧、跨 Scene 使用相同 `trackId` / `trackName` 时，UI 侧实例编号能保持一致。

## 2. 目标

- 同一个 XML 静态实例在所有帧、所有 Scene 中使用一致的 Xtreme `trackId` / `trackName`。
- Xtreme object 的 `id` 继续保持逐框唯一，不作为跨帧实例身份。
- 最终交付 KITTI label 不包含 `trackId` / `trackName`，避免影响下游 parser。
- 老格式 17 字段中间 KITTI label 继续可转换为 Xtreme JSON。
- 保持改动局部，不改变距离过滤、可见性过滤、点云点数过滤和 Xtreme 上传 API 流程。

## 3. 非目标

- 不改变静态障碍物是否可见的判断逻辑。
- 不改变最终 KITTI 的字段协议。
- 不为人工新增或动态目标生成稳定 track。
- 不改 Xtreme UI 或服务端代码。
- 不引入额外 sidecar 文件承载实例映射。

## 4. 核心决策

### 4.1 使用 Xtreme `trackId` / `trackName`

Xtreme UI 的实例编号由导入对象中的 `trackName` 稳定影响，`trackId` 作为机器可读的跨帧实例身份。`id` 仍然是逐对象 UUID，不能复用为实例身份。

### 4.2 实例身份在 XML 解析阶段绑定

实例编号必须在 `LoadLabelsFromXML()` 解析全局 XML 时绑定到 `BoundingBox`，不能在每帧 `GenerateKITTILabel()` 输出可见框时重新编号。

原因：每帧可见物体集合不同。如果按帧内可见顺序重新计数，同一全局实例会因为其他物体是否可见而拿到不同编号。

### 4.3 Track 字段只存在于中间链路

中间 KITTI label 可以扩展为 19 字段，用于 C++ 到 Python 的传递；最终 KITTI 仍输出现有 17 字段。

## 5. 数据流

```text
XML <tracklets>/<item>
  -> LoadLabelsFromXML()
       过滤未知类别
       绑定 BoundingBox.track_id / track_name
  -> GenerateKITTILabel()
       生成中间 KITTI 17 字段 + track_id + track_name
  -> convert_kitti_to_xtreme1_json()
       读取可选 track 字段
       写入 Xtreme result object 的 trackId / trackName
  -> Xtreme import
       服务端保存 trackId / trackName 到 classAttributes
  -> export / rebuild_result_tree_from_export_data()
       保留 trackId / trackName 到临时 result JSON
  -> parse_xtreme_to_kitti_lines()
       输出最终 17 字段 KITTI，不写 track 字段
```

## 6. 编号规则

### 6.1 `trackName`

`trackName` 是 UI 里显示的实例编号，使用数字字符串：

```text
trackName = XML 有效静态实例的全局递增序号，从 1 开始
```

有效静态实例定义：XML 中 `objectType` 命中 `class_name_to_label_` 映射，并进入 `annotation_status_.current_labels` 的 item。未知类别被跳过，不占编号。

### 6.2 `trackId`

`trackId` 是机器可读的稳定实例 ID：

```text
trackId = static_<normalized_class>_<zero_padded_index>
```

当前实现中 `normalized_class` 会把空白字符替换为 `_`，保留原有下划线；序号宽度为 6 位，左侧补零。

示例：

```text
object_type = Distance_Marker
trackName = 1
trackId = static_Distance_Marker_000001
```

## 7. 中间 KITTI 协议

现有中间 KITTI 已在标准 KITTI 字段后追加 `rx` / `rz`，总计 17 字段：

```text
class trunc occ alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry rx rz
```

本方案只在预标注中间产物中继续追加两个字段：

```text
class trunc occ alpha bbox_left bbox_top bbox_right bbox_bottom h w l x y z ry rx rz track_id track_name
```

Python 解析规则：

- `len(parts) >= 19`：读取 `parts[17]` 为 `trackId`，`parts[18]` 为 `trackName`。
- `len(parts) == 17`：老格式，继续转换，不写 track 字段。
- `len(parts) < 17`：跳过该行，保持原有容错行为。

## 8. Xtreme JSON 协议

中间 Xtreme object 示例：

```json
{
  "id": "per-object-uuid",
  "type": "3D_BOX",
  "className": "Distance_Marker",
  "modelClass": "Distance_Marker",
  "trackId": "static_Distance_Marker_000001",
  "trackName": "1",
  "contour": {
    "size3D": {"x": 0.69, "y": 1.78, "z": 1.74},
    "center3D": {"x": 1.0, "y": 2.0, "z": 0.5},
    "rotation3D": {"x": 0.0, "y": 0.0, "z": 1.57}
  }
}
```

字段约定：

- `id`：逐框 UUID，每个对象不同。
- `trackId`：同一静态实例跨帧、跨 Scene 一致。
- `trackName`：同一静态实例跨帧、跨 Scene 一致，作为 UI 编号。
- `modelClass`：与 `className` 同步写入，用于导出重建和类别回退兼容。

## 9. 实现落点

### 9.1 C++ 自动标注

相关文件：

- `automatic_annotation/include/automatic_annotation/gen_prompt_point.hpp`
- `automatic_annotation/src/gen_prompt_point.cpp`
- `automatic_annotation/include/automatic_annotation/static_track_id_utils.hpp`
- `automatic_annotation/src/static_track_id_utils.cpp`

实现要点：

- `BoundingBox` 增加 `track_id` 和 `track_name`。
- `LoadLabelsFromXML()` 在类别过滤、尺寸和位姿解析完成后，使用 `current_labels.size() + 1` 分配全局静态实例序号。
- `GenerateKITTILabel()` 只读取 `BoundingBox` 上已绑定的 track 字段，行尾追加到中间 label；不做任何递增或重排。
- `FormatStaticTrackName()` 和 `FormatStaticTrackId()` 独立到小工具文件，便于单元测试。

### 9.2 KITTI -> Xtreme 转换

相关文件：

- `pre_annotation_factory/scripts/replay_rosbag_main.py`

实现要点：

- `convert_kitti_to_xtreme1_json()` 兼容 17 字段和 19 字段。
- 19 字段输入会把 `trackId` / `trackName` 写入 Xtreme object。
- 17 字段输入不写 track 字段，保持历史数据兼容。
- `modelClass` 与 `className` 同步写入。

### 9.3 Xtreme 导出重建

相关文件：

- `pre_annotation_factory/my_package/workflow/xtreme_gateway.py`

实现要点：

- `rebuild_result_tree_from_export_data()` 从 Xtreme API annotation 中读取 object 顶层或 `classAttributes` 内的 `trackId` / `trackName`。
- 重建临时 `result/*.json` 时保留这些字段，避免 finalize 前丢失实例身份。

### 9.4 最终 KITTI 导出

相关文件：

- `pre_annotation_factory/scripts/MergeXtremeDynamic.py`

实现要点：

- `parse_xtreme_to_kitti_lines()` 忽略 `trackId` / `trackName`，最终仍输出 17 字段。
- 这是明确决议：最终 KITTI 下游 parser 不接受扩展字段。

## 10. 兼容性

- 老中间 label：17 字段继续可转 Xtreme。
- 新中间 label：19 字段会生成稳定实例字段。
- Xtreme 导出缺少 track 字段：finalize 按原有几何逻辑继续工作。
- 最终 KITTI：始终保持 17 字段，不暴露中间 track 扩展。

## 11. 风险与约束

### 11.1 XML 顺序变化会改变编号

当前 `trackName` 依赖 XML 有效 item 的顺序。如果上游 XML 重新排序，同一物体的编号会变化。当前接受这个约束，因为预标注任务内的 XML 是固定输入；如果未来要求跨不同 XML 版本编号稳定，需要引入基于地图坐标/尺寸/类别的哈希或外部实例 ID。

### 11.2 类别过滤会影响编号

未知类别不进入 `current_labels`，也不占编号。因此修改 `class_name_to_label_` 映射会影响后续实例编号。这个行为与“只给有效预标注对象编号”的目标一致。

### 11.3 中间 KITTI 不是最终协议

19 字段 KITTI 只用于 C++ 到 Python 的内部传递。任何对外交付或下游训练使用的 KITTI 都必须来自 finalize 后的 17 字段输出。

## 12. 验证

### 12.1 自动化测试

已覆盖：

- C++：`test_static_track_id_utils` 验证 `trackName` / `trackId` 格式。
- Python：19 字段 KITTI 转 Xtreme 后保留 `trackId` / `trackName`。
- Python：17 字段老 KITTI 转 Xtreme 时不写 track 字段。
- Python：Xtreme 导出重建保留 `trackId` / `trackName`。
- Python：最终 KITTI 输出不包含 track 字段。

常用验证命令：

```bash
python3 -m pytest \
  pre_annotation_factory/tests/test_annotationctl_pipeline.py \
  pre_annotation_factory/tests/test_xtreme_gateway.py \
  pre_annotation_factory/tests/test_merge_xtreme_dynamic.py

colcon test --packages-select automatic_annotation \
  --event-handlers console_direct+ \
  --ctest-args -R test_static_track_id_utils --output-on-failure
```

### 12.2 Xtreme 集成验证

使用 YinXiu 新工作流上传验证：

- job id：`8e7040921a74`
- Xtreme dataset id：`37`
- dataset name：`YinXiu_20260414_test0627`
- 导入状态：`PARSE_COMPLETED`
- 总帧数：`182`
- Scene 数：`5`

服务器端抽样 annotation 结果：

```text
classAttributes.trackId = static_Distance_Marker_000010
classAttributes.trackName = 10
```

本地上传包中同一静态实例跨多帧保留相同 `trackId` / `trackName`，Xtreme UI 检查结果符合预期。

## 13. 结论

该方案通过最小扩展中间链路，把 XML 静态实例身份稳定传递到 Xtreme 预标注结果；同时保持最终 KITTI 协议不变。核心原则是：实例编号只在 XML 全局实例层生成一次，所有帧级输出只携带这个身份，不重新编号。
