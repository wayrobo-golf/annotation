# 静态实例跨帧 Track ID 设计

## 背景

当前预标注流水线会从全局 XML 静态标注中生成 KITTI 中间标签，再由 Python 转成 Xtreme1 `result/*.json`。这些静态障碍物在 XML 中天然是实例级对象：每个 `<tracklets>/<item>` 表示一个全局固定物体，带有稳定的地图坐标、尺寸和姿态。

现有实现没有保留这个实例身份：

- `BoundingBox` 只保存类别、尺寸和位姿。
- `GenerateKITTILabel()` 只输出类别和几何字段。
- `convert_kitti_to_xtreme1_json()` 为每个 Xtreme object 生成随机 `id`，没有写 `trackId` / `trackName`。

结果是同一个静态物体出现在不同帧或不同 `Scene_XX` 时，Xtreme UI 右侧 Instances 中显示为不同编号。

本地验证已经确认：

- Xtreme1 会保留导入 JSON 中的 `trackId` 和 `trackName`。
- 在真实 `Scene_XX` 上传结构中，同一对象跨帧、跨 Scene 使用相同 `trackId` / `trackName` 后，API 返回也保持一致。
- `trackName` 会作为 UI 中可识别的实例编号来源。

## 目标

1. 同一个 XML 静态实例在所有帧、所有 Scene 中使用一致的 Xtreme `trackId` / `trackName`。
2. Xtreme 预标注结果中的 `id` 仍保持每个框唯一，不把 `id` 当作跨帧实例标识。
3. 最终交付 KITTI 数据集不包含 `trackId` / `trackName` 扩展字段，避免影响下游 parser。
4. 老格式中间标签继续兼容：没有 track 字段时仍可转换 Xtreme JSON。
5. 保持改动局部，不重构自动标注主链路。

## 非目标

- 不改变静态框可见性过滤、距离过滤、点云点数过滤逻辑。
- 不改变 Xtreme 上传、导入、导出的 API 流程。
- 不为人工新增的动态障碍物自动生成稳定静态 track。
- 不要求最终 KITTI label 保留实例 ID。
- 不改 Xtreme UI 或服务端代码。

## 设计总览

采用“XML 静态实例 -> C++ 中间 KITTI 扩展字段 -> Python Xtreme track 字段”的方案。

数据流如下：

```text
XML <tracklets>/<item>
  -> LoadLabelsFromXML() 绑定 BoundingBox.track_id / track_name
  -> GenerateKITTILabel() 在中间 label 行尾追加 track_id track_name
  -> convert_kitti_to_xtreme1_json() 读取扩展字段
  -> Xtreme result JSON 写入 trackId / trackName
  -> finalize 生成最终 KITTI 时剥离 track 字段
```

这样能最大化复用现有流水线，也避免为 Xtreme 单独引入新的 sidecar 文件。

## 实例编号规则

实例编号必须在 XML 解析阶段生成，并绑定在 `BoundingBox` 上。不能在 `GenerateKITTILabel()` 里按当前帧可见框重新编号。

原因是每帧可见物体集合不同。如果在每帧输出时重新计数，同一个全局物体会因为前面可见框数量不同而拿到不同编号，或者不同物体在不同帧拿到相同编号。

推荐规则：

```text
trackName: XML 中有效静态实例的全局递增编号，从 1 开始
trackId: static_<normalized_class>_<zero_padded_track_name>
```

示例：

```text
trackName = "17"
trackId = "static_Distance_Marker_000017"
```

其中“有效静态实例”指类别命中 `class_name_to_label_` 映射并进入 `current_labels` 的 XML item。跳过的未知类别不占编号。

## 中间 KITTI 扩展格式

现有 C++ 输出已经在标准 KITTI 字段后追加 `rx` 和 `rz`，形成 17 字段格式：

```text
class trunc occ alpha bbox h w l x y z ry rx rz
```

本次仅对预标注中间产物继续追加两个字段：

```text
class trunc occ alpha bbox h w l x y z ry rx rz track_id track_name
```

Python 转 Xtreme 时：

- `len(parts) >= 19`：读取 `parts[17]` 为 `trackId`，`parts[18]` 为 `trackName`。
- `len(parts) == 17`：老格式，继续转换，不写 track 字段。
- `len(parts) < 17`：维持当前行为，跳过该行。

## Xtreme JSON 输出

每个 object 输出：

```json
{
  "id": "per-object-uuid",
  "type": "3D_BOX",
  "className": "Distance_Marker",
  "modelClass": "Distance_Marker",
  "trackId": "static_Distance_Marker_000017",
  "trackName": "17",
  "contour": {
    "size3D": {"x": 0.69, "y": 1.78, "z": 1.74},
    "center3D": {"x": 1.0, "y": 2.0, "z": 0.5},
    "rotation3D": {"x": 0.0, "y": 0.0, "z": 1.57}
  }
}
```

`modelClass` 建议同步写入，便于导出重建时兼容当前 `className -> modelClass` 双回退逻辑。

## Finalize 和最终 KITTI

`finalize` 阶段从 Xtreme 导出后会调用 `parse_xtreme_to_kitti_lines()` 生成最终 KITTI label。根据决议，最终 KITTI 不写 `trackId` / `trackName`：

- `rebuild_result_tree_from_export_data()` 应保留 `trackId` / `trackName` 到临时 `result/*.json`，避免信息在中间环节丢失。
- `parse_xtreme_to_kitti_lines()` 可以读取这些字段用于调试或未来扩展，但输出 KITTI 行时只写现有 17 字段，不追加 track 字段。

## 错误处理和兼容性

1. 如果 `BoundingBox.track_id` 或 `track_name` 为空，C++ 仍可输出 label，但 Python 不写 Xtreme track 字段。
2. 如果老 label 没有 track 字段，Python 继续生成 Xtreme object，并保留随机 `id`。
3. 如果 Xtreme 导出的对象没有 `trackId` / `trackName`，finalize 继续按当前几何逻辑生成最终 KITTI。
4. `trackName` 只使用数字字符串，避免 UI 显示过长或混入类别名。

## 测试策略

### 单元测试

- Python：带 19 字段 label 的 KITTI 行转换后包含 `trackId` / `trackName`。
- Python：17 字段老 label 转换后不写 track 字段但仍生成 object。
- Python：Xtreme 导出重建时保留 `trackId` / `trackName`。
- Python：finalize 的 KITTI 输出不包含 track 字段。
- C++：实例编号 helper 对类别和序号生成稳定 `trackId` / `trackName`。

### 集成验证

使用真实 YinXiu `xtreme_upload` 子集：

1. 生成包含 `Scene_01` 和 `Scene_02` 的小测试包。
2. 给同一个静态实例写相同 `trackId` / `trackName`。
3. 上传本地 Xtreme。
4. 使用 `/api/data/findByPage?parentId=<scene_id>` 获取 Scene 内 frame。
5. 使用 `/api/annotate/data/listByDataIds` 确认跨 Scene frame 的 `trackId` / `trackName` 一致。
6. 人工打开 UI 确认 Instances 显示一致。

## 风险与缓解

### 风险：在帧级重新编号导致跨帧不一致

缓解：只在 XML 解析阶段生成编号，并存入 `BoundingBox`。`GenerateKITTILabel()` 只读取字段，不递增、不排序、不重排。

### 风险：最终 KITTI 被扩展字段污染

缓解：明确只有预标注中间 label 可带 track 字段。`MergeXtremeDynamic.py` 输出最终 KITTI 时保持 17 字段，不追加 track。

### 风险：老数据包转换失败

缓解：Python 转换逻辑兼容 17 字段和 19 字段。没有 track 字段时不写 `trackId` / `trackName`。

## 审核决议

- 最终 KITTI 不接受行尾 `trackId` / `trackName` 扩展字段。
- Xtreme UI 已确认相同 `trackName` 会显示为一致实例编号。
- 实例编号必须绑定 XML 全局实例，不能绑定帧内输出顺序。
