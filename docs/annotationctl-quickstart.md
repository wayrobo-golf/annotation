# annotationctl 使用说明

本文档对应当前已经落地的固定工作流：

1. 准备 `job.yaml`
2. 运行 `annotationctl submit`
3. 如 Xtreme 上传失败，运行 `annotationctl upload` 重试上传
4. 在 Xtreme1 中人工微调与补标动态障碍物
5. 运行 `annotationctl finalize`
6. 获取最终交付 KITTI 数据集

## 1. 运行环境

- 已完成 `3D_Detection_Annotation` 工作区编译
- 服务器或本机可访问 Xtreme1
- 预标注所需地图、XML、外参配置已准备完成
- 新机器环境初始化可参考 `docs/nusc_env_setup.md`
- 在仓库根目录准备 `.env`，写入 Xtreme Token：

```dotenv
XTREME1_TOKEN=<your_token>
```

`annotationctl.py` 启动时会自动读取仓库根目录 `.env`；如果当前 shell 里已经存在同名环境变量，则以 shell 中的值为准。

## 2. job.yaml 完整示例

说明：本节展示的是当前 `job.yaml` 可配置字段的完整示例，不是 `submit` 内部全部运行参数的完整展开。

```yaml
job_name: baoli_debug_resubmit
location: BaoLi_20260305_Debug

input:
  source_dir: /home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/BaoLi20260305_Debug
  recursive: false
  overwrite_extract: false
  qos_yaml_path: /home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/rosbag_play_qos_cfg.yaml

workspace:
  root_dir: /tmp/annotationctl_baoli_debug_resubmit

auto_annotation:
  base_yaml_path: /home/keyaoli/Code/Wayrobo/3D_Detection_Annotation/automatic_annotation/config/default.yaml
  extrinsics_source:
    path: /home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/BaoLi20260305_Debug/config.yaml
    quaternion_order: wxyz
  overrides:
    global_pc_map_addr: /home/keyaoli/Data/AutoAnnotation/MapAnnotation/bl_20260317/velodyne_points/data/colored_lidar_merged.pcd
    pc_annotation_file_addr: /home/keyaoli/Data/AutoAnnotation/MapAnnotation/bl_20260317/tracklet_labels.xml

xtreme:
  base_url: http://localhost:8190
  token_env: XTREME1_TOKEN
  dataset_name: BaoLi_20260305_Debug_Resubmit_20260416
  dataset_type: lidar_fusion

output:
  final_dataset_dir: /tmp/annotationctl_baoli_debug_resubmit/final_delivery
```

字段说明：

- `job_name`：任务名称，用于区分不同 job，便于人工识别。
- `location`：当前数据对应的地点或批次标识，会传给预标注和后处理流程。
- `input.source_dir`：输入数据目录，可以放原始 rosbag 目录，也可以放待解压的压缩包目录。
- `input.recursive`：是否递归扫描 `source_dir` 的子目录查找压缩包；`false` 只扫描顶层，`true` 会继续扫描下级目录。
- `input.overwrite_extract`：解压时如果目标位置已有同名内容，是否先删除旧内容再重新解压；`false` 不主动覆盖，`true` 会清理后重解。
- `input.qos_yaml_path`：可选的 rosbag 播放 QoS override 文件路径；如果配置且文件存在，`ros2 bag play` 会带上 `--qos-profile-overrides-path`。
- `workspace.root_dir`：annotationctl 工作目录，`jobs/<job_id>/`、中间产物、状态文件都会写到这里。
- `auto_annotation.base_yaml_path` 是模板 yaml
- `auto_annotation.extrinsics_source`：可选的原始外参 yaml 配置；工作流会读取其中的 `trans_imu2ins` / `quat_imu_to_ins` / `trans_lidar_to_imu` / `quat_lidar_to_imu` / `trans_cam_to_lidar` / `quat_cam_to_lidar`，自动生成 `default_tf_lidar_to_ins` 和 `default_tf_lcam_to_lidar`。
- `auto_annotation.overrides` 会生成到 job 私有 runtime yaml，而不会改仓库里的 `default.yaml`
- 如果 `auto_annotation.extrinsics_source` 和 `auto_annotation.overrides` 同时配置了 `default_tf_lidar_to_ins` 或 `default_tf_lcam_to_lidar`，以 `overrides` 中的手写值为准。
- `xtreme.base_url`：Xtreme1 服务地址。
- `xtreme.token_env`：Xtreme1 token 的环境变量名，只写变量名，不在 `job.yaml` 中写明文 token；默认可配合仓库根目录 `.env` 使用。
- `xtreme.dataset_name`：本次 submit 在 Xtreme1 中创建或复用的数据集名称。
- `xtreme.dataset_type`：Xtreme1 数据集类型，当前完整示例使用 `lidar_fusion`。
- `output.final_dataset_dir`：最终交付数据集的目标目录配置，供收尾阶段输出使用。

当前未开放到 `job.yaml`、而是由代码内固定传入的运行参数：

- `workspace_path`：固定为当前仓库根目录，用于回放和后处理阶段定位工作区。
- `xtreme1_output_dir`：固定写入 `<workspace.root_dir>/jobs/<job_id>/artifacts/xtreme_upload/`。
- `raw_data_archive_dir`：固定写入 `<workspace.root_dir>/jobs/<job_id>/artifacts/raw_origin/`。
- `preserve_full_raw_copy=False`：当前不会额外保留一份 `origin_full` 完整原始备份。
- `cleanup_share_data=False`：当前不会在流程结束后清空原始生成目录，而是走归档逻辑。

如果后续需要让这些参数也能在 `job.yaml` 中配置，需要先扩展 manifest 模型和 `submit` 管线；当前版本直接写在代码里。

## 3. 提交任务

```bash
conda run --live-stream -n nusc_env python -u pre_annotation_factory/scripts/annotationctl.py submit /path/to/job.yaml
```

`submit` 当前会分两段执行：

1. 本地回放 rosbag、生成 Xtreme 上传包、归档原始数据
2. 调用 Xtreme 接口创建数据集并上传导入

如果第 2 段失败，只要本地产物已经生成完成，就不需要重跑 rosbag，可直接使用下文的 `upload` 命令重试。

成功后会输出：

```text
job_id=<job_id>
status=waiting_human_annotation
```

关键产物会出现在：

```text
<workspace.root_dir>/jobs/<job_id>/
```

重点关注：

- `state.json`
- `artifacts/xtreme_upload/*.zip`
- `artifacts/raw_origin/`
- `runtime/auto_annotation.yaml`
- `jobs/<job_id>/job.yaml`

### 3.1 仅重试 Xtreme 上传

当 `state.json` 中的 `current_step` 停在 `uploading_xtreme`，且 `artifacts/xtreme_upload/*.zip` 已经生成时，可以直接重试上传：

```bash
conda run --live-stream -n nusc_env python -u pre_annotation_factory/scripts/annotationctl.py upload <job_id> --workspace <workspace.root_dir>
```

这个命令会复用：

- `jobs/<job_id>/job.yaml`
- `state.json`
- `artifacts/xtreme_upload/*.zip`

不会重新回放 rosbag，也不会重新生成本地打包产物。

对于旧版本工作流生成的失败 job，如果 `jobs/<job_id>/job.yaml` 不存在，`upload` 会优先回退使用 `<workspace.root_dir>/job.yaml`。如果工作区里也没有这份 manifest，可以显式指定：

```bash
conda run --live-stream -n nusc_env python -u pre_annotation_factory/scripts/annotationctl.py upload <job_id> --workspace <workspace.root_dir> --manifest /path/to/job.yaml
```

## 4. 查看状态

```bash
conda run -n nusc_env python pre_annotation_factory/scripts/annotationctl.py status <job_id> --workspace <workspace.root_dir>
```

## 5. Xtreme1 人工标注

在 Xtreme1 中打开本次 submit 创建的数据集，完成人工微调与动态障碍物标注。

当前自动上传会按 UI 中以下等价设置导入：

- `Data Format = Xtreme1`
- `Contains annotation result = true`
- `Result Type = Ground Truth`

## 6. 收尾生成最终交付数据集

```bash
conda run --live-stream -n nusc_env python -u pre_annotation_factory/scripts/annotationctl.py finalize <job_id> --workspace <workspace.root_dir>
```

成功后会输出：

```text
job_id=<job_id>
status=completed
```

最终产物位于：

- `artifacts/final_dataset/<dataset_name>/`
- `artifacts/final_dataset/<dataset_name>.zip`

同时 `state.json` 会记录：

- `xtreme.export_serial_number`
- `paths.normalized_export_dir`
- `paths.final_dataset_dir`
- `paths.final_zip_path`

## 7. 常见排查点

### 7.1 submit 后 Xtreme 中没有预标注框

先检查：

- `artifacts/xtreme_upload/Scene_XX/result/*.json` 是否非空
- 自动上传是否带了：
  - `dataFormat=XTREME1`
  - `resultType=GROUND_TRUTH`

如果 `submit` 或 `upload` 在 `uploading_xtreme` 阶段失败，优先检查：

- `state.json` 中的 `current_step` 是否为 `uploading_xtreme`
- `state.json` 中的 `last_error`
- `artifacts/xtreme_upload/*.zip` 是否已生成

如果 zip 已生成，可直接执行一次 `annotationctl upload <job_id> --workspace <workspace.root_dir>` 重试，不需要重新回放 rosbag。

### 7.2 submit 后全是空帧

优先检查 `job.yaml` 中的 `auto_annotation.overrides`：

- `global_pc_map_addr`
- `pc_annotation_file_addr`

如果外参由 `auto_annotation.extrinsics_source` 自动生成，继续检查：

- 原始外参 yaml 路径是否正确
- 原始外参 yaml 是否包含 `trans_imu2ins` / `quat_imu_to_ins` / `trans_lidar_to_imu` / `quat_lidar_to_imu` / `trans_cam_to_lidar` / `quat_cam_to_lidar`
- `quaternion_order` 是否与原始文件一致，默认是 `wxyz`
- `jobs/<job_id>/runtime/auto_annotation.yaml` 中生成的 `default_tf_lidar_to_ins` 和 `default_tf_lcam_to_lidar` 是否符合预期

### 7.3 finalize 失败

优先检查：

- Xtreme1 导出是否成功
- `artifacts/xtreme_export/normalized/Scene_XX/data/*.json` 是否存在
- `state.json` 中的 `last_error`

## 8. 多任务 manifest 管理建议

建议每一组数据使用一份新的 `job.yaml`，不要直接修改上一组已经提交过的 manifest 反复复用。

原因：

- 一个 `job.yaml` 对应一组独立输入、一套配置和一份独立产物
- `submit` 后原始 manifest 会被复制到 `jobs/<job_id>/job.yaml`
- 后续追溯某个 job 当时使用的 bag、地图、XML、外参时，会更清晰

推荐做法：

- 为每组数据单独保存一个 manifest 文件
- 复用上一份作为模板，只修改会随任务变化的字段

通常需要修改：

- `job_name`
- `location`
- `input.source_dir`
- `workspace.root_dir`
- `xtreme.dataset_name`
- 如果场景不同，再修改 `auto_annotation.overrides`

推荐命名方式例如：

- `jobs/baoli_20260305_debug.yaml`
- `jobs/yenan_20260318.yaml`
- `jobs/xihu_20260330_debug.yaml`

如果只是同一批数据重复试跑，也建议新建 manifest 或至少修改 `xtreme.dataset_name` 和 `workspace.root_dir`，避免和历史 job 混淆。
