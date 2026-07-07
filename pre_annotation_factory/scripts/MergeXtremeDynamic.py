import os
import sys
import json
import math
import shutil
import glob
import random
import bisect
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# 尝试导入 NuScenes 元数据生成函数
try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from my_package.io_modules.nusc_json_exporter import generate_nuscenes_metadata
except ImportError:
    print("⚠️ 警告: 未能找到 my_package.io_modules, NuScenes 元数据生成功能将不可用。")

from my_package.workflow.xtreme_rotation import xtreme_euler_to_matrix

# ================= 配置区 =================
# 1. Xtreme1 导出的标注数据根目录 (解压后的 Scene_XX 所在目录)
XTREME_EXPORT_ROOT = Path("/home/keyaoli/Data/AutoAnnotation/Xtreme/ExportFromXtreme/BaoLi_20260305-20260411012156")

# 2. 你的原始数据集归档目录 (_origin 后缀的那个文件夹)
RAW_ARCHIVE_ROOT = Path("/home/keyaoli/Data/AutoAnnotation/Auto_Annotation_Origin/3DBox_Annotation_20260407181828_BaoLi_20260305_origin")

# 3. 最终训练集输出目录
FINAL_KITTI_OUTPUT_DIR = Path("/home/keyaoli/Data/AutoAnnotation/Wayrobo_KITTI_Dataset/Debug")

# 数据集配置
SPLIT_RATIO = 0.8         # 80% 训练集, 20% 验证集
CONVERT_PCD_TO_BIN = True # 是否转换为 OpenPCDet 需要的 bin
RANDOM_SEED = 42
MAX_EMPTY_LABEL_RATIO = 0.10
XTREME_TIMESTAMP_MATCH_TOLERANCE_NS = 1000

# ================= 核心数学与转换工具 =================
R_xtreme2kitti = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0],
    [0.0, -1.0,  0.0]
])

def load_camera_config(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list): data = data[0]
    if "camera_internal" not in data: data = data[list(data.keys())[0]]
        
    intrinsics = data["camera_internal"]
    K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]],
                  [0, 0, 1]])
    img_w, img_h = data.get("width", 1920), data.get("height", 1080)
    T_lidar2cam = np.array(data["camera_external"]).reshape((4, 4), order='F')
    
    T_lidar2map = None
    if "tf_lidar_to_map" in data:
        T_lidar2map = np.array(data["tf_lidar_to_map"]).reshape((4, 4), order='F')
        
    return K, T_lidar2cam, T_lidar2map, img_w, img_h


def has_tf_lidar_to_map(config_path):
    with open(config_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    if "camera_internal" not in data:
        data = data[list(data.keys())[0]]
    return "tf_lidar_to_map" in data


def generate_calib(config_path, out_calib):
    K, T_lidar2cam, T_lidar2map, _, _ = load_camera_config(config_path)
    with open(out_calib, 'w') as f:
        P_str = " ".join([f"{x:.6f}" for x in np.pad(K, ((0,0),(0,1))).flatten()])
        f.writelines([f"P{i}: {P_str}\n" for i in range(4)])
        f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")
        Tr = " ".join([f"{x:.6f}" for x in T_lidar2cam[:3, :].flatten()])
        f.write(f"Tr_velo_to_cam: {Tr}\nTr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        if T_lidar2map is not None:
            Tr_velo2map = " ".join([f"{x:.6f}" for x in T_lidar2map[:3, :].flatten()])
            f.write(f"Tr_velo_to_map: {Tr_velo2map}\n")

def pcd_to_bin_fixed(pcd_path, bin_path):
    try:
        with open(pcd_path, 'rb') as f:
            num_fields = 4  # 默认兜底为 4 维
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                
                # 【新增】动态解析 PCD 的字段维度
                if line.startswith('FIELDS'):
                    # FIELDS x y z intensity timestamp -> 切片后减去开头的'FIELDS'
                    num_fields = len(line.split()) - 1 
                
                if line.startswith('DATA binary'): 
                    break
                if not line: return False
                
            # 读取二进制数据
            data = np.fromfile(f, dtype=np.float32)
            
        # 【修改】使用动态解析出的维度进行 reshape (自动兼容 4维 或 5维)
        points = data.reshape(-1, num_fields)
        
        # 保存为 bin
        points.tofile(bin_path)
        return True
        
    except ValueError as ve:
        print(f"  [Error] 维度匹配失败 {pcd_path}: {ve} (解析到的维度数: {num_fields})")
        return False
    except Exception as e:
        print(f"  [Error] 转换 PCD 失败 {pcd_path}: {e}")
        return False

# ================= 核心标签解析引擎 =================
def parse_xtreme_to_kitti_lines(xtreme_json_path, config_file):
    K, T_lidar2cam, _, img_w, img_h = load_camera_config(config_file)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    with open(xtreme_json_path, 'r') as f:
        xtreme_data = json.load(f)
        
    objects_to_write = []
    if xtreme_data and "objects" in xtreme_data[0]:
        for obj in xtreme_data[0].get("objects", []):
            if obj["type"] != "3D_BOX": continue
            
            # 【核心修复】：双重回退读取逻辑 (className -> modelClass)
            raw_class_name = obj.get('className')
            
            # 如果 className 是 None 或空字符串，则尝试读取 modelClass
            if not raw_class_name:
                raw_class_name = obj.get('modelClass')
                
            # 如果两个都为空，那说明这个框真的坏了，跳过它
            if not raw_class_name:
                print(f"⚠️ 警告: 发现缺失类别的异常目标 (className 与 modelClass 均为空)，已跳过。文件: {xtreme_json_path.name}")
                continue

            class_name = raw_class_name.replace(' ', '_')
            
            s, c, rot = obj["contour"]["size3D"], obj["contour"]["center3D"], obj["contour"]["rotation3D"]
            h, w, l = s["z"], s["y"], s["x"]
            
            R_lidar_to_box = xtreme_euler_to_matrix(rot)
            center_lidar = np.array([c["x"], c["y"], c["z"]])
            
            R_6dof = T_lidar2cam[:3, :3] @ R_lidar_to_box @ R_xtreme2kitti
            center_cam_geo = T_lidar2cam[:3, :3] @ center_lidar + T_lidar2cam[:3, 3]
            bottom_cam = center_cam_geo + R_6dof @ np.array([0.0, h/2.0, 0.0])
            rx, ry, rz = R.from_matrix(R_6dof).as_euler('xyz')
            
            corners_local = np.array([
                [ l/2, -h,  w/2], [ l/2, -h, -w/2], [ l/2, 0, -w/2], [ l/2, 0,  w/2],
                [-l/2, -h,  w/2], [-l/2, -h, -w/2], [-l/2, 0, -w/2], [-l/2, 0,  w/2]
            ]).T
            corners_cam = R_6dof @ corners_local + bottom_cam.reshape(3, 1)
            
            is_camera_visible = True
            bbox_left, bbox_top, bbox_right, bbox_bottom = 0.0, 0.0, 0.0, 0.0
            
            valid_idx = corners_cam[2, :] > 0.1
            if not np.any(valid_idx):
                is_camera_visible = False
            else:
                u = fx * corners_cam[0, valid_idx] / corners_cam[2, valid_idx] + cx
                v = fy * corners_cam[1, valid_idx] / corners_cam[2, valid_idx] + cy
                bbox_left = max(0.0, min(float(img_w) - 1.0, np.min(u)))
                bbox_right = max(0.0, min(float(img_w) - 1.0, np.max(u)))
                bbox_top = max(0.0, min(float(img_h) - 1.0, np.min(v)))
                bbox_bottom = max(0.0, min(float(img_h) - 1.0, np.max(v)))
                if bbox_right <= bbox_left or bbox_bottom <= bbox_top:
                    is_camera_visible = False
                    
            alpha = ry - math.atan2(bottom_cam[0], bottom_cam[2])
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            
            bbox_str = f"{bbox_left:.2f} {bbox_top:.2f} {bbox_right:.2f} {bbox_bottom:.2f}" if is_camera_visible else "-1.00 -1.00 -1.00 -1.00"
            trunc, occ = ("0.00", "0") if is_camera_visible else ("1.00", "3")
            alpha_val = f"{alpha:.2f}" if is_camera_visible else "-10.00"

            line = (f"{class_name} {trunc} {occ} {alpha_val} {bbox_str} "
                    f"{h:.4f} {w:.4f} {l:.4f} "
                    f"{bottom_cam[0]:.4f} {bottom_cam[1]:.4f} {bottom_cam[2]:.4f} "
                    f"{ry:.4f} {rx:.4f} {rz:.4f}\n")
            objects_to_write.append(line)
            
    return objects_to_write


def has_non_empty_label(label_path):
    if label_path is None or not Path(label_path).exists():
        return False

    with open(label_path, 'r', encoding='utf-8') as f:
        return any(line.strip() for line in f)


def compute_max_empty_frames(positive_count, max_empty_ratio):
    if positive_count < 0:
        raise ValueError("positive_count must be non-negative")
    if not 0.0 <= max_empty_ratio <= 1.0:
        raise ValueError("MAX_EMPTY_LABEL_RATIO must be between 0.0 and 1.0")
    if max_empty_ratio == 1.0:
        return None
    if positive_count == 0:
        return 0
    if max_empty_ratio == 0.0:
        return 0

    return int(math.floor((positive_count * max_empty_ratio) / (1.0 - max_empty_ratio)))


def parse_numeric_stem(path):
    try:
        return int(Path(path).stem)
    except ValueError:
        return None


def get_xtreme_frame_id_from_data(data_json_path):
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("name"):
        return str(data["name"])

    camera_config = data.get("cameraConfig") or {}
    if camera_config.get("filename"):
        return Path(camera_config["filename"]).stem

    return data_json_path.stem


def find_nearest_raw_frame_id(target_frame_id, raw_frame_ids, max_delta_ns):
    target_timestamp = parse_numeric_stem(Path(f"{target_frame_id}.json"))
    if target_timestamp is None:
        return target_frame_id if target_frame_id in raw_frame_ids else None

    numeric_raw_frames = []
    for raw_frame_id in raw_frame_ids:
        raw_timestamp = parse_numeric_stem(Path(f"{raw_frame_id}.json"))
        if raw_timestamp is not None:
            numeric_raw_frames.append((raw_timestamp, raw_frame_id))
    if not numeric_raw_frames:
        return None

    numeric_raw_frames.sort()
    raw_timestamps = [item[0] for item in numeric_raw_frames]
    insert_idx = bisect.bisect_left(raw_timestamps, target_timestamp)
    candidate_indices = [insert_idx]
    if insert_idx > 0:
        candidate_indices.append(insert_idx - 1)

    best_match = None
    for idx in candidate_indices:
        if idx >= len(numeric_raw_frames):
            continue
        raw_timestamp, raw_frame_id = numeric_raw_frames[idx]
        delta_ns = abs(raw_timestamp - target_timestamp)
        if delta_ns <= max_delta_ns and (
            best_match is None or delta_ns < best_match[0]
        ):
            best_match = (delta_ns, raw_frame_id)

    return best_match[1] if best_match else None


def build_xtreme_manifest_index(xtreme_scene_dir, raw_frame_ids):
    data_dir = xtreme_scene_dir / "data"
    data_json_paths = sorted(data_dir.glob("*.json"))
    if not data_json_paths:
        return None

    raw_to_xtreme = {}
    used_raw_frame_ids = set()
    for data_json_path in data_json_paths:
        xtreme_frame_id = get_xtreme_frame_id_from_data(data_json_path)
        raw_frame_id = find_nearest_raw_frame_id(
            xtreme_frame_id,
            raw_frame_ids,
            XTREME_TIMESTAMP_MATCH_TOLERANCE_NS,
        )
        if raw_frame_id is None:
            continue
        if raw_frame_id in used_raw_frame_ids:
            continue

        used_raw_frame_ids.add(raw_frame_id)
        raw_to_xtreme[raw_frame_id] = {
            "xtreme_frame_id": xtreme_frame_id,
            "data_path": data_json_path,
            "result_path": xtreme_scene_dir / "result" / f"{xtreme_frame_id}.json",
        }

    return raw_to_xtreme


def load_submit_frame_manifest(submit_manifest_path):
    if submit_manifest_path is None:
        raise FileNotFoundError("Missing submit manifest path for finalize.")

    manifest_path = Path(submit_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing submit manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Invalid submit manifest payload: {manifest_path}")

    return payload


def resolve_raw_scene_dir_for_frame(raw_archive_root: Path, scene_name: str, frame_id: str) -> Path:
    scene_matches = sorted(raw_archive_root.glob(f"data_record_*/{scene_name}"))
    matched_scene_dirs = [
        scene_dir
        for scene_dir in scene_matches
        if (scene_dir / "camera_config" / f"{frame_id}.json").exists()
    ]

    if len(matched_scene_dirs) != 1:
        raise FileNotFoundError(
            "Expected exactly one raw scene directory for "
            f"{scene_name}/{frame_id}, got {len(matched_scene_dirs)}"
        )

    return matched_scene_dirs[0]

# ================= 编译流水线 =================
def build_final_dataset(
    location_str,
    xtreme_export_root=None,
    raw_archive_root=None,
    final_kitti_output_dir=None,
    split_ratio=None,
    convert_pcd_to_bin=None,
    random_seed=None,
    max_empty_label_ratio=None,
    submit_manifest_path=None,
):
    print(f"\n{'='*50}")
    print("🚀 开始终极编译：融合 Xtreme1 人工精修与原始数据...")
    print(f"{'='*50}")

    xtreme_export_root = Path(xtreme_export_root or XTREME_EXPORT_ROOT)
    raw_archive_root = Path(raw_archive_root or RAW_ARCHIVE_ROOT)
    final_kitti_output_dir = Path(final_kitti_output_dir or FINAL_KITTI_OUTPUT_DIR)
    split_ratio = SPLIT_RATIO if split_ratio is None else split_ratio
    convert_pcd_to_bin = CONVERT_PCD_TO_BIN if convert_pcd_to_bin is None else convert_pcd_to_bin
    random_seed = RANDOM_SEED if random_seed is None else random_seed
    max_empty_label_ratio = (
        MAX_EMPTY_LABEL_RATIO if max_empty_label_ratio is None else max_empty_label_ratio
    )

    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_folder_name = f"3DBox_Annotation_Final_{current_time_str}_{location_str}"
    dataset_root_dir = final_kitti_output_dir / dataset_folder_name
    
    training_base = dataset_root_dir / "training"
    imagesets_dir = dataset_root_dir / "ImageSets"

    for fol in ["label_2", "calib", "image_2", "velodyne"]:
        (training_base / fol).mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = load_submit_frame_manifest(submit_manifest_path)
    scanned_tasks = []
    filtered_missing_tf_count = 0
    skipped_not_in_xtreme_manifest_count = 0
    xtreme_manifest_authoritative = any(xtreme_export_root.glob("Scene_*/data/*.json"))
    xtreme_manifest_index_cache = {}

    for item in manifest_items:
        scene_name = item["scene_name"]
        frame_id = item["frame_id"]
        img_ext = item["img_ext"]

        scene_dir = resolve_raw_scene_dir_for_frame(raw_archive_root, scene_name, frame_id)
        xtreme_scene_dir = xtreme_export_root / scene_name

        config_path = scene_dir / "camera_config" / f"{frame_id}.json"
        img_path = scene_dir / "camera_image_0" / f"{frame_id}{img_ext}"
        pcd_path = scene_dir / "lidar_point_cloud_0" / f"{frame_id}.pcd"
        raw_label_path = scene_dir / "label_2" / f"{frame_id}.txt"

        if not config_path.exists() or not img_path.exists() or not pcd_path.exists():
            raise FileNotFoundError(
                f"Submit manifest frame is missing raw artifacts: {scene_name}/{frame_id}"
            )

        if not has_tf_lidar_to_map(config_path):
            filtered_missing_tf_count += 1
            continue

        cache_key = (str(xtreme_scene_dir), str(scene_dir))
        if cache_key not in xtreme_manifest_index_cache:
            raw_frame_ids = {
                Path(path).stem
                for path in glob.glob(str(scene_dir / "camera_config" / "*.json"))
            }
            xtreme_manifest_index_cache[cache_key] = (
                build_xtreme_manifest_index(xtreme_scene_dir, raw_frame_ids)
                if xtreme_scene_dir.exists()
                else None
            )
        xtreme_manifest_index = xtreme_manifest_index_cache[cache_key]

        if xtreme_manifest_index is not None:
            xtreme_match = xtreme_manifest_index.get(frame_id)
            if xtreme_match is None:
                skipped_not_in_xtreme_manifest_count += 1
                continue

            xtreme_json_path = xtreme_match["result_path"]
            if xtreme_json_path.exists():
                label_source = "xtreme"
                target_label_path = xtreme_json_path
            else:
                label_source = "empty_missing_json"
                target_label_path = None
        elif xtreme_manifest_authoritative:
            skipped_not_in_xtreme_manifest_count += 1
            continue
        else:
            xtreme_json_path = xtreme_scene_dir / "result" / f"{frame_id}.json"
            if xtreme_json_path.exists():
                label_source = "xtreme"
                target_label_path = xtreme_json_path
            else:
                label_source = "empty_missing_json"
                target_label_path = None

        scanned_tasks.append({
            "frame_id": frame_id,
            "scene_name": scene_name,
            "config_path": config_path,
            "img_path": img_path,
            "pcd_path": pcd_path,
            "label_source": label_source,
            "label_path": target_label_path,
            "raw_label_path": raw_label_path,
            "kitti_lines": None
        })

    if not scanned_tasks:
        print(f"ℹ️ 已过滤 {filtered_missing_tf_count} 帧缺 tf_lidar_to_map 数据。")
        if skipped_not_in_xtreme_manifest_count:
            print(f"ℹ️ 已跳过 {skipped_not_in_xtreme_manifest_count} 帧未出现在 Xtreme data 清单中的原始帧。")
        print("❌ 未发现任何有效帧数据，整合终止。")
        return

    positive_tasks, empty_tasks = [], []
    for task in scanned_tasks:
        if task["label_source"] == "xtreme":
            kitti_lines = parse_xtreme_to_kitti_lines(task["label_path"], task["config_path"])
            task["kitti_lines"] = kitti_lines
            if kitti_lines:
                positive_tasks.append(task)
            else:
                task["label_source"] = "empty_xtreme_json"
                empty_tasks.append(task)
        else:
            task["kitti_lines"] = []
            empty_tasks.append(task)

    max_empty_frames = compute_max_empty_frames(len(positive_tasks), max_empty_label_ratio)
    if not positive_tasks and max_empty_frames == 0:
        raise ValueError("Merge would produce zero positive frames; cannot enforce empty-label cap with zero positive frames.")

    rng = random.Random(random_seed)
    if max_empty_frames is None or len(empty_tasks) <= max_empty_frames:
        kept_empty_tasks = list(empty_tasks)
    else:
        kept_empty_tasks = rng.sample(empty_tasks, max_empty_frames)

    dropped_empty_count = len(empty_tasks) - len(kept_empty_tasks)
    tasks = positive_tasks + kept_empty_tasks
    rng.shuffle(tasks)

    split_idx = int(len(tasks) * SPLIT_RATIO)
    train_ids, val_ids = [], []
    xtreme_used_count = 0
    empty_missing_json_count = 0
    empty_xtreme_json_count = 0

    print(f"🔎 扫描完毕。共捕获 {len(tasks)} 帧有效数据，开始转换装配...")

    for i, task in enumerate(tasks):
        frame_id = task["frame_id"]
        img_ext = task["img_path"].suffix
        
        out_label = training_base / "label_2" / f"{frame_id}.txt"
        out_calib = training_base / "calib" / f"{frame_id}.txt"
        out_bin = training_base / "velodyne" / f"{frame_id}.bin"
        out_img = training_base / "image_2" / f"{frame_id}{img_ext}"
        
        if task["label_source"] == "xtreme":
            with open(out_label, 'w') as f:
                f.writelines(task["kitti_lines"])
            xtreme_used_count += 1
        elif task["label_source"] in {"empty_missing_json", "empty_xtreme_json"}:
            with open(out_label, 'w', encoding='utf-8'):
                pass
            if task["label_source"] == "empty_missing_json":
                empty_missing_json_count += 1
                if has_non_empty_label(task["raw_label_path"]):
                    print(f"ℹ️ 帧 {frame_id} 在 Xtreme 场景中缺失导出 json，已按人工判空覆盖原始非空标注。")
            else:
                empty_xtreme_json_count += 1
        else:
            raise ValueError(f"未知的标签来源: {task['label_source']}")
            
        generate_calib(task["config_path"], out_calib)
        shutil.copy2(task["img_path"], out_img)
        
        if convert_pcd_to_bin:
            pcd_to_bin_fixed(task["pcd_path"], out_bin)
        else:
            shutil.copy2(task["pcd_path"], training_base / "velodyne" / f"{frame_id}.pcd")

        if i < int(len(tasks) * split_ratio): train_ids.append(frame_id)
        else: val_ids.append(frame_id)

        if (i + 1) % 50 == 0:
            print(f"  已装配 {i + 1} / {len(tasks)} 帧...")

    with open(imagesets_dir / "train.txt", "w") as f: f.write("\n".join(train_ids))
    with open(imagesets_dir / "val.txt", "w") as f: f.write("\n".join(val_ids))
    
    if 'generate_nuscenes_metadata' in globals():
        try:
            generate_nuscenes_metadata(str(dataset_root_dir), location_str)
        except Exception as e:
            print(f"⚠️ NuScenes Metadata 生成跳过: {e}")

    print(
        f"\n✅ 编译完成！共使用 Xtreme精修: {xtreme_used_count}帧, "
        f"缺失 json 判空: {empty_missing_json_count}帧, "
        f"空 Xtreme 结果判空: {empty_xtreme_json_count}帧, "
        f"丢弃空标签: {dropped_empty_count}帧。"
    )
    print(f"ℹ️ 已过滤 {filtered_missing_tf_count} 帧缺 tf_lidar_to_map 数据。")
    if skipped_not_in_xtreme_manifest_count:
        print(f"ℹ️ 已跳过 {skipped_not_in_xtreme_manifest_count} 帧未出现在 Xtreme data 清单中的原始帧。")
    print(f"📊 划分情况 -> 训练集: {len(train_ids)}帧，验证集: {len(val_ids)}帧。")
    
    print(f"📦 正在打包最终极训练数据集 (ZIP)...")
    zip_path = shutil.make_archive(
        base_name=str(dataset_root_dir), 
        format='zip',
        root_dir=final_kitti_output_dir,  
        base_dir=dataset_folder_name
    )
    print(f"🎉 终极交付准备就绪！数据集已打包至 -> {zip_path}")

    return {
        "dataset_root_dir": dataset_root_dir,
        "zip_path": Path(zip_path),
    }


if __name__ == "__main__":
    print("\n" + "="*50)
    location_str = input("👉 请输入本次数据集的【采集地点与时间】(例如: Binjiang_20260319): ").strip()
    if not location_str:
        location_str = "Unknown_Location"
        print("⚠️ 未输入标识，将使用默认名 'Unknown_Location'")
    print("="*50 + "\n")
    
    build_final_dataset(location_str)
