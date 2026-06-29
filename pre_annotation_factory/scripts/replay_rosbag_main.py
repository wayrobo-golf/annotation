import os
import sys
# 确保能够找到 my_package
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from my_package.io_modules.nusc_json_exporter import generate_nuscenes_metadata
import subprocess
import time
import signal
import json
import shutil
import glob
import random
import uuid
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from contextlib import contextmanager
from pathlib import Path

# ================= 配置区 =================
# 1. 你的工程工作空间绝对路径
WORKSPACE_PATH = "/home/keyaoli/Code/Wayrobo/3D_Detection_Annotation"

# 2. 存放所有 bag 的根目录
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/BaoLi20260305"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/BaoLi20260305_Debug"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/XiangXue20260323"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/XiangXue20260323_Debug"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/YeNan20260318"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/YeNan20260318_Debug"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/Binjiang20260319"
TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/Binjiang20260319_Debug"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/YinXiu20260414/20260414/extracted"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/DongZhuang20260411/2026_4_11/extracted"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/DongZhuang20260411_Debug"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/XiHu20260330"
# TARGET_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/XiHu20260330_Debug"

# 3. QoS 文件的绝对路径
QOS_YAML_PATH = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_Rosbag/rosbag_play_qos_cfg.yaml"

# 4. 参数配置文件 (default.yaml) 的绝对路径
YAML_CONFIG_PATH = "/home/keyaoli/Code/Wayrobo/3D_Detection_Annotation/automatic_annotation/config/default.yaml"

PACKAGE_NAME = "automatic_annotation"
NODE_NAME = "automatic_annotation_node"

# ================= 数据集输出配置 =================
# 【新增】全链路开关：是否生成并打包 KITTI 格式数据集 (开启需耗费较多点云转换时间和磁盘空间)
GENERATE_KITTI_DATASET = False

# KITTI 输出路径
KITTI_OUTPUT_DIR = "/home/keyaoli/Data/AutoAnnotation/Wayrobo_KITTI_Dataset/Debug"

# Xtreme1 输出路径
XTREME1_OUTPUT_DIR = "/home/keyaoli/Data/AutoAnnotation/DataRecord/Binjiang20260319/Xtreme1_Upload"

SPLIT_RATIO = 0.8         # 80% 划入 train.txt, 20% 划入 val.txt
CONVERT_PCD_TO_BIN = True # 是否自动将点云转为 OpenPCDet 必须的 bin 格式 (仅在 GENERATE_KITTI_DATASET=True 时有效)
RANDOM_SEED = 42

# 是否在转换打包完成后，彻底删除 ROS2 install/share 目录下的原始生成数据
CLEANUP_SHARE_DATA = False 
RAW_DATA_ARCHIVE_DIR = "/home/keyaoli/Data/AutoAnnotation/Auto_Annotation_Origin"
PRESERVE_FULL_RAW_COPY = True
FULL_RAW_DATA_ARCHIVE_DIR = "/home/keyaoli/Data/AutoAnnotation/Auto_Annotation_Origin_Full"
MAX_EMPTY_FRAME_RATIO = 0.1 # 0.1 表示空帧数量最多为有目标帧数量的 10%
# ==========================================================


def select_xtreme_upload_tasks(valid_tasks, empty_tasks, max_empty_frame_ratio, random_seed):
    if valid_tasks:
        max_empty_count = int(len(valid_tasks) * max_empty_frame_ratio)
        random.seed(random_seed)
        random.shuffle(empty_tasks)
        kept_empty_tasks = empty_tasks[:max_empty_count]
        discarded_empty_tasks = empty_tasks[max_empty_count:]
    else:
        kept_empty_tasks = list(empty_tasks)
        discarded_empty_tasks = []

    tasks = valid_tasks + kept_empty_tasks
    random.seed(random_seed)
    random.shuffle(tasks)
    return kept_empty_tasks, discarded_empty_tasks, tasks


def write_submit_frame_manifest(manifest_path: Path, tasks):
    payload = []
    for (
        frame_id,
        _config_file,
        _label_path,
        _pcd_path,
        img_path,
        scene_name,
        _scene_dir,
    ) in tasks:
        payload.append(
            {
                "scene_name": scene_name,
                "frame_id": frame_id,
                "img_ext": Path(img_path).suffix,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path

def get_bags_to_process(path):
    """智能解析路径：支持单个 .db3 文件，或遍历目录下的所有 bag"""
    bags = []
    if os.path.isfile(path) and path.endswith('.db3'):
        bags.append(path)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "metadata.yaml" in files or any(f.endswith('.db3') for f in files):
                bags.append(root)
    return sorted(list(set(bags)))

def process_bag(bag_path, scene_index):
    """回放单个 Bag 并触发 C++ 节点生成标注数据"""
    SETUP_SCRIPT = os.path.join(WORKSPACE_PATH, "install/setup.bash")
    output_folder_name = f"Scene_{scene_index:02d}"
    
    ros_cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {SETUP_SCRIPT} && "
        f"ros2 run {PACKAGE_NAME} {NODE_NAME} --ros-args "
        f"--params-file {YAML_CONFIG_PATH} "
        f"-p use_sim_time:=true "
        f"-p custom_record_folder_name:={output_folder_name}"
    )

    print(f"\n{'='*50}")
    print(f"⏳ 正在启动 C++ 标注节点 (Scene {scene_index:02d})...")
    print(f"📁 目标目录: data/{output_folder_name}")
    print(f"{'='*50}")
    
    node_process = subprocess.Popen(
        ros_cmd, 
        shell=True, 
        executable="/bin/bash",
        preexec_fn=os.setsid 
    )
    
    print("⏲️ 等待节点初始化和加载地图 (15s)...")
    time.sleep(15) 

    bag_name = os.path.basename(bag_path).replace('.db3', '')
    
    bag_cmd = [
        "ros2", "bag", "play", bag_path, 
        "--clock"
    ]
    
    qos_yaml_path = str(QOS_YAML_PATH) if QOS_YAML_PATH is not None else None
    if qos_yaml_path and os.path.exists(qos_yaml_path):
        print(f"⚙️ 加载 QoS 配置文件: {qos_yaml_path}")
        bag_cmd.extend(["--qos-profile-overrides-path", qos_yaml_path])
    
    print(f"▶️ 开始播放 Bag...")
    subprocess.run(bag_cmd) 
    time.sleep(5) 
    
    print("✅ Bag 播放完毕，正在关闭 C++ 节点...")
    
    try:
        os.killpg(os.getpgid(node_process.pid), signal.SIGINT)
        node_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("⚠️ 节点未能按时退出，强制终止...")
        os.killpg(os.getpgid(node_process.pid), signal.SIGKILL)
        
    print(f"🎉 {bag_name} 处理完成！\n")


# ================= 数据后处理与转换引擎 =================

def pcd_to_bin_fixed(pcd_path, bin_path):
    """高速无损 PCD 转 BIN"""
    try:
        with open(pcd_path, 'rb') as f:
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('DATA binary'):
                    break
                if not line: return False
            data = np.fromfile(f, dtype=np.float32)
        points = data.reshape(-1, 4)
        points.tofile(bin_path)
        return True
    except Exception as e:
        print(f"  [Error] 转换 PCD 失败 {pcd_path}: {e}")
        return False

def load_camera_config(config_path):
    """从 JSON 中提取内参、Lidar->Cam 外参以及 Map->Lidar 外参"""
    with open(config_path, 'r') as f: 
        data = json.load(f)
        
    if isinstance(data, list): 
        data = data[0]
    if "camera_internal" not in data: 
        data = data[list(data.keys())[0]]
    
    intrinsics = data["camera_internal"]
    K = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]],
                  [0, 0, 1]])
    
    T_lidar2cam = np.array(data["camera_external"]).reshape((4, 4), order='F')
    
    T_lidar2map = None
    if "tf_lidar_to_map" in data:
        T_lidar2map = np.array(data["tf_lidar_to_map"]).reshape((4, 4), order='F')
        
    return K, T_lidar2cam, T_lidar2map

def generate_calib(config_path, out_calib):
    """生成 KITTI 标准的 calib.txt"""
    K, T_lidar2cam, T_lidar2map = load_camera_config(config_path)
    
    with open(out_calib, 'w') as f:
        P_str = " ".join([f"{x:.6f}" for x in np.pad(K, ((0,0),(0,1))).flatten()])
        f.writelines([f"P{i}: {P_str}\n" for i in range(4)])
        f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")
        Tr = " ".join([f"{x:.6f}" for x in T_lidar2cam[:3, :].flatten()])
        f.write(f"Tr_velo_to_cam: {Tr}\nTr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n")
        if T_lidar2map is not None:
            Tr_velo2map = " ".join([f"{x:.6f}" for x in T_lidar2map[:3, :].flatten()])
            f.write(f"Tr_velo_to_map: {Tr_velo2map}\n")

def convert_kitti_to_xtreme1_json(kitti_txt_path, config_path, out_json_path):
    """【核心计算模块】复刻 C++ 几何逻辑，将 Camera 系的 KITTI TXT 转换为 Lidar 系的 Xtreme1 JSON"""
    _, T_lidar2cam, _ = load_camera_config(config_path)
    
    # 计算相机到雷达的外参逆矩阵 (C++ 中的 lcam2lidar)
    T_cam2lidar = np.linalg.inv(T_lidar2cam)
    R_cam2lidar = T_cam2lidar[:3, :3]
    t_cam2lidar = T_cam2lidar[:3, 3]

    # 【新增核心修正】：构建从 Xtreme1 Local (ROS FLU) 到 KITTI Local 的坐标系转换逆矩阵
    # 作用：把 Xtreme1 默认的 (X前, Y左, Z上) 映射回 KITTI 的 (X前, Y下, Z右)
    R_kitti2ros_inv = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0]
    ])

    xtreme1_objects = []

    if os.path.exists(kitti_txt_path):
        with open(kitti_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17: continue
                
                class_name = parts[0]
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                ry = float(parts[14])
                rx, rz = float(parts[15]), float(parts[16])
                track_id = parts[17] if len(parts) >= 19 else None
                track_name = parts[18] if len(parts) >= 19 else None
                
                # --- A. 构建相机系下的旋转矩阵 (R_6dof) ---
                rot_cam = R.from_euler('xyz', [rx, ry, rz])
                R_6dof = rot_cam.as_matrix()
                
                # --- B. 计算雷达系下的几何中心 ---
                T_cam = np.array([x, y, z])
                geometric_center_local = np.array([0.0, -h/2.0, 0.0]) # KITTI 局部系 Y 轴向下
                geometric_center_cam = R_6dof.dot(geometric_center_local) + T_cam
                center_lidar = R_cam2lidar.dot(geometric_center_cam) + t_cam2lidar
                
                # --- C. 计算雷达系下的欧拉角 (引入 R_kitti2ros_inv 修正侧翻问题) ---
                # 矩阵相乘顺序：先将 Xtreme1 局部系转为 KITTI 局部系，再转到 Camera，最后转到 LiDAR
                R_lidar = R_cam2lidar.dot(R_6dof).dot(R_kitti2ros_inv)
                rx_l, ry_l, rz_l = R.from_matrix(R_lidar).as_euler('xyz')
                
                # 组装 Xtreme1 Object
                obj = {
                    "id": str(uuid.uuid4()),
                    "type": "3D_BOX",
                    "className": class_name,
                    "modelClass": class_name,
                    "contour": {
                        "size3D": {"x": l, "y": w, "z": h},
                        "center3D": {"x": center_lidar[0], "y": center_lidar[1], "z": center_lidar[2]},
                        "rotation3D": {"x": rx_l, "y": ry_l, "z": rz_l}
                    }
                }
                if track_id and track_name:
                    obj["trackId"] = track_id
                    obj["trackName"] = track_name
                xtreme1_objects.append(obj)
                
    with open(out_json_path, 'w') as f:
        json.dump({"objects": xtreme1_objects}, f, indent=2)


def build_datasets(workspace_path, location_str):
    """统一构建数据集，并执行打包与深度清理"""
    print(f"\n{'='*50}")
    print("🚀 开始多模态数据集清洗与构建流水线...")
    print(f"⚙️  配置状态 -> 生成 KITTI: {'✅ 开启' if GENERATE_KITTI_DATASET else '🚫 关闭 (仅生成 Xtreme1)'}")
    print(f"{'='*50}")

    current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_folder_name = f"3DBox_Annotation_{current_time_str}_{location_str}"
    xtreme1_folder_name = f"Xtreme1_Upload_{current_time_str}_{location_str}"
    full_archive_target_dir = None
    archive_target_dir = None
    
    # 1. 准备目标路径
    xtreme1_root_dir = os.path.join(XTREME1_OUTPUT_DIR, xtreme1_folder_name)
    os.makedirs(xtreme1_root_dir, exist_ok=True)
    
    if GENERATE_KITTI_DATASET:
        kitti_root_dir = os.path.join(KITTI_OUTPUT_DIR, dataset_folder_name)
        training_base = os.path.join(kitti_root_dir, "training")
        imagesets_dir = os.path.join(kitti_root_dir, "ImageSets")
        for fol in ["label_2", "calib", "image_2", "velodyne"]:
            os.makedirs(os.path.join(training_base, fol), exist_ok=True)
        os.makedirs(imagesets_dir, exist_ok=True)

    # 2. 扫描有效数据与空帧清洗
    share_data_dir = os.path.join(workspace_path, "install", PACKAGE_NAME, "share", PACKAGE_NAME, "data")
    scene_dirs = glob.glob(os.path.join(share_data_dir, "data_record_*", "Scene_*"))

    if not scene_dirs:
        print(f"❌ 未在 {share_data_dir} 找到任何生成的数据目录，无法整合！")
        return

    valid_tasks = []
    empty_tasks = []

    for scene_dir in sorted(scene_dirs):
        scene_name = os.path.basename(scene_dir)
        config_files = glob.glob(os.path.join(scene_dir, "camera_config", "*.json"))
        
        for config_file in config_files:
            frame_id = os.path.basename(config_file).split('.')[0]
            label_path = os.path.join(scene_dir, "label_2", f"{frame_id}.txt")
            pcd_path = os.path.join(scene_dir, "lidar_point_cloud_0", f"{frame_id}.pcd")
            
            img_matches = glob.glob(os.path.join(scene_dir, "camera_image_0", f"{frame_id}.*"))
            img_path = img_matches[0] if img_matches else None

            if os.path.exists(label_path) and os.path.exists(pcd_path) and img_path:
                is_empty = True
                with open(label_path, 'r') as f:
                    if any(line.strip() for line in f):
                        is_empty = False
                
                # 【优化1关联】加入了 scene_dir，供后续斩草除根清理时使用
                task_tuple = (frame_id, config_file, label_path, pcd_path, img_path, scene_name, scene_dir)
                if is_empty:
                    empty_tasks.append(task_tuple)
                else:
                    valid_tasks.append(task_tuple)

    kept_empty_tasks, discarded_empty_tasks, tasks = select_xtreme_upload_tasks(
        valid_tasks,
        empty_tasks,
        MAX_EMPTY_FRAME_RATIO,
        RANDOM_SEED,
    )
    discarded_empty_count = len(discarded_empty_tasks)

    print(f"🔎 发现有目标帧: {len(valid_tasks)} 帧，原始空帧: {len(empty_tasks)} 帧。")
    print(f"⚖️ 保留 {len(kept_empty_tasks)} 个负样本，拟彻底销毁 {discarded_empty_count} 个多余空帧及其所有附属副产物。")
    print(f"🚀 共整合 {len(tasks)} 帧数据，正在转换流水线...")

    submit_manifest_path = Path(xtreme1_root_dir) / "submit_frames.json"
    write_submit_frame_manifest(submit_manifest_path, tasks)

    # 3. 核心大循环：处理保留下来的帧
    split_idx = int(len(tasks) * SPLIT_RATIO)
    train_ids, val_ids = [], []

    for i, (frame_id, config_file, label_path, pcd_path, img_path, scene_name, _) in enumerate(tasks):
        img_ext = os.path.splitext(img_path)[1] 

        # ================= 路线 A: KITTI 构建 (受控) =================
        if GENERATE_KITTI_DATASET:
            out_label = os.path.join(training_base, "label_2", f"{frame_id}.txt")
            out_calib = os.path.join(training_base, "calib", f"{frame_id}.txt")
            out_bin = os.path.join(training_base, "velodyne", f"{frame_id}.bin")
            out_img = os.path.join(training_base, "image_2", f"{frame_id}{img_ext}")

            shutil.copy2(label_path, out_label)
            generate_calib(config_file, out_calib)
            shutil.copy2(img_path, out_img)
            if CONVERT_PCD_TO_BIN:
                pcd_to_bin_fixed(pcd_path, out_bin)
            else:
                shutil.copy2(pcd_path, os.path.join(training_base, "velodyne", f"{frame_id}.pcd"))

            if i < split_idx: train_ids.append(frame_id)
            else: val_ids.append(frame_id)

        # ================= 路线 B: Xtreme1 构建 =================
        scene_dir_xtreme = os.path.join(xtreme1_root_dir, scene_name)
        
        for sub_dir in ["camera_config", "camera_image_0", "lidar_point_cloud_0", "result"]:
            os.makedirs(os.path.join(scene_dir_xtreme, sub_dir), exist_ok=True)
            
        shutil.copy2(config_file, os.path.join(scene_dir_xtreme, "camera_config", f"{frame_id}.json"))
        shutil.copy2(img_path, os.path.join(scene_dir_xtreme, "camera_image_0", f"{frame_id}{img_ext}"))
        shutil.copy2(pcd_path, os.path.join(scene_dir_xtreme, "lidar_point_cloud_0", f"{frame_id}.pcd"))
        
        out_xtreme_json = os.path.join(scene_dir_xtreme, "result", f"{frame_id}.json")
        convert_kitti_to_xtreme1_json(label_path, config_file, out_xtreme_json)

        if (i + 1) % 50 == 0:
            print(f"  已转换 {i + 1} / {len(tasks)} 帧...")

    # ================= 收尾：生成索引与打包 =================
    if GENERATE_KITTI_DATASET:
        with open(os.path.join(imagesets_dir, "train.txt"), "w") as f: f.write("\n".join(train_ids))
        with open(os.path.join(imagesets_dir, "val.txt"), "w") as f: f.write("\n".join(val_ids))
        print(f"✅ KITTI 转换完成！训练集: {len(train_ids)}帧，验证集: {len(val_ids)}帧。")
        generate_nuscenes_metadata(kitti_root_dir, location_str)
        print(f"📦 正在打包 KITTI 为 ZIP...")
        kitti_zip = shutil.make_archive(
            base_name=kitti_root_dir, 
            format='zip',
            root_dir=KITTI_OUTPUT_DIR,  
            base_dir=dataset_folder_name
        )
        print(f"🎉 训练数据打包至 -> {kitti_zip}")
    else:
        print(f"✅ Xtreme1 转换完成！(已跳过 KITTI 生成)")

    print(f"📦 正在打包 Xtreme1 为 ZIP...")
    xtreme1_zip = shutil.make_archive(
        base_name=xtreme1_root_dir, 
        format='zip',
        root_dir=XTREME1_OUTPUT_DIR,  
        base_dir=xtreme1_folder_name
    )
    print(f"🎉 Xtreme1 标注数据打包至 -> {xtreme1_zip}")

    # ================= 深度清理与归档逻辑 =================
    data_record_dirs = set(os.path.dirname(scene_dir) for scene_dir in scene_dirs)

    if PRESERVE_FULL_RAW_COPY:
        full_archive_folder_name = f"{dataset_folder_name}_origin_full"
        full_archive_target_dir = os.path.join(FULL_RAW_DATA_ARCHIVE_DIR, full_archive_folder_name)
        print(f"\n💾 正在备份完整原始生成数据...")

        try:
            os.makedirs(full_archive_target_dir, exist_ok=False)
        except FileExistsError as e:
            raise RuntimeError(f"完整原始数据备份目录已存在，请先处理后再重试: {full_archive_target_dir}") from e

        for dr_dir in sorted(data_record_dirs):
            backup_target_dir = os.path.join(full_archive_target_dir, os.path.basename(dr_dir))
            try:
                shutil.copytree(dr_dir, backup_target_dir, copy_function=shutil.copy2)
                print(f"  [+] 已完整备份: {os.path.basename(dr_dir)}")
            except Exception as e:
                raise RuntimeError(f"完整原始数据备份失败，已中止后续裁剪与归档: {dr_dir} -> {e}") from e

        print(f"✨ 完整原始数据已备份至 -> {full_archive_target_dir}")

    if CLEANUP_SHARE_DATA:
        print(f"\n🧹 [彻底模式] 正在清理 ROS 2 install/share 目录下的原始生成数据...")
        for dr_dir in data_record_dirs:
            try:
                shutil.rmtree(dr_dir)
            except Exception as e:
                print(f"  ⚠️ 无法删除 {dr_dir}: {e}")
        print("✨ 原始生成数据全局清理完毕！")
    else:
        print(f"\n📦 正在归档原始生成数据...")
        if discarded_empty_tasks:
            print(f"  ✂️ [斩草除根] 正在彻底清理 {discarded_empty_count} 个多余空帧及其所有验证副产物...")
            # 【优化1落地】遍历所有被丢弃的帧，递归查找该 Scene 下包含此 frame_id 的所有文件并物理删除
            for task in discarded_empty_tasks:
                frame_id = task[0]
                scene_dir = task[-1]  # 取出上面存好的 scene_dir
                
                # os.walk 递归遍历该 Scene_XX 下的所有文件夹 (如 depth_image, fuse_image 等)
                for root_path, _, file_names in os.walk(scene_dir):
                    for file_name in file_names:
                        if str(frame_id) in file_name:  # 只要文件名包含这个时间戳，无差别击杀
                            file_to_delete = os.path.join(root_path, file_name)
                            try:
                                os.remove(file_to_delete)
                            except: 
                                pass 
            print("  ✂️ 修剪完成，副产物已全部净化！")

        archive_folder_name = f"{dataset_folder_name}_origin"
        archive_target_dir = os.path.join(RAW_DATA_ARCHIVE_DIR, archive_folder_name)
        os.makedirs(archive_target_dir, exist_ok=True)
        
        for dr_dir in data_record_dirs:
            try:
                shutil.move(dr_dir, archive_target_dir)
                print(f"  [+] 已归档精简版数据: {os.path.basename(dr_dir)}")
            except Exception as e:
                print(f"  ⚠️ 归档失败 {dr_dir}: {e}")
                
        print(f"✨ 数据已安全归档至 -> {archive_target_dir}")

    return {
        "xtreme_upload_dir": Path(xtreme1_root_dir),
        "xtreme_zip_path": Path(xtreme1_zip),
        "submit_manifest_path": submit_manifest_path,
        "raw_archive_dir": Path(archive_target_dir) if archive_target_dir else None,
        "full_raw_archive_dir": Path(full_archive_target_dir) if full_archive_target_dir else None,
    }


def run_xtreme_upload_bundle(workspace_path, location_str):
    global GENERATE_KITTI_DATASET

    previous = GENERATE_KITTI_DATASET
    GENERATE_KITTI_DATASET = False
    try:
        build_datasets(workspace_path, location_str)
    finally:
        GENERATE_KITTI_DATASET = previous


@contextmanager
def temporary_config(**overrides):
    previous = {}
    for key, value in overrides.items():
        previous[key] = globals()[key]
        globals()[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            globals()[key] = value


def run_auto_annotation_job(
    target_path,
    location_str,
    *,
    workspace_path=WORKSPACE_PATH,
    qos_yaml_path=QOS_YAML_PATH,
    yaml_config_path=YAML_CONFIG_PATH,
    xtreme1_output_dir=XTREME1_OUTPUT_DIR,
    raw_data_archive_dir=RAW_DATA_ARCHIVE_DIR,
    preserve_full_raw_copy=PRESERVE_FULL_RAW_COPY,
    full_raw_data_archive_dir=FULL_RAW_DATA_ARCHIVE_DIR,
    cleanup_share_data=CLEANUP_SHARE_DATA,
    max_empty_frame_ratio=MAX_EMPTY_FRAME_RATIO,
):
    overrides = {
        "WORKSPACE_PATH": str(workspace_path),
        "QOS_YAML_PATH": str(qos_yaml_path) if qos_yaml_path is not None else None,
        "YAML_CONFIG_PATH": str(yaml_config_path),
        "XTREME1_OUTPUT_DIR": str(xtreme1_output_dir),
        "RAW_DATA_ARCHIVE_DIR": str(raw_data_archive_dir),
        "PRESERVE_FULL_RAW_COPY": preserve_full_raw_copy,
        "FULL_RAW_DATA_ARCHIVE_DIR": str(full_raw_data_archive_dir),
        "CLEANUP_SHARE_DATA": cleanup_share_data,
        "MAX_EMPTY_FRAME_RATIO": max_empty_frame_ratio,
        "GENERATE_KITTI_DATASET": False,
    }

    with temporary_config(**overrides):
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"找不到目标路径: {target_path}")

        bag_list = get_bags_to_process(str(target_path))
        if not bag_list:
            raise ValueError(f"在 {target_path} 中没有找到任何有效的 ROS2 Bag！")

        for index, bag in enumerate(bag_list, start=1):
            process_bag(bag, index)

        return build_datasets(str(workspace_path), location_str)


if __name__ == "__main__":
    if not os.path.exists(TARGET_PATH):
        print(f"❌ 找不到目标路径: {TARGET_PATH}")
        exit(1)

    bag_list = get_bags_to_process(TARGET_PATH)
    
    if not bag_list:
        print(f"❌ 在 {TARGET_PATH} 中没有找到任何有效的 ROS2 Bag！")
        exit(1)
        
    print(f"🔎 找到了 {len(bag_list)} 个 Bag 准备处理。")
    
    print("\n" + "="*50)
    location_str = input("👉 请输入本次数据集的【采集地点与时间】(例如: YinXiu_20260327): ").strip()
    if not location_str:
        location_str = "Unknown_Location"
        print("⚠️ 未输入标识，将使用默认名 'Unknown_Location'")
    print("="*50 + "\n")
    
    # 1. 自动化回放并触发 C++ 节点执行核心标注
    for index, bag in enumerate(bag_list, start=1):
        process_bag(bag, index)
        
    print("\n🏆 所有 Bag 均已播放并抽取完毕！")

    # 2. 自动化后处理：按需构建数据集与斩草除根
    build_datasets(WORKSPACE_PATH, location_str)
