#include "automatic_annotation/gen_prompt_point.hpp"

#include <pcl/filters/crop_box.h>
#include <pcl/io/pcd_io.h>

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
namespace automatic_annotation {
namespace {

constexpr int64_t kPoseCacheCoverageWaitTimeoutMs = 30;
constexpr int64_t kPoseCacheCoveragePollIntervalMs = 1;

std::string ToLowerCopy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

RawDeskewMode ParseRawDeskewMode(const std::string& value) {
  const std::string normalized = ToLowerCopy(value);
  if (normalized == "off") {
    return RawDeskewMode::Off;
  }
  if (normalized == "accurate") {
    return RawDeskewMode::Accurate;
  }
  return RawDeskewMode::Fast;
}

RawDeskewReferenceTime ParseRawDeskewReferenceTime(const std::string& value) {
  const std::string normalized = ToLowerCopy(value);
  if (normalized == "scan_start") {
    return RawDeskewReferenceTime::ScanStart;
  }
  if (normalized == "scan_end") {
    return RawDeskewReferenceTime::ScanEnd;
  }
  return RawDeskewReferenceTime::HeaderStamp;
}

RawDeskewTimestampPolarity ParseRawDeskewTimestampPolarity(
    const std::string& value) {
  const std::string normalized = ToLowerCopy(value);
  if (normalized == "from_scan_start") {
    return RawDeskewTimestampPolarity::FromScanStart;
  }
  return RawDeskewTimestampPolarity::FromScanEnd;
}

RawDeskewPoseSource ParseRawDeskewPoseSource(const std::string& value) {
  const std::string normalized = ToLowerCopy(value);
  if (normalized == "tf") {
    return RawDeskewPoseSource::Tf;
  }
  return RawDeskewPoseSource::InsOdom;
}

RawDeskewMissingPosePolicy ParseRawDeskewMissingPosePolicy(
    const std::string& value) {
  const std::string normalized = ToLowerCopy(value);
  if (normalized == "skip") {
    return RawDeskewMissingPosePolicy::Skip;
  }
  return RawDeskewMissingPosePolicy::FallbackFast;
}

const char* RawDeskewModeToString(RawDeskewMode mode) {
  switch (mode) {
    case RawDeskewMode::Off:
      return "off";
    case RawDeskewMode::Fast:
      return "fast";
    case RawDeskewMode::Accurate:
      return "accurate";
  }
  return "off";
}

const char* RawDeskewReferenceTimeToString(RawDeskewReferenceTime reference) {
  switch (reference) {
    case RawDeskewReferenceTime::HeaderStamp:
      return "header_stamp";
    case RawDeskewReferenceTime::ScanStart:
      return "scan_start";
    case RawDeskewReferenceTime::ScanEnd:
      return "scan_end";
  }
  return "header_stamp";
}

const char* RawDeskewTimestampPolarityToString(
    RawDeskewTimestampPolarity polarity) {
  switch (polarity) {
    case RawDeskewTimestampPolarity::FromScanStart:
      return "from_scan_start";
    case RawDeskewTimestampPolarity::FromScanEnd:
      return "from_scan_end";
  }
  return "from_scan_end";
}

const char* RawDeskewPoseSourceToString(RawDeskewPoseSource pose_source) {
  switch (pose_source) {
    case RawDeskewPoseSource::InsOdom:
      return "ins_odom";
    case RawDeskewPoseSource::Tf:
      return "tf";
  }
  return "ins_odom";
}

const char* RawDeskewMissingPosePolicyToString(
    RawDeskewMissingPosePolicy policy) {
  switch (policy) {
    case RawDeskewMissingPosePolicy::FallbackFast:
      return "fallback_fast";
    case RawDeskewMissingPosePolicy::Skip:
      return "skip";
  }
  return "fallback_fast";
}

}  // namespace

GenPromptPoint::GenPromptPoint(const rclcpp::Node::SharedPtr& node)
    : node_(node) {
  callback_group_lidar_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  callback_group_pose_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  callback_group_camera_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
  INFO("AutoAnnotation GenPromptPoint constructor called, Version {}",
       ConstValue::kVersion);
  package_share_path_ =
      ament_index_cpp::get_package_share_directory(ConstValue::kPackageName);
  INFO("Package path: {}", package_share_path_);
  InitConfigFromParamsServer();
  InitClassMappingAndReflact();
  InitMapAndLabel();
  InitPublishAndSubscription();
  InitTfListener();
}

void GenPromptPoint::InitConfigFromParamsServer() {
  InitGlobalMapAndLabelAddr();
  InitPointCloudTopic();
  InitPointCloudSettings();
  InitRawDeskewSettings();
  InitImageSettings();
  InitCameraTopics();
  InitCameraIntrinsicsTopics();
  InitSaveFolder();
  InitAutoAnnotation();
  InitDefaultTfSettings();
}

void GenPromptPoint::InitGlobalMapAndLabelAddr() {
  node_config_.global_pc_map_addr =
      get_param_value_form_params_server<std::string>(
          "global_pc_map_addr", ConstValue::kDefaultGlobalPCMapAddr);
  node_config_.pc_annotation_file_addr =
      get_param_value_form_params_server<std::string>(
          "pc_annotation_file_addr", ConstValue::kDefaultPCAnnotationFileAddr);
}

void GenPromptPoint::InitPointCloudTopic() {
  node_config_.point_cloud_type = get_param_value_form_params_server<int64_t>(
      "point_cloud_type", ConstValue::kPointCloudTypeOriginal);
  if (node_config_.point_cloud_type == 0) {
    annotation_status_.pc_type = AutoAnnotationStatus::PointCloudType::Original;
    node_config_.point_cloud_topic =
        get_param_value_form_params_server<std::string>(
            "origin_point_cloud_topic",
            ConstValue::kDefaultOriginPointCloudTopic);
  } else if (node_config_.point_cloud_type == 1) {
    annotation_status_.pc_type =
        AutoAnnotationStatus::PointCloudType::Undistorted;
    node_config_.point_cloud_topic =
        get_param_value_form_params_server<std::string>(
            "undistorted_point_cloud_topic",
            ConstValue::kDefaultUndistortedPointCloudTopic);
  } else {
    ERROR("Invalid point_cloud_type: {}. Must be 0 or 1.",
          node_config_.point_cloud_type);
    throw std::invalid_argument("Invalid point_cloud_type");
  }
}

void GenPromptPoint::InitPointCloudSettings() {
  node_config_.target_accumulate_pc_num =
      get_param_value_form_params_server<int64_t>(
          "target_accumulate_pc_num", ConstValue::kTargetPointCloudAccumulate);
  node_config_.publish_accumulated_pc =
      get_param_value_form_params_server<bool>(
          "publish_accumulated_pc", ConstValue::kDefaultPublishAccumulatedPC);
}

void GenPromptPoint::InitRawDeskewSettings() {
  std::string raw_deskew_mode =
      get_param_value_form_params_server<std::string>("raw_deskew_mode", "off");
  node_config_.raw_deskew_mode = ParseRawDeskewMode(raw_deskew_mode);

  node_config_.raw_deskew_fixed_frame =
      get_param_value_form_params_server<std::string>("raw_deskew_fixed_frame",
                                                      ConstValue::kInsMap);

  std::string raw_deskew_reference_time =
      get_param_value_form_params_server<std::string>(
          "raw_deskew_reference_time", "header_stamp");
  node_config_.raw_deskew_reference_time =
      ParseRawDeskewReferenceTime(raw_deskew_reference_time);

  node_config_.raw_deskew_timestamp_unit_sec =
      get_param_value_form_params_server<double>(
          "raw_deskew_timestamp_unit_sec", 1e-6);

  std::string raw_deskew_timestamp_polarity =
      get_param_value_form_params_server<std::string>(
          "raw_deskew_timestamp_polarity", "from_scan_end");
  node_config_.raw_deskew_timestamp_polarity =
      ParseRawDeskewTimestampPolarity(raw_deskew_timestamp_polarity);

  node_config_.raw_deskew_bucket_count_fast =
      get_param_value_form_params_server<int64_t>("raw_deskew_bucket_count_fast",
                                                  10);
  node_config_.raw_deskew_bucket_count_accurate =
      get_param_value_form_params_server<int64_t>(
          "raw_deskew_bucket_count_accurate", 50);
  node_config_.raw_deskew_publish_debug_cloud =
      get_param_value_form_params_server<bool>(
          "raw_deskew_publish_debug_cloud", false);
  node_config_.raw_deskew_save_debug_cloud =
      get_param_value_form_params_server<bool>("raw_deskew_save_debug_cloud",
                                               false);
  node_config_.raw_deskew_min_valid_ratio =
      get_param_value_form_params_server<double>("raw_deskew_min_valid_ratio",
                                                 0.9);
  node_config_.raw_deskew_save_folder =
      get_param_value_form_params_server<std::string>(
          "raw_deskew_save_folder", "lidar_point_cloud_deskewed_0");
  node_config_.raw_deskew_pose_source = ParseRawDeskewPoseSource(
      get_param_value_form_params_server<std::string>(
          "raw_deskew_pose_source", "ins_odom"));
  node_config_.raw_deskew_pose_topic =
      get_param_value_form_params_server<std::string>(
          "raw_deskew_pose_topic", "/novatel/oem7/ins_odom");
  node_config_.raw_deskew_missing_pose_policy =
      ParseRawDeskewMissingPosePolicy(
          get_param_value_form_params_server<std::string>(
              "raw_deskew_missing_pose_policy", "fallback_fast"));
  node_config_.raw_deskew_pose_cache_duration_sec =
      get_param_value_form_params_server<double>(
          "raw_deskew_pose_cache_duration_sec", 30.0);

  INFO(
      "Raw deskew resolved. mode={} pose_source={} pose_topic={} "
      "missing_pose_policy={} pose_cache_duration_sec={:.1f}",
      RawDeskewModeToString(node_config_.raw_deskew_mode),
      RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source),
      node_config_.raw_deskew_pose_topic,
      RawDeskewMissingPosePolicyToString(
          node_config_.raw_deskew_missing_pose_policy),
      node_config_.raw_deskew_pose_cache_duration_sec);
}

void GenPromptPoint::InitImageSettings() {
  node_config_.image_source_type = get_param_value_form_params_server<int64_t>(
      "image_source_type", ConstValue::kImageSourceLeftCamera);
  node_config_.image_save_interval =
      get_param_value_form_params_server<int64_t>(
          "image_save_interval", ConstValue::kDefaultImageSaveInterval);
}

void GenPromptPoint::InitCameraTopics() {
  node_config_.left_image_topic =
      get_param_value_form_params_server<std::string>(
          "left_image_topic", ConstValue::kDefaultLeftImageTopic);
  node_config_.right_image_topic =
      get_param_value_form_params_server<std::string>(
          "right_image_topic", ConstValue::kDefaultRightImageTopic);
}

void GenPromptPoint::InitCameraIntrinsicsTopics() {
  node_config_.left_camera_intrinsic_topic =
      get_param_value_form_params_server<std::string>(
          "left_camera_intrinsic_topic",
          ConstValue::kDefaultLeftCameraIntrinsicTopic);

  node_config_.right_camera_intrinsic_topic =
      get_param_value_form_params_server<std::string>(
          "right_camera_intrinsic_topic",
          ConstValue::kDefaultRightCameraIntrinsicTopic);
}

void GenPromptPoint::InitAutoAnnotation() {
  node_config_.auto_annotation = get_param_value_form_params_server<bool>(
      "auto_annotation", ConstValue::kDefaultAutoAnnotation);
  node_config_.save_raw_image =
      get_param_value_form_params_server<bool>("save_raw_image", true);
  node_config_.save_synced_pcd =
      get_param_value_form_params_server<bool>("save_synced_pcd", true);
  node_config_.generate_depth_map =
      get_param_value_form_params_server<bool>("generate_depth_map", false);
  node_config_.generate_semantic_mask =
      get_param_value_form_params_server<bool>("generate_semantic_mask", false);
  node_config_.generate_kitti_label =
      get_param_value_form_params_server<bool>("generate_kitti_label", true);
  node_config_.generate_nusc_label =
      get_param_value_form_params_server<bool>("generate_nusc_label", false);
  node_config_.generate_xtreme1_json =
      get_param_value_form_params_server<bool>("generate_xtreme1_json", false);
  node_config_.enable_visual_verify =
      get_param_value_form_params_server<bool>("enable_visual_verify", false);
}

void GenPromptPoint::InitDefaultTfSettings() {
  node_config_.tf_wait_timeout_sec = get_param_value_form_params_server<double>(
      "tf_wait_timeout_sec", ConstValue::kDefaultTfWaitTimeoutSec);

  std::vector<double> zeros = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  // 左相机 -> 雷达
  auto lcam_vec = get_param_value_form_params_server<std::vector<double>>(
      "default_tf_lcam_to_lidar", zeros);
  node_config_.default_isometry_lcam2lidar = ConvertXYZRPYToIsometry(lcam_vec);
  annotation_status_.isometry_lcam2lidar =
      node_config_.default_isometry_lcam2lidar;

  // 右相机 -> 雷达
  auto rcam_vec = get_param_value_form_params_server<std::vector<double>>(
      "default_tf_rcam_to_lidar", zeros);
  node_config_.default_isometry_rcam2lidar = ConvertXYZRPYToIsometry(rcam_vec);
  annotation_status_.isometry_rcam2lidar =
      node_config_.default_isometry_rcam2lidar;

  // 雷达 -> 车体
  auto lidar2ins_vec = get_param_value_form_params_server<std::vector<double>>(
      "default_tf_lidar_to_ins", zeros);
  node_config_.default_isometry_lidar2ins =
      ConvertXYZRPYToIsometry(lidar2ins_vec);
  annotation_status_.isometry_lidar2ins =
      node_config_.default_isometry_lidar2ins;
}

Eigen::Isometry3d GenPromptPoint::ConvertXYZRPYToIsometry(
    const std::vector<double>& params) {
  Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
  if (params.size() != 6) {
    ERROR(
        "Default TF params size must be 6 (x, y, z, roll, pitch, yaw). Using "
        "Identity Matrix.");
    return iso;
  }

  iso.translation() = Eigen::Vector3d(params[0], params[1], params[2]);

  // 将欧拉角从 度 转换为 弧度
  double r = params[3] * M_PI / 180.0;
  double p = params[4] * M_PI / 180.0;
  double y = params[5] * M_PI / 180.0;

  Eigen::AngleAxisd rollAngle(r, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(p, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(y, Eigen::Vector3d::UnitZ());

  // 顺序: yaw * pitch * roll
  Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
  iso.rotate(q);

  return iso;
}

void GenPromptPoint::InitSaveFolder() {
  std::string custom_folder_name =
      get_param_value_form_params_server<std::string>(
          "custom_record_folder_name", "");
  std::string data_record_folder;
  if (custom_folder_name.empty()) {
    // 如果没有传入，就按原来的逻辑用时间戳命名
    data_record_folder = "data/data_record_" + get_current_localtime_str();
    INFO("No custom folder name provided. Using default folder name: {}",
         data_record_folder);
  } else {
    // 如果传入了，就用传入的名字（比如 Bag 的名字）
    data_record_folder = "data/data_record_" + get_current_localtime_str() +
                         "/" + custom_folder_name;
    INFO("Custom folder name provided: {}. Using folder name: {}",
         custom_folder_name, data_record_folder);
  }
  std::filesystem::path package_share_path(package_share_path_);
  std::filesystem::path data_record_relative_folder(data_record_folder);
  std::filesystem::path data_record_folder_path =
      package_share_path / data_record_relative_folder;
  // 从ROS参数服务器获取存储路径 (相对于功能包share文件夹的相对路径)
  annotation_status_.data_record_root_folder = data_record_folder_path.string();
  node_config_.left_camera_save_folder =
      get_param_value_form_params_server<std::string>(
          "left_camera_fig_save_folder",
          ConstValue::kDefaultLeftCameraFigSaveFolder);
  node_config_.right_camera_save_folder =
      get_param_value_form_params_server<std::string>(
          "right_camera_fig_save_folder",
          ConstValue::kDefaultRightCameraFigSaveFolder);
  node_config_.accumulate_pc_save_folder =
      get_param_value_form_params_server<std::string>(
          "accumulate_pc_save_folder",
          ConstValue::kDefaultAccumulatePointCloudSaveFolder);
  // 通过文件系统确认路径存在及其权限
  std::filesystem::path accumulate_pc_save_relative_folder(
      node_config_.accumulate_pc_save_folder);
  std::filesystem::path accumulate_pc_save_folder =
      data_record_folder_path / accumulate_pc_save_relative_folder;
  if (EnsureDirectoryWithFullPermissions(accumulate_pc_save_folder)) {
    annotation_status_.accumulate_pc_save_folder =
        accumulate_pc_save_folder.string();
  } else {
    throw std::runtime_error(
        "Unable to get full permission for saving point cloud");
  }

  if (node_config_.raw_deskew_save_debug_cloud) {
    std::filesystem::path raw_deskew_save_relative_folder(
        node_config_.raw_deskew_save_folder);
    std::filesystem::path raw_deskew_save_folder =
        data_record_folder_path / raw_deskew_save_relative_folder;
    if (EnsureDirectoryWithFullPermissions(raw_deskew_save_folder)) {
      annotation_status_.raw_deskew_save_folder =
          raw_deskew_save_folder.string();
    } else {
      throw std::runtime_error(
          "Unable to get full permission for saving raw deskew point cloud");
    }
  }

  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceLeftCamera)) {
    INFO("Checking if left_camera_save_folder is useable.");
    std::filesystem::path left_original_image_save_relative_folder(
        node_config_.left_camera_save_folder);
    std::filesystem::path left_fuse_image_save_relative_folder("fuse_image");
    std::filesystem::path left_depth_image_save_relative_folder("depth_image");
    std::filesystem::path left_semantic_image_save_relative_folder(
        "semantic_image");
    std::filesystem::path left_label_save_relative_folder("label_2");

    std::filesystem::path left_original_image_save_folder =
        data_record_folder_path / left_original_image_save_relative_folder;

    std::filesystem::path left_fuse_image_save_folder =
        data_record_folder_path / left_fuse_image_save_relative_folder;

    std::filesystem::path left_depth_image_save_folder =
        data_record_folder_path / left_depth_image_save_relative_folder;

    std::filesystem::path left_semantic_image_save_folder =
        data_record_folder_path / left_semantic_image_save_relative_folder;

    std::filesystem::path left_label_save_folder =
        data_record_folder_path / left_label_save_relative_folder;

    if (EnsureDirectoryWithFullPermissions(left_original_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(left_fuse_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(left_depth_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(left_semantic_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(left_label_save_folder)) {
      annotation_status_.left_original_image_save_folder =
          left_original_image_save_folder.string();
      annotation_status_.left_fuse_image_save_folder =
          left_fuse_image_save_folder.string();
      annotation_status_.left_depth_image_save_folder =
          left_depth_image_save_folder.string();
      annotation_status_.left_semantic_image_save_folder =
          left_semantic_image_save_folder.string();
      annotation_status_.left_label_save_folder =
          left_label_save_folder.string();
    } else {
      throw std::runtime_error(
          "Unable to get full permission for saving image");
    }
  }

  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceRightCamera)) {
    INFO("Checking if right_camera_save_folder is useable.");
    std::filesystem::path right_original_image_save_relative_folder(
        node_config_.right_camera_save_folder);
    std::filesystem::path right_fuse_image_save_relative_folder("/fuse_image");
    std::filesystem::path right_depth_image_save_relative_folder(
        "/depth_image");
    std::filesystem::path right_semantic_image_save_relative_folder(
        "/semantic_image");

    std::filesystem::path right_original_image_save_folder =
        data_record_folder_path / right_original_image_save_relative_folder;
    std::filesystem::path right_fuse_image_save_folder =
        data_record_folder_path / right_fuse_image_save_relative_folder;
    std::filesystem::path right_depth_image_save_folder =
        data_record_folder_path / right_depth_image_save_relative_folder;
    std::filesystem::path right_semantic_image_save_folder =
        data_record_folder_path / right_semantic_image_save_relative_folder;
    if (EnsureDirectoryWithFullPermissions(right_original_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(right_fuse_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(right_depth_image_save_folder) &&
        EnsureDirectoryWithFullPermissions(right_semantic_image_save_folder)) {
      annotation_status_.right_original_image_save_folder =
          right_original_image_save_folder.string();
      annotation_status_.right_fuse_image_save_folder =
          right_fuse_image_save_folder.string();
      annotation_status_.right_depth_image_save_folder =
          right_depth_image_save_folder.string();
      annotation_status_.right_semantic_image_save_folder =
          right_semantic_image_save_folder.string();
    } else {
      throw std::runtime_error(
          "Unable to get full permission for saving image");
    }
  }
}

bool GenPromptPoint::EnsureDirectoryWithFullPermissions(
    const std::filesystem::path& dir_path) {
  namespace fs = std::filesystem;
  try {
    // 检查目录是否存在
    if (fs::exists(dir_path)) {
      // 目录存在，检查是否是目录（而不是文件）
      if (!fs::is_directory(dir_path)) {
        ERROR("dir_path is exists but not a directory!");
        return false;
      }

      // 获取当前权限
      fs::perms current_perms = fs::status(dir_path).permissions();

      // 检查是否有全部权限（读、写、执行）
      bool has_full_perms = (current_perms & fs::perms::all) == fs::perms::all;

      if (!has_full_perms) {
        // 设置全部权限
        fs::permissions(dir_path, fs::perms::all,
                        fs::perm_options::add | fs::perm_options::nofollow);
        INFO("add all permissions for dir_path.");
      } else {
        INFO("dir_path is exists and have all permissions.");
      }
    } else {
      // 目录不存在，递归创建
      bool created = fs::create_directories(dir_path);
      if (created) {
        INFO("create dir: {}.", dir_path.string());
      }

      // 为新创建的目录设置全部权限
      fs::permissions(dir_path, fs::perms::all,
                      fs::perm_options::replace | fs::perm_options::nofollow);
      INFO("get all permission for dir_path.");
    }

    // 验证目录现在是否存在且有权限
    if (fs::exists(dir_path) && fs::is_directory(dir_path)) {
      // 再次检查权限
      fs::perms final_perms = fs::status(dir_path).permissions();
      bool has_full_perms = (final_perms & fs::perms::all) == fs::perms::all;

      if (has_full_perms) {
        return true;
      } else {
        WARN("dir_path: {}, permission is not full.", dir_path.string());
        return true;  // 目录已创建，即使权限不完美也算成功
      }
    }
  } catch (const fs::filesystem_error& e) {
    ERROR("Catch filesystem_error Error:{}", e.what());
    return false;
  } catch (const std::exception& e) {
    ERROR("Catch std::exception error:{}", e.what());
    return false;
  }

  return false;
}

void GenPromptPoint::InitClassMappingAndReflact() {
  // ... 其他
  class_name_to_label_["Distance Marker"] = "Distance_Marker";
  class_name_to_label_["Structure"] = "Unloading_Station";
  class_name_to_label_["flag"] = "Pole";
  class_name_to_label_["Chipping Net"] = "Chipping_Net";
  class_name_to_label_["Bucket"] = "Bucket";
  class_name_to_label_["Golf_Cart"] = "Golf_Cart";
  class_name_to_id_["Distance_Marker"] = 1;
  class_name_to_id_["Unloading_Station"] = 2;
  class_name_to_id_["Pole"] = 3;
  class_name_to_id_["Chipping_Net"] = 4;
  class_name_to_id_["Bucket"] = 5;
  class_name_to_id_["Golf_Cart"] = 6;
}

/**
 * @brief 初始化全局地图和标注框文件
 *
 */
void GenPromptPoint::InitMapAndLabel() {
  if (!node_config_.auto_annotation) {
    INFO(
        "User select do not excute auto annotation, the map and label will not "
        "be loaded.");
    return;
  }
  // 1. 加载全局 PCD 地图
  if (!node_config_.global_pc_map_addr.empty()) {
    std::filesystem::path map_path(node_config_.global_pc_map_addr);

    if (std::filesystem::exists(map_path)) {
      INFO("Loading global map from: {}", map_path.string());

      // 创建 PointXYZ 类型的指针
      pcl::PointCloud<pcl::PointXYZ>::Ptr temp_map(
          new pcl::PointCloud<pcl::PointXYZ>);

      // PCL 会自动加载 x,y,z 并忽略文件中的 rgb 字段，也不会因为找不到
      // intensity 而报错
      if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_path.string(), *temp_map) ==
          -1) {
        ERROR("Failed to read PCD file: {}", map_path.string());
      } else {
        // 设置 Frame ID (重要：用于后续 TF 变换)
        temp_map->header.frame_id = ConstValue::kInsMap;

        // 存入 annotation_status_
        annotation_status_.global_map = temp_map;
        INFO("Global map loaded successfully. Size: {} points.",
             temp_map->size());
      }
    } else {
      WARN("Global map path does not exist: {}", map_path.string());
    }
  } else {
    WARN("Global map address parameter is empty.");
  }

  // 2. 加载 XML 标注文件
  if (!node_config_.pc_annotation_file_addr.empty()) {
    std::filesystem::path xml_path(node_config_.pc_annotation_file_addr);

    if (std::filesystem::exists(xml_path)) {
      INFO("Loading annotations from: {}", xml_path.string());
      LoadLabelsFromXML(xml_path.string());
    } else {
      WARN("Annotation file path does not exist: {}", xml_path.string());
    }
  } else {
    INFO("Annotation file address parameter is empty.");
  }

  if (annotation_status_.global_map &&
      !annotation_status_.current_labels.empty()) {
    // ExtractCloudsFromBoxes();
  }
}

void GenPromptPoint::InitPublishAndSubscription() {
  CreateAccumulatePointCloudPublisher();
  CreateRawDeskewPointCloudPublisher();
  CreateInsOdomSubscription();
  CreatePointCloudSubscription();
  CreateImageSubscription();
  CreateCameraIntrinsicsSubscription();
}

void GenPromptPoint::CreateAccumulatePointCloudPublisher() {
  if (node_config_.publish_accumulated_pc) {
    INFO(
        "Configured to publish accumulated point cloud, initializing "
        "publisher.");
    accumulated_pc_pub_ =
        node_->create_publisher<sensor_msgs::msg::PointCloud2>(
            "debug/accumulated_pc", 1);
  } else {
    INFO(
        "Not configured to publish accumulated point cloud, skipping publisher "
        "initialization.");
  }
}

void GenPromptPoint::CreateRawDeskewPointCloudPublisher() {
  if (node_config_.raw_deskew_publish_debug_cloud) {
    INFO("Configured to publish raw deskewed point cloud, initializing "
         "publisher.");
    raw_deskewed_pc_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "debug/raw_deskewed_pc", 1);
  } else {
    INFO("Not configured to publish raw deskewed point cloud, skipping "
         "publisher initialization.");
  }
}

void GenPromptPoint::CreateInsOdomSubscription() {
  if (node_config_.raw_deskew_pose_source != RawDeskewPoseSource::InsOdom) {
    INFO("Raw deskew pose source is {}, skipping INS odom subscription.",
         RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source));
    return;
  }
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_pose_;
  INFO("Subscribing to deskew pose topic: {}", node_config_.raw_deskew_pose_topic);
  ins_odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      node_config_.raw_deskew_pose_topic,
      rclcpp::SensorDataQoS().keep_last(ConstValue::kLidarQueueSize * 20),
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        InsOdomCallback(msg);
      },
      options);
}

void GenPromptPoint::CreatePointCloudSubscription() {
  // 准备订阅选项，指定回调组
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_lidar_;
  // 订阅雷达话题
  if (node_config_.point_cloud_type == 0) {
    // 创建点云缓存
    INFO("Initialize queue_original");
    annotation_status_.queue_original.clear();
    annotation_status_.queue_deskewed.clear();
    // 创建订阅
    INFO("Subscribing to original point cloud topic: {}",
         node_config_.point_cloud_topic);
    pointcloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        node_config_.point_cloud_topic,
        rclcpp::SensorDataQoS().keep_last(ConstValue::kLidarQueueSize),
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          auto pcl_cloud = this->ConvertMsgToPCLPointCloud<LivoxPoint>(msg);
          if (pcl_cloud) {
            uint32_t first_raw_timestamp = 0;
            uint32_t last_raw_timestamp = 0;
            uint32_t max_raw_timestamp = 0;
            if (!pcl_cloud->points.empty()) {
              first_raw_timestamp = pcl_cloud->points.front().timestamp;
              last_raw_timestamp = pcl_cloud->points.back().timestamp;
              for (const auto& pt : pcl_cloud->points) {
                max_raw_timestamp = std::max(max_raw_timestamp, pt.timestamp);
              }
            }
            THROTTLEINFO(
                1,
                "Raw cloud input snapshot. header_ns={} frame_id={} size={} "
                "first_raw_ts={} last_raw_ts={} max_raw_ts={} deskew_mode={}",
                rclcpp::Time(msg->header.stamp).nanoseconds(),
                msg->header.frame_id, pcl_cloud->size(), first_raw_timestamp,
                last_raw_timestamp, max_raw_timestamp,
                RawDeskewModeToString(node_config_.raw_deskew_mode));
            if (node_config_.raw_deskew_mode == RawDeskewMode::Off) {
              AccumulatePointCloudBack<LivoxPoint>(
                  pcl_cloud, annotation_status_.queue_original, msg->header);
              return;
            }

            auto deskewed_cloud =
                DeskewOriginalPointCloudToPointXYZIT(pcl_cloud, msg->header);
            if (!deskewed_cloud || deskewed_cloud->empty()) {
              THROTTLEWARN(2,
                           "Skipping raw cloud at {} because deskew failed or "
                           "produced too few valid points.",
                           rclcpp::Time(msg->header.stamp).nanoseconds());
              return;
            }
            PublishOrSaveDeskewedPointCloud(deskewed_cloud, msg->header);
            AccumulatePointCloudBack<PointXYZIT>(
                deskewed_cloud, annotation_status_.queue_deskewed,
                msg->header);
          }
        },
        options);
  } else if (node_config_.point_cloud_type == 1) {
    // 创建点云缓存
    INFO("Initialize queue_undistorted");
    annotation_status_.queue_undistorted.clear();
    // 创建订阅
    INFO("Subscribing to undistorted point cloud topic: {}",
         node_config_.point_cloud_topic);
    auto custom_qos =
        rclcpp::QoS(rclcpp::KeepLast(ConstValue::kUndistortedLidarQueueSize));
    custom_qos.reliable();             // 必须匹配发布者的 RELIABLE
    custom_qos.durability_volatile();  // 匹配发布者的 VOLATILE
    pointcloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        node_config_.point_cloud_topic, custom_qos,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
          auto pcl_cloud =
              this->ConvertMsgToPCLPointCloud<UndistortedLivoxPoint>(msg);
          if (pcl_cloud) {
            AccumulatePointCloudBack<UndistortedLivoxPoint>(
                pcl_cloud, annotation_status_.queue_undistorted, msg->header);
          }
        },
        options);
  } else {
    ERROR("Invalid point_cloud_type: {}. Must be 0 or 1.",
          node_config_.point_cloud_type);
    return;
  }
}

void GenPromptPoint::CreateImageSubscription() {
  // 准备订阅选项，指定相机组
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_camera_;
  // 订阅图像话题
  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceLeftCamera)) {
    INFO("Subscribing to original left camera image topic: {}",
         node_config_.left_image_topic);
    left_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        node_config_.left_image_topic,
        rclcpp::SensorDataQoS().keep_last(ConstValue::kCameraQueueSize),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          HandleLeftImageMsg(msg);
        },
        options);
  }
  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceRightCamera)) {
    INFO("Subscribing to original right camera image topic: {}",
         node_config_.right_image_topic);
    right_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        node_config_.right_image_topic,
        rclcpp::SensorDataQoS().keep_last(ConstValue::kCameraQueueSize),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          HandleRightImageMsg(msg);
        },
        options);
  }
}

void GenPromptPoint::HandleLeftImageMsg(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  annotation_status_.left_image_rcv_count++;
  if (annotation_status_.left_image_rcv_count %
          node_config_.image_save_interval !=
      0)
    return;

  try {
    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    INFO("Recive Image timestamp: {}", timestamp);

    // ================== 1. 基础数据准备 ==================
    cv::Mat fig = ConvertRosImageToCvMat(msg);
    if (fig.empty()) return;

    if (!annotation_status_.has_left_intrinsics) {
      WARN("Waiting for intrinsics... Image dropped.");
      return;
    }

    // ================== 2. 获取并补偿点云 (时间对齐) ==================
    pcl::PointCloud<PointXYZIT>::Ptr cloud_in_lidar_raw(
        new pcl::PointCloud<PointXYZIT>);
    int64_t img_time_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    bool has_cached_cloud = false;
    int64_t latest_cloud_ts_ns = 0;
    double cache_lag_ms = 0.0;
    {
      std::lock_guard<std::mutex> lock(locks_.mutex_cache_cloud);
      has_cached_cloud = annotation_status_.latest_accumulated_cloud != nullptr;
      if (has_cached_cloud) {
        latest_cloud_ts_ns =
            annotation_status_.latest_cloud_timestamp.nanoseconds();
        cache_lag_ms = (img_time_ns - latest_cloud_ts_ns) / 1e6;
      }
    }
    THROTTLEINFO(
        1,
        "Image sync precheck. img_time_ns={} has_cached_cloud={} "
        "latest_cloud_ts_ns={} cache_lag_ms={:.3f}",
        img_time_ns, has_cached_cloud, latest_cloud_ts_ns, cache_lag_ms);
    if (!WaitForSyncedPointCloud(img_time_ns, cloud_in_lidar_raw)) return;

    rclcpp::Time img_time = msg->header.stamp;
    rclcpp::Time lidar_time(cloud_in_lidar_raw->header.stamp * 1000);

    // 计算从 Lidar(t_lidar) 到 Lidar(t_img) 的自车运动补偿矩阵
    Eigen::Matrix4f tf_lidar_compensation;
    GetTimeInterpolatedTransform(
        ConstValue::kLidarLink, img_time, ConstValue::kLidarLink, lidar_time,
        ConstValue::kInsMap, Eigen::Isometry3d::Identity(),
        tf_lidar_compensation);

    // 生成【图像时刻对齐】的 Lidar 点云
    pcl::PointCloud<PointXYZIT>::Ptr aligned_lidar_cloud(
        new pcl::PointCloud<PointXYZIT>);
    pcl::transformPointCloud(*cloud_in_lidar_raw, *aligned_lidar_cloud,
                             tf_lidar_compensation);
    aligned_lidar_cloud->header.stamp =
        img_time.nanoseconds() / 1000;  // 覆写时间戳为图像时间

    // ================== 3. 保存基础传感器数据 ==================
    if (node_config_.save_raw_image) {
      SaveOriginImageToFile(msg, fig,
                            annotation_status_.left_original_image_save_folder,
                            "", timestamp);
    }
    if (node_config_.save_synced_pcd) {
      std::string pcd_filename = annotation_status_.accumulate_pc_save_folder +
                                 "/" + std::to_string(timestamp) + ".pcd";
      if (pcl::io::savePCDFileBinary(pcd_filename, *aligned_lidar_cloud) == 0) {
        INFO("Saved Aligned PCD: {}", timestamp);
      }
    }

    // ================== 4. 自动标注流水线 (Auto Annotation) ==================
    if (!node_config_.auto_annotation) return;

    Eigen::Matrix4f tf_map_to_cam, tf_map_to_lidar;
    if (!GetMapToCameraAndLidarTf(img_time, tf_map_to_cam, tf_map_to_lidar)) {
      WARN("Skipping annotation for timestamp {} due to missing TF.",
           timestamp);
      // 虽然没 TF，但可以保底输出一个空的 json 避免文件缺失
      if (node_config_.generate_xtreme1_json)
        GenerateCameraExtrinsic(timestamp, fig.cols, fig.rows,
                                Eigen::Matrix4f::Identity());
      return;
    }

    // 将对齐后的雷达点云转到相机系 (因为点云已经是 t_img
    // 时刻，直接用静态外参即可！)
    Eigen::Matrix4f static_lidar_to_cam =
        annotation_status_.isometry_lcam2lidar.inverse().matrix().cast<float>();
    pcl::PointCloud<PointXYZIT>::Ptr cloud_in_cam(
        new pcl::PointCloud<PointXYZIT>);
    pcl::transformPointCloud(*aligned_lidar_cloud, *cloud_in_cam,
                             static_lidar_to_cam);

    // --- 4.1 深度图与语义分割 ---
    cv::Mat current_scan_depth_f32;
    if (node_config_.generate_depth_map ||
        node_config_.generate_semantic_mask) {
      if (node_config_.generate_depth_map) {
        cv::Mat fused_img, raw_depth_u16;
        GenerateDepthAndFusion(
            cloud_in_cam, fig, annotation_status_.left_intrinsic_data,
            fused_img, raw_depth_u16, current_scan_depth_f32);
        SaveOriginImageToFile(msg, raw_depth_u16,
                              annotation_status_.left_depth_image_save_folder,
                              "depth", timestamp);
        if (node_config_.enable_visual_verify) {
          SaveOriginImageToFile(msg, fused_img,
                                annotation_status_.left_fuse_image_save_folder,
                                "dep_fuse", timestamp);
        }
      }

      if (node_config_.generate_semantic_mask) {
        cv::Mat semantic_mask;
        GenerateSemanticMask(current_scan_depth_f32,
                             annotation_status_.left_intrinsic_data,
                             tf_map_to_cam, semantic_mask);
        SaveOriginImageToFile(
            msg, semantic_mask,
            annotation_status_.left_semantic_image_save_folder, "sem_mask",
            timestamp);

        if (node_config_.enable_visual_verify) {
          cv::Mat vis_img;
          GenerateSemanticVisualization(fig, semantic_mask, tf_map_to_cam,
                                        annotation_status_.left_intrinsic_data,
                                        vis_img);
          SaveOriginImageToFile(msg, vis_img,
                                annotation_status_.left_fuse_image_save_folder,
                                "sem_fuse", timestamp);
        }
      }
    }

    // --- 4.2 KITTI / NuScenes 3D Bounding Box ---
    if (node_config_.generate_kitti_label) {
      GenerateKITTILabel(
          timestamp, fig, tf_map_to_cam, annotation_status_.left_intrinsic_data,
          annotation_status_.left_label_save_folder, cloud_in_cam);

      if (node_config_.enable_visual_verify) {
        cv::Mat verify_img;
        VerifyKITTILabel(timestamp, fig, annotation_status_.left_intrinsic_data,
                         annotation_status_.left_label_save_folder, verify_img);
        SaveOriginImageToFile(msg, verify_img,
                              annotation_status_.left_fuse_image_save_folder,
                              "kitti_verify", timestamp);
        // 使用对齐后的点云进行验证
        VerifyKITTILabelInPointCloud(
            timestamp, aligned_lidar_cloud,
            annotation_status_.isometry_lcam2lidar,
            annotation_status_.left_label_save_folder,
            annotation_status_.left_fuse_image_save_folder);
      }
    }

    if (node_config_.generate_nusc_label) {
      // 【NEXT ACTION】这里就是我们接下来要写 NuScenes 格式收集逻辑的地方！
      // INFO("NuScenes data collection triggered for {}", timestamp);
    }

    // --- 4.3 其他格式导出 ---
    if (node_config_.generate_xtreme1_json) {
      GenerateCameraExtrinsic(timestamp, fig.cols, fig.rows, tf_map_to_lidar);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count() /
                1000.0;
    INFO("Handle Left Image Pipeline Done. Time taken: {:.2f} ms", ms);
  } catch (cv_bridge::Exception& e) {
    ERROR("Catch cv_bridge Error: {}", e.what());
  }
}

void GenPromptPoint::HandleRightImageMsg(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  auto start_time = std::chrono::high_resolution_clock::now();
  annotation_status_.right_image_rcv_count++;
  if (annotation_status_.right_image_rcv_count %
          node_config_.image_save_interval !=
      0) {
    return;
  }
  try {
    // 将 ROS 消息转换为 OpenCV Mat
    // 使用 BGR8 是为了兼容后续大多数标注工具
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");

    // 4. 生成文件名（建议使用时间戳或序号）
    // 使用毫秒级时间戳可以避免文件名重复
    int64_t timestamp = msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec;
    std::string filename = node_config_.right_camera_save_folder + "/img_" +
                           std::to_string(timestamp) + ".png";

    // 提取cv::Mat
    cv::Mat fig = cv_ptr->image;

    // 5. 保存图像
    // cv::imwrite 会根据后缀名自动选择压缩方式，PNG 为无损压缩
    bool success = cv::imwrite(filename, fig);
    if (success) {
      INFO("save image:{} success.", timestamp);
    } else {
      WARN("save image:{} failed.", timestamp);
    }

    // 记录结束时间并计算差值
    auto end_time = std::chrono::high_resolution_clock::now();
    // 计算耗时（单位：毫秒）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    double ms = duration.count() / 1000.0;
    INFO("Handle Right Image Done Time taken: {} ms", ms);
  } catch (cv_bridge::Exception& e) {
    ERROR("Catch cv_bridge Error: {}", e.what());
  }
}

cv::Mat GenPromptPoint::ConvertRosImageToCvMat(
    const sensor_msgs::msg::Image::SharedPtr& msg) {
  cv::Mat out_img;
  try {
    if (msg->encoding == "jpeg" || msg->encoding == "jpg" ||
        msg->encoding == "png") {
      // 压缩格式：直接解码 byte 数组
      out_img = cv::imdecode(msg->data, cv::IMREAD_COLOR);
      if (out_img.empty()) {
        ERROR("Failed to decode compressed image data (encoding: {}).",
              msg->encoding);
      }
    } else {
      // 原始格式：使用 cv_bridge 转换
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
      out_img = cv_ptr->image;
    }
  } catch (const cv_bridge::Exception& e) {
    ERROR("Catch cv_bridge Error: {}", e.what());
  } catch (const std::exception& e) {
    ERROR("Standard exception during image conversion: {}", e.what());
  }

  return out_img;
}

void GenPromptPoint::VerifyKITTILabel(int64_t timestamp,
                                      const cv::Mat& original_image,
                                      const std::vector<double>& intrinsics,
                                      const std::string& label_folder,
                                      cv::Mat& out_vis_image) {
  // 1. 初始化输出图像为原图的拷贝
  original_image.copyTo(out_vis_image);

  std::string label_file =
      label_folder + "/" + std::to_string(timestamp) + ".txt";
  std::ifstream ifs(label_file);
  if (!ifs.is_open()) {
    WARN("Verification skipped: Cannot open label file {}", label_file);
    return;
  }

  double fx = intrinsics[0], cx = intrinsics[2], fy = intrinsics[4],
         cy = intrinsics[5];
  std::string line;
  int valid_box_count = 0;

  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string type;
    float trunc, occ, alpha, left, top, right, bottom, h, w, l, x, y, z, ry;
    float rx = 0.0f, rz = 0.0f;

    // 解析 KITTI 基础参数
    ss >> type >> trunc >> occ >> alpha >> left >> top >> right >> bottom >>
        h >> w >> l >> x >> y >> z >> ry;
    // 解析 17 参数格式中的 rx 和 rz
    if (ss >> rx) {
      ss >> rz;
    }

    if (left < 0) continue;
    valid_box_count++;

    // ==================== 1. 准备局部坐标系下的 8 个顶点 ====================
    float half_l = l / 2.0f;
    float half_w = w / 2.0f;
    float h_full = h;
    // 局部坐标定义：底面中心为原点，Y轴负方向为高度方向
    std::vector<Eigen::Vector3f> corners_local = {
        {half_l, -h_full, half_w},  {half_l, -h_full, -half_w},
        {half_l, 0, -half_w},       {half_l, 0, half_w},
        {-half_l, -h_full, half_w}, {-half_l, -h_full, -half_w},
        {-half_l, 0, -half_w},      {-half_l, 0, half_w}};

    // ==================== 2. 计算 6-DoF 旋转矩阵与位姿 ====================
    tf2::Quaternion q;
    q.setRPY(rx, ry, rz);
    tf2::Matrix3x3 mat(q);
    Eigen::Matrix3f R_6dof;
    R_6dof << mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2],
        mat[2][0], mat[2][1], mat[2][2];
    Eigen::Vector3f T(x, y, z);

    // 计算几何中心 (相机系)
    Eigen::Vector3f geo_center_local(0.0f, -h / 2.0f, 0.0f);
    Eigen::Vector3f geo_center_cam = R_6dof * geo_center_local + T;

    // ==================== 3. 打印详细日志 ====================
    INFO("Box [{}] Verification Details:", type);
    INFO("  -> BottomCenter (Cam): X:{:.3f}, Y:{:.3f}, Z:{:.3f}", x, y, z);
    INFO("  -> GeoCenter    (Cam): X:{:.3f}, Y:{:.3f}, Z:{:.3f}",
         geo_center_cam.x(), geo_center_cam.y(), geo_center_cam.z());
    INFO("  -> Corners Local (Rel to BottomCenter):");
    for (size_t i = 0; i < corners_local.size(); ++i) {
      INFO("     [{}] X:{:7.3f}, Y:{:7.3f}, Z:{:7.3f}", i, corners_local[i].x(),
           corners_local[i].y(), corners_local[i].z());
    }
    INFO("  -> Corners in Camera Frame (R_6dof * Local + T):");
    for (size_t i = 0; i < corners_local.size(); ++i) {
      Eigen::Vector3f pt_cam = R_6dof * corners_local[i] + T;
      INFO("     [{}] X:{:7.3f}, Y:{:7.3f}, Z:{:7.3f}", i, pt_cam.x(),
           pt_cam.y(), pt_cam.z());
    }
    INFO("  -> local2cam Matrix:");
    INFO("     [{:7.3f}, {:7.3f}, {:7.3f}]", R_6dof(0, 0), R_6dof(0, 1),
         R_6dof(0, 2));
    INFO("     [{:7.3f}, {:7.3f}, {:7.3f}]", R_6dof(1, 0), R_6dof(1, 1),
         R_6dof(1, 2));
    INFO("     [{:7.3f}, {:7.3f}, {:7.3f}]", R_6dof(2, 0), R_6dof(2, 1),
         R_6dof(2, 2));

    // ==================== 4. 绘制 2D Box (绿色) ====================
    cv::Scalar color_2d(0, 255, 0);
    cv::rectangle(out_vis_image, cv::Point(left, top), cv::Point(right, bottom),
                  color_2d, 2);
    cv::putText(out_vis_image, type, cv::Point(left, top - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color_2d, 2);

    // ==================== 5. 绘制标准 KITTI 4-DoF 框 (橙色对照)
    // ====================
    Eigen::Matrix3f R_y;
    R_y << std::cos(ry), 0, std::sin(ry), 0, 1, 0, -std::sin(ry), 0,
        std::cos(ry);
    std::vector<cv::Point> pts_2d_kitti;
    bool valid_3d_kitti = true;
    for (const auto& pt_local : corners_local) {
      Eigen::Vector3f pt_cam = R_y * pt_local + T;
      if (pt_cam.z() < 0.1) {
        valid_3d_kitti = false;
        break;
      }
      pts_2d_kitti.push_back(
          cv::Point(std::round(fx * pt_cam.x() / pt_cam.z() + cx),
                    std::round(fy * pt_cam.y() / pt_cam.z() + cy)));
    }
    if (valid_3d_kitti && pts_2d_kitti.size() == 8) {
      cv::Scalar color_3d_kitti(0, 165, 255);
      for (int i = 0; i < 4; ++i) {
        cv::line(out_vis_image, pts_2d_kitti[i], pts_2d_kitti[(i + 1) % 4],
                 color_3d_kitti, 2);
        cv::line(out_vis_image, pts_2d_kitti[i + 4],
                 pts_2d_kitti[((i + 1) % 4) + 4], color_3d_kitti, 2);
        cv::line(out_vis_image, pts_2d_kitti[i], pts_2d_kitti[i + 4],
                 color_3d_kitti, 2);
      }
    }

    // ==================== 6. 绘制全量 6-DoF 框 (紫色) ====================
    std::vector<cv::Point> pts_2d_comp;
    bool valid_3d_comp = true;
    for (const auto& pt_local : corners_local) {
      Eigen::Vector3f pt_cam = R_6dof * pt_local + T;
      if (pt_cam.z() < 0.1) {
        valid_3d_comp = false;
        break;
      }
      pts_2d_comp.push_back(
          cv::Point(std::round(fx * pt_cam.x() / pt_cam.z() + cx),
                    std::round(fy * pt_cam.y() / pt_cam.z() + cy)));
    }

    if (valid_3d_comp && pts_2d_comp.size() == 8) {
      cv::Scalar color_3d_comp(255, 0, 255);
      for (int i = 0; i < 4; ++i) {
        cv::line(out_vis_image, pts_2d_comp[i], pts_2d_comp[(i + 1) % 4],
                 color_3d_comp, 2);
        cv::line(out_vis_image, pts_2d_comp[i + 4],
                 pts_2d_comp[((i + 1) % 4) + 4], color_3d_comp, 2);
        cv::line(out_vis_image, pts_2d_comp[i], pts_2d_comp[i + 4],
                 color_3d_comp, 2);
      }

      // === [新增特性] 逆向解析车头朝向并绘制箭头 ===
      // 1. 提取旋转矩阵第一列作为前向向量 (X轴)
      Eigen::Vector3f v_forward(R_6dof(0, 0), R_6dof(1, 0), R_6dof(2, 0));
      // 2. 计算前脸中心 (局部系下 X向为 l/2, Y向为高度中心 -h/2)
      Eigen::Vector3f front_center_cam =
          T + R_6dof * Eigen::Vector3f(l / 2.0f, -h / 2.0f, 0.0f);
      // 3. 沿 v_forward 延伸 2 米
      Eigen::Vector3f pointer_tip_cam = front_center_cam + 2.0f * v_forward;

      // 确保指示针的起点和终点都在相机前方
      if (front_center_cam.z() > 0.1 && pointer_tip_cam.z() > 0.1) {
        cv::Point p1(
            std::round(fx * front_center_cam.x() / front_center_cam.z() + cx),
            std::round(fy * front_center_cam.y() / front_center_cam.z() + cy));
        cv::Point p2(
            std::round(fx * pointer_tip_cam.x() / pointer_tip_cam.z() + cx),
            std::round(fy * pointer_tip_cam.y() / pointer_tip_cam.z() + cy));

        // 绘制亮绿色箭头，thickness=3, tipLength=0.2
        cv::arrowedLine(out_vis_image, p1, p2, cv::Scalar(0, 255, 0), 3, 8, 0,
                        0.2);
      }
    }
  }
  ifs.close();

  INFO("Verified KITTI Label (Valid Boxes: {})", valid_box_count);
}

void GenPromptPoint::SaveOriginImageToFile(
    const sensor_msgs::msg::Image::SharedPtr& msg,
    const cv::Mat& fig,           // 改为常量引用，避免拷贝
    const std::string& save_dir,  // 将路径传入，增加通用性
    const std::string& file_name_before_stamp, int64_t timestamp) {
  // 1. 确定后缀名
  std::string extension = ".png";
  // 2. 只有当原图确实是 jpeg，且当前要保存的矩阵(fig)是普通 8 位 3
  // 通道彩色图时，才允许使用 jpg 这样就能保护 16 位的深度图 (CV_16UC1) 和 1
  // 通道的语义图 (CV_8UC1) 不被 jpg 破坏
  if ((msg->encoding == "jpeg" || msg->encoding == "jpg") &&
      fig.type() == CV_8UC3) {
    extension = ".jpg";
  }

  // 2. 拼接完整路径
  // 使用 std::string::append 或直接 + 拼接
  std::string ori_filename;
  if (file_name_before_stamp.empty()) {
    ori_filename = save_dir + "/" + std::to_string(timestamp) + extension;
  } else {
    ori_filename = save_dir + "/" + file_name_before_stamp + "_" +
                   std::to_string(timestamp) + extension;
  }

  // 3. 执行写入
  try {
    if (cv::imwrite(ori_filename, fig)) {
      INFO("Saved Image ({}): {}", (extension == ".jpg" ? "JPEG" : "PNG"),
           ori_filename);
    } else {
      ERROR("Failed to write image to: {}", ori_filename);
    }
  } catch (const cv::Exception& e) {
    ERROR("OpenCV exception during imwrite: {}", e.what());
  }
}

void GenPromptPoint::CreateCameraIntrinsicsSubscription() {
  rclcpp::QoS qos_profile(1);
  qos_profile.reliability(rclcpp::ReliabilityPolicy::Reliable);
  qos_profile.durability(rclcpp::DurabilityPolicy::TransientLocal);
  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceLeftCamera)) {
    INFO("Subscribing to original left camera intrinsic topic: {}",
         node_config_.left_camera_intrinsic_topic);
    left_camera_intrinsics_sub_ =
        node_->create_subscription<std_msgs::msg::Float64MultiArray>(
            node_config_.left_camera_intrinsic_topic, qos_profile,
            std::bind(&GenPromptPoint::LeftIntrinsicsCallback, this,
                      std::placeholders::_1));
  }
  if (node_config_.image_source_type &
      (1 << ConstValue::kImageSourceRightCamera)) {
    INFO("Subscribing to original right camera intrinsic topic: {}",
         node_config_.right_camera_intrinsic_topic);
    right_camera_intrinsics_sub_ =
        node_->create_subscription<std_msgs::msg::Float64MultiArray>(
            node_config_.right_camera_intrinsic_topic, qos_profile,
            std::bind(&GenPromptPoint::RightIntrinsicsCallback, this,
                      std::placeholders::_1));
  }
}

void GenPromptPoint::InitTfListener() {
  // 1. 初始化 Buffer
  // 必须传入 clock，这样 buffer 才能知道消息的时间戳与系统时间的对应关系
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node_->get_clock());

  // 2. 初始化 Listener
  // Listener 创建后会立即开始监听 /tf 和 /tf_static 话题，并将数据填入 buffer
  // 注意：Listener 必须保持存活，不能是局部变量，否则出了作用域就会停止接收数据
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  tf_check_timer_ = node_->create_wall_timer(
      std::chrono::milliseconds(ConstValue::kCheckTfPeriodMs),
      std::bind(&GenPromptPoint::CheckTfCallback, this));
}

bool GenPromptPoint::LookUpTransform(
    const std::string& target_frame,  // 父坐标系
    const std::string& source_frame,  // 子坐标系
    geometry_msgs::msg::TransformStamped& out_transform) {
  // 1. 安全检查
  if (!tf_buffer_) {
    ERROR("TF Buffer is not initialized!");
    return false;
  }
  try {
    // 2. 查询变换
    // 使用 timeout (例如 50ms) 比 TimePointZero 更稳健，能容忍微小的网络延迟
    // 如果你坚持要非阻塞，保持 tf2::TimePointZero 即可
    out_transform = tf_buffer_->lookupTransform(target_frame, source_frame,
                                                tf2::TimePointZero,
                                                std::chrono::milliseconds(20));
    return true;  // 成功
  } catch (const tf2::TransformException& ex) {
    // 3. 异常处理
    // 使用 WARN 而不是 ERROR，因为在系统启动初期这是常见情况
    WARN("Cannot get coordinate transform {} -> {}: {}", source_frame,
         target_frame, ex.what());
    return false;  // 失败
  }
}

void GenPromptPoint::CheckTfCallback() {
  std::lock_guard<std::mutex> lock(
      locks_.mutex_use_tf);  // 确保同一时间只有一个线程在访问 TF 相关的状态

  // 计数器累加
  tf_check_count_++;
  INFO("Check Tf Use: {}s", tf_check_count_ * ConstValue::kCheckTfPeriodMs /
                                ConstValue::kSeconds2Milliseconds);

  geometry_msgs::msg::TransformStamped transform_result_test;

  // 1. 测试 左相机 -> 雷达 (静态外参)
  if (!annotation_status_.has_tf_left_camera_link_2_lidar_link) {
    if (LookUpTransform(ConstValue::kLidarLink, ConstValue::kCameraLeftLink,
                        transform_result_test)) {
      INFO("Real Transform found: {} -> {}. Switching to live TF!",
           ConstValue::kCameraLeftLink, ConstValue::kLidarLink);
      annotation_status_.isometry_lcam2lidar =
          tf2::transformToEigen(transform_result_test);
      print_transform_eigen_isometry("LeftCamera -> Lidar",
                                     annotation_status_.isometry_lcam2lidar);
      annotation_status_.has_tf_left_camera_link_2_lidar_link =
          true;  // 真正拿到了才置位
    } else {
      // 查不到就不管，因为 isometry_lcam2lidar 里已经是默认值了。使用
      // THROTTLEWARN 防刷屏。
      THROTTLEWARN(5, "Waiting for live TF {} -> {}. Using default meanwhile.",
                   ConstValue::kCameraLeftLink, ConstValue::kLidarLink);
    }
  }

  // 2. 测试 右相机 -> 雷达 (静态外参)
  if (!annotation_status_.has_tf_right_camera_link_2_lidar_link) {
    if (LookUpTransform(ConstValue::kLidarLink, ConstValue::kCameraRightLink,
                        transform_result_test)) {
      INFO("Real Transform found: {} -> {}. Switching to live TF!",
           ConstValue::kCameraRightLink, ConstValue::kLidarLink);
      annotation_status_.isometry_rcam2lidar =
          tf2::transformToEigen(transform_result_test);
      print_transform_eigen_isometry("RightCamera -> Lidar",
                                     annotation_status_.isometry_rcam2lidar);
      annotation_status_.has_tf_right_camera_link_2_lidar_link = true;
    } else {
      THROTTLEWARN(5, "Waiting for live TF {} -> {}. Using default meanwhile.",
                   ConstValue::kCameraRightLink, ConstValue::kLidarLink);
    }
  }

  // 3. 测试 雷达 -> 车体 (静态外参)
  if (!annotation_status_.has_tf_lidar_link_2_ins_link) {
    if (LookUpTransform(ConstValue::kInsLink, ConstValue::kLidarLink,
                        transform_result_test)) {
      INFO("Real Transform found: {} -> {}. Switching to live TF!",
           ConstValue::kLidarLink, ConstValue::kInsLink);
      annotation_status_.isometry_lidar2ins =
          tf2::transformToEigen(transform_result_test);
      print_transform_eigen_isometry("Lidar -> InsLink",
                                     annotation_status_.isometry_lidar2ins);
      annotation_status_.has_tf_lidar_link_2_ins_link = true;
    } else {
      THROTTLEWARN(5, "Waiting for live TF {} -> {}. Using default meanwhile.",
                   ConstValue::kLidarLink, ConstValue::kInsLink);
    }
  }

  // 4. 校验全部前置静态条件是否满足
  bool is_all_ready = true;
  if (!annotation_status_.has_tf_left_camera_link_2_lidar_link)
    is_all_ready = false;
  if (!annotation_status_.has_tf_right_camera_link_2_lidar_link)
    is_all_ready = false;
  if (!annotation_status_.has_tf_lidar_link_2_ins_link) is_all_ready = false;

  if (is_all_ready) {
    INFO(
        "All required static TF relationships are ready. Timer will be "
        "cancelled.");
    tf_check_timer_->cancel();
  }
}

void GenPromptPoint::LeftIntrinsicsCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // 1. 检查数据是否为空
  if (msg->data.empty()) {
    WARN("Received empty left camera intrinsics!");
  }
  // 2. 解析行数和列数 (从 layout 中读取)
  // MultiArray 的 layout.dim 通常按顺序存储维度：dim[0]是行，dim[1]是列
  int rows = 0;
  int cols = 0;

  if (msg->layout.dim.size() >= 2) {
    rows = msg->layout.dim[0].size;
    cols = msg->layout.dim[1].size;
  } else {
    // 如果发布者比较懒，没填 layout，但数据长度是 9，我们默认它是 3x3
    if (msg->data.size() == ConstValue::kDefaultSizeofIntrinsic) {
      rows = 3;
      cols = 3;
      WARN("Layout info missing, assuming 3x3 based on data size.");
    } else {
      ERROR("Layout empty and data size is {}. Cannot determine rows/cols!",
            msg->data.size());
      return;
    }
  }
  // 3. 将数据保存到成员变量
  // 这里直接发生内存拷贝
  annotation_status_.left_intrinsic_data = msg->data;
  annotation_status_.has_left_intrinsics = true;  // 标记数据已就绪
  INFO("Print Left camera intrinsics");
  INFO(">>> Left Camera Intrinsics Received <<<");
  print_matrix_as_numpy(msg->data, rows, cols);
}

void GenPromptPoint::InsOdomCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (!msg) {
    return;
  }

  const int64_t pose_time_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  Eigen::Isometry3d ins_pose = Eigen::Isometry3d::Identity();
  ins_pose.translation() =
      Eigen::Vector3d(msg->pose.pose.position.x, msg->pose.pose.position.y,
                      msg->pose.pose.position.z);

  Eigen::Quaterniond q(msg->pose.pose.orientation.w,
                       msg->pose.pose.orientation.x,
                       msg->pose.pose.orientation.y,
                       msg->pose.pose.orientation.z);
  if (q.norm() < 1e-6) {
    THROTTLEWARN(2, "Received invalid INS odom quaternion at {}.",
                 pose_time_ns);
    return;
  }
  q.normalize();
  ins_pose.linear() = q.toRotationMatrix();

  {
    std::lock_guard<std::mutex> lock(locks_.mutex_pose_cache);
    if (!annotation_status_.ins_pose_cache.empty() &&
        pose_time_ns < annotation_status_.ins_pose_cache.back().time_ns) {
      auto insert_pos = std::upper_bound(
          annotation_status_.ins_pose_cache.begin(),
          annotation_status_.ins_pose_cache.end(), pose_time_ns,
          [](int64_t target_time_ns,
             const AutoAnnotationStatus::TimedPoseSample& sample) {
            return target_time_ns < sample.time_ns;
          });
      annotation_status_.ins_pose_cache.insert(insert_pos,
                                               {pose_time_ns, ins_pose});
    } else {
      annotation_status_.ins_pose_cache.push_back({pose_time_ns, ins_pose});
    }
    PrunePoseCacheLocked(pose_time_ns);
  }
}

void GenPromptPoint::RightIntrinsicsCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
  // 1. 检查数据是否为空
  if (msg->data.empty()) {
    WARN("Received empty right camera intrinsics!");
  }
  // 2. 解析行数和列数 (从 layout 中读取)
  // MultiArray 的 layout.dim 通常按顺序存储维度：dim[0]是行，dim[1]是列
  int rows = 0;
  int cols = 0;

  if (msg->layout.dim.size() >= 2) {
    rows = msg->layout.dim[0].size;
    cols = msg->layout.dim[1].size;
  } else {
    // 如果发布者比较懒，没填 layout，但数据长度是 9，我们默认它是 3x3
    if (msg->data.size() == ConstValue::kDefaultSizeofIntrinsic) {
      rows = 3;
      cols = 3;
      WARN(
          "Layout info missing, assuming 3x3 based on data size (Right "
          "Camera).");
    } else {
      ERROR(
          "Layout empty and data size is {}. Cannot determine rows/cols for "
          "Right Camera!",
          msg->data.size());
      return;
    }
  }

  // 3. 将数据保存到成员变量
  // 这里直接发生内存拷贝
  // 注意：请确保 annotation_status_ 结构体里定义了 right_intrinsic_data
  annotation_status_.right_intrinsic_data = msg->data;
  annotation_status_.has_right_intrinsics = true;  // 标记数据已就绪
  INFO("Print Right camera intrinsics");
  INFO(">>> Right Camera Intrinsics Received <<<");
  print_matrix_as_numpy(msg->data, rows, cols);
}

void GenPromptPoint::ExtractCloudsFromBoxes() {
  if (!annotation_status_.global_map ||
      annotation_status_.global_map->empty()) {
    WARN("Global map is empty, cannot extract clouds.");
    return;
  }

  if (annotation_status_.current_labels.empty()) {
    WARN("No labels found, cannot extract clouds.");
    return;
  }

  INFO("Starting to extract point clouds for {} labels...",
       annotation_status_.current_labels.size());

  // 清空之前的缓存
  annotation_status_.object_clouds.clear();

  // 初始化 CropBox 滤波器
  pcl::CropBox<pcl::PointXYZ> crop_box_filter;
  crop_box_filter.setInputCloud(annotation_status_.global_map);

  for (size_t i = 0; i < annotation_status_.current_labels.size(); ++i) {
    auto& box = annotation_status_.current_labels[i];
    INFO(
        "Debug Box [{}]: tx={:.2f}, ty={:.2f}, tz={:.2f} | rx={:.2f}, "
        "ry={:.2f}, rz={:.2f} | l={:.2f}, w={:.2f}, "
        "h={:.2f}",
        box.object_type, box.tx, box.ty, box.tz, box.rx, box.ry, box.rz, box.l,
        box.w, box.h);

    // 3. 创建结果点云容器
    pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // 4. 设置 CropBox 参数
    // CropBox
    // 的原理是：定义一个未旋转的盒子(MinMax)，然后设置平移和旋转让它去匹配地图上的位置

    // 设置盒子的尺寸范围 (相对于盒子中心)
    // 注意：XML中的 l=x方向长度, w=y方向宽度, h=z方向高度
    Eigen::Vector4f min_pt(-box.l / 2.0, -box.w / 2.0, -box.h / 2.0, 1.0f);
    Eigen::Vector4f max_pt(box.l / 2.0, box.w / 2.0, box.h / 2.0, 1.0f);
    crop_box_filter.setMin(min_pt);
    crop_box_filter.setMax(max_pt);

    // 设置位姿 (Translation)
    crop_box_filter.setTranslation(Eigen::Vector3f(box.tx, box.ty, box.tz));

    // 设置旋转 (Rotation - Euler Angles in Radians)
    // CropBox 接受 Vector3f(roll, pitch, yaw)
    crop_box_filter.setRotation(Eigen::Vector3f(box.rx, box.ry, box.rz));

    // 3. 执行过滤
    crop_box_filter.filter(*object_cloud);

    // 4. 保存结果
    annotation_status_.object_clouds.push_back(object_cloud);

    // 打印日志：看看每个框里抠出了多少个点
    INFO("Label [{}] ({}) -> Extracted {} points.", i, box.object_type,
         object_cloud->size());
  }

  INFO("Extraction finished. Total object clouds: {}",
       annotation_status_.object_clouds.size());
}

void GenPromptPoint::LoadLabelsFromXML(const std::string& xml_path) {
  tinyxml2::XMLDocument doc;

  // 加载文件
  tinyxml2::XMLError error = doc.LoadFile(xml_path.c_str());
  if (error != tinyxml2::XML_SUCCESS) {
    ERROR("TinyXML2 failed to load file. ErrorID: {}", (int)error);
    return;
  }

  // 1. 获取根节点 <boost_serialization>
  tinyxml2::XMLElement* root = doc.RootElement();
  if (!root) {
    ERROR("XML format error: No root element.");
    return;
  }

  // 2. 获取 <tracklets> 节点
  tinyxml2::XMLElement* tracklets_node = root->FirstChildElement("tracklets");
  if (!tracklets_node) {
    ERROR("XML format error: No <tracklets> element.");
    return;
  }

  // 清空现有标签
  annotation_status_.current_labels.clear();

  // 3. 遍历所有的 <item> (每一个 item 代表一个物体)
  // FirstChildElement("item") 会自动跳过 <count>, <item_version> 等其他节点
  tinyxml2::XMLElement* item_node = tracklets_node->FirstChildElement("item");

  int count = 0;
  while (item_node) {
    BoundingBox box;

    // --- 解析基础信息 ---
    tinyxml2::XMLElement* type_elem =
        item_node->FirstChildElement("objectType");
    if (!type_elem || !type_elem->GetText()) {
      WARN("Skip item: missing objectType.");
      item_node = item_node->NextSiblingElement("item");
      continue;
    }

    std::string xml_type = type_elem->GetText();

    // 统一检查类别是否存在于 label 映射中
    auto it = class_name_to_label_.find(xml_type);
    if (it == class_name_to_label_.end()) {
      // 如果不在关心的类别列表中，直接跳过
      item_node = item_node->NextSiblingElement("item");
      continue;
    }

    // 命中类别，直接赋值
    box.object_type = it->second;

    // 解析尺寸 (W, H, L)
    double raw_h, raw_w, raw_l;
    item_node->FirstChildElement("h")->QueryDoubleText(&raw_h);
    item_node->FirstChildElement("w")->QueryDoubleText(&raw_w);
    item_node->FirstChildElement("l")->QueryDoubleText(&raw_l);

    // 纠正映射：
    box.h = raw_l;  // 真实的垂直高度 (Z轴)
    box.l = raw_w;  // 真实的水平长度 (X轴)
    box.w = raw_h;  // 真实的水平宽度 (Y轴)

    // --- 解析位姿信息 (Nested inside <poses> -> <item>) ---
    tinyxml2::XMLElement* poses_node = item_node->FirstChildElement("poses");
    if (poses_node) {
      // 这里的 item 是 poses 内部的 item，代表具体的某一帧位姿
      tinyxml2::XMLElement* pose_item = poses_node->FirstChildElement("item");
      if (pose_item) {
        // 解析位置 Translation
        pose_item->FirstChildElement("tx")->QueryDoubleText(&box.tx);
        pose_item->FirstChildElement("ty")->QueryDoubleText(&box.ty);
        pose_item->FirstChildElement("tz")->QueryDoubleText(&box.tz);

        // 解析旋转 Rotation
        pose_item->FirstChildElement("rx")->QueryDoubleText(&box.rx);
        pose_item->FirstChildElement("ry")->QueryDoubleText(&box.ry);
        pose_item->FirstChildElement("rz")->QueryDoubleText(&box.rz);  // Yaw

        Eigen::AngleAxisd rot_z(box.rz, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd rot_y(box.ry, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rot_x(box.rx, Eigen::Vector3d::UnitX());
        Eigen::Translation3d translation(box.tx, box.ty, box.tz);

        // 构建 Box 在 Map 下的位姿，并直接求逆存入结构体
        Eigen::Affine3d box_pose = translation * rot_z * rot_y * rot_x;
        box.map_to_local_tf = box_pose.inverse();

        // 解析状态
        pose_item->FirstChildElement("state")->QueryIntText(&box.state);
        pose_item->FirstChildElement("occlusion")->QueryIntText(&box.occlusion);
        pose_item->FirstChildElement("truncation")
            ->QueryIntText(&box.truncation);
      }
    }

    // 存入全局状态
    annotation_status_.current_labels.push_back(box);
    count++;

    // 寻找下一个兄弟节点 <item>
    item_node = item_node->NextSiblingElement("item");
  }

  INFO("Successfully loaded {} bounding boxes from XML.", count);

  // (可选) 打印前几个加载的数据进行调试
  if (!annotation_status_.current_labels.empty()) {
    const auto& first = annotation_status_.current_labels[0];
    INFO(
        "Debug First Label: Type={}, Pos=({:.2f}, {:.2f}, {:.2f}), "
        "Size=({:.2f}, {:.2f}, {:.2f}), Rotation=({:.2f}, "
        "{:.2f}, {:.2f})",
        first.object_type, first.tx, first.ty, first.tz, first.l, first.w,
        first.h, first.rx, first.ry, first.rz);
  }
}

void GenPromptPoint::GenerateSemanticMask(const cv::Mat& current_scan_depth,
                                          const std::vector<double>& intrinsics,
                                          const Eigen::Matrix4f& tf_map_to_cam,
                                          cv::Mat& out_semantic_mask) {
  if (annotation_status_.object_clouds.size() !=
      annotation_status_.current_labels.size()) {
    WARN(
        "Semantic mask generation skipped: object_clouds are not extracted. "
        "Did you load the global map?");
    // 初始化一个全白的图直接返回，防止后续逻辑报错
    out_semantic_mask =
        cv::Mat(current_scan_depth.size(), CV_8UC1, cv::Scalar(255));
    return;
  }
  auto start_time = std::chrono::high_resolution_clock::now();
  // ==================== 1. 初始化 ====================
  // 初始化输出语义图 (背景设为 255)
  out_semantic_mask =
      cv::Mat(current_scan_depth.size(), CV_8UC1, cv::Scalar(255));

  // 初始化 Label 的 Z-Buffer (用于处理 Box 之间的遮挡)
  cv::Mat label_z_buffer(current_scan_depth.size(), CV_32FC1,
                         cv::Scalar(std::numeric_limits<float>::infinity()));

  // 解析内参
  double fx = intrinsics[0];
  double cx = intrinsics[2];
  double fy = intrinsics[4];
  double cy = intrinsics[5];

  INFO("Start GenerateSemanticMask. Total Labels: {}, Depth Map Size: [{}x{}]",
       annotation_status_.current_labels.size(), current_scan_depth.cols,
       current_scan_depth.rows);
  // ==================== [需求1] 解决LiDAR稀疏透视问题 ====================
  // 策略：对深度图进行腐蚀（Min-Filter），让前景物体（小深度值）向周围的空洞（Inf）扩张。
  // 这样，如果有物体挡在前面，它的“遮挡判定区”会比实际雷达点大一圈，防止背景漏过来。

  cv::Mat safe_occlusion_depth;
  current_scan_depth.copyTo(safe_occlusion_depth);

  // Kernel Size: 5x5 通常适合 Livox 这种花瓣扫描，能填补大部分间隙
  int occlusion_kernel_size = 5;
  cv::Mat occlusion_kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(occlusion_kernel_size, occlusion_kernel_size));
  // 注意：深度越小越近，erode 是取最小值，所以是让“近处物体”变大
  cv::erode(safe_occlusion_depth, safe_occlusion_depth, occlusion_kernel);

  // ==================== 统计变量初始化 ====================
  int total_boxes_processed = 0;
  int total_boxes_drawn = 0;
  size_t total_pixels_drawn = 0;

  // ==================== 2. 遍历标注框并投影 ====================
  for (size_t i = 0; i < annotation_status_.current_labels.size(); ++i) {
    const auto& box = annotation_status_.current_labels[i];
    auto cloud_in_map = annotation_status_.object_clouds[i];

    if (!cloud_in_map || cloud_in_map->empty()) continue;

    // --- 视锥剔除 (粗略) ---
    Eigen::Vector4f center_in_map(box.tx, box.ty, box.tz, 1.0);
    Eigen::Vector4f center_in_cam = tf_map_to_cam * center_in_map;
    if (center_in_cam.z() < 0.1) {
      INFO("Label [{}] skipped: Behind camera (z={:.2f})", i,
           center_in_cam.z());
      continue;
    }

    // --- 获取类别 ID ---
    uint8_t class_id = 0;
    if (class_name_to_id_.find(box.object_type) != class_name_to_id_.end()) {
      class_id = class_name_to_id_[box.object_type];
    } else {
      continue;
    }
    total_boxes_processed++;

    // ==================== 3D 空间内缩 (Inner Core Filtering)
    // ====================
    const Eigen::Affine3d& map_to_box = box.map_to_local_tf;

    // 设定缩放比例 (0.8 表示只取中间 80% 的点)
    double shrink_ratio = 0.8;
    double half_l_threshold = (box.l * shrink_ratio) / 2.0;
    double half_w_threshold = (box.w * shrink_ratio) / 2.0;
    double half_h_threshold = (box.h * shrink_ratio) / 2.0;

    // --- 坐标转换：Map -> Camera ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_cam(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_in_map, *cloud_in_cam, tf_map_to_cam);

    int pts_total = cloud_in_cam->size();
    int pts_passed_shrink = 0;
    int pts_passed_occlusion = 0;
    int pts_drawn = 0;

    // 遍历点云
    // 注意：我们需要原始 cloud_in_map 的点来做 3D 内缩判断
    for (size_t pt_idx = 0; pt_idx < cloud_in_cam->points.size(); ++pt_idx) {
      const auto& pt_cam =
          cloud_in_cam->points[pt_idx];  // 相机系下的点 (用于投影)
      const auto& pt_map =
          cloud_in_map->points[pt_idx];  // 地图系下的点 (用于3D收缩判断)

      if (pt_cam.z < 0.5) {
        continue;
      }

      // [核心代码] 3D 内缩检查
      Eigen::Vector3d pt_vec_map(pt_map.x, pt_map.y, pt_map.z);
      Eigen::Vector3d pt_local =
          map_to_box * pt_vec_map;  // 转到 Box 局部坐标系

      // 只要有一个维度超出了收缩后的范围，就丢弃该点
      if (std::abs(pt_local.x()) > half_l_threshold ||
          std::abs(pt_local.y()) > half_w_threshold ||
          std::abs(pt_local.z()) > half_h_threshold) {
        continue;
      }
      pts_passed_shrink++;  // 统计通过收缩的点

      // --- 投影到像素 ---
      int u = std::round((fx * pt_cam.x) / pt_cam.z + cx);
      int v = std::round((fy * pt_cam.y) / pt_cam.z + cy);

      // --- 边界与遮挡检查 ---
      if (u >= 0 && u < out_semantic_mask.cols && v >= 0 &&
          v < out_semantic_mask.rows) {
        // 使用 [需求1] 生成的 safe_occlusion_depth 进行判断
        float occlusion_z = safe_occlusion_depth.at<float>(v, u);

        // 容差设为 0.2m，配合膨胀后的深度图，非常保守
        float occlusion_tolerance = 0.2f;

        // 如果该像素（或其邻域）有障碍物，且比当前 Box 点更近，则认为被遮挡
        bool is_occluded = false;
        if (!std::isinf(occlusion_z) &&
            pt_cam.z > (occlusion_z + occlusion_tolerance)) {
          is_occluded = true;
        }

        if (is_occluded) continue;

        pts_passed_occlusion++;  // 统计通过遮挡检查的点

        // Box 间遮挡判断 (Z-Buffer)
        float stored_depth = label_z_buffer.at<float>(v, u);
        if (pt_cam.z < stored_depth) {
          label_z_buffer.at<float>(v, u) = pt_cam.z;
          out_semantic_mask.at<uint8_t>(v, u) = class_id;
          pts_drawn++;  // 最终绘制的点
        }
      }
    }
    if (pts_drawn > 0) {
      total_boxes_drawn++;
      total_pixels_drawn += pts_drawn;
    } else {
      // [日志] 关键：如果一个框在视锥内，但是 0 个点被画出来，这通常意味着问题
      // 可能是收缩太狠了，或者是完全被遮挡了
      if (pts_total > 0 && pts_passed_shrink == 0) {
        INFO(
            "Label [{}] ({}) invisible: All {} points removed by 3D Shrink "
            "(Ratio 0.8).",
            i, box.object_type, pts_total);
      } else if (pts_passed_shrink > 0 && pts_passed_occlusion == 0) {
        INFO(
            "Label [{}] ({}) invisible: All points occluded by dynamic "
            "objects.",
            i, box.object_type);
      }
    }
  }

  // ==================== 2D 语义图腐蚀 (Mask Erosion) ====================
  // 策略：对生成的 Mask 再做一次“瘦身”。
  // 目的：消除光栅化带来的锯齿边缘，进一步确保 Mask 位于物体内部。

  // 原理：背景是 255 (亮)，物体是 ID (暗)。
  // 我们使用 dilate (膨胀) 操作，让亮色区域（背景）吞噬暗色区域（物体）。
  // 效果等于物体的 Mask 变小了。

  int mask_shrink_size = 3;  // 3x3 核，温和地去掉边缘的一圈像素
  cv::Mat shrink_kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(mask_shrink_size, mask_shrink_size));

  // 注意：只针对非背景区域进行操作，防止把本来就很小的远处物体完全抹掉（可选保护逻辑）
  // 这里直接全局操作，假设 SAM2 对极小物体本身就很难处理
  cv::dilate(out_semantic_mask, out_semantic_mask, shrink_kernel);

  auto end_time = std::chrono::high_resolution_clock::now();
  double cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time - start_time)
                       .count() /
                   1000.0;

  // 这是最关键的一行日志，告诉你当前帧生成了什么
  INFO(
      "Semantic Mask Generated in {:.2f}ms. Labels: Total={}, "
      "Visible/Drawn={}, Pixels={}. (ErosionK={}, "
      "ShrinkRatio={})",
      cost_ms, annotation_status_.current_labels.size(), total_boxes_drawn,
      total_pixels_drawn, occlusion_kernel_size,
      0.8);  // 直接打印参数值方便回溯
}

bool GenPromptPoint::WaitForSyncedPointCloud(
    const int64_t target_time_ns, pcl::PointCloud<PointXYZIT>::Ptr& out_cloud) {
  // 建议：缩短睡眠时间，增加重试次数。
  // 这样能更频繁地检查，第一时间截获点云，总等待时间不变。
  int max_retry_count = 100;
  int sleep_ms = 10;
  const double max_allowed_diff = ConstValue::kLidarToCameraMaxTimeDiff;

  for (int i = 0; i < max_retry_count; ++i) {
    {
      std::lock_guard<std::mutex> lock(locks_.mutex_cache_cloud);

      bool has_cloud = false;
      int64_t latest_cloud_ts_ns = 0;

      if (annotation_status_.latest_accumulated_cloud) {
        latest_cloud_ts_ns =
            annotation_status_.latest_cloud_timestamp.nanoseconds();
        has_cloud = true;
      }

      if (has_cloud) {
        double time_diff = std::abs(target_time_ns - latest_cloud_ts_ns) / 1e9;

        // 核心退出逻辑 A：如果点云时间已经追上或越过图像时间
        if (latest_cloud_ts_ns >= target_time_ns) {
          pcl::copyPointCloud(*annotation_status_.latest_accumulated_cloud,
                              *out_cloud);
          INFO("Sync Success (Crossed). Diff: {:.3f}s", time_diff);
          return true;
        }

        // 核心退出逻辑 B：点云时间还没追上图像，但误差已经达标
        if (time_diff <= max_allowed_diff) {
          pcl::copyPointCloud(*annotation_status_.latest_accumulated_cloud,
                              *out_cloud);
          INFO("Sync Success (Caught early). Diff: {:.3f}s", time_diff);
          return true;
        }
      }
    }  // 释放锁

    // 如果都不满足（点云太老，还没追上来），释放锁睡 sleep_ms 再查
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
  }

  bool has_cloud = false;
  int64_t latest_cloud_ts_ns = 0;
  uint64_t latest_cloud_header_us = 0;
  size_t latest_cloud_size = 0;
  {
    std::lock_guard<std::mutex> lock(locks_.mutex_cache_cloud);
    if (annotation_status_.latest_accumulated_cloud) {
      has_cloud = true;
      latest_cloud_ts_ns = annotation_status_.latest_cloud_timestamp.nanoseconds();
      latest_cloud_header_us =
          annotation_status_.latest_accumulated_cloud->header.stamp;
      latest_cloud_size = annotation_status_.latest_accumulated_cloud->size();
    }
  }
  const double diff_ms =
      has_cloud ? std::abs(target_time_ns - latest_cloud_ts_ns) / 1e6 : -1.0;
  WARN(
      "Time sync timeout. target_time_ns={} has_cloud={} latest_cloud_ts_ns={} "
      "diff_ms={:.3f} retry_count={} latest_cloud_header_us={} "
      "latest_cloud_size={}",
      target_time_ns, has_cloud, latest_cloud_ts_ns, diff_ms, max_retry_count,
      latest_cloud_header_us, latest_cloud_size);
  return false;
}

void GenPromptPoint::GenerateSemanticVisualization(
    const cv::Mat& original_image, const cv::Mat& semantic_mask,
    const Eigen::Matrix4f& tf_map_to_cam, const std::vector<double>& intrinsics,
    cv::Mat& out_vis_image) {
  // 1. 初始化可视化图像为原图的一份拷贝
  original_image.copyTo(out_vis_image);

  // 2. 建立颜色映射表 (BGR 格式)
  // 可以根据 class_name_to_id_ 的定义来调整颜色
  std::map<uint8_t, cv::Scalar> color_map = {
      {1, cv::Scalar(0, 0, 255)},  // Sand: 红色
      {2, cv::Scalar(0, 255, 0)},  // Distance Marker: 绿色
      {3, cv::Scalar(255, 0, 0)}   // Structure: 蓝色
  };

  // 3. 为语义 Mask 赋予颜色，并与原图混合 (Alpha Blending)
  cv::Mat colored_mask = cv::Mat::zeros(original_image.size(), CV_8UC3);
  cv::Mat mask_valid = cv::Mat::zeros(original_image.size(), CV_8UC1);

  for (int y = 0; y < semantic_mask.rows; ++y) {
    for (int x = 0; x < semantic_mask.cols; ++x) {
      uint8_t class_id = semantic_mask.at<uint8_t>(y, x);
      if (class_id != 255 && color_map.find(class_id) != color_map.end()) {
        colored_mask.at<cv::Vec3b>(y, x) =
            cv::Vec3b(color_map[class_id][0], color_map[class_id][1],
                      color_map[class_id][2]);
        mask_valid.at<uint8_t>(y, x) = 255;  // 标记哪些像素被涂色了
      }
    }
  }

  // 叠加带颜色的 Mask (权重: 原图 0.6, Mask 0.6)
  cv::Mat blended;
  cv::addWeighted(out_vis_image, 0.6, colored_mask, 0.6, 0.0, blended);
  // 只在有 Mask 的地方替换为融合图像，保持背景原样
  blended.copyTo(out_vis_image, mask_valid);

  // 4. 解析内参用于 Box 投影
  double fx = intrinsics[0], cx = intrinsics[2];
  double fy = intrinsics[4], cy = intrinsics[5];

  // 5. 遍历并绘制 3D Bounding Box
  for (const auto& box : annotation_status_.current_labels) {
    // 获取 3D Box 在局部坐标系下的 8 个顶点
    // double hl = box.w / 2.0;
    // double hw = box.h / 2.0;
    // double hh = box.l / 2.0;

    double hl = box.l / 2.0;
    double hw = box.w / 2.0;
    double hh = box.h / 2.0;

    std::vector<Eigen::Vector3d> corners_local = {
        {hl, hw, hh},    {hl, -hw, hh},
        {-hl, -hw, hh},  {-hl, hw, hh},  // 顶部 4 个点 (0,1,2,3)
        {hl, hw, -hh},   {hl, -hw, -hh},
        {-hl, -hw, -hh}, {-hl, hw, -hh}  // 底部 4 个点 (4,5,6,7)
    };

    Eigen::Affine3d local_to_map =
        box.map_to_local_tf.inverse();  // 从局部转回全局 Map
    std::vector<cv::Point> pts_2d;
    bool valid_box = true;

    Eigen::Vector4f center_cam =
        tf_map_to_cam * Eigen::Vector4f(box.tx, box.ty, box.tz, 1.0f);
    INFO("Box [{}] Distance to Camera Z: {:.2f} meters", box.object_type,
         center_cam.z());

    // 投影 8 个顶点
    for (const auto& pt_local : corners_local) {
      Eigen::Vector3d pt_map = local_to_map * pt_local;  // 转到 Map 坐标系
      Eigen::Vector4f pt_map_h(pt_map.x(), pt_map.y(), pt_map.z(), 1.0f);
      Eigen::Vector4f pt_cam = tf_map_to_cam * pt_map_h;  // 转到 Camera 坐标系

      // 如果有任何一个顶点在相机背后，为了防止投影畸变，直接跳过这个框的绘制
      if (pt_cam.z() < 0.1) {
        valid_box = false;
        break;
      }

      int u = std::round((fx * pt_cam.x()) / pt_cam.z() + cx);
      int v = std::round((fy * pt_cam.y()) / pt_cam.z() + cy);
      pts_2d.push_back(cv::Point(u, v));
    }

    if (!valid_box || pts_2d.size() != 8) continue;

    // 画线段连接 8 个顶点构成 3D 框
    cv::Scalar box_color(0, 255, 255);  // 黄色框
    int thickness = 2;
    for (int i = 0; i < 4; ++i) {
      // 顶面
      cv::line(out_vis_image, pts_2d[i], pts_2d[(i + 1) % 4], box_color,
               thickness);
      // 底面
      cv::line(out_vis_image, pts_2d[i + 4], pts_2d[((i + 1) % 4) + 4],
               box_color, thickness);
      // 侧面的柱子
      cv::line(out_vis_image, pts_2d[i], pts_2d[i + 4], box_color, thickness);
    }
  }
}

void GenPromptPoint::GenerateDepthAndFusion(
    const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_cam,
    const cv::Mat& original_image, const std::vector<double>& intrinsic_data,
    cv::Mat& out_fused_image, cv::Mat& out_raw_depth_u16,
    cv::Mat& out_raw_depth_f32) {
  // 1. 初始化深度图 (单通道 float32，初始值为无穷大)
  cv::Mat depth_map(original_image.rows, original_image.cols, CV_32FC1,
                    cv::Scalar(std::numeric_limits<float>::infinity()));

  // 2. 解析内参
  double fx = intrinsic_data[0], cx = intrinsic_data[2];
  double fy = intrinsic_data[4], cy = intrinsic_data[5];

  // 3. 投影并填充 Z-Buffer
  for (const auto& pt : cloud_in_cam->points) {
    if (pt.z < 0.5 || pt.z > 100.0) continue;

    int u = std::round((fx * pt.x) / pt.z + cx);
    int v = std::round((fy * pt.y) / pt.z + cy);

    if (u >= 0 && u < depth_map.cols && v >= 0 && v < depth_map.rows) {
      float current_depth = depth_map.at<float>(v, u);
      if (pt.z < current_depth) {
        depth_map.at<float>(v, u) = pt.z;
      }
    }
  }

  // 4.1 创建有效区域掩码
  cv::Mat valid_mask;
  cv::inRange(depth_map, 0.5, 50.0, valid_mask);

  // 4.2 生成伪彩色深度图
  cv::Mat visual_depth;
  {
    cv::Mat clipped_depth;
    depth_map.copyTo(clipped_depth);
    double max_visual_dist = 10.0;
    cv::threshold(clipped_depth, clipped_depth, max_visual_dist,
                  max_visual_dist, cv::THRESH_TRUNC);
    cv::Mat norm_depth;
    cv::normalize(clipped_depth, norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(norm_depth, visual_depth, cv::COLORMAP_JET);

    cv::Mat black_bg = cv::Mat::zeros(visual_depth.size(), visual_depth.type());
    visual_depth.copyTo(black_bg, valid_mask);
    visual_depth = black_bg;
  }

  // === [核心修改：压暗背景与点云增强] ===

  // A. 压暗背景：将原图亮度降低为 40%
  cv::Mat dimmed_bg;
  original_image.convertTo(dimmed_bg, -1, 0.4, 0);

  // B. 点云“加粗”：使用 3x3 椭圆核进行膨胀，让点云更醒目
  cv::Mat thick_visual_depth;
  cv::Mat thick_valid_mask;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::dilate(visual_depth, thick_visual_depth, kernel);
  cv::dilate(valid_mask, thick_valid_mask, kernel);

  // C. 图像融合：以压暗的图为底，将加粗的点云贴上去
  dimmed_bg.copyTo(out_fused_image);
  thick_visual_depth.copyTo(out_fused_image, thick_valid_mask);

  // 保持原始深度的输出 (用于后续语义遮挡计算)
  depth_map.copyTo(out_raw_depth_f32);

  // 5. 生成 16位 原始深度图逻辑
  cv::patchNaNs(depth_map, 0.0);
  cv::threshold(depth_map, depth_map, 65.0, 0.0, cv::THRESH_TOZERO_INV);
  depth_map.convertTo(out_raw_depth_u16, CV_16UC1, 1000.0);
}

bool GenPromptPoint::GetTimeInterpolatedTransform(
    const std::string& target_frame, const rclcpp::Time& target_time,
    const std::string& source_frame, const rclcpp::Time& source_time,
    const std::string& fixed_frame, const Eigen::Isometry3d& fallback_extrinsic,
    Eigen::Matrix4f& out_tf_mat) {
  out_tf_mat = Eigen::Matrix4f::Identity();
  if (!tf_buffer_) {
    WARN("TF Buffer not initialized. Using fallback extrinsic.");
    out_tf_mat = fallback_extrinsic.matrix().cast<float>();
    return false;
  }

  try {
    // 请求高级 TF 变换 (Time-travel Transform)
    // 它的含义是：将 source_time 时刻的 source_frame 坐标系，
    // 以 fixed_frame 为静止参考，转换到 target_time 时刻的 target_frame
    // 坐标系。
    geometry_msgs::msg::TransformStamped transform_stamped =
        tf_buffer_->lookupTransform(target_frame, target_time, source_frame,
                                    source_time, fixed_frame,
                                    rclcpp::Duration::from_seconds(0.05));

    Eigen::Affine3d affine = tf2::transformToEigen(transform_stamped);
    out_tf_mat = affine.matrix().cast<float>();

    // double diff_ms = (target_time.nanoseconds() - source_time.nanoseconds())
    // / 1e6; INFO("Motion compensation applied. {} -> {} Time diff: {:.2f} ms",
    // source_frame, target_frame, diff_ms);
    return true;
  } catch (const tf2::TransformException& ex) {
    // 如果查不到插值（比如刚好缺少里程计 TF），回退到静态外参保底
    WARN("Time interpolation TF failed: {}. Fallback to static extrinsic.",
         ex.what());
    out_tf_mat = fallback_extrinsic.matrix().cast<float>();
    return false;
  }
}

pcl::PointCloud<PointXYZIT>::Ptr
GenPromptPoint::DeskewOriginalPointCloudToPointXYZIT(
    const pcl::PointCloud<LivoxPoint>::Ptr& input_cloud,
    const std_msgs::msg::Header& header) {
  auto deskew_begin_time = std::chrono::high_resolution_clock::now();
  if (!input_cloud || input_cloud->empty()) {
    return nullptr;
  }

  pcl::PointCloud<PointXYZIT>::Ptr output_cloud(
      new pcl::PointCloud<PointXYZIT>());
  output_cloud->reserve(input_cloud->size());

  uint32_t max_raw_timestamp = 0;
  for (const auto& pt : input_cloud->points) {
    max_raw_timestamp = std::max(max_raw_timestamp, pt.timestamp);
  }

  if (max_raw_timestamp == 0) {
    THROTTLEWARN(5,
                 "Raw deskew is enabled but all Livox timestamps are zero at "
                 "{}. Forwarding frame without deskew.",
                 rclcpp::Time(header.stamp).nanoseconds());
    for (const auto& pt : input_cloud->points) {
      PointXYZIT out_pt;
      out_pt.x = pt.x;
      out_pt.y = pt.y;
      out_pt.z = pt.z;
      out_pt.intensity = pt.intensity;
      out_pt.timestamp = 0.0f;
      output_cloud->push_back(out_pt);
    }
    output_cloud->header.stamp =
        rclcpp::Time(header.stamp).nanoseconds() / 1000;
    output_cloud->header.frame_id = header.frame_id;
    output_cloud->width = static_cast<uint32_t>(output_cloud->size());
    output_cloud->height = 1;
    output_cloud->is_dense = false;
    return output_cloud;
  }

  const int64_t header_stamp_ns = rclcpp::Time(header.stamp).nanoseconds();
  int64_t scan_start_time_ns = header_stamp_ns;
  int64_t scan_end_time_ns = header_stamp_ns;
  const int64_t max_offset_ns = static_cast<int64_t>(std::llround(
      static_cast<double>(max_raw_timestamp) *
      node_config_.raw_deskew_timestamp_unit_sec * 1e9));
  if (node_config_.raw_deskew_timestamp_polarity ==
      RawDeskewTimestampPolarity::FromScanStart) {
    scan_end_time_ns = header_stamp_ns + max_offset_ns;
  } else {
    scan_start_time_ns = header_stamp_ns - max_offset_ns;
  }
  const int64_t reference_time_ns = ResolveDeskewReferenceTimeNs(
      header_stamp_ns, scan_start_time_ns, scan_end_time_ns);
  THROTTLEINFO(
      1,
      "Deskew window snapshot. mode={} header_ns={} frame_id={} max_raw_ts={} "
      "unit_sec={:.9f} polarity={} ref_mode={} scan_start_ns={} "
      "scan_end_ns={} reference_time_ns={}",
      RawDeskewModeToString(node_config_.raw_deskew_mode), header_stamp_ns,
      header.frame_id, max_raw_timestamp, node_config_.raw_deskew_timestamp_unit_sec,
      RawDeskewTimestampPolarityToString(
          node_config_.raw_deskew_timestamp_polarity),
      RawDeskewReferenceTimeToString(
          node_config_.raw_deskew_reference_time),
      scan_start_time_ns, scan_end_time_ns, reference_time_ns);

  if (scan_start_time_ns == scan_end_time_ns) {
    for (const auto& pt : input_cloud->points) {
      PointXYZIT out_pt;
      out_pt.x = pt.x;
      out_pt.y = pt.y;
      out_pt.z = pt.z;
      out_pt.intensity = pt.intensity;
      out_pt.timestamp = 0.0f;
      output_cloud->push_back(out_pt);
    }
    output_cloud->header.stamp =
        rclcpp::Time(header.stamp).nanoseconds() / 1000;
    output_cloud->header.frame_id = header.frame_id;
    output_cloud->width = static_cast<uint32_t>(output_cloud->size());
    output_cloud->height = 1;
    output_cloud->is_dense = false;
    return output_cloud;
  }

  const std::string fixed_frame = node_config_.raw_deskew_fixed_frame;
  const std::string sensor_frame = header.frame_id;
  const std::string ins_frame = ConstValue::kInsLink;
  Eigen::Isometry3d reference_pose = Eigen::Isometry3d::Identity();
  if (!LookupDeskewPoseAtTime(fixed_frame, sensor_frame, ins_frame,
                              rclcpp::Time(reference_time_ns),
                              reference_pose)) {
    WARN(
        "Deskew skipped because reference pose lookup failed. mode={} "
        "header_ns={} reference_time_ns={} scan_start_ns={} scan_end_ns={}",
        RawDeskewModeToString(node_config_.raw_deskew_mode), header_stamp_ns,
        reference_time_ns, scan_start_time_ns, scan_end_time_ns);
    return nullptr;
  }

  size_t valid_points = 0;
  DeskewPoseLookupStats pose_stats;
  bool deskew_success = false;
  const char* effective_path = RawDeskewModeToString(node_config_.raw_deskew_mode);

  if (node_config_.raw_deskew_mode == RawDeskewMode::Fast) {
    deskew_success =
        RunFastDeskew(input_cloud, header, max_raw_timestamp, reference_pose,
                      scan_start_time_ns, scan_end_time_ns, output_cloud,
                      valid_points, &pose_stats);
  } else {
    const size_t sample_count = static_cast<size_t>(
        std::max<int64_t>(2, node_config_.raw_deskew_bucket_count_accurate));
    std::vector<DeskewPoseSample> trajectory_samples;
    int64_t sampled_reference_time_ns = 0;
    int64_t sampled_scan_start_ns = 0;
    int64_t sampled_scan_end_ns = 0;
    if (SampleDeskewTrajectory(header, max_raw_timestamp, sample_count,
                               trajectory_samples, sampled_reference_time_ns,
                               sampled_scan_start_ns, sampled_scan_end_ns,
                               &pose_stats)) {
      Eigen::Isometry3d sampled_reference_pose = reference_pose;
      if (InterpolatePoseAtTime(trajectory_samples, sampled_reference_time_ns,
                                sampled_reference_pose)) {
        pose_stats.cache_hit_count = trajectory_samples.size();
        for (const auto& pt : input_cloud->points) {
          const int64_t point_time_ns =
              ComputePointAbsoluteTimeNs(pt.timestamp, header, max_raw_timestamp);
          Eigen::Isometry3d point_pose = Eigen::Isometry3d::Identity();
          if (!InterpolatePoseAtTime(trajectory_samples, point_time_ns,
                                     point_pose)) {
            continue;
          }

          Eigen::Vector3d point_local(pt.x, pt.y, pt.z);
          Eigen::Vector3d point_ref =
              sampled_reference_pose.inverse() * point_pose * point_local;
          PointXYZIT out_pt;
          out_pt.x = static_cast<float>(point_ref.x());
          out_pt.y = static_cast<float>(point_ref.y());
          out_pt.z = static_cast<float>(point_ref.z());
          out_pt.intensity = pt.intensity;
          out_pt.timestamp = 0.0f;
          output_cloud->push_back(out_pt);
          ++valid_points;
        }
        deskew_success = true;
      } else {
        WARN(
            "Deskew skipped because reference pose interpolation failed. "
            "header_ns={} reference_time_ns={} sample_count={}",
            header_stamp_ns, sampled_reference_time_ns,
            trajectory_samples.size());
      }
    } else {
      WARN(
          "Deskew trajectory sampling failed. header_ns={} sample_count={} "
          "scan_start_ns={} scan_end_ns={} reference_time_ns={} "
          "pose_source={} missing_pose_policy={} failure_reason={} "
          "cache_oldest_ns={} cache_newest_ns={} cache_wait_ms={}",
          header_stamp_ns, sample_count, sampled_scan_start_ns,
          sampled_scan_end_ns, sampled_reference_time_ns,
          RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source),
          RawDeskewMissingPosePolicyToString(
              node_config_.raw_deskew_missing_pose_policy),
          pose_stats.failure_reason, pose_stats.cache_oldest_time_ns,
          pose_stats.cache_newest_time_ns, pose_stats.cache_wait_ms);
    }

    if (!deskew_success &&
        node_config_.raw_deskew_missing_pose_policy ==
            RawDeskewMissingPosePolicy::FallbackFast) {
      pose_stats.fallback_fast = true;
      effective_path = "fallback_fast";
      THROTTLEWARN(
          1,
          "Accurate deskew falling back to fast. header_ns={} pose_source={} "
          "cache_sample_count={} cache_hit_count={} failure_reason={} "
          "cache_oldest_ns={} cache_newest_ns={} cache_wait_ms={}",
          header_stamp_ns,
          RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source),
          pose_stats.cache_sample_count, pose_stats.cache_hit_count,
          pose_stats.failure_reason, pose_stats.cache_oldest_time_ns,
          pose_stats.cache_newest_time_ns, pose_stats.cache_wait_ms);
      deskew_success =
          RunFastDeskew(input_cloud, header, max_raw_timestamp, reference_pose,
                        scan_start_time_ns, scan_end_time_ns, output_cloud,
                        valid_points, &pose_stats);
    }
  }

  if (!deskew_success) {
    return nullptr;
  }

  const double valid_point_ratio =
      static_cast<double>(valid_points) /
      static_cast<double>(input_cloud->size());
  if (valid_point_ratio < node_config_.raw_deskew_min_valid_ratio) {
    WARN(
        "Deskew skipped because valid point ratio is too low. header_ns={} "
        "valid_points={} total_points={} ratio={:.3f} threshold={:.3f}",
        header_stamp_ns, valid_points, input_cloud->size(), valid_point_ratio,
        node_config_.raw_deskew_min_valid_ratio);
    return nullptr;
  }

  output_cloud->header.stamp = header_stamp_ns / 1000;
  output_cloud->header.frame_id = header.frame_id;
  output_cloud->width = static_cast<uint32_t>(output_cloud->size());
  output_cloud->height = 1;
  output_cloud->is_dense = false;
  const double deskew_cost_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now() - deskew_begin_time)
          .count() /
      1000.0;
  THROTTLEINFO(
      1,
      "Deskew success snapshot. header_ns={} mode={} effective_path={} "
      "input_points={} valid_points={} valid_ratio={:.3f} "
      "pose_source={} cache_sample_count={} cache_hit_count={} "
      "cache_snapshot_size={} cache_wait_ms={} failure_reason={} "
      "fallback_fast={} cost_ms={:.2f} output_header_us={}",
      header_stamp_ns, RawDeskewModeToString(node_config_.raw_deskew_mode),
      effective_path, input_cloud->size(), valid_points, valid_point_ratio,
      RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source),
      pose_stats.cache_sample_count, pose_stats.cache_hit_count,
      pose_stats.cache_snapshot_size, pose_stats.cache_wait_ms,
      pose_stats.failure_reason, pose_stats.fallback_fast, deskew_cost_ms,
      output_cloud->header.stamp);
  return output_cloud;
}

bool GenPromptPoint::LookupDeskewPoseAtTime(const std::string& fixed_frame,
                                            const std::string& sensor_frame,
                                            const std::string& ins_frame,
                                            const rclcpp::Time& query_time,
                                            Eigen::Isometry3d& out_pose) {
  if (node_config_.raw_deskew_pose_source == RawDeskewPoseSource::InsOdom &&
      LookupDeskewPoseFromCacheAtTime(query_time.nanoseconds(), out_pose)) {
    return true;
  }
  if (!tf_buffer_) {
    return false;
  }

  try {
    geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_->lookupTransform(
        fixed_frame, sensor_frame, query_time,
        rclcpp::Duration::from_seconds(0.02));
    out_pose = tf2::transformToEigen(tf_msg);
    return true;
  } catch (const tf2::TransformException&) {
    try {
      geometry_msgs::msg::TransformStamped tf_ins_msg =
          tf_buffer_->lookupTransform(fixed_frame, ins_frame, query_time,
                                      rclcpp::Duration::from_seconds(0.05));
      Eigen::Isometry3d iso_ins_to_map = tf2::transformToEigen(tf_ins_msg);
      out_pose = iso_ins_to_map * annotation_status_.isometry_lidar2ins;
      THROTTLEWARN(5,
                   "Deskew direct TF {}->{} failed at {}. Using "
                   "Map->InsLink(dynamic) * Lidar->InsLink(static/default).",
                   sensor_frame, fixed_frame, query_time.nanoseconds());
      return true;
    } catch (const tf2::TransformException& ex2) {
      THROTTLEWARN(5, "Deskew pose lookup failed at {}: {}",
                   query_time.nanoseconds(), ex2.what());
      return false;
    }
  }
}

bool GenPromptPoint::LookupDeskewPoseFromCacheAtTime(
    int64_t query_time_ns, Eigen::Isometry3d& out_pose) {
  std::vector<DeskewPoseSample> samples;
  samples.reserve(2);
  {
    std::lock_guard<std::mutex> lock(locks_.mutex_pose_cache);
    if (annotation_status_.ins_pose_cache.empty()) {
      return false;
    }
    auto upper = std::lower_bound(
        annotation_status_.ins_pose_cache.begin(),
        annotation_status_.ins_pose_cache.end(), query_time_ns,
        [](const AutoAnnotationStatus::TimedPoseSample& sample,
           int64_t target_time_ns) { return sample.time_ns < target_time_ns; });
    if (upper == annotation_status_.ins_pose_cache.begin()) {
      if (upper == annotation_status_.ins_pose_cache.end() ||
          upper->time_ns != query_time_ns) {
        return false;
      }
      samples.push_back({upper->time_ns, upper->pose});
    } else if (upper == annotation_status_.ins_pose_cache.end()) {
      const auto& last = annotation_status_.ins_pose_cache.back();
      if (last.time_ns != query_time_ns) {
        return false;
      }
      samples.push_back({last.time_ns, last.pose});
    } else {
      if (upper->time_ns == query_time_ns) {
        samples.push_back({upper->time_ns, upper->pose});
      } else {
        const auto& left = *(upper - 1);
        const auto& right = *upper;
        if (query_time_ns < left.time_ns || query_time_ns > right.time_ns) {
          return false;
        }
        samples.push_back({left.time_ns, left.pose});
        samples.push_back({right.time_ns, right.pose});
      }
    }
  }

  Eigen::Isometry3d ins_pose = Eigen::Isometry3d::Identity();
  if (!InterpolatePoseAtTime(samples, query_time_ns, ins_pose)) {
    return false;
  }
  out_pose = ins_pose * annotation_status_.isometry_lidar2ins;
  return true;
}

bool GenPromptPoint::GetPoseCacheCoverageSnapshot(
    int64_t& out_oldest_time_ns, int64_t& out_newest_time_ns) const {
  std::lock_guard<std::mutex> lock(locks_.mutex_pose_cache);
  if (annotation_status_.ins_pose_cache.empty()) {
    out_oldest_time_ns = 0;
    out_newest_time_ns = 0;
    return false;
  }
  out_oldest_time_ns = annotation_status_.ins_pose_cache.front().time_ns;
  out_newest_time_ns = annotation_status_.ins_pose_cache.back().time_ns;
  return true;
}

bool GenPromptPoint::WaitForPoseCacheCoverage(
    int64_t scan_start_time_ns, int64_t scan_end_time_ns, int64_t timeout_ms,
    int64_t poll_interval_ms, int64_t& out_waited_ms,
    int64_t& out_oldest_time_ns, int64_t& out_newest_time_ns) const {
  out_waited_ms = 0;
  while (out_waited_ms <= timeout_ms) {
    if (!GetPoseCacheCoverageSnapshot(out_oldest_time_ns, out_newest_time_ns)) {
      if (out_waited_ms == timeout_ms) {
        return false;
      }
    } else if (out_oldest_time_ns <= scan_start_time_ns &&
               out_newest_time_ns >= scan_end_time_ns) {
      return true;
    } else if (out_oldest_time_ns > scan_start_time_ns) {
      return false;
    }

    if (out_waited_ms == timeout_ms) {
      return false;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(poll_interval_ms));
    out_waited_ms = std::min(timeout_ms, out_waited_ms + poll_interval_ms);
  }
  return false;
}

bool GenPromptPoint::CopyPoseCacheWindowSnapshot(
    int64_t scan_start_time_ns, int64_t scan_end_time_ns,
    std::vector<DeskewPoseSample>& out_snapshot,
    int64_t& out_oldest_time_ns, int64_t& out_newest_time_ns) const {
  out_snapshot.clear();
  std::lock_guard<std::mutex> lock(locks_.mutex_pose_cache);
  if (annotation_status_.ins_pose_cache.empty()) {
    out_oldest_time_ns = 0;
    out_newest_time_ns = 0;
    return false;
  }

  out_oldest_time_ns = annotation_status_.ins_pose_cache.front().time_ns;
  out_newest_time_ns = annotation_status_.ins_pose_cache.back().time_ns;
  if (out_oldest_time_ns > scan_start_time_ns ||
      out_newest_time_ns < scan_end_time_ns) {
    return false;
  }

  auto begin = std::lower_bound(
      annotation_status_.ins_pose_cache.begin(),
      annotation_status_.ins_pose_cache.end(), scan_start_time_ns,
      [](const AutoAnnotationStatus::TimedPoseSample& sample,
         int64_t target_time_ns) { return sample.time_ns < target_time_ns; });
  if (begin != annotation_status_.ins_pose_cache.begin()) {
    --begin;
  }

  auto end = std::lower_bound(
      annotation_status_.ins_pose_cache.begin(),
      annotation_status_.ins_pose_cache.end(), scan_end_time_ns,
      [](const AutoAnnotationStatus::TimedPoseSample& sample,
         int64_t target_time_ns) { return sample.time_ns < target_time_ns; });
  if (end == annotation_status_.ins_pose_cache.end()) {
    return false;
  }
  ++end;

  out_snapshot.reserve(static_cast<size_t>(std::distance(begin, end)));
  for (auto it = begin; it != end; ++it) {
    out_snapshot.push_back({it->time_ns, it->pose});
  }
  return !out_snapshot.empty();
}

void GenPromptPoint::PrunePoseCacheLocked(int64_t latest_time_ns) {
  const int64_t cache_duration_ns = static_cast<int64_t>(
      std::llround(node_config_.raw_deskew_pose_cache_duration_sec * 1e9));
  const int64_t keep_after_ns = latest_time_ns - cache_duration_ns;
  while (!annotation_status_.ins_pose_cache.empty() &&
         annotation_status_.ins_pose_cache.front().time_ns < keep_after_ns) {
    annotation_status_.ins_pose_cache.pop_front();
  }
}

bool GenPromptPoint::UsePoseCacheForAccurateDeskew() const {
  return node_config_.raw_deskew_pose_source == RawDeskewPoseSource::InsOdom;
}

bool GenPromptPoint::RunFastDeskew(
    const pcl::PointCloud<LivoxPoint>::Ptr& input_cloud,
    const std_msgs::msg::Header& header, uint32_t max_raw_timestamp,
    const Eigen::Isometry3d& reference_pose, int64_t scan_start_time_ns,
    int64_t scan_end_time_ns, pcl::PointCloud<PointXYZIT>::Ptr& output_cloud,
    size_t& out_valid_points, DeskewPoseLookupStats* out_stats) {
  if (!input_cloud || !output_cloud) {
    return false;
  }
  const std::string fixed_frame = node_config_.raw_deskew_fixed_frame;
  const std::string sensor_frame = header.frame_id;
  const std::string ins_frame = ConstValue::kInsLink;
  const size_t bucket_count = static_cast<size_t>(
      std::max<int64_t>(1, node_config_.raw_deskew_bucket_count_fast));
  std::vector<DeskewPoseSample> bucket_samples;
  std::vector<bool> bucket_valid(bucket_count, false);
  bucket_samples.reserve(bucket_count);
  size_t success_samples = 0;

  for (size_t i = 0; i < bucket_count; ++i) {
    double alpha =
        (static_cast<double>(i) + 0.5) / static_cast<double>(bucket_count);
    int64_t sample_time_ns =
        scan_start_time_ns +
        static_cast<int64_t>((scan_end_time_ns - scan_start_time_ns) * alpha);
    Eigen::Isometry3d sample_pose = Eigen::Isometry3d::Identity();
    if (LookupDeskewPoseAtTime(fixed_frame, sensor_frame, ins_frame,
                               rclcpp::Time(sample_time_ns), sample_pose)) {
      bucket_samples.push_back({sample_time_ns, sample_pose});
      bucket_valid[i] = true;
      ++success_samples;
    } else {
      bucket_samples.push_back(
          {sample_time_ns, Eigen::Isometry3d::Identity()});
    }
  }

  if (out_stats) {
    out_stats->cache_sample_count = bucket_count;
    out_stats->cache_hit_count = success_samples;
  }

  const double sample_valid_ratio =
      bucket_count == 0
          ? 0.0
          : static_cast<double>(success_samples) /
                static_cast<double>(bucket_count);
  if (sample_valid_ratio < node_config_.raw_deskew_min_valid_ratio) {
    WARN(
        "Deskew skipped because bucket pose valid ratio is too low. "
        "header_ns={} ratio={:.3f} threshold={:.3f} bucket_count={} "
        "scan_start_ns={} scan_end_ns={} reference_time_ns={}",
        rclcpp::Time(header.stamp).nanoseconds(), sample_valid_ratio,
        node_config_.raw_deskew_min_valid_ratio, bucket_count,
        scan_start_time_ns, scan_end_time_ns,
        ResolveDeskewReferenceTimeNs(rclcpp::Time(header.stamp).nanoseconds(),
                                     scan_start_time_ns, scan_end_time_ns));
    return false;
  }

  for (const auto& pt : input_cloud->points) {
    const int64_t point_time_ns =
        ComputePointAbsoluteTimeNs(pt.timestamp, header, max_raw_timestamp);
    double normalized =
        static_cast<double>(point_time_ns - scan_start_time_ns) /
        static_cast<double>(scan_end_time_ns - scan_start_time_ns);
    normalized = std::clamp(normalized, 0.0, 0.999999);
    const size_t bucket_idx = std::min(
        bucket_count - 1,
        static_cast<size_t>(normalized * static_cast<double>(bucket_count)));
    if (!bucket_valid[bucket_idx]) {
      continue;
    }
    const auto& bucket_sample = bucket_samples[bucket_idx];

    Eigen::Vector3d point_local(pt.x, pt.y, pt.z);
    Eigen::Vector3d point_ref =
        reference_pose.inverse() * bucket_sample.pose * point_local;
    PointXYZIT out_pt;
    out_pt.x = static_cast<float>(point_ref.x());
    out_pt.y = static_cast<float>(point_ref.y());
    out_pt.z = static_cast<float>(point_ref.z());
    out_pt.intensity = pt.intensity;
    out_pt.timestamp = 0.0f;
    output_cloud->push_back(out_pt);
    ++out_valid_points;
  }
  return true;
}

bool GenPromptPoint::SampleDeskewTrajectory(
    const std_msgs::msg::Header& header, uint32_t max_raw_timestamp,
    size_t sample_count, std::vector<DeskewPoseSample>& out_samples,
    int64_t& out_reference_time_ns, int64_t& out_scan_start_time_ns,
    int64_t& out_scan_end_time_ns, DeskewPoseLookupStats* out_stats) {
  out_samples.clear();
  if (sample_count == 0) {
    return false;
  }

  const int64_t header_stamp_ns = rclcpp::Time(header.stamp).nanoseconds();
  const int64_t max_offset_ns = static_cast<int64_t>(std::llround(
      static_cast<double>(max_raw_timestamp) *
      node_config_.raw_deskew_timestamp_unit_sec * 1e9));

  if (node_config_.raw_deskew_timestamp_polarity ==
      RawDeskewTimestampPolarity::FromScanStart) {
    out_scan_start_time_ns = header_stamp_ns;
    out_scan_end_time_ns = header_stamp_ns + max_offset_ns;
  } else {
    out_scan_end_time_ns = header_stamp_ns;
    out_scan_start_time_ns = header_stamp_ns - max_offset_ns;
  }
  out_reference_time_ns = ResolveDeskewReferenceTimeNs(
      header_stamp_ns, out_scan_start_time_ns, out_scan_end_time_ns);

  if (out_stats) {
    out_stats->cache_sample_count = sample_count;
    out_stats->cache_hit_count = 0;
    out_stats->used_pose_cache = UsePoseCacheForAccurateDeskew();
  }

  if (UsePoseCacheForAccurateDeskew()) {
    int64_t waited_ms = 0;
    int64_t oldest_time_ns = 0;
    int64_t newest_time_ns = 0;
    if (out_stats) {
      out_stats->failure_reason = "none";
    }
    const bool has_coverage = WaitForPoseCacheCoverage(
        out_scan_start_time_ns, out_scan_end_time_ns,
        kPoseCacheCoverageWaitTimeoutMs, kPoseCacheCoveragePollIntervalMs,
        waited_ms, oldest_time_ns, newest_time_ns);
    if (out_stats) {
      out_stats->cache_wait_ms = waited_ms;
      out_stats->cache_oldest_time_ns = oldest_time_ns;
      out_stats->cache_newest_time_ns = newest_time_ns;
    }
    if (!has_coverage) {
      std::string reason = "cache_empty";
      if (oldest_time_ns != 0 || newest_time_ns != 0) {
        if (oldest_time_ns > out_scan_start_time_ns) {
          reason = "cache_before_scan_start";
        } else if (newest_time_ns < out_scan_end_time_ns) {
          reason = "cache_behind_scan_end";
        }
      }
      if (out_stats) {
        out_stats->failure_reason = reason;
      }
      WARN(
          "Deskew pose cache coverage wait failed. header_ns={} "
          "reason={} waited_ms={} cache_oldest_ns={} cache_newest_ns={} "
          "scan_start_ns={} scan_end_ns={} reference_time_ns={}",
          header_stamp_ns, reason, waited_ms, oldest_time_ns, newest_time_ns,
          out_scan_start_time_ns, out_scan_end_time_ns, out_reference_time_ns);
      return false;
    }

    std::vector<DeskewPoseSample> cache_snapshot;
    if (!CopyPoseCacheWindowSnapshot(out_scan_start_time_ns, out_scan_end_time_ns,
                                     cache_snapshot, oldest_time_ns,
                                     newest_time_ns)) {
      if (out_stats) {
        out_stats->failure_reason = "coverage_gap_inside_window";
      }
      WARN(
          "Deskew pose cache snapshot failed. header_ns={} "
          "reason=coverage_gap_inside_window cache_oldest_ns={} "
          "cache_newest_ns={} scan_start_ns={} scan_end_ns={} "
          "reference_time_ns={}",
          header_stamp_ns, oldest_time_ns, newest_time_ns,
          out_scan_start_time_ns, out_scan_end_time_ns, out_reference_time_ns);
      return false;
    }

    if (out_stats) {
      out_stats->cache_snapshot_size = cache_snapshot.size();
      out_stats->cache_oldest_time_ns = oldest_time_ns;
      out_stats->cache_newest_time_ns = newest_time_ns;
    }

    Eigen::Isometry3d reference_pose = Eigen::Isometry3d::Identity();
    if (!InterpolatePoseAtTime(cache_snapshot, out_reference_time_ns,
                               reference_pose)) {
      if (out_stats) {
        out_stats->failure_reason = "reference_pose_missing";
      }
      WARN(
          "Deskew reference pose interpolation failed. header_ns={} "
          "reason=reference_pose_missing snapshot_size={} cache_oldest_ns={} "
          "cache_newest_ns={} reference_time_ns={}",
          header_stamp_ns, cache_snapshot.size(), oldest_time_ns,
          newest_time_ns, out_reference_time_ns);
      return false;
    }

    out_samples.reserve(sample_count + 1);
    size_t success_count = 0;
    for (size_t i = 0; i < sample_count; ++i) {
      int64_t sample_time_ns = out_reference_time_ns;
      if (sample_count > 1 && out_scan_start_time_ns != out_scan_end_time_ns) {
        double alpha = static_cast<double>(i) /
                       static_cast<double>(sample_count - 1);
        sample_time_ns =
            out_scan_start_time_ns +
            static_cast<int64_t>(
                (out_scan_end_time_ns - out_scan_start_time_ns) * alpha);
      }
      Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
      if (!InterpolatePoseAtTime(cache_snapshot, sample_time_ns, pose)) {
        if (out_stats) {
          out_stats->failure_reason = "coverage_gap_inside_window";
        }
        WARN(
            "Deskew sample interpolation failed. header_ns={} "
            "reason=coverage_gap_inside_window sample_time_ns={} "
            "snapshot_size={} cache_oldest_ns={} cache_newest_ns={} "
            "scan_start_ns={} scan_end_ns={}",
            header_stamp_ns, sample_time_ns, cache_snapshot.size(),
            oldest_time_ns, newest_time_ns, out_scan_start_time_ns,
            out_scan_end_time_ns);
        return false;
      }
      out_samples.push_back({sample_time_ns, pose});
      ++success_count;
    }

    bool has_reference_sample = false;
    for (const auto& sample : out_samples) {
      if (sample.time_ns == out_reference_time_ns) {
        has_reference_sample = true;
        break;
      }
    }
    if (!has_reference_sample) {
      out_samples.push_back({out_reference_time_ns, reference_pose});
    }

    std::sort(out_samples.begin(), out_samples.end(),
              [](const DeskewPoseSample& lhs, const DeskewPoseSample& rhs) {
                return lhs.time_ns < rhs.time_ns;
              });
    out_samples.erase(
        std::unique(
            out_samples.begin(), out_samples.end(),
            [](const DeskewPoseSample& lhs, const DeskewPoseSample& rhs) {
              return lhs.time_ns == rhs.time_ns;
            }),
        out_samples.end());

    if (out_stats) {
      out_stats->cache_hit_count = out_samples.size();
    }
    const double sample_valid_ratio =
        sample_count == 0
            ? 0.0
            : static_cast<double>(success_count) /
                  static_cast<double>(sample_count);
    THROTTLEINFO(
        1,
        "Deskew sampled trajectory snapshot. header_ns={} pose_source={} "
        "cache_oldest_ns={} cache_newest_ns={} wait_ms={} "
        "snapshot_size={} sample_count={} valid_sample_count={} "
        "sample_valid_ratio={:.3f} scan_start_ns={} scan_end_ns={} "
        "reference_time_ns={}",
        header_stamp_ns,
        RawDeskewPoseSourceToString(node_config_.raw_deskew_pose_source),
        oldest_time_ns, newest_time_ns, waited_ms, cache_snapshot.size(),
        sample_count, out_samples.size(), sample_valid_ratio,
        out_scan_start_time_ns, out_scan_end_time_ns, out_reference_time_ns);
    if (out_samples.size() < 2 ||
        sample_valid_ratio < node_config_.raw_deskew_min_valid_ratio) {
      if (out_stats) {
        out_stats->failure_reason = "trajectory_interpolation_failed";
      }
      WARN(
          "Deskew sampled trajectory invalid. header_ns={} "
          "reason=trajectory_interpolation_failed valid_sample_count={} "
          "sample_valid_ratio={:.3f} threshold={:.3f}",
          header_stamp_ns, out_samples.size(), sample_valid_ratio,
          node_config_.raw_deskew_min_valid_ratio);
      return false;
    }
    return true;
  }

  const std::string fixed_frame = node_config_.raw_deskew_fixed_frame;
  const std::string sensor_frame = header.frame_id;
  const std::string ins_frame = ConstValue::kInsLink;

  size_t success_count = 0;
  size_t attempt_count = 0;
  out_samples.reserve(sample_count + 1);
  for (size_t i = 0; i < sample_count; ++i) {
    int64_t sample_time_ns = out_reference_time_ns;
    if (sample_count > 1 && out_scan_start_time_ns != out_scan_end_time_ns) {
      double alpha = static_cast<double>(i) /
                     static_cast<double>(sample_count - 1);
      sample_time_ns =
          out_scan_start_time_ns +
          static_cast<int64_t>((out_scan_end_time_ns - out_scan_start_time_ns) *
                               alpha);
    }
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    ++attempt_count;
    if (LookupDeskewPoseAtTime(fixed_frame, sensor_frame, ins_frame,
                               rclcpp::Time(sample_time_ns), pose)) {
      out_samples.push_back({sample_time_ns, pose});
      ++success_count;
    }
  }

  bool has_reference_sample = false;
  for (const auto& sample : out_samples) {
    if (sample.time_ns == out_reference_time_ns) {
      has_reference_sample = true;
      break;
    }
  }
  if (!has_reference_sample) {
    Eigen::Isometry3d reference_pose = Eigen::Isometry3d::Identity();
    ++attempt_count;
    if (LookupDeskewPoseAtTime(fixed_frame, sensor_frame, ins_frame,
                               rclcpp::Time(out_reference_time_ns),
                               reference_pose)) {
      out_samples.push_back({out_reference_time_ns, reference_pose});
      ++success_count;
    }
  }

  std::sort(out_samples.begin(), out_samples.end(),
            [](const DeskewPoseSample& lhs, const DeskewPoseSample& rhs) {
              return lhs.time_ns < rhs.time_ns;
            });
  out_samples.erase(
      std::unique(out_samples.begin(), out_samples.end(),
                  [](const DeskewPoseSample& lhs, const DeskewPoseSample& rhs) {
                    return lhs.time_ns == rhs.time_ns;
                  }),
      out_samples.end());

  const double sample_valid_ratio =
      attempt_count == 0
          ? 0.0
          : static_cast<double>(success_count) /
                static_cast<double>(attempt_count);
  if (out_samples.size() < 2 ||
      sample_valid_ratio < node_config_.raw_deskew_min_valid_ratio) {
    WARN("Deskew trajectory sampling at {} only has {} valid samples "
         "(ratio {:.3f}, threshold {:.3f}).",
         header_stamp_ns, out_samples.size(), sample_valid_ratio,
         node_config_.raw_deskew_min_valid_ratio);
    return false;
  }
  if (out_stats) {
    out_stats->cache_hit_count = out_samples.size();
  }
  return true;
}

bool GenPromptPoint::InterpolatePoseAtTime(
    const std::vector<DeskewPoseSample>& samples, int64_t query_time_ns,
    Eigen::Isometry3d& out_pose) const {
  if (samples.empty()) {
    return false;
  }
  if (samples.size() == 1) {
    out_pose = samples.front().pose;
    return true;
  }

  if (query_time_ns <= samples.front().time_ns) {
    if (query_time_ns < samples.front().time_ns) {
      return false;
    }
    out_pose = samples.front().pose;
    return true;
  }
  if (query_time_ns >= samples.back().time_ns) {
    if (query_time_ns > samples.back().time_ns) {
      return false;
    }
    out_pose = samples.back().pose;
    return true;
  }

  auto upper = std::lower_bound(
      samples.begin(), samples.end(), query_time_ns,
      [](const DeskewPoseSample& sample, int64_t target_time_ns) {
        return sample.time_ns < target_time_ns;
      });
  if (upper == samples.end()) {
    return false;
  }
  if (upper->time_ns == query_time_ns) {
    out_pose = upper->pose;
    return true;
  }
  if (upper == samples.begin()) {
    return false;
  }

  const auto& right = *upper;
  const auto& left = *(upper - 1);
  const int64_t duration_ns = right.time_ns - left.time_ns;
  if (duration_ns <= 0) {
    return false;
  }

  const double alpha =
      static_cast<double>(query_time_ns - left.time_ns) /
      static_cast<double>(duration_ns);
  Eigen::Quaterniond left_q(left.pose.rotation());
  Eigen::Quaterniond right_q(right.pose.rotation());
  left_q.normalize();
  right_q.normalize();

  Eigen::Isometry3d interpolated_pose = Eigen::Isometry3d::Identity();
  interpolated_pose.translation() =
      (1.0 - alpha) * left.pose.translation() +
      alpha * right.pose.translation();
  interpolated_pose.linear() = left_q.slerp(alpha, right_q).toRotationMatrix();
  out_pose = interpolated_pose;
  return true;
}

int64_t GenPromptPoint::ComputePointAbsoluteTimeNs(
    uint32_t raw_timestamp, const std_msgs::msg::Header& header,
    uint32_t max_raw_timestamp) const {
  (void)max_raw_timestamp;
  const int64_t header_stamp_ns = rclcpp::Time(header.stamp).nanoseconds();
  const int64_t offset_ns = static_cast<int64_t>(std::llround(
      static_cast<double>(raw_timestamp) *
      node_config_.raw_deskew_timestamp_unit_sec * 1e9));
  if (node_config_.raw_deskew_timestamp_polarity ==
      RawDeskewTimestampPolarity::FromScanStart) {
    return header_stamp_ns + offset_ns;
  }
  return header_stamp_ns - offset_ns;
}

int64_t GenPromptPoint::ResolveDeskewReferenceTimeNs(
    int64_t header_stamp_ns, int64_t scan_start_time_ns,
    int64_t scan_end_time_ns) const {
  switch (node_config_.raw_deskew_reference_time) {
    case RawDeskewReferenceTime::ScanStart:
      return scan_start_time_ns;
    case RawDeskewReferenceTime::ScanEnd:
      return scan_end_time_ns;
    case RawDeskewReferenceTime::HeaderStamp:
    default:
      return header_stamp_ns;
  }
}

void GenPromptPoint::PublishOrSaveDeskewedPointCloud(
    const pcl::PointCloud<PointXYZIT>::Ptr& deskewed_cloud,
    const std_msgs::msg::Header& header) {
  if (!deskewed_cloud || deskewed_cloud->empty()) {
    return;
  }

  if (raw_deskewed_pc_pub_) {
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*deskewed_cloud, output_msg);
    output_msg.header = header;
    raw_deskewed_pc_pub_->publish(output_msg);
  }

  if (node_config_.raw_deskew_save_debug_cloud &&
      !annotation_status_.raw_deskew_save_folder.empty()) {
    const int64_t timestamp_ns = rclcpp::Time(header.stamp).nanoseconds();
    const std::string file_path = annotation_status_.raw_deskew_save_folder +
                                  "/" + std::to_string(timestamp_ns) + ".pcd";
    if (pcl::io::savePCDFileBinary(file_path, *deskewed_cloud) == 0) {
      INFO("Saved raw deskewed PCD: {}", timestamp_ns);
    } else {
      WARN("Failed to save raw deskewed PCD: {}", timestamp_ns);
    }
  }
}

void GenPromptPoint::GenerateKITTILabel(
    int64_t timestamp, const cv::Mat& fig, const Eigen::Matrix4f& tf_map_to_cam,
    const std::vector<double>& intrinsics, const std::string& save_folder,
    const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_cam) {
  if (annotation_status_.current_labels.empty()) return;

  std::string label_filename =
      save_folder + "/" + std::to_string(timestamp) + ".txt";
  std::ofstream label_ofs(label_filename);

  if (!label_ofs.is_open()) {
    ERROR("Failed to open label file for writing: {}", label_filename);
    return;
  }

  double fx = intrinsics[0], cx = intrinsics[2], fy = intrinsics[4],
         cy = intrinsics[5];
  int img_w = fig.cols, img_h = fig.rows;
  int saved_count = 0;

  // 提前计算 相机坐标系 -> 全局Map坐标系 的变换矩阵
  Eigen::Matrix4f tf_cam_to_map = tf_map_to_cam.inverse();

  for (const auto& box : annotation_status_.current_labels) {
    Eigen::Affine3d local_to_map = box.map_to_local_tf.inverse();
    double hl = box.l / 2.0, hw = box.w / 2.0, hh = box.h / 2.0;

    Eigen::Vector3d bottom_local_check(0.0, 0.0, -box.h / 2.0);
    Eigen::Vector4f bottom_cam_check =
        tf_map_to_cam *
        (local_to_map * bottom_local_check).cast<float>().homogeneous();
    float dist_to_cam = bottom_cam_check.head<3>().norm();

    if (dist_to_cam > 30.0f) {
      INFO(
          "--> Dropped! [{}] was discarded: bottom center distance to camera "
          "({:.2f}m) > 30m.",
          box.object_type, dist_to_cam);
      continue;
    }

    // ================= 1. 点云可见性检测 (高性能 AABB 过滤 + 精确检测)
    // =================
    int visible_point_count = 0;
    Eigen::Matrix4f cam_to_local =
        box.map_to_local_tf.cast<float>() * tf_cam_to_map;
    float margin = 0.0f;

    // --- 【性能优化核心】：计算 3D 框在 Camera 坐标系下的 AABB (粗略外包围盒)
    // ---
    float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9, min_z = 1e9,
          max_z = -1e9;
    std::vector<Eigen::Vector3d> corners_local = {
        {hl, hw, hh},  {hl, -hw, hh},  {-hl, -hw, hh},  {-hl, hw, hh},
        {hl, hw, -hh}, {hl, -hw, -hh}, {-hl, -hw, -hh}, {-hl, hw, -hh}};

    for (const auto& pt_local : corners_local) {
      Eigen::Vector4f pt_cam =
          tf_map_to_cam * (local_to_map * pt_local).cast<float>().homogeneous();
      min_x = std::min(min_x, pt_cam.x());
      max_x = std::max(max_x, pt_cam.x());
      min_y = std::min(min_y, pt_cam.y());
      max_y = std::max(max_y, pt_cam.y());
      min_z = std::min(min_z, pt_cam.z());
      max_z = std::max(max_z, pt_cam.z());
    }

    // 给 AABB 加上容差
    min_x -= margin;
    max_x += margin;
    min_y -= margin;
    max_y += margin;
    min_z -= margin;
    max_z += margin;

    // 提取 3x3 旋转和 3x1 平移，比 4x4 齐次矩阵运算快很多
    Eigen::Matrix3f R = cam_to_local.block<3, 3>(0, 0);
    Eigen::Vector3f t = cam_to_local.block<3, 1>(0, 3);

    for (const auto& pt : cloud_in_cam->points) {
      // 1.1 快速剔除：如果点根本不在粗略包围盒内，直接跳过 (瞬间过滤 99% 的点)
      // 注意这里没有判断 pt.z < 0.1，以便能捕获相机背后的雷达点
      if (pt.x < min_x || pt.x > max_x || pt.y < min_y || pt.y > max_y ||
          pt.z < min_z || pt.z > max_z) {
        continue;
      }

      // 1.2 精确计算：只有非常靠近框的点，才进行矩阵乘法坐标系转换判断
      Eigen::Vector3f pt_cam(pt.x, pt.y, pt.z);
      Eigen::Vector3f pt_local = R * pt_cam + t;

      if (std::abs(pt_local.x()) <= hl + margin &&
          std::abs(pt_local.y()) <= hw + margin &&
          std::abs(pt_local.z()) <= hh + margin) {
        visible_point_count++;
      }
    }

    INFO(
        "Check KITTI Label - Object: [{}], Center(X:{:.2f}, Y:{:.2f}, "
        "Z:{:.2f}), Visible LiDAR Points: {}",
        box.object_type, box.tx, box.ty, box.tz, visible_point_count);

    // [重要判定]：如果该 3D 框内击中的雷达点少于 50
    // 个，认为不可见/被遮挡(幽灵框)，直接丢弃！ 你也可以根据
    // ConstValue::kThresholdPcCheck 调整这里的 50
    if (visible_point_count < ConstValue::kThresholdPcCheck) {
      WARN(
          "--> Dropped! [{}] was discarded because visible_point_count ({}) < "
          "{}.",
          box.object_type, visible_point_count, ConstValue::kThresholdPcCheck);
      continue;
    }

    // ================= 2. 相机视野与 2D 框计算 =================
    Eigen::Vector3d center_local(0, 0, 0);
    Eigen::Vector3d bottom_local(0, 0, -box.h / 2.0);

    Eigen::Vector4f center_cam =
        tf_map_to_cam *
        (local_to_map * center_local).cast<float>().homogeneous();
    Eigen::Vector4f bottom_cam =
        tf_map_to_cam *
        (local_to_map * bottom_local).cast<float>().homogeneous();

    float bbox_left = img_w, bbox_top = img_h, bbox_right = 0, bbox_bottom = 0;
    bool is_camera_visible = true;

    // 2.1 如果中心点在相机背后，直接标记为相机不可见
    if (center_cam.z() < 0.1) {
      is_camera_visible = false;
    } else {
      // 2.2 判断是否有顶点在相机前方并计算 2D 投影边界
      bool any_point_in_front = false;
      for (const auto& pt_local : corners_local) {
        Eigen::Vector4f pt_cam =
            tf_map_to_cam *
            (local_to_map * pt_local).cast<float>().homogeneous();
        if (pt_cam.z() > 0.1) {
          any_point_in_front = true;
          float u = (fx * pt_cam.x()) / pt_cam.z() + cx;
          float v = (fy * pt_cam.y()) / pt_cam.z() + cy;
          bbox_left = std::min(bbox_left, u);
          bbox_top = std::min(bbox_top, v);
          bbox_right = std::max(bbox_right, u);
          bbox_bottom = std::max(bbox_bottom, v);
        }
      }

      if (!any_point_in_front) {
        is_camera_visible = false;
      } else {
        // 2.3 限制在图像边界内
        bbox_left =
            std::max(0.0f, std::min(static_cast<float>(img_w) - 1, bbox_left));
        bbox_top =
            std::max(0.0f, std::min(static_cast<float>(img_h) - 1, bbox_top));
        bbox_right =
            std::max(0.0f, std::min(static_cast<float>(img_w) - 1, bbox_right));
        bbox_bottom = std::max(
            0.0f, std::min(static_cast<float>(img_h) - 1, bbox_bottom));

        // 2.4 如果边界框无效（比如在图像外面被截断裁没了），也视为不可见
        if (bbox_right <= bbox_left || bbox_bottom <= bbox_top) {
          is_camera_visible = false;
        }
      }
    }

    // ================= 3. 参数计算与写入 =================
    Eigen::Matrix3f R_local_to_cam =
        tf_map_to_cam.block<3, 3>(0, 0) * local_to_map.linear().cast<float>();
    // Eigen::Vector3f forward_cam = R_local_to_cam * Eigen::Vector3f(1.0f,
    // 0.0f, 0.0f); float ry = std::atan2(-forward_cam.z(), forward_cam.x());
    // 使得旋转矩阵真正匹配下游 Dataloader 使用的 KITTI 局部顶点定义
    Eigen::Matrix3f R_kitti2ros;
    R_kitti2ros << 1, 0, 0, 0, 0, 1, 0, -1, 0;

    // 3. 计算真正的 KITTI_Local -> Camera 旋转矩阵
    Eigen::Matrix3f R_kitti_to_cam = R_local_to_cam * R_kitti2ros;

    // 利用 tf2 工具类提取完整的欧拉角 (rx: 绕X轴, ry: 绕Y轴, rz: 绕Z轴)
    tf2::Matrix3x3 mat(
        R_kitti_to_cam(0, 0), R_kitti_to_cam(0, 1), R_kitti_to_cam(0, 2),
        R_kitti_to_cam(1, 0), R_kitti_to_cam(1, 1), R_kitti_to_cam(1, 2),
        R_kitti_to_cam(2, 0), R_kitti_to_cam(2, 1), R_kitti_to_cam(2, 2));
    double rx_double, ry_double, rz_double;
    mat.getRPY(rx_double, ry_double, rz_double);
    float rx = static_cast<float>(rx_double);
    float ry = static_cast<float>(ry_double);
    float rz = static_cast<float>(rz_double);

    // 正常计算 alpha (观测角)
    // Eigen::Vector3f forward_cam = R_kitti_to_cam * Eigen::Vector3f(1.0f,
    // 0.0f, 0.0f);
    float alpha = ry - std::atan2(bottom_cam.x(), bottom_cam.z());
    while (alpha > M_PI) alpha -= 2 * M_PI;
    while (alpha < -M_PI) alpha += 2 * M_PI;

    std::string class_name = box.object_type;
    std::replace(class_name.begin(), class_name.end(), ' ', '_');

    // 根据相机可见性输出不同的 2D Box、截断率、遮挡状态和 Alpha
    std::string truncation = "0.00";
    std::string occlusion = "0";
    std::string bbox_str;
    std::string alpha_str;

    if (is_camera_visible) {
      // 相机可见，正常输出
      std::ostringstream bbox_oss;
      bbox_oss << std::fixed << std::setprecision(2) << bbox_left << " "
               << bbox_top << " " << bbox_right << " " << bbox_bottom;
      bbox_str = bbox_oss.str();

      std::ostringstream alpha_oss;
      alpha_oss << std::fixed << std::setprecision(2) << alpha;
      alpha_str = alpha_oss.str();
    } else {
      // 相机不可见，输出无效值
      truncation = "1.00";  // 1.00 表示被完全截断/不在图像内
      occlusion = "3";      // 3 表示未知/完全不可见
      bbox_str = "-1.00 -1.00 -1.00 -1.00";
      alpha_str = "-10.00";  // KITTI约定俗成的无效角度值
    }

    label_ofs << std::fixed << std::setprecision(4) << class_name << " "
              << truncation << " " << occlusion << " " << alpha_str << " "
              << bbox_str << " " << std::setprecision(4) << box.h << " "
              << box.w << " " << box.l << " " << bottom_cam.x() << " "
              << bottom_cam.y() << " " << bottom_cam.z() << " " << ry << " "
              << rx << " " << rz << "\n";

    saved_count++;
  }
  label_ofs.close();
  INFO("Saved KITTI Label: {} (Total {} valid objects)", timestamp,
       saved_count);
}

void GenPromptPoint::VerifyKITTILabelInPointCloud(
    int64_t timestamp,
    const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_lidar_frame,
    const Eigen::Isometry3d& lcam2lidar, const std::string& label_folder,
    const std::string& save_folder) {
  std::string label_file =
      label_folder + "/" + std::to_string(timestamp) + ".txt";
  std::ifstream ifs(label_file);
  if (!ifs.is_open()) {
    WARN("PointCloud Verification skipped: Cannot open label file {}",
         label_file);
    return;
  }

  // 1. 初始化背景点云（灰色）
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  // 优化：提前分配内存，防止大点云 push_back 导致内存重分配崩溃
  vis_cloud->points.reserve(cloud_in_lidar_frame->points.size() + 10000);

  for (const auto& pt : cloud_in_lidar_frame->points) {
    pcl::PointXYZRGB p_rgb;
    p_rgb.x = pt.x;
    p_rgb.y = pt.y;
    p_rgb.z = pt.z;
    p_rgb.r = 180;
    p_rgb.g = 180;
    p_rgb.b = 180;
    vis_cloud->points.push_back(p_rgb);
  }

  // ==================== 绘制工具函数 ====================
  // 绘制单条线段的 Lambda 函数 (用于画指示针)
  auto draw_line = [&](const Eigen::Vector3f& p1, const Eigen::Vector3f& p2,
                       uint8_t r, uint8_t g, uint8_t b) {
    float dist = (p1 - p2).norm();
    int num_points = std::max(10, static_cast<int>(dist / 0.01f));
    for (int i = 0; i <= num_points; ++i) {
      float ratio = static_cast<float>(i) / num_points;
      Eigen::Vector3f pt = p1 + ratio * (p2 - p1);
      pcl::PointXYZRGB p_rgb;
      p_rgb.x = pt.x();
      p_rgb.y = pt.y();
      p_rgb.z = pt.z();
      p_rgb.r = r;
      p_rgb.g = g;
      p_rgb.b = b;
      vis_cloud->points.push_back(p_rgb);
    }
  };

  // 绘制框线的 Lambda 函数 (复用画单线的逻辑，保持整洁)
  auto draw_box_edges = [&](const std::vector<Eigen::Vector3f>& corners_lidar,
                            uint8_t r, uint8_t g, uint8_t b) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                              {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                              {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (const auto& edge : edges) {
      draw_line(corners_lidar[edge.first], corners_lidar[edge.second], r, g, b);
    }
  };

  std::string line;
  int valid_box_count = 0;

  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string type;
    float trunc, occ, alpha, left, top, right, bottom, h, w, l, x, y, z, ry;
    float rx = 0.0f, rz = 0.0f;

    // 解析 17 参数格式
    ss >> type >> trunc >> occ >> alpha >> left >> top >> right >> bottom >>
        h >> w >> l >> x >> y >> z >> ry;
    if (ss >> rx) ss >> rz;

    // 局部顶点定义
    float half_l = l / 2.0f;
    float half_w = w / 2.0f;
    float h_full = h;
    std::vector<Eigen::Vector3f> corners_local = {
        {half_l, -h_full, half_w},  {half_l, -h_full, -half_w},
        {half_l, 0, -half_w},       {half_l, 0, half_w},
        {-half_l, -h_full, half_w}, {-half_l, -h_full, -half_w},
        {-half_l, 0, -half_w},      {-half_l, 0, half_w}};

    Eigen::Vector3f T(x, y, z);

    // ==================== 2. 计算 6-DoF 位姿与雷达系中心 ====================
    tf2::Quaternion q;
    q.setRPY(rx, ry, rz);
    tf2::Matrix3x3 mat(q);
    Eigen::Matrix3f R_6dof;
    R_6dof << mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2],
        mat[2][0], mat[2][1], mat[2][2];

    // A. 计算雷达系下的底面中心 (BottomCenter)
    Eigen::Vector3f center_lidar_bottom =
        lcam2lidar.linear().cast<float>() * T +
        lcam2lidar.translation().cast<float>();

    // B. 计算雷达系下的几何中心 (GeoCenter)
    Eigen::Vector3f geometric_center_local(0.0f, -h_full / 2.0f, 0.0f);
    Eigen::Vector3f geometric_center_cam = R_6dof * geometric_center_local + T;
    Eigen::Vector3f center_lidar_geo =
        lcam2lidar.linear().cast<float>() * geometric_center_cam +
        lcam2lidar.translation().cast<float>();

    // C. 计算雷达系下的旋转角
    Eigen::Matrix3f R_lidar = lcam2lidar.linear().cast<float>() * R_6dof;
    tf2::Matrix3x3 mat_lidar(R_lidar(0, 0), R_lidar(0, 1), R_lidar(0, 2),
                             R_lidar(1, 0), R_lidar(1, 1), R_lidar(1, 2),
                             R_lidar(2, 0), R_lidar(2, 1), R_lidar(2, 2));
    double rx_l, ry_l, rz_l;
    mat_lidar.getRPY(rx_l, ry_l, rz_l);

    // ==================== 3. 打印详细日志 (雷达系) ====================
    INFO("Box [{}] LiDAR Verification Details:", type);
    INFO("  -> BottomCenter (LiDAR): X:{:.3f}, Y:{:.3f}, Z:{:.3f}",
         center_lidar_bottom.x(), center_lidar_bottom.y(),
         center_lidar_bottom.z());
    INFO("  -> GeoCenter    (LiDAR): X:{:.3f}, Y:{:.3f}, Z:{:.3f}",
         center_lidar_geo.x(), center_lidar_geo.y(), center_lidar_geo.z());
    INFO("  -> Rotation RPY (LiDAR): rx:{:.3f}, ry:{:.3f}, rz:{:.3f}", rx_l,
         ry_l, rz_l);

    // 计算 8 个顶点坐标
    std::vector<Eigen::Vector3f> corners_comp_lidar;
    for (size_t i = 0; i < corners_local.size(); ++i) {
      Eigen::Vector3f pt_cam = R_6dof * corners_local[i] + T;
      Eigen::Vector3f pt_lidar = lcam2lidar.linear().cast<float>() * pt_cam +
                                 lcam2lidar.translation().cast<float>();
      corners_comp_lidar.push_back(pt_lidar);
    }

    // 绘制紫色 6-DoF 框
    draw_box_edges(corners_comp_lidar, 255, 0, 255);

    // ==================== [新增] 绘制车头指示针 ====================
    // 提取相机坐标系下的前向向量 (X轴)
    Eigen::Vector3f v_fwd_cam(R_6dof(0, 0), R_6dof(1, 0), R_6dof(2, 0));
    // 计算前脸中心：在局部坐标系中，X向是 half_l，Y向取几何中心 (-h_full
    // / 2.0f)
    Eigen::Vector3f f_cnt_cam =
        T + R_6dof * Eigen::Vector3f(half_l, -h_full / 2.0f, 0.0f);
    // 针尖向外延伸 2 米
    Eigen::Vector3f tip_cam = f_cnt_cam + 2.0f * v_fwd_cam;

    // 转换到雷达坐标系
    Eigen::Vector3f f_cnt_lidar =
        lcam2lidar.linear().cast<float>() * f_cnt_cam +
        lcam2lidar.translation().cast<float>();
    Eigen::Vector3f tip_lidar = lcam2lidar.linear().cast<float>() * tip_cam +
                                lcam2lidar.translation().cast<float>();

    // 绘制亮绿色指示针
    draw_line(f_cnt_lidar, tip_lidar, 0, 255, 0);

    // ==================== 4. 绘制标准 KITTI 框 (红色，对照组)
    // ====================
    Eigen::Matrix3f R_y;
    R_y << std::cos(ry), 0, std::sin(ry), 0, 1, 0, -std::sin(ry), 0,
        std::cos(ry);
    std::vector<Eigen::Vector3f> corners_kitti_lidar;
    for (const auto& pt_local : corners_local) {
      Eigen::Vector3f pt_cam = R_y * pt_local + T;
      Eigen::Vector3f pt_lidar = lcam2lidar.linear().cast<float>() * pt_cam +
                                 lcam2lidar.translation().cast<float>();
      corners_kitti_lidar.push_back(pt_lidar);
    }
    draw_box_edges(corners_kitti_lidar, 255, 0, 0);
  }
  ifs.close();

  // ==================== [关键] 设置 Header 信息 ====================
  // 如果不设置这些，很多点云查看器会拒绝渲染
  vis_cloud->width = vis_cloud->points.size();
  vis_cloud->height = 1;
  vis_cloud->is_dense = true;

  std::string save_path =
      save_folder + "/" + std::to_string(timestamp) + "_verify.pcd";
  pcl::io::savePCDFileBinary(save_path, *vis_cloud);
  INFO("Verified KITTI Label in LiDAR Frame & Saved PCD: {}", save_path);
}

void GenPromptPoint::GenerateCameraExtrinsic(
    int64_t timestamp, int width, int height,
    const Eigen::Matrix4f& tf_map_to_lidar) {
  INFO("Generate Extrinsic: {}", timestamp);
  // 路径生成逻辑保持不变
  std::string config_dir =
      annotation_status_.data_record_root_folder + "/camera_config";
  if (!std::filesystem::exists(config_dir)) {
    std::filesystem::create_directories(config_dir);
  }

  std::string json_path =
      config_dir + "/" + std::to_string(timestamp) + ".json";
  std::ofstream ofs(json_path);
  if (!ofs.is_open()) {
    ERROR("Failed to create camera_config json at {}", json_path);
    return;
  }

  // 获取外参: Lidar -> Camera
  Eigen::Isometry3d tf_lidar2cam =
      annotation_status_.isometry_lcam2lidar.inverse();
  Eigen::Matrix4f ext = tf_lidar2cam.matrix().cast<float>();

  // 获取内参
  const auto& K = annotation_status_.left_intrinsic_data;
  double fx = K[0], cx = K[2], fy = K[4], cy = K[5];

  Eigen::Matrix4f tf_lidar_to_map = tf_map_to_lidar.inverse();

  // 1. 先写入到 camera_external 数组结束 (注意结尾不加逗号，逗号留在后面动态加)
  ofs << "[\n"
      << "  {\n"
      << "    \"camera_internal\": {\n"
      << "      \"fx\": " << fx << ",\n"
      << "      \"fy\": " << fy << ",\n"
      << "      \"cx\": " << cx << ",\n"
      << "      \"cy\": " << cy << "\n"
      << "    },\n"
      << "    \"width\": " << width << ",\n"
      << "    \"height\": " << height << ",\n"
      << "    \"camera_external\": [\n"
      << "      " << ext(0, 0) << ",\n"
      << "      " << ext(1, 0) << ",\n"
      << "      " << ext(2, 0) << ",\n"
      << "      " << ext(3, 0) << ",\n"
      << "      " << ext(0, 1) << ",\n"
      << "      " << ext(1, 1) << ",\n"
      << "      " << ext(2, 1) << ",\n"
      << "      " << ext(3, 1) << ",\n"
      << "      " << ext(0, 2) << ",\n"
      << "      " << ext(1, 2) << ",\n"
      << "      " << ext(2, 2) << ",\n"
      << "      " << ext(3, 2) << ",\n"
      << "      " << ext(0, 3) << ",\n"
      << "      " << ext(1, 3) << ",\n"
      << "      " << ext(2, 3) << ",\n"
      << "      " << ext(3, 3) << "\n"
      << "    ]";

  // 2. 校验 tf_map_to_lidar 是否为单位阵。加入 1e-4 的容差防止浮点数精度问题
  if (!tf_lidar_to_map.isIdentity(1e-4)) {
    ofs << ",\n"
        << "    \"tf_lidar_to_map\": [\n"  // <--- 注意这里键名改了
        << "      " << tf_lidar_to_map(0, 0) << ",\n"
        << "      " << tf_lidar_to_map(1, 0) << ",\n"
        << "      " << tf_lidar_to_map(2, 0) << ",\n"
        << "      " << tf_lidar_to_map(3, 0) << ",\n"
        << "      " << tf_lidar_to_map(0, 1) << ",\n"
        << "      " << tf_lidar_to_map(1, 1) << ",\n"
        << "      " << tf_lidar_to_map(2, 1) << ",\n"
        << "      " << tf_lidar_to_map(3, 1) << ",\n"
        << "      " << tf_lidar_to_map(0, 2) << ",\n"
        << "      " << tf_lidar_to_map(1, 2) << ",\n"
        << "      " << tf_lidar_to_map(2, 2) << ",\n"
        << "      " << tf_lidar_to_map(3, 2) << ",\n"
        << "      " << tf_lidar_to_map(0, 3) << ",\n"
        << "      " << tf_lidar_to_map(1, 3) << ",\n"
        << "      " << tf_lidar_to_map(2, 3) << ",\n"
        << "      " << tf_lidar_to_map(3, 3) << "\n"
        << "    ]";
  } else {
    INFO("Check tf_lidar_to_map is Identity");
  }

  // 3. 写入最后的 rowMajor 标志 (不管上面有没有写
  // tf_map_to_lidar，前面都要加逗号)
  ofs << ",\n"
      << "    \"rowMajor\": false\n"
      << "  }\n"
      << "]\n";

  ofs.close();
}

bool GenPromptPoint::GetMapToCameraAndLidarTf(
    const rclcpp::Time& img_time, Eigen::Matrix4f& out_tf_map_to_cam,
    Eigen::Matrix4f& out_tf_map_to_lidar) {
  std::string fixed_frame = ConstValue::kInsMap;
  try {
    // 1. 尝试直接查询 Map -> CameraLeft
    geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_->lookupTransform(
        ConstValue::kCameraLeftLink, fixed_frame, img_time,
        rclcpp::Duration::from_seconds(0.02));

    Eigen::Affine3d tf_map_to_cam_affine = tf2::transformToEigen(tf_msg);
    out_tf_map_to_cam = tf_map_to_cam_affine.matrix().cast<float>();

    // 2. 利用静态外参计算 Map -> Lidar
    out_tf_map_to_lidar =
        (annotation_status_.isometry_lcam2lidar * tf_map_to_cam_affine)
            .matrix()
            .cast<float>();
    return true;

  } catch (tf2::TransformException& ex) {
    // 降级策略：利用 InsLink 作为桥梁
    std::string ins_frame = ConstValue::kInsLink;
    try {
      geometry_msgs::msg::TransformStamped tf_ins_msg =
          tf_buffer_->lookupTransform(fixed_frame, ins_frame, img_time,
                                      rclcpp::Duration::from_seconds(0.03));

      Eigen::Isometry3d iso_ins_to_map = tf2::transformToEigen(tf_ins_msg);

      // T_cam_to_map = T_ins_to_map * T_lidar_to_ins * T_cam_to_lidar
      Eigen::Isometry3d iso_lcam_to_map =
          iso_ins_to_map * annotation_status_.isometry_lidar2ins *
          annotation_status_.isometry_lcam2lidar;
      out_tf_map_to_cam = iso_lcam_to_map.inverse().matrix().cast<float>();

      // T_lidar_to_map = T_ins_to_map * T_lidar_to_ins
      Eigen::Isometry3d iso_lidar_to_map =
          iso_ins_to_map * annotation_status_.isometry_lidar2ins;
      out_tf_map_to_lidar = iso_lidar_to_map.inverse().matrix().cast<float>();

      WARN("Direct TF Map->Camera failed. Fallback to Map->InsLink used.");
      return true;

    } catch (tf2::TransformException& ex2) {
      WARN("TF lookup failed completely for this frame: {}", ex2.what());
      return false;
    }
  }
}

std::string GenPromptPoint::get_current_localtime_str() {
  // 获取当前时间点
  auto now = std::chrono::system_clock::now();

  // 转换为time_t
  std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

  // 转换为tm结构（本地时间）
  std::tm* now_tm = std::localtime(&now_time_t);

  // 使用stringstream格式化输出
  std::stringstream ss;
  ss << std::put_time(now_tm, "%Y%m%d%H%M%S");

  return ss.str();
}

void GenPromptPoint::print_matrix_as_numpy(const std::vector<double>& data,
                                           const int rows, const int cols) {
  INFO("Dimensions: {} rows x {} cols", rows, cols);
  INFO("Matrix Values:");
  size_t idx = 0;
  // 设置打印宽度 (例如 10) 和精度 (例如 4)
  const int width = 12;
  const int precision = 4;
  for (int i = 0; i < rows; ++i) {
    std::stringstream ss;
    // Numpy 风格：第一行开头是 "[[", 后续行开头是 " ["
    // (这就产生了一个空格的缩进，视觉对齐)
    if (i == 0) {
      ss << "[[";
    } else {
      ss << " [";
    }
    for (int j = 0; j < cols; ++j) {
      if (idx < data.size()) {
        // 使用 fixed 和 setprecision 控制小数位，setw 控制对齐
        ss << std::fixed << std::setprecision(precision) << std::setw(width)
           << data[idx];

        // 只有不是最后一列时才加逗号(Numpy
        // 其实不加逗号，只加空格，但为了清晰保留空格即可) 如果你严格想要 Numpy
        // 纯数字风格，就不加逗号，只留空格
        if (j < cols - 1) {
          ss << " ";
        }
        idx++;
      }
    }
    // 行尾处理：如果是最后一行，加 "]]"，否则加 "]"
    if (i == rows - 1) {
      ss << "]]";
    } else {
      ss << "]";
    }
    // 打印这一整行
    INFO("{}", ss.str());
  }
}

void GenPromptPoint::print_transform_eigen_isometry(
    const std::string& name, const Eigen::Isometry3d& transform) {
  // 1. 提取平移
  auto translation = transform.translation();

  // 2. 提取旋转并转换为欧拉角 (使用 TF2 工具类更稳定，避免 Eigen 的多解问题)
  // 将 Eigen 旋转矩阵构造为 tf2 矩阵
  tf2::Matrix3x3 mat(transform(0, 0), transform(0, 1), transform(0, 2),
                     transform(1, 0), transform(1, 1), transform(1, 2),
                     transform(2, 0), transform(2, 1), transform(2, 2));

  double roll, pitch, yaw;
  mat.getRPY(roll, pitch, yaw);

  // 4. 打印日志
  // 格式示例: [TF] Lidar->Camera | T:[1.50, 0.20, 0.05] | R(deg):[0.00,
  // 0.00, 90.00]
  INFO("[TF] {} | T:[{:.3f}, {:.3f}, {:.3f}] | R(deg):[{:.3f}, {:.3f}, {:.3f}]",
       name, translation.x(), translation.y(), translation.z(),
       roll * ConstValue::kRad2Deg, pitch * ConstValue::kRad2Deg,
       yaw * ConstValue::kRad2Deg);
}

}  // namespace automatic_annotation
