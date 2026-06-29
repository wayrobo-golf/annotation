#ifndef AUTOMATIC_ANNOTATION_GEN_PROMPT_POINT_HPP
#define AUTOMATIC_ANNOTATION_GEN_PROMPT_POINT_HPP

// 必须在包含 PCL 头文件前定义，确保自定义点类型被正确处理
#include <cv_bridge_with_opencv411/cv_bridge.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#define PCL_NO_PRECOMPILE
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tinyxml2.h>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <string>
#include <thread>
#include <tf2_eigen/tf2_eigen.hpp>
#include <vector>

#include "automatic_annotation/const_value.hpp"
#include "common_utils/logger.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace automatic_annotation {
// 1. 自定义点云结构体
struct LivoxPoint {
  PCL_ADD_POINT4D;
  float intensity;
  uint8_t tag;
  uint8_t line;
  uint32_t timestamp;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

struct UndistortedLivoxPoint {
  // float x;
  // float y;
  // float z;
  // float unused_padding;  // 必须加这个！占用 12, 13, 14, 15 字节
  PCL_ADD_POINT4D;  // 方法2:  使用PCL宏定义点结构体，自动添加padding
  float intensity;  // 偏移 16

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

struct PointXYZIT {
  PCL_ADD_POINT4D;
  float intensity;
  float timestamp;  // 相对时间差（年龄），单位：秒

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

enum class RawDeskewMode { Off = 0, Fast = 1, Accurate = 2 };

enum class RawDeskewReferenceTime {
  HeaderStamp = 0,
  ScanStart = 1,
  ScanEnd = 2
};

enum class RawDeskewTimestampPolarity {
  FromScanStart = 0,
  FromScanEnd = 1
};

enum class RawDeskewPoseSource { InsOdom = 0, Tf = 1 };

enum class RawDeskewMissingPosePolicy {
  FallbackFast = 0,
  Skip = 1
};

struct BoundingBox {
  // === 1. 基础属性 (对应 XML 外层 item) ===
  std::string
      object_type;  // 对应 <objectType> (例如: "Sand", "Distance Marker")
  std::string track_id;    // Xtreme1 跨帧实例身份
  std::string track_name;  // Xtreme1 UI 右侧 Instances 中显示的编号

  // 尺寸 (Dimensions)
  double h = 0.0;  // 对应 <h> (Height)
  double w = 0.0;  // 对应 <w> (Width)
  double l = 0.0;  // 对应 <l> (Length)

  // === 2. 位姿属性 (对应 XML 内部 poses -> item) ===
  // 位置 (Translation)
  double tx = 0.0;  // 对应 <tx> (X坐标)
  double ty = 0.0;  // 对应 <ty> (Y坐标)
  double tz = 0.0;  // 对应 <tz> (Z坐标)

  // 旋转 (Rotation) - 欧拉角
  double rx = 0.0;  // 对应 <rx>
  double ry = 0.0;  // 对应 <ry>
  double rz = 0.0;  // 对应 <rz> (通常是 Yaw 角)

  // === 3. 状态属性 ===
  int state = 0;       // 对应 <state>
  int occlusion = 0;   // 对应 <occlusion>
  int truncation = 0;  // 对应 <truncation>

  // 存储从 Map 坐标系到 Box 局部坐标系的变换矩阵 (逆变换)
  // 必须初始化为单位矩阵，防止未计算时崩溃
  Eigen::Affine3d map_to_local_tf = Eigen::Affine3d::Identity();
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 获取中心点 Vector
  Eigen::Vector3d GetCenter() const { return Eigen::Vector3d(tx, ty, tz); }

  // 获取尺寸 Vector (l, w, h)
  Eigen::Vector3d GetSize() const { return Eigen::Vector3d(l, w, h); }

  // 获取旋转
  Eigen::Vector3d GetRotation() const { return Eigen::Vector3d(rx, ry, rz); }
};

struct AutoAnnotationConfig {
  int64_t point_cloud_type = -1;  // 0: 原始点云, 1: 去畸变点云
  std::string point_cloud_topic;
  int64_t target_accumulate_pc_num = -1;
  bool publish_accumulated_pc = false;
  std::string accumulate_pc_save_folder;
  RawDeskewMode raw_deskew_mode = RawDeskewMode::Off;
  std::string raw_deskew_fixed_frame = ConstValue::kInsMap;
  RawDeskewReferenceTime raw_deskew_reference_time =
      RawDeskewReferenceTime::HeaderStamp;
  double raw_deskew_timestamp_unit_sec = 1e-6;
  RawDeskewTimestampPolarity raw_deskew_timestamp_polarity =
      RawDeskewTimestampPolarity::FromScanEnd;
  int64_t raw_deskew_bucket_count_fast = 10;
  int64_t raw_deskew_bucket_count_accurate = 50;
  bool raw_deskew_publish_debug_cloud = false;
  bool raw_deskew_save_debug_cloud = false;
  double raw_deskew_min_valid_ratio = 0.9;
  std::string raw_deskew_save_folder;
  RawDeskewPoseSource raw_deskew_pose_source = RawDeskewPoseSource::InsOdom;
  std::string raw_deskew_pose_topic = "/novatel/oem7/ins_odom";
  RawDeskewMissingPosePolicy raw_deskew_missing_pose_policy =
      RawDeskewMissingPosePolicy::FallbackFast;
  double raw_deskew_pose_cache_duration_sec = 30.0;

  int64_t image_source_type = -1;    // 1:左目相机 2:右目相机  3:双目相机
  int64_t image_save_interval = -1;  // 存储图片的间隔
  std::string left_image_topic;
  std::string right_image_topic;
  std::string left_camera_intrinsic_topic;
  std::string right_camera_intrinsic_topic;
  std::string left_camera_save_folder;
  std::string right_camera_save_folder;

  bool auto_annotation = false;
  bool save_raw_image = true;           // 是否保存原始图像
  bool save_synced_pcd = true;          // 是否保存对齐到图像时刻的点云
  bool generate_depth_map = false;      // 是否生成 16位深度图与伪彩图
  bool generate_semantic_mask = false;  // 是否生成语义分割 Mask
  bool generate_kitti_label = true;     // 是否生成 KITTI 格式 txt 标签
  bool generate_nusc_label = false;     // 是否收集 NuScenes 格式数据
  bool generate_xtreme1_json = true;    // 是否生成用于 Xtreme1 的标定 json
  bool enable_visual_verify = false;  // 是否生成用于人类观测的 2D/3D 投影画框图
  std::string global_pc_map_addr;
  std::string pc_annotation_file_addr;

  double tf_wait_timeout_sec = 3.0;
  Eigen::Isometry3d default_isometry_lcam2lidar = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d default_isometry_rcam2lidar = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d default_isometry_lidar2ins = Eigen::Isometry3d::Identity();
};

enum class CameraRole { Left = 0, Right = 1 };

struct PendingImageTask {
  CameraRole role = CameraRole::Left;
  int64_t timestamp_ns = 0;
  sensor_msgs::msg::Image::SharedPtr image_msg;
};

struct AccumulatedCloudSnapshot {
  int64_t timestamp_ns = 0;
  pcl::PointCloud<PointXYZIT>::Ptr cloud;
};

struct MatchedImageTask {
  CameraRole role = CameraRole::Left;
  int64_t timestamp_ns = 0;
  sensor_msgs::msg::Image::SharedPtr image_msg;
  pcl::PointCloud<PointXYZIT>::Ptr cloud_snapshot;
};

struct AutoAnnotationStatus {
  // 左相机参数
  bool has_left_intrinsics = false;
  int left_intrinsic_rows = 0;
  int left_intrinsic_cols = 0;
  std::vector<double> left_intrinsic_data;
  // 右相机参数
  bool has_right_intrinsics = false;
  int right_intrinsic_rows = 0;
  int right_intrinsic_cols = 0;
  std::vector<double> right_intrinsic_data;
  // 图像统计
  uint32_t left_image_rcv_count = 0;
  uint32_t left_raw_image_rcv_count = 0;
  uint32_t right_image_rcv_count = 0;
  // 点云类型
  enum class PointCloudType { Original = 0, Undistorted = 1 };
  PointCloudType pc_type = PointCloudType::Original;
  // 点云缓存
  std::deque<pcl::PointCloud<LivoxPoint>::Ptr> queue_original;
  std::deque<pcl::PointCloud<UndistortedLivoxPoint>::Ptr> queue_undistorted;
  std::deque<pcl::PointCloud<PointXYZIT>::Ptr> queue_deskewed;
  std::deque<PendingImageTask> pending_image_tasks;
  std::deque<MatchedImageTask> matched_image_tasks;
  std::deque<AccumulatedCloudSnapshot> accumulated_cloud_history;
  pcl::PointCloud<PointXYZIT>::Ptr latest_accumulated_cloud;
  rclcpp::Time latest_cloud_timestamp;
  // 全局地图缓存
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_map = nullptr;
  // 标注框
  std::vector<BoundingBox> current_labels;
  // 存放每个标注框对应的点云
  // 数组索引与 current_labels 的索引一一对应
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> object_clouds;
  // 外参
  bool has_tf_left_camera_link_2_lidar_link = false;
  Eigen::Isometry3d isometry_lcam2lidar = Eigen::Isometry3d::Identity();
  bool has_tf_right_camera_link_2_lidar_link = false;
  Eigen::Isometry3d isometry_rcam2lidar = Eigen::Isometry3d::Identity();
  bool has_tf_lidar_link_2_ins_link = false;
  Eigen::Isometry3d isometry_lidar2ins = Eigen::Isometry3d::Identity();
  struct TimedPoseSample {
    int64_t time_ns = 0;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  };
  std::deque<TimedPoseSample> ins_pose_cache;
  // 存储路径
  std::string data_record_root_folder;   // 记录总根目录，用于放 JSON
  bool has_saved_camera_config = false;  // 标记是否已经生成过配置文件
  std::string accumulate_pc_save_folder;
  std::string raw_deskew_save_folder;
  std::string left_label_save_folder;
  std::string left_original_image_save_folder;
  std::string left_fuse_image_save_folder;
  std::string left_depth_image_save_folder;
  std::string left_semantic_image_save_folder;
  std::string right_original_image_save_folder;
  std::string right_fuse_image_save_folder;
  std::string right_depth_image_save_folder;
  std::string right_semantic_image_save_folder;
};

class GenPromptPoint {
 public:
  explicit GenPromptPoint(const rclcpp::Node::SharedPtr& node);
  ~GenPromptPoint();

 private:
  rclcpp::Node::SharedPtr node_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      pointcloud_sub_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_image_sub_ =
      nullptr;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
      left_raw_image_sub_ = nullptr;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_image_sub_ =
      nullptr;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      left_camera_intrinsics_sub_ = nullptr;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr
      right_camera_intrinsics_sub_ = nullptr;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ins_odom_sub_ =
      nullptr;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_ = nullptr;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_ = nullptr;
  rclcpp::TimerBase::SharedPtr tf_check_timer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      accumulated_pc_pub_ = nullptr;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      raw_deskewed_pc_pub_ = nullptr;
  rclcpp::CallbackGroup::SharedPtr callback_group_lidar_ = nullptr;
  rclcpp::CallbackGroup::SharedPtr callback_group_pose_ = nullptr;
  rclcpp::CallbackGroup::SharedPtr callback_group_camera_ = nullptr;
  rclcpp::CallbackGroup::SharedPtr callback_group_raw_image_ = nullptr;

  AutoAnnotationConfig node_config_;
  AutoAnnotationStatus annotation_status_;
  int tf_check_count_ = 0;
  struct LockGroup {
    std::mutex mutex_use_tf;
    std::mutex mutex_cache_cloud;
    std::mutex mutex_pose_cache;
    std::mutex mutex_image_tasks;
  };
  mutable LockGroup locks_;
  std::condition_variable pending_image_task_cv_;
  std::condition_variable matched_image_task_cv_;
  std::thread image_match_thread_;
  std::thread image_worker_thread_;
  std::atomic<bool> stop_async_processing_{false};
  std::map<std::string, uint8_t> class_name_to_id_;
  std::map<std::string, std::string> class_name_to_label_;
  std::string package_share_path_;

  // 初始化相关的成员函数
  void InitConfigFromParamsServer();
  void InitDefaultTfSettings();  // 初始化默认外参
  Eigen::Isometry3d ConvertXYZRPYToIsometry(
      const std::vector<double>& params);  // 将数组转换为变换矩阵
  void InitGlobalMapAndLabelAddr();
  void InitPointCloudTopic();
  void InitPointCloudSettings();
  void InitRawDeskewSettings();
  void InitImageSettings();
  void InitCameraTopics();
  void InitCameraIntrinsicsTopics();
  void InitSaveFolder();
  void InitAutoAnnotation();
  bool EnsureDirectoryWithFullPermissions(
      const std::filesystem::path& dir_path);
  void InitMapAndLabel();
  void InitClassMappingAndReflact();
  void InitPublishAndSubscription();
  void CreateAccumulatePointCloudPublisher();
  void CreateRawDeskewPointCloudPublisher();
  void CreatePointCloudSubscription();
  void CreateInsOdomSubscription();
  void CreateImageSubscription();
  void CreateCameraIntrinsicsSubscription();
  void StartAsyncProcessingThreads();
  void InitTfListener();
  // 回调函数
  void LeftIntrinsicsCallback(std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void RightIntrinsicsCallback(std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void InsOdomCallback(nav_msgs::msg::Odometry::SharedPtr msg);
  void CheckTfCallback();
  // 业务函数
  void EnqueueLeftImageTask(const sensor_msgs::msg::Image::SharedPtr msg);
  void HandleLeftRawImageSaveMsg(const sensor_msgs::msg::Image::SharedPtr msg);
  void HandleLeftImageMsg(const sensor_msgs::msg::Image::SharedPtr msg);
  void HandleRightImageMsg(const sensor_msgs::msg::Image::SharedPtr msg);
  void MatchPendingImageTasks();
  void ProcessMatchedImageTasks();
  void ProcessMatchedLeftImageTask(const MatchedImageTask& task);
  void RunImageMatchLoop();
  void RunImageWorkerLoop();
  void AppendAccumulatedCloudSnapshot(
      const pcl::PointCloud<PointXYZIT>::Ptr& cloud, int64_t timestamp_ns);
  void PrunePendingImageTasksLocked(int64_t now_ns);
  void PruneAccumulatedCloudHistoryLocked(int64_t now_ns);
  bool TryBuildMatchedTaskLocked(MatchedImageTask& out_task,
                                 std::string& out_reason);
  bool LookUpTransform(const std::string& target_frame,
                       const std::string& source_frame,
                       geometry_msgs::msg::TransformStamped& out_transform);
  void ExtractCloudsFromBoxes();
  void LoadLabelsFromXML(const std::string& xml_path);
  // 将 ROS Image 消息转换为 OpenCV Mat，支持压缩格式和未压缩格式
  cv::Mat ConvertRosImageToCvMat(const sensor_msgs::msg::Image::SharedPtr& msg);
  void SaveOriginImageToFile(const sensor_msgs::msg::Image::SharedPtr& msg,
                             const cv::Mat& fig, const std::string& save_dir,
                             const std::string& file_name_before_stamp,
                             int64_t timestamp);
  /**
   * @brief 生成 KITTI Benchmark 格式的 2D/3D 目标检测标签并保存为 txt
   * @param timestamp 当前帧时间戳，用于文件命名
   * @param fig 当前帧图像，用于获取图像宽高进行边界截断
   * @param tf_map_to_cam 世界/地图坐标系到相机坐标系的变换矩阵
   * @param intrinsics 相机内参
   * @param save_folder 标签保存的目录 (例如 label_2)
   */
  void GenerateKITTILabel(int64_t timestamp, const cv::Mat& fig,
                          const Eigen::Matrix4f& tf_map_to_cam,
                          const std::vector<double>& intrinsics,
                          const std::string& save_folder,
                          const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_cam);
  /**
   * @brief 生成语义掩码
   * @param current_scan_depth 当前帧雷达生成的深度图 (用于判断动态遮挡)
   * @param intrinsics 相机内参
   * @param tf_map_to_cam 世界坐标系到相机坐标系的变换矩阵
   * @param out_semantic_mask 输出的语义掩码 (单通道 uint8)
   */
  void GenerateSemanticMask(const cv::Mat& current_scan_depth,
                            const std::vector<double>& intrinsics,
                            const Eigen::Matrix4f& tf_map_to_cam,
                            cv::Mat& out_semantic_mask);
  /**
   * @brief 阻塞等待并获取与指定时间戳同步的点云
   * * @param target_time_ns 目标时间戳（通常是图像时间，纳秒）
   * @param out_cloud 输出的点云指针（需要预先 new 出来）
   * @return true 同步成功，out_cloud 已填充数据
   * @return false 同步超时失败，out_cloud 内容未定义
   */
  bool WaitForSyncedPointCloud(const int64_t target_time_ns,
                               pcl::PointCloud<PointXYZIT>::Ptr& out_cloud);
  /**
   * @brief 根据相机坐标系下的点云和原始图像，生成融合图和16位深度图
   * @param cloud_in_cam 输入：已转换到相机坐标系的点云
   * @param original_image 输入：原始图像 (BGR)
   * @param intrinsic_data 输入：相机内参 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
   * @param out_fused_image 输出：点云投影融合后的彩色图 (BGR)
   * @param out_raw_depth_u16 输出：16位深度图 (单位mm)，用于保存
   * @param out_raw_depth_f32 输出：32位浮点深度图 (单位mm)，用于语义掩码生成
   */
  void GenerateDepthAndFusion(
      const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_cam,
      const cv::Mat& original_image, const std::vector<double>& intrinsic_data,
      cv::Mat& out_fused_image, cv::Mat& out_raw_depth_u16,
      cv::Mat& out_raw_depth_f32);
  /**
   * @brief 生成用于人类观测的可视化图像 (包含彩色 Mask 和 3D Box 投影)
   */
  void GenerateSemanticVisualization(const cv::Mat& original_image,
                                     const cv::Mat& semantic_mask,
                                     const Eigen::Matrix4f& tf_map_to_cam,
                                     const std::vector<double>& intrinsics,
                                     cv::Mat& out_vis_image);

  /**
   * @brief 读取生成的 KITTI txt 标签，逆向解析并在图像上绘制 2D 和 3D
   * 框进行验证
   * @param timestamp 当前帧时间戳 (用于读取 txt)
   * @param original_image 原始图像
   * @param intrinsics 相机内参
   * @param label_folder txt 标签所在的文件夹
   * @param out_vis_image 输出的验证图像
   */
  void VerifyKITTILabel(int64_t timestamp, const cv::Mat& original_image,
                        const std::vector<double>& intrinsics,
                        const std::string& label_folder,
                        cv::Mat& out_vis_image);
  /**
   * @brief 在雷达坐标系点云中验证 KITTI 标签，模拟 训练模型的 Dataloader
   * @param timestamp 时间戳
   * @param cloud_in_lidar_frame 雷达坐标系下的原始点云
   * @param lcam2lidar 左相机到雷达的外参 (用于将标签从 Camera 反算回 LiDAR)
   * @param label_folder txt 标签存放路径
   * @param save_folder 验证 PCD 保存路径
   */
  void VerifyKITTILabelInPointCloud(
      int64_t timestamp,
      const pcl::PointCloud<PointXYZIT>::Ptr& cloud_in_lidar_frame,
      const Eigen::Isometry3d& lcam2lidar, const std::string& label_folder,
      const std::string& save_folder);
  /**
   * @brief
   * @param target_frame 目标坐标系 (例如相机坐标系 kCameraLeftLink)
   * @param target_time  目标时间 (例如图像曝光时间)
   * @param source_frame 源坐标系 (例如雷达坐标系 kLidarLink)
   * @param source_time  源时间 (例如点云采集时间)
   * @param fixed_frame  固定参考系 (例如世界/里程计坐标系 kInsMap)
   * @param fallback_extrinsic 如果插值失败，回退使用的静态外参 (Source ->
   * Target)
   * @param out_tf_mat   输出的 4x4 浮点型变换矩阵
   * @return true        插值成功
   * @return false       插值失败，已使用 fallback_extrinsic 填充 out_tf_mat
   */
  bool GetTimeInterpolatedTransform(const std::string& target_frame,
                                    const rclcpp::Time& target_time,
                                    const std::string& source_frame,
                                    const rclcpp::Time& source_time,
                                    const std::string& fixed_frame,
                                    const Eigen::Isometry3d& fallback_extrinsic,
                                    Eigen::Matrix4f& out_tf_mat);
  struct DeskewPoseSample {
    int64_t time_ns = 0;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  };
  struct DeskewPoseLookupStats {
    size_t cache_sample_count = 0;
    size_t cache_hit_count = 0;
    size_t cache_snapshot_size = 0;
    bool used_pose_cache = false;
    bool fallback_fast = false;
    int64_t cache_wait_ms = 0;
    int64_t cache_oldest_time_ns = 0;
    int64_t cache_newest_time_ns = 0;
    std::string failure_reason = "none";
  };
  pcl::PointCloud<PointXYZIT>::Ptr DeskewOriginalPointCloudToPointXYZIT(
      const pcl::PointCloud<LivoxPoint>::Ptr& input_cloud,
      const std_msgs::msg::Header& header);
  bool LookupDeskewPoseAtTime(const std::string& fixed_frame,
                              const std::string& sensor_frame,
                              const std::string& ins_frame,
                              const rclcpp::Time& query_time,
                              Eigen::Isometry3d& out_pose);
  bool SampleDeskewTrajectory(const std_msgs::msg::Header& header,
                              uint32_t max_raw_timestamp, size_t sample_count,
                              std::vector<DeskewPoseSample>& out_samples,
                              int64_t& out_reference_time_ns,
                              int64_t& out_scan_start_time_ns,
                              int64_t& out_scan_end_time_ns,
                              DeskewPoseLookupStats* out_stats = nullptr);
  bool InterpolatePoseAtTime(const std::vector<DeskewPoseSample>& samples,
                             int64_t query_time_ns,
                             Eigen::Isometry3d& out_pose) const;
  bool LookupDeskewPoseFromCacheAtTime(int64_t query_time_ns,
                                       Eigen::Isometry3d& out_pose);
  bool GetPoseCacheCoverageSnapshot(int64_t& out_oldest_time_ns,
                                    int64_t& out_newest_time_ns) const;
  bool WaitForPoseCacheCoverage(int64_t scan_start_time_ns,
                                int64_t scan_end_time_ns,
                                int64_t timeout_ms,
                                int64_t poll_interval_ms,
                                int64_t& out_waited_ms,
                                int64_t& out_oldest_time_ns,
                                int64_t& out_newest_time_ns) const;
  bool CopyPoseCacheWindowSnapshot(
      int64_t scan_start_time_ns, int64_t scan_end_time_ns,
      std::vector<DeskewPoseSample>& out_snapshot,
      int64_t& out_oldest_time_ns,
      int64_t& out_newest_time_ns) const;
  void PrunePoseCacheLocked(int64_t latest_time_ns);
  bool RunFastDeskew(const pcl::PointCloud<LivoxPoint>::Ptr& input_cloud,
                     const std_msgs::msg::Header& header,
                     uint32_t max_raw_timestamp,
                     const Eigen::Isometry3d& reference_pose,
                     int64_t scan_start_time_ns, int64_t scan_end_time_ns,
                     pcl::PointCloud<PointXYZIT>::Ptr& output_cloud,
                     size_t& out_valid_points,
                     DeskewPoseLookupStats* out_stats = nullptr);
  bool UsePoseCacheForAccurateDeskew() const;
  int64_t ComputePointAbsoluteTimeNs(uint32_t raw_timestamp,
                                     const std_msgs::msg::Header& header,
                                     uint32_t max_raw_timestamp) const;
  int64_t ResolveDeskewReferenceTimeNs(int64_t header_stamp_ns,
                                       int64_t scan_start_time_ns,
                                       int64_t scan_end_time_ns) const;
  void PublishOrSaveDeskewedPointCloud(
      const pcl::PointCloud<PointXYZIT>::Ptr& deskewed_cloud,
      const std_msgs::msg::Header& header);
  void GenerateCameraExtrinsic(int64_t timestamp, int width, int height,
                               const Eigen::Matrix4f& tf_map_to_lidar);
  bool GetMapToCameraAndLidarTf(const rclcpp::Time& img_time,
                                Eigen::Matrix4f& out_tf_map_to_cam,
                                Eigen::Matrix4f& out_tf_map_to_lidar);
  template <typename T>
  typename pcl::PointCloud<T>::Ptr ConvertMsgToPCLPointCloud(
      const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 3. 将 ROS PointCloud2 转换为 PCL PointCloud
    typename pcl::PointCloud<T>::Ptr pcl_cloud(new pcl::PointCloud<T>);
    pcl::fromROSMsg(*msg, *pcl_cloud);
    if (pcl_cloud->empty()) {
      ERROR("Converted PCL point cloud is empty!");
      return nullptr;
    }
    return pcl_cloud;
  }

  template <typename PointT>
  void AccumulatePointCloudBack(
      const typename pcl::PointCloud<PointT>::Ptr& input_cloud,
      std::deque<typename pcl::PointCloud<PointT>::Ptr>& cloud_queue,
      const std_msgs::msg::Header& header) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (!input_cloud || input_cloud->empty()) {
      return;
    }
    // ================= Step 1: 准备变换 =================
    std::string fixed_frame = ConstValue::kInsMap;
    std::string sensor_frame = header.frame_id;
    std::string ins_frame = ConstValue::kInsLink;
    if (!tf_buffer_) {
      return;
    }
    Eigen::Matrix4f transform_lidar_to_map = Eigen::Matrix4f::Identity();
    bool has_transform = false;
    try {
      geometry_msgs::msg::TransformStamped tf_msg =
          tf_buffer_->lookupTransform(fixed_frame, sensor_frame, header.stamp,
                                      rclcpp::Duration::from_seconds(0.02));

      Eigen::Affine3d affine = tf2::transformToEigen(tf_msg);
      transform_lidar_to_map = affine.matrix().cast<float>();
      has_transform = true;
    } catch (tf2::TransformException& ex) {
      try {
        geometry_msgs::msg::TransformStamped tf_ins_msg =
            tf_buffer_->lookupTransform(fixed_frame, ins_frame, header.stamp,
                                        rclcpp::Duration::from_seconds(0.05));
        auto end_time1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time1 - start_time);
        double ms1 = duration1.count() / 1000.0;
        INFO(
            "AccumulatePointCloud From Begin to Finish Step 1 lookupTransform "
            "INSmap and Inslink took {} ms",
            ms1);

        Eigen::Isometry3d iso_ins_to_map = tf2::transformToEigen(tf_ins_msg);

        // 使用我们在 CheckTfCallback 中维护的 Lidar -> InsLink 外参
        // (包含默认值降级) T_lidar_to_map = T_ins_to_map * T_lidar_to_ins
        Eigen::Isometry3d iso_lidar_to_map =
            iso_ins_to_map * annotation_status_.isometry_lidar2ins;

        transform_lidar_to_map = iso_lidar_to_map.matrix().cast<float>();
        has_transform = true;

        // 打印降级警告，每 5 秒最多打印一次，防止刷屏
        THROTTLEWARN(5,
                     "Direct TF {}->{} failed. Fallback: Map->InsLink(dynamic) "
                     "* Lidar->InsLink(static/default) used.",
                     sensor_frame, fixed_frame);
      } catch (tf2::TransformException& ex2) {
        // 如果连定位 TF 都查不到，那就真的无法进行运动补偿了
        THROTTLEWARN(5,
                     "Accumulation failed: Both direct and fallback TF lookup "
                     "failed. Error: {}",
                     ex2.what());
        return;
      }
    }
    auto end_time1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time1 - start_time);
    double ms1 = duration1.count() / 1000.0;
    INFO("AccumulatePointCloud From Begin to Finish Step 1 took {} ms", ms1);
    // ================= Step 2: 转换到 Map 帧并入队 =================
    if (has_transform) {
      typename pcl::PointCloud<PointT>::Ptr cloud_in_map(
          new pcl::PointCloud<PointT>());
      pcl::transformPointCloud(*input_cloud, *cloud_in_map,
                               transform_lidar_to_map);

      // 【修复1】给入队的点云赋予正确的时间戳和 frame_id，保持与 Front 一致
      cloud_in_map->header.stamp =
          rclcpp::Time(header.stamp).nanoseconds() / 1000;  // PCL用微秒
      cloud_in_map->header.frame_id = fixed_frame;

      cloud_queue.push_back(cloud_in_map);
      auto end_time2 = std::chrono::high_resolution_clock::now();
      auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time2 - start_time);
      double ms2 = duration2.count() / 1000.0;
      INFO("AccumulatePointCloud From Begin to Finish Step 2 took {} ms", ms2);
    }
    // ================= Step 3: 维护滑动窗口 =================
    while (cloud_queue.size() >
           static_cast<size_t>(node_config_.target_accumulate_pc_num)) {
      cloud_queue.pop_front();
    }

    auto end_time3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time3 - start_time);
    double ms3 = duration3.count() / 1000.0;
    INFO("AccumulatePointCloud From Begin to Finish Step 3 took {} ms", ms3);
    // ================= Step 4: 拼接并转换回当前帧并注入时间戳
    // =================
    if (!cloud_queue.empty()) {
      // 创建新的 5 维点云用于存放最终结果
      pcl::PointCloud<PointXYZIT>::Ptr final_cloud_local(
          new pcl::PointCloud<PointXYZIT>());

      // 提取最新帧（目标帧）的绝对时间（秒）
      double target_time_s = rclcpp::Time(header.stamp).nanoseconds() / 1e9;
      float oldest_age_s = 0.0f;
      float newest_age_s = std::numeric_limits<float>::max();

      // 预分配内存，避免频繁扩容
      size_t total_points = 0;
      for (const auto& frame : cloud_queue) total_points += frame->size();
      final_cloud_local->reserve(total_points);

      // 计算从 Map 到当前雷达系的逆变换
      Eigen::Matrix4f transform_map_to_current_lidar =
          transform_lidar_to_map.inverse();

      for (const auto& frame : cloud_queue) {
        // 计算当前历史帧的相对年龄
        // PCL 的 header.stamp 存储的是微秒，需转为纳秒再转秒
        double frame_time_s = (frame->header.stamp * 1000.0) / 1e9;
        float age = static_cast<float>(target_time_s - frame_time_s);

        // 容错处理：防止由于系统时间抖动出现极小的负数
        if (age < 0.0f) age = 0.0f;
        oldest_age_s = std::max(oldest_age_s, age);
        newest_age_s = std::min(newest_age_s, age);

        // 将该帧点云从 Map 转到 最新一帧的雷达坐标系
        typename pcl::PointCloud<PointT>::Ptr transformed_frame(
            new pcl::PointCloud<PointT>());
        pcl::transformPointCloud(*frame, *transformed_frame,
                                 transform_map_to_current_lidar);

        // 提取 XYZ 和 Intensity，强行注入 Age，存入 5维点云
        for (const auto& pt : transformed_frame->points) {
          PointXYZIT pt_5d;
          pt_5d.x = pt.x;
          pt_5d.y = pt.y;
          pt_5d.z = pt.z;
          pt_5d.intensity = pt.intensity;
          pt_5d.timestamp = age;  // 注入相对时间差（如 0.0, 0.1, 0.2 ...）
          final_cloud_local->push_back(pt_5d);
        }
      }

      auto end_time4 = std::chrono::high_resolution_clock::now();
      double ms4 = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time4 - start_time)
                       .count() /
                   1000.0;
      INFO("AccumulatePointCloud Step 4 (Merge & Age Injection) took {:.2f} ms",
           ms4);

      // 更新全局缓存，供图像回调使用（统一化处理）
      uint64_t current_stamp_us = cloud_queue.back()->header.stamp;
      final_cloud_local->header.stamp = current_stamp_us;
      final_cloud_local->header.frame_id = sensor_frame;

      {
        std::lock_guard<std::mutex> lock(locks_.mutex_cache_cloud);
        annotation_status_.latest_accumulated_cloud =
            final_cloud_local;  // 统一覆盖
        annotation_status_.latest_cloud_timestamp = header.stamp;
        const int64_t snapshot_time_ns = rclcpp::Time(header.stamp).nanoseconds();
        AccumulatedCloudSnapshot snapshot;
        snapshot.timestamp_ns = snapshot_time_ns;
        snapshot.cloud.reset(new pcl::PointCloud<PointXYZIT>(*final_cloud_local));
        annotation_status_.accumulated_cloud_history.push_back(
            std::move(snapshot));
        while (!annotation_status_.accumulated_cloud_history.empty() &&
               snapshot_time_ns -
                       annotation_status_.accumulated_cloud_history.front()
                           .timestamp_ns >
                   ConstValue::kCloudHistoryRetentionNs) {
          annotation_status_.accumulated_cloud_history.pop_front();
        }
        while (annotation_status_.accumulated_cloud_history.size() >
               ConstValue::kAccumulatedCloudHistoryLimit) {
          annotation_status_.accumulated_cloud_history.pop_front();
        }
      }
      pending_image_task_cv_.notify_one();
      if (newest_age_s == std::numeric_limits<float>::max()) {
        newest_age_s = 0.0f;
      }
      THROTTLEINFO(
          1,
          "Accumulated cache snapshot. input_header_ns={} final_header_us={} "
          "queue_size={} final_cloud_size={} oldest_age_s={:.3f} "
          "newest_age_s={:.3f}",
          rclcpp::Time(header.stamp).nanoseconds(), current_stamp_us,
          cloud_queue.size(), final_cloud_local->size(), oldest_age_s,
          newest_age_s);

      auto end_time5 = std::chrono::high_resolution_clock::now();
      double ms5 = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_time5 - start_time)
                       .count() /
                   1000.0;
      INFO("AccumulatePointCloud Step 4 Update cache took {:.2f} ms", ms5);

      // ================= Step 5: 发布 =================
      if (accumulated_pc_pub_) {
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*final_cloud_local,
                      output_msg);  // PCL 会自动打包 5 个维度
        output_msg.header = header;
        accumulated_pc_pub_->publish(output_msg);
      }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    double ms = duration.count() / 1000.0;
    INFO("AccumulatePointCloud (Base: Latest) took {} ms", ms);
  }

  // 工具函数
  template <typename T>
  T get_param_value_form_params_server(const std::string& name,
                                       const T& default_val) {
    if (!node_->has_parameter(name)) {
      node_->declare_parameter<T>(name, default_val);
    }
    T value = node_->get_parameter(name).get_value<T>();
    // 如果是基本类型或字符串，正常打印值；如果是 vector
    // 等复杂类型，只打印变量名
    if constexpr (std::is_fundamental_v<T> || std::is_same_v<T, std::string>) {
      INFO("Parameter {} is set to {}", name, value);
    } else {
      INFO("Parameter {} (complex type) has been loaded.", name);
    }

    return value;
  }
  std::string get_current_localtime_str();
  void print_matrix_as_numpy(const std::vector<double>& data, const int rows,
                             const int cols);
  void print_transform_eigen_isometry(const std::string& name,
                                      const Eigen::Isometry3d& transform);
};

}  // namespace automatic_annotation

// 2. 【关键】向 PCL 系统注册必须在全局作用域
// 注意：第一个参数要写完整的命名空间路径 automatic_annotation::LivoxPoint
POINT_CLOUD_REGISTER_POINT_STRUCT(
    automatic_annotation::LivoxPoint,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
        uint8_t, tag, tag)(uint8_t, line, line)(uint32_t, timestamp, timestamp))

// 注册宏必须放在全局命名空间
POINT_CLOUD_REGISTER_POINT_STRUCT(
    automatic_annotation::UndistortedLivoxPoint,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity))

// 注册宏必须放在全局命名空间
POINT_CLOUD_REGISTER_POINT_STRUCT(automatic_annotation::PointXYZIT,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, timestamp, timestamp))

#endif  // AUTOMATIC_ANNOTATION_GEN_PROMPT_POINT_HPP
