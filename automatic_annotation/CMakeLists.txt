cmake_minimum_required(VERSION 3.8)
project(automatic_annotation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找 ROS 依赖和标准 CMake 包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(common_utils REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge_with_opencv411 REQUIRED)
find_package(OpenCV 4.11 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

# 2. 【关键修改】查找 TinyXML2 (改用 PkgConfig)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TINYXML2 REQUIRED tinyxml2)

include_directories(
  include
)

add_executable(automatic_annotation_node 
  src/automatic_annotation_node.cpp 
  src/image_matching_utils.cpp
  src/static_track_id_utils.cpp
  src/gen_prompt_point.cpp
)

# 3. 链接 ROS 相关的依赖 (ament 依赖)
ament_target_dependencies(automatic_annotation_node
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
  # 注意：这里移除了 PCL, OpenCV 和 tinyxml2，我们在下面手动链接
)

# 4. 手动链接非 Ament 依赖 (TinyXML2, PCL, OpenCV)
target_include_directories(automatic_annotation_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TINYXML2_INCLUDE_DIRS}  # 添加 TinyXML2 头文件路径
)

target_link_libraries(automatic_annotation_node
  ${OpenCV_LIBS}            # OpenCV 库
  ${PCL_LIBRARIES}          # PCL 库
  ${TINYXML2_LIBRARIES}     # 添加 TinyXML2 库
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(
    test_image_matching_utils
    test/test_image_matching_utils.cpp
    src/image_matching_utils.cpp
  )
  if(TARGET test_image_matching_utils)
    target_include_directories(test_image_matching_utils PRIVATE include)
  endif()

  ament_add_gtest(
    test_static_track_id_utils
    test/test_static_track_id_utils.cpp
    src/static_track_id_utils.cpp
  )
  if(TARGET test_static_track_id_utils)
    target_include_directories(test_static_track_id_utils PRIVATE include)
  endif()
endif()

install(TARGETS automatic_annotation_node 
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
