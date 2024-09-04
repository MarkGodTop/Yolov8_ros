cmake_minimum_required(VERSION 3.5.0) # 选择较高的版本
project(plan_and_control)

# 添加编译选项
add_compile_options(-std=c++14)

# 设置OpenCV目录
set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4/)
set(AIRSIM_ROOT /mnt/d/AirSim)

add_subdirectory("${AIRSIM_ROOT}/cmake/rpclib_wrapper" rpclib_wrapper)
add_subdirectory("${AIRSIM_ROOT}/cmake/AirLib" AirLib)
add_subdirectory("${AIRSIM_ROOT}/cmake/MavLinkCom" MavLinkCom)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wstrict-null-sentinel -Wno-unused")
set(CXX_EXP_LIB "-nostdinc++ -I/usr/include/c++/8 -I/usr/include/x86_64-linux-gnu/c++/8 -nodefaultlibs 
-l/usr/lib/x86_64-linux-gnu/libc++.so -l/usr/lib/x86_64-linux-gnu/libc++abi.so
-lm -lc -lgcc_s -lgcc  
-lstdc++fs -fmax-errors=10")

set(RPC_LIB_INCLUDES " ${AIRSIM_ROOT}/external/rpclib/rpclib-2.3.0/include")
set(RPC_LIB rpc) # name of .a file with lib prefix
message(STATUS "found RPC_LIB_INCLUDES=${RPC_LIB_INCLUDES}")
# 查找所需的包
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# 包含目录
include_directories(${JSONCPP_INCLUDE_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})


# Catkin工作空间设置
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  visualization_msgs
  OpenCV
  cv_bridge
  image_transport
  sensor_msgs
  yolov8_ros_msgs
  message_generation
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs visualization_msgs yolov8_ros_msgs
)

# 包含AirSim相关目录
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${AIRSIM_ROOT}/AirLib/deps/eigen3
  ${AIRSIM_ROOT}/AirLib/include
  ${RPC_LIB_INCLUDES}
  ${AIRSIM_ROOT}/MavLinkCom/include
  ${AIRSIM_ROOT}/MavLinkCom/common_utils
)

# 添加可执行文件


add_executable(trajectory_replan_node src/trajectory_replan_node.cpp)
target_link_libraries(trajectory_replan_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
  ${JSONCPP_LIBRARIES}
  AirLib
)