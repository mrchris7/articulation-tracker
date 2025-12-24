cmake_minimum_required(VERSION 3.0.2)
project(articulation_tracker)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_generation
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs cv_bridge image_transport message_runtime
)

catkin_install_python(PROGRAMS
  scripts/sam2_icg_tracker_ros.py
  scripts/zed_simulator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

