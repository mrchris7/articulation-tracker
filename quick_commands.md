
# Setup
cd /home/hydra/chris_ws/articulation_tracker_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash


# Tracker (with simulated camera)
roslaunch articulation_tracker sam2_icg_tracking.launch use_simulated_camera:=true data_root:=/home/hydra/chris_ws/articulated_ws/src/data/scene0010_00


# Tracker (with live camera)
roslaunch zed_wrapper zed2i.launch
roslaunch articulation_tracker sam2_icg_tracking.launch use_simulated_camera:=false 


# Tracker (with live camera + fixed initial pose)
roslaunch zed_wrapper zed2i.launch
roslaunch articulation_tracker sam2_icg_tracking.launch use_simulated_camera:=false detector_yaml:=/home/hydra/chris_ws/articulation_tracker_ws/src/articulation-tracker/meta_files/detector_static_mar9.yaml


# Monitor (example usage)
roslaunch articulation_tracker monitor.launch

# Pose Configurator
roslaunch articulation_tracker pose_configurator.launch use_simulated_camera:=false detector_yaml:=/home/hydra/chris_ws/articulation_tracker_ws/src/articulation-tracker/meta_files/detector.yaml


# Setup robot
. ~/ros_master_159.sh
source /home/hydra/tiago_core/catkin_ws/devel/setup.bash 
