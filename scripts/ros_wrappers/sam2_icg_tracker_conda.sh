#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chris_articulation_ros

# Let ROS/OpenCV use system libraries
# Do NOT override LD_LIBRARY_PATH for GDAL
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5

source /opt/ros/noetic/setup.bash
source ~/chris_ws/articulation_tracker_ws/devel/setup.bash

exec python $(rospack find articulation_tracker)/scripts/sam2_icg_tracker_ros.py "$@"