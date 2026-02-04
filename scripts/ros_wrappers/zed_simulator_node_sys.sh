#!/bin/bash
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/noetic/setup.bash
source ~/chris_ws/articulation_tracker_ws/devel/setup.bash

exec /usr/bin/python3 \
  $(rospack find articulation_tracker)/scripts/zed_simulator_node.py "$@"