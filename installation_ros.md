# Installation (ROS-Integrated Mode)

This method uses ROS for communication between the ZED camera (zed-ros-wrapper), the articulation tracker and the ICG tracker.

## Requirements
- Python 3.8
- CUDA Support
- ROS Noetic
- ZED SDK
- zed-ros-wrapper


## 1. Set up Catkin Workspace
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

## 2. Create and activate a Conda environment
```bash
conda create -n articulation python=3.8 -y
conda activate articulation
```

## 3. Install ROS packages
```bash
conda install -c conda-forge ros-noetic-rospy ros-noetic-std-msgs ros-noetic-geometry-msgs
conda install -c conda-forge rospkg catkin_pkg pyyaml empy setuptools
```


## 4. Set PYTHONPATH for ROS
```bash
export PYTHONPATH=$PYTHONPATH:/root/miniconda3/envs/articulation/lib/python3.8/site-packages
```
(Adjust the path to match your conda installation location)


## 5. Install PyTorch and TorchVision
Install PyTorch>=2.3.1 and TorchVision>=0.18.1 by following the official instructions [here](https://pytorch.org/get-started/locally/).
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
```


## 6. Install SAM-2
Clone the [SAM-2 repository](https://github.com/facebookresearch/sam2):
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 
```

Install required Python dependencies:
```bash
pip install matplotlib opencv-python opencv-contrib-python
```

Install SAM-2 in editable mode:
```bash
pip install -e .
```
If there are dependency conflicts (e.g. PyTorch version warnings), reinstall using the flag ```--no-deps```.

Download SAM-2 model checkpoints:
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

## 7. Setup ICG
Next, get a slightly modified version of the [ICG](https://github.com/DLR-RM/3DObjectTracking/tree/master/ICG) pose estimation framework.
This fork extends the original implementation by adding the ROS commuication for the articulation tracker package.
 

Clone the custom branch that includes ROS communication:
```bash
git clone --branch articulation_tracker_ros https://github.com/mrchris7/pose-estimation.git
```

Install system dependencies:
```bash
sudo apt-get install libglfw3-dev libeigen3-dev libglew-dev
```


## 8. Setup the Articulation Tracker

Clone this ROS package:
```bash
https://github.com/mrchris7/articulation-tracker.git
cd articulation-tracker
```

Add additional Python dependencies:
```bash
pip install open3d hydra-core iopath
```


Make the nodes executable:
```bash
chmod +x src/articulation_tracker/scripts/*.py
```


# Usage

## Live ZED Camera

1. First, launch the ZED camera node (from zed-ros-wrapper):
```bash
roslaunch zed_wrapper zed.launch
```

2. Then launch the tracking system:
```bash
roslaunch articulation_tracker sam2_icg_tracking.launch \
    use_simulated_camera:=false \
    camera_metafile:=path/to/zed_color.yaml \
    body_metafile:=path/to/handle.yaml
```


## Simulated ZED Camera (Recorded Data)

Launch all nodes:
```bash
roslaunch articulation_tracker sam2_icg_tracking.launch \
    use_simulated_camera:=true \
    data_root:=/path/to/zed_stream \
    camera_metafile:=path/to/zed_color.yaml \
    body_metafile:=path/to/handle.yaml
```
