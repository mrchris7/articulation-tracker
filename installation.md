# Installation (Direct API Mode)

This project uses the Python API (Pyzed) of the ZED camera and Python bindings for direct communication between the articulation tracker and the ICG tracker.

## Requirements
- Python >=3.11
- CUDA Support
- ZED SDK

## 1. Create and activate a Conda environment
```bash
conda create -n articulation python=3.11 -y
conda activate articulation
```

## 2. Install PyTorch and TorchVision
Install PyTorch>=2.3.1 and TorchVision>=0.18.1 by following the official instructions [here](https://pytorch.org/get-started/locally/).


## 3. Install SAM-2
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

## 4. Install ICG (with Python bindings)
Next, install a slightly modified version of the [ICG](https://github.com/DLR-RM/3DObjectTracking/tree/master/ICG) pose estimation framework.
This fork extends the original implementation by adding Python bindings enabling seamless integration with our Python-based codebase.
 

Clone the custom branch that includes the Python bindings:
```bash
git clone --branch pybindings https://github.com/mrchris7/pose-estimation.git
cd pose-estimation
```

Install system dependencies:
```bash
sudo apt-get install libglfw3-dev libeigen3-dev libglew-dev
```

Build the project:
```bash
cd build
cmake ..
cmake --build .
```

## 5. Install the Articulation Tracker

Clone this repository:
```bash
https://github.com/mrchris7/articulation-tracker.git
cd articulation-tracker
```

Add additional Python dependencies:
```bash
pip install open3d
```


# Usage

## Live ZED Camera

```bash
python sam2_icg_tracker.py \
    --camera_source zed \
    --camera_metafile path/to/zed_color.yaml \
    --body_metafile path/to/handle.yaml \
    --cad_model_path path/to/handle.obj \
    --progress_mode rotation \
    --goal_rotation 90 \
    --render_model
```

## Simulated ZED Camera (Recorded Data)
```bash
python sam2_icg_tracker.py \
    --camera_source recorded \
    --recorded_root path/to/zed_stream \
    --camera_metafile path/to/zed_color.yaml \
    --body_metafile path/to/handle.yaml \
    --cad_model_path path/to/handle.obj \
    --progress_mode rotation \
    --goal_rotation 90 \
    --render_model
```