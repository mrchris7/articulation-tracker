"""
Offline streamer that mimics the ZED camera interface but reads frames from disk.
"""
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


class RecordedZEDStreamer:
    """
    Simple RGB-D streamer that replays pre-recorded frames from disk.
    Attempts to mirror the subset of the ZEDStreamer API used by the tracker:
      - set_params / start / get_frame / get_status / stop
      - `intrinsic`, `width`, and `height` attributes
    """

    def __init__(
        self,
        color_dir: str,
        depth_dir: str,
        intrinsic_path: str = None,
        depth_scale: float = 1000.0,
    ) -> None:
        self.color_dir = color_dir
        self.depth_dir = depth_dir
        self.intrinsic_path = intrinsic_path
        self.depth_scale = depth_scale

        self.started = False
        self.frame_indices: List[int] = []
        self.frame_ptr = 0

        self.width = None
        self.height = None
        self.fps = None
        self.close = 0.2
        self.far = 10.0

        self._color_paths: List[str] = []
        self._depth_paths: List[str] = []
        self._intrinsic_original = None
        self.intrinsic = None

    def set_params(self, **kwargs):
        self.width = kwargs.get("width")
        self.height = kwargs.get("height")
        self.fps = kwargs.get("fps")
        self.close = kwargs.get("close", self.close)
        self.far = kwargs.get("far", self.far)

    def start(self):
        color_files = self._collect_files(self.color_dir, (".jpg", ".png"))
        depth_files = self._collect_files(self.depth_dir, (".png", ".npy"))

        color_map = {self._stem_to_int(path): path for path in color_files}
        depth_map = {self._stem_to_int(path): path for path in depth_files}

        shared_indices = sorted(set(color_map.keys()) & set(depth_map.keys()))
        if not shared_indices:
            raise RuntimeError(
                f"No overlapping frame ids between {self.color_dir} and {self.depth_dir}"
            )

        self.frame_indices = shared_indices
        self._color_paths = [color_map[idx] for idx in self.frame_indices]
        self._depth_paths = [depth_map[idx] for idx in self.frame_indices]
        self.frame_ptr = 0

        sample_image = cv2.imread(self._color_paths[0], cv2.IMREAD_COLOR)
        if sample_image is None:
            raise RuntimeError(f"Failed to read sample color frame from {self._color_paths[0]}")

        original_size = (sample_image.shape[1], sample_image.shape[0])  # (W, H)
        self._intrinsic_original = self._load_intrinsic_matrix(original_size)
        self.intrinsic = self._intrinsic_original.copy()

        # Default width/height align with original resolution if not provided
        if self.width is None:
            self.width = original_size[0]
        if self.height is None:
            self.height = original_size[1]

        # Update intrinsic if requested resolution differs
        if (self.width, self.height) != original_size:
            self.intrinsic = self._resize_intrinsic(
                self._intrinsic_original, original_size, (self.width, self.height)
            )

        self.started = True

    def get_status(self):
        return self.started and self.frame_ptr < len(self.frame_indices)

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.started or self.frame_ptr >= len(self.frame_indices):
            return None, None

        color_path = self._color_paths[self.frame_ptr]
        depth_path = self._depth_paths[self.frame_ptr]
        self.frame_ptr += 1

        color = cv2.imread(color_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if color is None or depth_raw is None:
            return None, None

        # Convert depth to meters
        depth = depth_raw.astype(np.float32)
        if self.depth_scale and self.depth_scale != 0:
            depth /= self.depth_scale

        # Resize if requested
        target_size = (self.width, self.height)
        if target_size != (color.shape[1], color.shape[0]):
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        # Clamp invalid depths
        mask_invalid = (depth < self.close) | (depth > self.far) | np.isnan(depth)
        depth[mask_invalid] = 0.0

        return color, depth

    def stop(self):
        self.started = False

    # ------------------------------------------------------------------
    def _collect_files(self, directory: str, extensions: Tuple[str, ...]) -> List[str]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        if not files:
            raise RuntimeError(f"No frames with extensions {extensions} found in {directory}")
        return files

    def _stem_to_int(self, path: str) -> int:
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem)

    def _load_intrinsic_matrix(self, original_size: Tuple[int, int]) -> np.ndarray:
        if self.intrinsic_path is None:
            raise FileNotFoundError("Intrinsic file path not provided for recorded streamer.")

        with open(self.intrinsic_path, "r", encoding="utf-8") as f:
            rows = [list(map(float, line.strip().split())) for line in f if line.strip()]

        matrix = np.array(rows, dtype=np.float32)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(f"Intrinsic file {self.intrinsic_path} does not contain a valid matrix.")

        return matrix[:3, :3]

    def _resize_intrinsic(
        self, intrinsic: np.ndarray, original_size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> np.ndarray:
        sx = new_size[0] / float(original_size[0])
        sy = new_size[1] / float(original_size[1])
        scaled = intrinsic.copy()
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] *= sx
        scaled[1, 2] *= sy
        return scaled


