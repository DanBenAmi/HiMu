"""Video processing utilities for frame extraction."""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video loading and frame extraction."""

    def __init__(
        self,
        video_path: str,
        fps: float = 1.0,
    ):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract (default: 1 fps)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.target_fps = fps
        self.cap = None
        self.video_fps = None
        self.total_frames = None
        self.duration = None

    def __enter__(self):
        """Context manager entry."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cap is not None:
            self.cap.release()

    def extract_frames(self, max_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract frames at specified FPS.

        Args:
            max_frames: Maximum number of frames to extract (optional)

        Returns:
            Tuple of (frames, timestamps)
            - frames: Array of shape (num_frames, height, width, 3) in RGB
            - timestamps: Array of timestamps in seconds
        """
        if self.cap is None:
            raise RuntimeError("VideoProcessor not initialized. Use as context manager.")

        # Calculate frame interval as float to avoid accumulating rounding errors
        frame_interval = self.video_fps / self.target_fps
        if frame_interval < 1.0:
            frame_interval = 1.0

        frames_list = []
        timestamps_list = []
        frame_count = 0
        next_frame_to_extract = 0.0  # Use float accumulator

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Extract frame when we reach or pass the next extraction point
            if frame_count >= next_frame_to_extract:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp
                timestamp = frame_count / self.video_fps

                frames_list.append(frame_rgb)
                timestamps_list.append(timestamp)

                # Increment the accumulator by the float interval
                next_frame_to_extract += frame_interval

                if max_frames and len(frames_list) >= max_frames:
                    break

            frame_count += 1

        if not frames_list:
            raise RuntimeError("No frames extracted from video")

        frames = np.array(frames_list, dtype=np.uint8)
        timestamps = np.array(timestamps_list, dtype=np.float32)

        return frames, timestamps

    def extract_frames_at_indices(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract only frames at specific indices (memory efficient).

        Uses cv2 seeking to skip unnecessary frames.

        Args:
            indices: Frame indices to extract (0-based, at target FPS).

        Returns:
            Tuple of (frames, timestamps)
        """
        if self.cap is None:
            raise RuntimeError("VideoProcessor not initialized. Use as context manager.")

        frame_interval = self.video_fps / self.target_fps
        if frame_interval < 1.0:
            frame_interval = 1.0

        sorted_indices = sorted(set(indices))

        frames_list = []
        timestamps_list = []

        for target_idx in sorted_indices:
            video_frame_num = int(target_idx * frame_interval)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_num)
            ret, frame = self.cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = video_frame_num / self.video_fps

                frames_list.append(frame_rgb)
                timestamps_list.append(timestamp)
            else:
                log.warning(f"Failed to seek to frame {video_frame_num} (index {target_idx})")

        if not frames_list:
            raise RuntimeError(f"No frames extracted from video at indices {indices}")

        frames = np.array(frames_list, dtype=np.uint8)
        timestamps = np.array(timestamps_list, dtype=np.float32)

        return frames, timestamps

    def get_video_info(self) -> dict:
        """Get video metadata."""
        if self.cap is None:
            raise RuntimeError("VideoProcessor not initialized. Use as context manager.")

        return {
            "path": str(self.video_path),
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }


def format_timestamp(seconds: float) -> str:
    """Format timestamp as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
