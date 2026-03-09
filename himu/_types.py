"""Data types for the HiMu frame selection pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class FrameSelectionResult:
    """Result of HiMu frame selection.

    Attributes:
        frame_indices: Sorted indices of selected frames.
        timestamps: Timestamps (seconds) of selected frames.
        scores: Satisfaction score for each selected frame.
        truth_curve: Full per-frame satisfaction curve (all frames).
        tree: The logic tree used for selection.
        num_frames: Total number of frames in the video.
        fps: Frame rate used for extraction.
        best_frame_idx: Index of the highest-scoring frame.
        best_timestamp: Timestamp of the highest-scoring frame.
        best_score: Score of the highest-scoring frame.
    """

    frame_indices: np.ndarray
    timestamps: np.ndarray
    scores: np.ndarray
    truth_curve: np.ndarray
    tree: Dict[str, Any]
    num_frames: int
    fps: float
    best_frame_idx: int
    best_timestamp: float
    best_score: float
