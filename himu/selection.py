"""Frame selection strategies for HiMu."""

import math

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class FrameSelectionStrategy(ABC):
    """Abstract base class for frame selection strategies."""

    @abstractmethod
    def select(self, truth_curve: np.ndarray, top_k: int, fps: float = 1.0) -> np.ndarray:
        """
        Select top-K frames from truth curve.

        Args:
            truth_curve: Per-frame truth scores (num_frames,)
            top_k: Number of frames to select
            fps: Frames per second (used for time-based constraints)

        Returns:
            Array of selected frame indices
        """
        pass


class ScoreRankedSelection(FrameSelectionStrategy):
    """
    Simple score-ranked selection.

    Select frames with the highest truth scores. Optionally enforce minimum
    temporal gap between selected frames and/or a sliding-window density
    constraint to prevent clustering.
    """

    def __init__(
        self,
        min_gap: Optional[int] = None,
        max_frames_per_window: Optional[int] = None,
        window_seconds: float = 4.0,
    ):
        self.min_gap = min_gap
        self.max_frames_per_window = max_frames_per_window
        self.window_seconds = window_seconds

    def select(self, truth_curve: np.ndarray, top_k: int, fps: float = 1.0) -> np.ndarray:
        sorted_indices = np.argsort(truth_curve)[::-1]

        if self.min_gap is None and self.max_frames_per_window is None:
            return sorted_indices[:top_k]

        window_frames = self.window_seconds * fps
        selected = []
        skipped = []

        for idx in sorted_indices:
            if len(selected) >= top_k:
                break

            # Check min_gap constraint
            if self.min_gap is not None:
                if selected and not all(abs(idx - s) >= self.min_gap for s in selected):
                    skipped.append(idx)
                    continue

            # Check window density constraint
            if self.max_frames_per_window is not None:
                nearby = sum(1 for s in selected if abs(idx - s) < window_frames)
                if nearby >= self.max_frames_per_window:
                    skipped.append(idx)
                    continue

            selected.append(idx)

        # Fallback: fill remaining budget from skipped frames (by score order)
        for idx in skipped:
            if len(selected) >= top_k:
                break
            selected.append(idx)

        return np.array(selected)


class PASSSelection(FrameSelectionStrategy):
    """Peak-And-Spread Selection (PASS).

    3-phase algorithm with adaptive prominence:
      Phase 1: Detect peaks with adaptive prominence (halving cascade fallback),
               pick top int(sqrt(K)) peaks.
      Phase 2: For each peak, add int(sqrt(K)/2) - 1 best neighboring frames
               within a +-surround_half window.
      Phase 3: Greedily fill remaining budget from all unselected frames by score.
    """

    def __init__(self, prominence_ratio: float = 0.05, min_prominence: float = 1e-6):
        self.prominence_ratio = prominence_ratio
        self.min_prominence = min_prominence

    def select(self, truth_curve: np.ndarray, top_k: int, fps: float = 1.0) -> np.ndarray:
        from scipy.signal import find_peaks as scipy_find_peaks

        num_frames = len(truth_curve)
        top_k = min(top_k, num_frames)

        # Small budget: just return top-scored frames
        if top_k < 8:
            return np.argsort(truth_curve)[::-1][:top_k]

        sqrt_n = math.sqrt(top_k)
        num_peaks = max(1, int(sqrt_n))
        extra_per_peak = max(0, int(sqrt_n / 2) - 1)
        peak_distance = max(1, int(sqrt_n * fps))
        surround_half = max(1, int(sqrt_n / 2 * fps))

        # --- Adaptive prominence with halving cascade ---
        curve_range = float(truth_curve.max() - truth_curve.min())
        prominence = max(self.min_prominence, self.prominence_ratio * curve_range)

        peaks = np.array([], dtype=int)
        for _ in range(5):
            peaks, _ = scipy_find_peaks(
                truth_curve, distance=peak_distance, prominence=prominence
            )
            if len(peaks) > 0:
                break
            prominence *= 0.5

        # Rank detected peaks by score descending
        if len(peaks) > 0:
            peak_order = np.argsort(truth_curve[peaks])[::-1]
            ranked_peaks = peaks[peak_order]
        else:
            ranked_peaks = np.argsort(truth_curve)[::-1]

        selected_set = set()
        selected_list = []

        def _add(idx):
            idx = int(idx)
            if idx not in selected_set:
                selected_set.add(idx)
                selected_list.append(idx)

        # Phase 1: Pick top num_peaks peaks
        for p in ranked_peaks[:num_peaks]:
            _add(int(p))

        # Phase 2: Surround sampling
        peak_list = list(selected_list)
        for peak_idx in peak_list:
            lo = max(0, peak_idx - surround_half)
            hi = min(num_frames, peak_idx + surround_half + 1)
            candidates = [i for i in range(lo, hi) if i not in selected_set]
            candidates.sort(key=lambda i: truth_curve[i], reverse=True)
            for c in candidates[:extra_per_peak]:
                _add(c)
                if len(selected_list) >= top_k:
                    break
            if len(selected_list) >= top_k:
                break

        # Phase 3: Greedy fill
        if len(selected_list) < top_k:
            all_by_score = np.argsort(truth_curve)[::-1]
            for idx in all_by_score:
                _add(int(idx))
                if len(selected_list) >= top_k:
                    break

        return np.array(selected_list[:top_k])


def create_selector(
    mode: str = "pass",
    min_gap: Optional[int] = None,
    max_frames_per_window: Optional[int] = None,
    window_seconds: float = 4.0,
    peak_prominence: float = 0.01,
) -> FrameSelectionStrategy:
    """
    Factory function to create frame selector.

    Args:
        mode: Selection algorithm -- "pass" or "score_ranked"
        min_gap: Minimum temporal gap between selected frames (score_ranked only)
        max_frames_per_window: Max frames in any time window (score_ranked only)
        window_seconds: Window size for density constraint (score_ranked only)
        peak_prominence: Prominence ratio for adaptive peak detection (pass only)
    """
    if mode == "pass":
        return PASSSelection(prominence_ratio=peak_prominence)
    elif mode == "score_ranked":
        return ScoreRankedSelection(
            min_gap=min_gap,
            max_frames_per_window=max_frames_per_window,
            window_seconds=window_seconds,
        )
    else:
        raise ValueError(f"Unknown selection mode: {mode}. Available: 'pass', 'score_ranked'")
