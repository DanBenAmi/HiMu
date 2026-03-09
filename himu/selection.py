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


class PeakBasedSelection(FrameSelectionStrategy):
    """Peak-based hierarchical frame selection.

    3-phase algorithm derived from budget n = top_k:
      Phase 1: find_peaks with distance constraint, pick top sqrt(n) peaks.
      Phase 2: For each peak, pick best sqrt(n)/2 frames within +-window.
      Phase 3: Greedily fill remaining budget from all unselected frames by score.
    """

    def __init__(self, prominence: float = 0.01):
        self.prominence = prominence

    def select(self, truth_curve: np.ndarray, top_k: int, fps: float = 1.0) -> np.ndarray:
        from scipy.signal import find_peaks as scipy_find_peaks

        n = top_k
        sqrt_n = math.sqrt(n)
        num_peaks = max(1, int(sqrt_n))
        extra_per_peak = max(1, int(sqrt_n) // 2)
        peak_distance = max(1, int(sqrt_n * fps))
        surround_half = max(1, int(sqrt_n / 2 * fps))

        num_frames = len(truth_curve)
        selected_set = set()
        selected_list = []

        def _add(idx):
            if idx not in selected_set:
                selected_set.add(idx)
                selected_list.append(idx)

        # Phase 1: Peak detection
        peaks, _ = scipy_find_peaks(
            truth_curve, distance=peak_distance, prominence=self.prominence
        )

        if len(peaks) > 0:
            peak_scores = truth_curve[peaks]
            sorted_peak_idx = np.argsort(peak_scores)[::-1][:num_peaks]
            top_peaks = peaks[sorted_peak_idx]
            for p in top_peaks:
                _add(int(p))
        else:
            sorted_all = np.argsort(truth_curve)[::-1]
            for i in range(min(num_peaks, len(sorted_all))):
                _add(int(sorted_all[i]))

        # Phase 2: Surround sampling
        peak_list = list(selected_list)
        for peak_idx in peak_list:
            window_start = max(0, peak_idx - surround_half)
            window_end = min(num_frames, peak_idx + surround_half + 1)

            candidates = [i for i in range(window_start, window_end) if i not in selected_set]
            candidates.sort(key=lambda i: truth_curve[i], reverse=True)
            for c in candidates[:extra_per_peak]:
                _add(c)
                if len(selected_list) >= top_k:
                    break
            if len(selected_list) >= top_k:
                break

        # Phase 3: Greedy fill
        if len(selected_list) < top_k:
            sorted_all = np.argsort(truth_curve)[::-1]
            for idx in sorted_all:
                idx = int(idx)
                if idx not in selected_set:
                    _add(idx)
                    if len(selected_list) >= top_k:
                        break

        return np.array(selected_list[:top_k])


class TieredPeakSelection(FrameSelectionStrategy):
    """Tiered peak-based frame selection with adaptive prominence (peak_nms_v2).

    Improvements over PeakBasedSelection:
      1. Adaptive prominence: uses a fraction of the curve's dynamic range
         instead of an absolute threshold, with a halving cascade fallback.
      2. Tiered interleaving: output is ordered so that any prefix of length
         8, 16, 32, 64, ... contains a well-distributed mix of peaks,
         surround frames, and high-scoring fill frames.
    """

    PEAKS_PER_TIER = 2
    SURROUND_PER_PEAK = 2

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

        # Precompute surround candidates for each peak
        peak_surrounds = {}
        for p in ranked_peaks:
            lo = max(0, int(p) - surround_half)
            hi = min(num_frames, int(p) + surround_half + 1)
            cands = [i for i in range(lo, hi) if i != int(p)]
            cands.sort(key=lambda i: truth_curve[i], reverse=True)
            peak_surrounds[int(p)] = cands

        # Pre-sort all frames by score for fill phases
        all_by_score = np.argsort(truth_curve)[::-1]

        # --- Tiered interleaving ---
        selected_set = set()
        selected_list = []

        def _add(idx):
            idx = int(idx)
            if idx not in selected_set:
                selected_set.add(idx)
                selected_list.append(idx)

        tiers = self._compute_tier_schedule(top_k)
        peak_cursor = 0
        surround_cursors = {}

        for tier_budget, num_new_peaks in tiers:
            # Emit new peaks + their surrounds
            added = 0
            while added < num_new_peaks and peak_cursor < len(ranked_peaks):
                p = int(ranked_peaks[peak_cursor])
                peak_cursor += 1
                if p in selected_set:
                    continue

                _add(p)
                added += 1
                surround_cursors[p] = 0

                # Add surrounds for this peak
                cands = peak_surrounds.get(p, [])
                emitted = 0
                cursor = 0
                while emitted < self.SURROUND_PER_PEAK and cursor < len(cands):
                    s = cands[cursor]
                    cursor += 1
                    if s not in selected_set:
                        _add(s)
                        emitted += 1
                surround_cursors[p] = cursor

                if len(selected_list) >= tier_budget:
                    break

            # Fill remaining tier budget with global best-scored unselected frames
            if len(selected_list) < tier_budget:
                for idx in all_by_score:
                    if len(selected_list) >= tier_budget:
                        break
                    _add(int(idx))

        return np.array(selected_list[:top_k])

    @staticmethod
    def _compute_tier_schedule(top_k: int):
        """Compute tier schedule: list of (tier_budget, num_new_peaks)."""
        tiers = []
        budget = 8
        while budget <= top_k:
            tiers.append((min(budget, top_k), 2))
            budget *= 2

        if not tiers or tiers[-1][0] < top_k:
            tiers.append((top_k, 2))

        return tiers


def create_selector(
    mode: str = "score_ranked",
    min_gap: Optional[int] = None,
    max_frames_per_window: Optional[int] = None,
    window_seconds: float = 4.0,
    peak_prominence: float = 0.01,
) -> FrameSelectionStrategy:
    """
    Factory function to create frame selector.

    Args:
        mode: Selection algorithm -- "score_ranked", "peak_nms", or "peak_nms_v2"
        min_gap: Minimum temporal gap between selected frames (score_ranked only)
        max_frames_per_window: Max frames in any time window (score_ranked only)
        window_seconds: Window size for density constraint (score_ranked only)
        peak_prominence: peak_nms: absolute; peak_nms_v2: ratio of curve range
    """
    if mode == "peak_nms":
        return PeakBasedSelection(prominence=peak_prominence)
    elif mode == "peak_nms_v2":
        return TieredPeakSelection(prominence_ratio=peak_prominence)
    elif mode == "score_ranked":
        return ScoreRankedSelection(
            min_gap=min_gap,
            max_frames_per_window=max_frames_per_window,
            window_seconds=window_seconds,
        )
    else:
        raise ValueError(f"Unknown selection mode: {mode}. Available: 'score_ranked', 'peak_nms', 'peak_nms_v2'")
