"""Score normalization for HiMu expert outputs."""

from typing import Dict

import numpy as np


class ScoreNormalizer:
    """
    Normalize expert scores to fuzzy truth values in (0, 1).

    Supports multiple normalization methods for ablation studies.
    """

    def __init__(
        self,
        method: str = "robust",
        delta: float = 1e-6,
        gamma: float = 3.0
    ):
        """
        Initialize score normalizer.

        Args:
            method: Normalization method ("robust", "minmax", "zscore")
            delta: Small constant for numerical stability
            gamma: Sigmoid scaling factor for robust/zscore methods
        """
        self.method = method
        self.delta = delta
        self.gamma = gamma

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to fuzzy truth values in (0, 1).

        Args:
            scores: Raw expert scores

        Returns:
            Normalized scores in (0, 1)
        """
        if self.method == "robust":
            return self._robust_normalize(scores)
        elif self.method == "minmax":
            return self._minmax_normalize(scores)
        elif self.method == "zscore":
            return self._zscore_normalize(scores)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _robust_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Robust normalization using median and MAD (Median Absolute Deviation).

        Pipeline:
        1. Robust scaling: (s - median) / (MAD + delta)
        2. Sigmoid projection: sigmoid(gamma * s_normalized)
        """
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        # Robust scaling
        s_normalized = (scores - median) / (mad + self.delta)

        # Sigmoid projection to (0, 1)
        return self._sigmoid(self.gamma * s_normalized)

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        # Ensure scores are in [0, 1] first
        scores = np.clip(scores, 0, 1)

        min_val = scores.min()
        max_val = scores.max()

        if max_val - min_val > self.delta:
            return (scores - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(scores)

    def _zscore_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Z-score normalization with sigmoid projection."""
        mean = scores.mean()
        std = scores.std()

        s_normalized = (scores - mean) / (std + self.delta)

        return self._sigmoid(self.gamma * s_normalized)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Stable sigmoid implementation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    # ------------------------------------------------------------------
    # Joint normalization: shared statistics across sibling signals
    # ------------------------------------------------------------------

    def normalize_joint(
        self, scores_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Normalize multiple signals jointly using shared statistics.

        Computes statistics (median/MAD, min/max, or mean/std) from the
        concatenation of all signals, then applies those shared statistics
        to each signal individually.  This preserves relative magnitude
        differences between sibling signals from the same expert.

        Args:
            scores_dict: Mapping from query string to raw score array.

        Returns:
            Mapping from query string to normalized score array in (0, 1).
        """
        if not scores_dict:
            return {}

        # Single signal: identical to independent normalization
        if len(scores_dict) == 1:
            key = next(iter(scores_dict))
            return {key: self.normalize(scores_dict[key])}

        # Concatenate all signals to compute shared statistics
        all_scores = np.concatenate(list(scores_dict.values()))

        if self.method == "robust":
            return self._robust_normalize_joint(scores_dict, all_scores)
        elif self.method == "minmax":
            return self._minmax_normalize_joint(scores_dict, all_scores)
        elif self.method == "zscore":
            return self._zscore_normalize_joint(scores_dict, all_scores)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def _robust_normalize_joint(
        self,
        scores_dict: Dict[str, np.ndarray],
        all_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Joint robust normalization using shared median and MAD."""
        median = np.median(all_scores)
        mad = np.median(np.abs(all_scores - median))

        result = {}
        for query, scores in scores_dict.items():
            s_normalized = (scores - median) / (mad + self.delta)
            result[query] = self._sigmoid(self.gamma * s_normalized)
        return result

    def _minmax_normalize_joint(
        self,
        scores_dict: Dict[str, np.ndarray],
        all_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Joint min-max normalization using shared min and max."""
        all_clipped = np.clip(all_scores, 0, 1)
        min_val = all_clipped.min()
        max_val = all_clipped.max()

        result = {}
        if max_val - min_val > self.delta:
            for query, scores in scores_dict.items():
                clipped = np.clip(scores, 0, 1)
                result[query] = (clipped - min_val) / (max_val - min_val)
        else:
            for query, scores in scores_dict.items():
                result[query] = np.zeros_like(scores)
        return result

    def _zscore_normalize_joint(
        self,
        scores_dict: Dict[str, np.ndarray],
        all_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Joint z-score normalization using shared mean and std."""
        mean = all_scores.mean()
        std = all_scores.std()

        result = {}
        for query, scores in scores_dict.items():
            s_normalized = (scores - mean) / (std + self.delta)
            result[query] = self._sigmoid(self.gamma * s_normalized)
        return result
