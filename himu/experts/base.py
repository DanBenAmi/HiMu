"""Base expert interface and utilities."""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict
import logging

log = logging.getLogger(__name__)

# Default models weights directory
DEFAULT_MODELS_WEIGHTS_DIR = Path.home() / ".cache" / "himu" / "models"


def get_models_weights_dir(weights_dir: Optional[str] = None) -> Path:
    """Get the models weights directory, creating it if necessary.

    Args:
        weights_dir: Optional custom weights directory path

    Returns:
        Path to the models weights directory
    """
    if weights_dir:
        path = Path(weights_dir)
    else:
        path = DEFAULT_MODELS_WEIGHTS_DIR

    path.mkdir(parents=True, exist_ok=True)
    return path


class BaseExpert(ABC):
    """Abstract base class for expert models."""

    @abstractmethod
    def compute_scores(self, frames: np.ndarray, query: str) -> np.ndarray:
        """
        Compute truth scores for each frame.

        Args:
            frames: Array of shape (num_frames, height, width, 3) in RGB
            query: Text query to ground

        Returns:
            Array of shape (num_frames,) with scores in [0, 1]
        """
        pass

    def compute_batch_scores(
        self,
        frames: np.ndarray,
        queries: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Compute truth scores for multiple queries in parallel.

        Default implementation falls back to sequential processing.
        Subclasses can override for optimized batch processing.
        """
        return {query: self.compute_scores(frames, query) for query in queries}
