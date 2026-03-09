"""Bandwidth-matched Gaussian smoothing for HiMu expert signals."""

import numpy as np
from typing import Optional, Dict
from scipy.ndimage import gaussian_filter1d


# Modality mapping for experts
EXPERT_MODALITY_MAP = {
    "YOLO": "visual",
    "CLIP": "visual",
    "OCR": "visual",
    "ASR": "speech",
    "CLAP": "audio",
}


class BandwidthMatchedSmoother:
    """
    Apply modality-specific Gaussian smoothing to expert signals.

    Different modalities have different temporal precisions:
    - Visual signals (CLIP, YOLO, OCR) are frame-precise
    - Speech signals (ASR/Whisper) have timing noise from word boundaries (~1-2s)
    - Audio signals (CLAP) have window-level granularity (~2s)

    Bandwidth-matched smoothing resolves cross-modal asynchrony by applying
    modality-appropriate kernels before composition.
    """

    DEFAULT_SIGMAS = {
        "visual": 0.5,    # Frame-precise signals (seconds)
        "speech": 1.5,    # Word-boundary timing noise (seconds)
        "audio": 2.0,     # Window-level granularity (seconds)
    }

    def __init__(
        self,
        sigmas: Optional[Dict[str, float]] = None,
        fps: float = 1.0
    ):
        """
        Initialize bandwidth-matched smoother.

        Args:
            sigmas: Custom sigma values for each modality (seconds).
                   If None, uses DEFAULT_SIGMAS.
            fps: Frames per second (for converting sigma from seconds to frames).
        """
        self.sigmas = sigmas or self.DEFAULT_SIGMAS.copy()
        self.fps = fps

    def smooth_signal(
        self,
        signal: np.ndarray,
        modality: str
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing with bandwidth-matched sigma.

        Args:
            signal: Expert output signal (num_frames,)
            modality: One of "visual", "speech", "audio"

        Returns:
            Smoothed signal
        """
        if modality not in self.sigmas:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Expected one of {list(self.sigmas.keys())}"
            )

        # Get sigma in seconds and convert to frames
        sigma_seconds = self.sigmas[modality]
        sigma_frames = sigma_seconds * self.fps

        # sigma=0 means no smoothing — return signal unchanged
        if sigma_frames <= 0:
            return signal

        # Apply Gaussian filter
        # mode='nearest' pads with edge values to avoid boundary artifacts
        smoothed = gaussian_filter1d(
            signal,
            sigma=sigma_frames,
            mode='nearest'
        )

        return smoothed

    def set_sigma(self, modality: str, sigma: float):
        """Set custom sigma for a specific modality."""
        self.sigmas[modality] = sigma
