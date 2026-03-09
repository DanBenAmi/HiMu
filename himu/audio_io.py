"""Audio extraction utilities for HiMu."""

import os
import sys
import numpy as np
from contextlib import contextmanager
from typing import List, Optional
import warnings


@contextmanager
def _suppress_c_stderr():
    """Temporarily redirect C-level stderr to /dev/null.

    Suppresses messages from C libraries (e.g., ffmpeg av1 codec warnings)
    that bypass Python's logging/warnings system.
    """
    stderr_fd = sys.stderr.fileno()
    old = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(old, stderr_fd)
        os.close(old)
        os.close(devnull)


class AudioProcessor:
    """Extract audio segments aligned with video frames."""

    @staticmethod
    def extract_audio_for_frames(
        video_path: str,
        timestamps: np.ndarray,
        fps: float,
        sample_rate: int = 16000,
        window_expansion: float = 0.5
    ) -> List[np.ndarray]:
        """
        Extract audio chunks aligned with frame timestamps.

        For each frame at timestamp t, extracts audio from a window around t.
        The window size is based on the frame rate.

        Args:
            video_path: Path to video file
            timestamps: Frame timestamps in seconds (num_frames,)
            fps: Frames per second used for frame extraction
            sample_rate: Target audio sample rate (Hz)
            window_expansion: Factor to expand window beyond 1/fps

        Returns:
            List of audio chunks (num_frames,), each as np.ndarray of shape (samples,)
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio processing. "
                "Install with: pip install 'himu[asr]' or pip install 'himu[clap]'"
            )

        try:
            with _suppress_c_stderr():
                audio, sr = librosa.load(
                    video_path,
                    sr=sample_rate,
                    mono=True
                )
        except Exception as e:
            warnings.warn(f"Failed to load audio from {video_path}: {e}")
            return [np.array([], dtype=np.float32) for _ in timestamps]

        audio_chunks = []
        window_size = (1.0 / fps) * (1.0 + window_expansion)

        for t in timestamps:
            start_sec = max(0, t - window_size / 2)
            end_sec = t + window_size / 2

            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)

            if start_sample < end_sample:
                chunk = audio[start_sample:end_sample]
            else:
                chunk = np.array([], dtype=np.float32)

            audio_chunks.append(chunk)

        return audio_chunks

    @staticmethod
    def extract_full_audio(
        video_path: str,
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Extract complete audio track from video.

        Args:
            video_path: Path to video file
            sample_rate: Target audio sample rate (Hz)

        Returns:
            Audio array (samples,) or None if extraction fails
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio processing. "
                "Install with: pip install 'himu[asr]' or pip install 'himu[clap]'"
            )

        try:
            with _suppress_c_stderr():
                audio, sr = librosa.load(
                    video_path,
                    sr=sample_rate,
                    mono=True
                )
            return audio
        except Exception as e:
            warnings.warn(f"Failed to load audio from {video_path}: {e}")
            return None

    @staticmethod
    def get_audio_duration(video_path: str) -> Optional[float]:
        """Get duration of audio track in seconds."""
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for audio processing. "
                "Install with: pip install 'himu[asr]' or pip install 'himu[clap]'"
            )

        try:
            duration = librosa.get_duration(path=video_path)
            return duration
        except Exception as e:
            warnings.warn(f"Failed to get audio duration from {video_path}: {e}")
            return None
