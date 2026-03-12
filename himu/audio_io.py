"""Audio extraction utilities for HiMu."""

import io
import shutil
import subprocess
import numpy as np
from typing import List, Optional
import warnings


def _load_audio(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from a video/audio file using ffmpeg directly.

    Avoids librosa's fragile audioread fallback chain which breaks when
    C-level stderr is suppressed.

    Returns:
        Audio as float32 numpy array (samples,), mono, at target sample rate.

    Raises:
        RuntimeError: If ffmpeg is not found or audio extraction fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with: "
            "apt install ffmpeg  /  conda install ffmpeg"
        )

    cmd = [
        ffmpeg,
        "-i", str(video_path),
        "-vn",                      # no video
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),    # target sample rate
        "-f", "f32le",              # raw 32-bit float PCM
        "-loglevel", "error",       # suppress noisy codec warnings
        "pipe:1",                   # write to stdout
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        stderr_msg = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}) on {video_path}: {stderr_msg}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError(f"No audio stream found in {video_path}")

    return audio


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
            audio = _load_audio(video_path, sample_rate)
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
            return _load_audio(video_path, sample_rate)
        except Exception as e:
            warnings.warn(f"Failed to load audio from {video_path}: {e}")
            return None

    @staticmethod
    def get_audio_duration(video_path: str) -> Optional[float]:
        """Get duration of audio track in seconds."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            warnings.warn("ffprobe not found on PATH, cannot get audio duration")
            return None

        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return float(result.stdout.decode().strip())
        except Exception as e:
            warnings.warn(f"Failed to get audio duration from {video_path}: {e}")
            return None
