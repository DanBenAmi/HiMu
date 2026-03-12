"""Unified feature caching for HiMu.

Caches query-independent features per video so subsequent queries on the same
video skip expensive expert inference.  Cached features:
  - CLIP frame embeddings (.npz)
  - OCR per-frame text detections (.json)
  - ASR transcript segments (.json)
  - CLAP frame embeddings (.npz)
OVD is NOT cached because it is query-conditioned.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def _video_cache_key(video_path: str, fps: float) -> str:
    """Compute a deterministic cache key from the resolved video path and fps."""
    resolved = str(Path(video_path).resolve())
    raw = f"{resolved}:{fps}"
    return hashlib.sha256(raw.encode()).hexdigest()


class FeatureCache:
    """Unified feature cache backed by a directory on disk.

    Directory layout::

        <cache_dir>/
          <sha256_key>/
            clip.npz          # embeddings: (num_frames, embed_dim)
            ocr.json           # List[List[[text, confidence]]]
            asr.json           # List[[start, end, text]]
            clap.npz           # embeddings: (num_frames, embed_dim)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_dir(self, video_path: str, fps: float) -> Path:
        key = _video_cache_key(video_path, fps)
        return self.cache_dir / key

    # ------------------------------------------------------------------
    # CLIP
    # ------------------------------------------------------------------

    def has_clip(self, video_path: str, fps: float) -> bool:
        return (self._key_dir(video_path, fps) / "clip.npz").exists()

    def load_clip(self, video_path: str, fps: float) -> np.ndarray:
        path = self._key_dir(video_path, fps) / "clip.npz"
        data = np.load(str(path))
        return data["embeddings"]

    def save_clip(self, video_path: str, fps: float, embeddings: np.ndarray) -> None:
        d = self._key_dir(video_path, fps)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(d / "clip.npz"), embeddings=embeddings)
        log.info(f"Saved CLIP cache: {embeddings.shape}")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def has_ocr(self, video_path: str, fps: float) -> bool:
        return (self._key_dir(video_path, fps) / "ocr.json").exists()

    def load_ocr(self, video_path: str, fps: float) -> List[List[Tuple[str, float]]]:
        path = self._key_dir(video_path, fps) / "ocr.json"
        with open(path, "r") as f:
            raw = json.load(f)
        # Convert inner lists back to tuples
        return [[tuple(item) for item in frame] for frame in raw]

    def save_ocr(self, video_path: str, fps: float, texts: List[List[Tuple[str, float]]]) -> None:
        d = self._key_dir(video_path, fps)
        d.mkdir(parents=True, exist_ok=True)
        # Convert tuples to lists for JSON serialization
        serializable = [[list(item) for item in frame] for frame in texts]
        with open(d / "ocr.json", "w") as f:
            json.dump(serializable, f)
        log.info(f"Saved OCR cache: {len(texts)} frames")

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def has_asr(self, video_path: str, fps: float) -> bool:
        return (self._key_dir(video_path, fps) / "asr.json").exists()

    def load_asr(self, video_path: str, fps: float) -> List[Tuple[float, float, str]]:
        path = self._key_dir(video_path, fps) / "asr.json"
        with open(path, "r") as f:
            raw = json.load(f)
        return [tuple(seg) for seg in raw]

    def save_asr(self, video_path: str, fps: float, segments: List[Tuple[float, float, str]]) -> None:
        d = self._key_dir(video_path, fps)
        d.mkdir(parents=True, exist_ok=True)
        serializable = [list(seg) for seg in segments]
        with open(d / "asr.json", "w") as f:
            json.dump(serializable, f)
        log.info(f"Saved ASR cache: {len(segments)} segments")

    # ------------------------------------------------------------------
    # CLAP
    # ------------------------------------------------------------------

    def has_clap(self, video_path: str, fps: float) -> bool:
        return (self._key_dir(video_path, fps) / "clap.npz").exists()

    def load_clap(self, video_path: str, fps: float) -> np.ndarray:
        path = self._key_dir(video_path, fps) / "clap.npz"
        data = np.load(str(path))
        return data["embeddings"]

    def save_clap(self, video_path: str, fps: float, embeddings: np.ndarray) -> None:
        d = self._key_dir(video_path, fps)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(d / "clap.npz"), embeddings=embeddings)
        log.info(f"Saved CLAP cache: {embeddings.shape}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_cache_status(self, video_path: str, fps: float) -> Dict[str, bool]:
        """Return which features are cached for a given video + fps."""
        return {
            "CLIP": self.has_clip(video_path, fps),
            "OCR": self.has_ocr(video_path, fps),
            "ASR": self.has_asr(video_path, fps),
            "CLAP": self.has_clap(video_path, fps),
        }
