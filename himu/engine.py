"""Expert orchestration engine for HiMu.

Handles expert lifecycle, routing inputs to the right expert interface,
and integrating with FeatureCache.  Currently executes all experts
sequentially on a single GPU; the API surface supports multi-GPU
allocation for future parallel execution.
"""

import gc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cache import FeatureCache
from .config import HiMuConfig
from .experts import ExpertFactory

log = logging.getLogger(__name__)

# Approximate peak VRAM in GB per expert (for auto-allocation)
_VRAM_COST = {
    "CLIP": 6.0,
    "YOLO": 4.0,
    "OCR": 2.0,
    "ASR": 3.0,
    "CLAP": 2.0,
}


class MultiGPUEngine:
    """Orchestrates expert execution with optional multi-GPU support.

    Current implementation: sequential execution on a single device.
    The ``gpu_map`` parameter and ``_auto_allocate`` method define the
    full multi-GPU API surface; parallel execution via ThreadPoolExecutor
    is marked as a TODO.
    """

    def __init__(
        self,
        config: HiMuConfig,
        device: str = "cuda",
        cache: Optional[FeatureCache] = None,
    ):
        self.config = config
        self.device = device
        self.cache = cache
        self.gpu_map = config.gpu_map  # Dict[str, List[int]] or None
        self._expert_cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Expert lifecycle
    # ------------------------------------------------------------------

    def _get_expert(self, expert_type: str):
        """Get or create an expert instance (lazy, cached in memory)."""
        if expert_type not in self._expert_cache:
            expert_device = self.device
            # TODO: use self.gpu_map to route experts to specific GPUs
            log.info(f"Initializing {expert_type} expert on {expert_device}...")

            model_name = self.config.get_expert_model_name(expert_type)
            clip_config = None
            if expert_type == "CLIP":
                clip_config = {"preset": self.config.clip_preset}

            self._expert_cache[expert_type] = ExpertFactory.create_expert(
                expert_type=expert_type,
                device=expert_device,
                clip_config=clip_config,
                weights_dir=self.config.weights_dir,
                model_name=model_name,
            )
        return self._expert_cache[expert_type]

    # ------------------------------------------------------------------
    # Auto-allocation (placeholder)
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_allocate(expert_types: List[str], num_gpus: int) -> Dict[str, List[int]]:
        """Greedy assignment of experts to GPUs by VRAM cost.

        TODO: implement actual multi-GPU allocation.  For now returns
        all experts on GPU 0.
        """
        return {et: [0] for et in expert_types}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_experts(
        self,
        grouped_leaves: Dict[str, List[Tuple[int, str]]],
        frames: Optional[np.ndarray],
        timestamps: np.ndarray,
        full_audio: Optional[np.ndarray],
        audio_chunks: Optional[List[np.ndarray]],
        fps: float,
        video_path: str,
        subtitle_segments: Optional[List[Tuple[float, float, str]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run all required experts and return raw scores per expert.

        Args:
            grouped_leaves: ``{expert_type: [(node_id, query), ...]}``
            frames: Extracted video frames (num_frames, H, W, 3) or None.
            timestamps: Frame timestamps (num_frames,).
            full_audio: Full-length audio waveform for ASR or None.
            audio_chunks: Per-frame audio chunks for CLAP or None.
            fps: Frame extraction rate.
            video_path: Path to video (for cache key).
            subtitle_segments: Optional pre-loaded subtitle segments for ASR.

        Returns:
            ``{expert_type: {query: scores_array}}``
        """
        num_frames = len(timestamps)
        results: Dict[str, Dict[str, np.ndarray]] = {}

        for expert_type, node_queries in grouped_leaves.items():
            if not self.config.is_expert_enabled(expert_type):
                log.info(f"Skipping disabled expert: {expert_type}")
                continue

            queries = [q for _, q in node_queries]
            batch_scores = self._run_single_expert(
                expert_type=expert_type,
                queries=queries,
                frames=frames,
                timestamps=timestamps,
                full_audio=full_audio,
                audio_chunks=audio_chunks,
                fps=fps,
                video_path=video_path,
                num_frames=num_frames,
                subtitle_segments=subtitle_segments,
            )
            results[expert_type] = batch_scores

        return results

    # ------------------------------------------------------------------
    # Per-expert routing
    # ------------------------------------------------------------------

    def _run_single_expert(
        self,
        expert_type: str,
        queries: List[str],
        frames: Optional[np.ndarray],
        timestamps: np.ndarray,
        full_audio: Optional[np.ndarray],
        audio_chunks: Optional[List[np.ndarray]],
        fps: float,
        video_path: str,
        num_frames: int,
        subtitle_segments: Optional[List[Tuple[float, float, str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Route a single expert type to the correct inference path."""

        # ----- OCR with disk cache -----
        if expert_type == "OCR" and self.cache and self.cache.has_ocr(video_path, fps):
            cached_texts = self.cache.load_ocr(video_path, fps)
            return self._compute_ocr_scores_from_cache(queries, cached_texts, num_frames)

        # ----- CLIP with disk cache -----
        cached_clip = None
        if expert_type == "CLIP" and self.cache and self.cache.has_clip(video_path, fps):
            cached_clip = self.cache.load_clip(video_path, fps)

        # ----- ASR with disk cache -----
        if expert_type == "ASR" and self.cache and self.cache.has_asr(video_path, fps):
            cached_segments = self.cache.load_asr(video_path, fps)
            subtitle_segments = cached_segments

        # Load expert model
        expert = self._get_expert(expert_type)

        if expert_type == "CLIP":
            return self._run_clip(expert, frames, queries, cached_clip)

        elif expert_type == "OCR":
            batch_scores = expert.compute_batch_scores(frames, queries)
            # Save to cache if available
            if self.cache and hasattr(expert, 'detect_text_batch'):
                texts = expert.detect_text_batch(frames)
                self.cache.save_ocr(video_path, fps, texts)
            return batch_scores

        elif expert_type == "YOLO":
            return expert.compute_batch_scores(
                frames, queries, batch_size=self.config.yolo.batch_size
            )

        elif expert_type == "ASR":
            return self._run_asr(
                expert, queries, full_audio, timestamps, fps, subtitle_segments
            )

        elif expert_type == "CLAP":
            if audio_chunks is None:
                log.warning("Audio chunks not available for CLAP")
                return {q: np.zeros(num_frames, dtype=np.float32) for q in queries}
            return expert.compute_batch_scores(audio_chunks, queries)

        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def _run_clip(
        self, expert, frames, queries, cached_embeddings=None
    ) -> Dict[str, np.ndarray]:
        """Run CLIP expert with optional cached embeddings."""
        if cached_embeddings is not None and len(cached_embeddings) > 0:
            expected_dim = expert.get_embedding_dim()
            actual_dim = cached_embeddings.shape[-1]
            if actual_dim != expected_dim:
                log.warning(
                    f"Cached CLIP embeddings dim ({actual_dim}) != model dim ({expected_dim}). "
                    f"Recomputing."
                )
                cached_embeddings = None
            elif frames is not None:
                len_diff = len(cached_embeddings) - len(frames)
                if len_diff == 0:
                    pass
                elif abs(len_diff) <= 2 and len_diff > 0:
                    cached_embeddings = cached_embeddings[len_diff:]
                elif abs(len_diff) <= 2 and len_diff < 0:
                    pass  # frames longer by 1-2, cache still usable
                else:
                    min_len = min(len(cached_embeddings), len(frames))
                    log.warning(
                        f"CLIP cache length ({len(cached_embeddings)}) differs from "
                        f"frames ({len(frames)}) by {abs(len_diff)}, truncating to {min_len}"
                    )
                    cached_embeddings = cached_embeddings[len(cached_embeddings) - min_len:]

        if self.config.clip_templates:
            return expert.compute_batch_scores_with_templates(
                frames, queries,
                templates=self.config.clip_templates,
                cached_embeddings=cached_embeddings,
            )
        else:
            return expert.compute_batch_scores(
                frames, queries, cached_embeddings=cached_embeddings
            )

    @staticmethod
    def _run_asr(
        expert, queries, full_audio, timestamps, fps, subtitle_segments=None
    ) -> Dict[str, np.ndarray]:
        """Run ASR expert with appropriate audio source."""
        if subtitle_segments is not None:
            dummy_audio = full_audio if full_audio is not None else np.zeros(16000, dtype=np.float32)
            return expert.compute_batch_scores(
                dummy_audio, queries, timestamps, fps,
                pre_segments=subtitle_segments
            )
        elif full_audio is None or len(full_audio) == 0:
            log.warning("Full audio not available for ASR")
            return {q: np.zeros(len(timestamps), dtype=np.float32) for q in queries}
        else:
            return expert.compute_batch_scores(full_audio, queries, timestamps, fps)

    # ------------------------------------------------------------------
    # OCR cache scoring (no model needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ocr_scores_from_cache(
        queries: List[str],
        cached_texts: List[List[tuple]],
        num_frames: int,
    ) -> Dict[str, np.ndarray]:
        """Score OCR queries against cached text detections without loading model."""
        cache_len = len(cached_texts)
        results = {}

        for query in queries:
            scores = np.zeros(num_frames, dtype=np.float32)
            query_lower = query.lower().strip()

            for i in range(min(cache_len, num_frames)):
                text_results = cached_texts[i]
                if text_results:
                    scores[i] = _match_query_to_texts(query_lower, text_results)

            results[query] = scores

        log.info(
            f"Computed OCR scores from cache for {len(queries)} queries "
            f"({num_frames} frames, model not loaded)"
        )
        return results


# ------------------------------------------------------------------
# Text matching utilities (shared with selector)
# ------------------------------------------------------------------

def _match_query_to_texts(query: str, text_results: List[tuple]) -> float:
    """Match a query against detected OCR texts from a frame."""
    if not text_results:
        return 0.0

    detected_texts = [text.lower().strip() for text, _ in text_results]
    best_score = 0.0
    for text in detected_texts:
        if query in text:
            return 1.0
        score = _text_similarity(query, text)
        best_score = max(best_score, score)
    return best_score


def _text_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    dist = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dist[i][0] = i
    for j in range(len2 + 1):
        dist[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist[len1][len2] / max_len))
