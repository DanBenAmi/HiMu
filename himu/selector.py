"""HiMuSelector — end-to-end HiMu frame selection pipeline."""

import gc
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._types import FrameSelectionResult
from .cache import FeatureCache
from .config import (
    HiMuConfig,
    DEFAULT_CONFIG,
    VISUAL_ONLY_CONFIG,
    FAST_CONFIG,
    ABLATION_NO_SMOOTHING_CONFIG,
    ABLATION_ADDITIVE_CONFIG,
)
from .engine import MultiGPUEngine
from .experts import ExpertFactory
from .experts.asr import ASRExpert
from .llm import BaseLLM, OpenAILLM, create_llm
from .logic import LogicEngine
from .normalization import ScoreNormalizer
from .selection import create_selector
from .smoothing import BandwidthMatchedSmoother, EXPERT_MODALITY_MAP
from .video_io import VideoProcessor
from .audio_io import AudioProcessor

log = logging.getLogger(__name__)


class HiMuSelector:
    """End-to-end HiMu frame selector.

    Implements a five-stage pipeline:

    1. **Query Parsing** — LLM decomposes the question into a logic tree.
    2. **Expert Signal Extraction** — multimodal experts ground atomic predicates.
    3. **Normalization + Smoothing** — robust normalization then bandwidth-matched
       Gaussian smoothing.
    4. **Fuzzy Logic Composition** — bottom-up tree evaluation with fuzzy operators.
    5. **Frame Selection** — choose the top-K frames from the satisfaction curve.
    """

    def __init__(
        self,
        config: Optional[HiMuConfig] = None,
        llm: Optional[BaseLLM] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            config: Pipeline configuration (uses DEFAULT_CONFIG if None).
            llm: LLM for query parsing (creates OpenAILLM if None).
            device: Device override (uses config.device if None).
            cache_dir: Directory for feature caching (overrides config.cache_dir).
            verbose: Enable detailed per-question logging.
        """
        self.config = config or HiMuConfig()
        self.device = device or self.config.device
        self.verbose = verbose

        # LLM (Stage 1)
        self.llm = llm or OpenAILLM(
            model=self.config.llm_model,
            seed=self.config.llm_seed,
        )

        # Smoother (Stage 3)
        if self.config.smoothing.enabled:
            self.smoother = BandwidthMatchedSmoother(
                sigmas={
                    "visual": self.config.smoothing.visual_sigma,
                    "speech": self.config.smoothing.speech_sigma,
                    "audio": self.config.smoothing.audio_sigma,
                },
                fps=self.config.fps,
            )
        else:
            self.smoother = None

        # Logic engine (Stage 4)
        self.logic_engine = LogicEngine(
            kappa=self.config.composition.kappa,
            fps=self.config.fps,
            and_mode=self.config.composition.and_mode,
        )

        # Normalizer
        self.normalizer = ScoreNormalizer(
            method=self.config.normalization.method,
            delta=self.config.normalization.delta,
            gamma=self.config.normalization.gamma,
        )

        # Frame selector (Stage 5)
        self.frame_selector = create_selector(
            mode=self.config.selection.mode,
            min_gap=self.config.selection.min_gap,
            max_frames_per_window=self.config.selection.max_frames_per_window,
            window_seconds=self.config.selection.window_seconds,
            peak_prominence=self.config.selection.peak_prominence,
        )

        # Cache
        effective_cache_dir = cache_dir or self.config.cache_dir
        self.cache = FeatureCache(effective_cache_dir) if effective_cache_dir else None
        self._memory_cache: dict = {}

        # Engine
        self.engine = MultiGPUEngine(
            config=self.config,
            device=self.device,
            cache=self.cache,
            memory_cache=self._memory_cache,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_frames(
        self,
        video_path: str,
        question: str,
        candidates: Optional[List[str]] = None,
        num_frames: int = 16,
        fps: Optional[float] = None,
        subtitle_segments: Optional[List[Tuple[float, float, str]]] = None,
    ) -> FrameSelectionResult:
        """Select the most question-relevant frames from a video.

        Args:
            video_path: Path to the video file.
            question: Natural language question about the video.
            candidates: Optional MCQ answer options. If provided they are
                appended to the question as ``Options: ["A", "B", ...]``
                before the LLM call.
            num_frames: Number of frames to select (K).
            fps: Frame extraction rate (uses config.fps if None).
            subtitle_segments: Optional pre-loaded subtitle segments
                ``[(start, end, text), ...]``. If provided, ASR uses these
                instead of running Whisper.

        Returns:
            :class:`FrameSelectionResult` with selected frame indices,
            timestamps, scores, truth curve, and metadata.
        """
        fps = fps or self.config.fps

        # Format query with candidates
        query = question
        if candidates:
            opts = json.dumps(candidates)
            query = f"{question} Options: {opts}"

        if self.verbose:
            log.info("=" * 80)
            log.info("HiMu: Hierarchical Multimodal Frame Selection")
            log.info("=" * 80)
            log.info(f"Query: {query}\n")
        else:
            log.info(f"HiMu | Query: {query[:100]}")

        # =============================================================
        # STAGE 1: Query Parsing
        # =============================================================
        enabled_audio = {
            "ASR": self.config.asr.enabled,
            "CLAP": self.config.clap.enabled,
        }
        tree = self.llm.parse_query_to_tree(query, enabled_audio_experts=enabled_audio)

        if self.verbose:
            log.info(f"Logical Tree:\n{json.dumps(tree, indent=2)}\n")

        # =============================================================
        # STAGE 2: Expert Signal Extraction
        # =============================================================
        with VideoProcessor(video_path, fps=fps) as vp:
            frames_arr, timestamps_arr = vp.extract_frames()
        if self.verbose:
            log.info(f"Extracted {len(frames_arr)} frames")

        full_audio = None
        audio_chunks = None
        if self.config.asr.enabled and subtitle_segments is None:
            full_audio = AudioProcessor.extract_full_audio(video_path, sample_rate=16000)
        if self.config.clap.enabled:
            audio_chunks = AudioProcessor.extract_audio_for_frames(
                video_path, timestamps_arr, fps=fps
            )

        leaf_nodes = self.logic_engine.collect_leaf_nodes(tree)
        grouped_leaves = self.logic_engine.group_leaves_by_expert(leaf_nodes)

        expert_results = self.engine.run_experts(
            grouped_leaves=grouped_leaves,
            frames=frames_arr,
            timestamps=timestamps_arr,
            full_audio=full_audio,
            audio_chunks=audio_chunks,
            fps=fps,
            video_path=video_path,
            subtitle_segments=subtitle_segments,
        )

        # =============================================================
        # Normalization + Smoothing (Stage 3)
        # =============================================================
        frame_scores: Dict[int, np.ndarray] = {}

        for expert_type, node_queries in grouped_leaves.items():
            if not self.config.is_expert_enabled(expert_type):
                continue

            batch_scores = expert_results.get(expert_type, {})

            # Joint normalization for same-expert siblings
            if self.config.normalization.joint and len(node_queries) > 1:
                normalized_dict = self._joint_normalize(expert_type, node_queries, batch_scores)
            else:
                normalized_dict = None

            for node_id, q in node_queries:
                raw = batch_scores.get(q)
                if raw is None:
                    continue

                if normalized_dict is not None:
                    normed = normalized_dict[q]
                else:
                    normed = self.normalizer.normalize(raw)

                # Bandwidth-matched smoothing
                if self.smoother:
                    modality = EXPERT_MODALITY_MAP.get(expert_type, "visual")
                    smoothed = self.smoother.smooth_signal(normed, modality)
                else:
                    smoothed = normed

                frame_scores[node_id] = smoothed

                if self.verbose:
                    log.info(
                        f"  Node {node_id} ({expert_type}): '{q}' — "
                        f"[{smoothed.min():.3f}, {smoothed.max():.3f}], "
                        f"mean: {smoothed.mean():.3f}"
                    )

        # Free heavy data
        n_frames = len(timestamps_arr)
        del frames_arr, full_audio, audio_chunks
        gc.collect()

        # =============================================================
        # STAGE 4: Logical Composition
        # =============================================================
        truth_curve = self._compose(tree, frame_scores, n_frames)

        if self.verbose:
            log.info(
                f"Truth curve: [{truth_curve.min():.3f}, {truth_curve.max():.3f}]"
            )

        # =============================================================
        # STAGE 5: Frame Selection
        # =============================================================
        top_indices = self.frame_selector.select(truth_curve, num_frames, fps=fps)
        scores = truth_curve[top_indices]
        ts = timestamps_arr[top_indices]

        best_idx = top_indices[0]
        best_ts = float(timestamps_arr[best_idx])
        best_score = float(truth_curve[best_idx])

        if self.verbose:
            log.info(
                f"Selected {len(top_indices)} frames, "
                f"best: idx={best_idx} t={best_ts:.1f}s score={best_score:.3f}"
            )

        return FrameSelectionResult(
            frame_indices=top_indices,
            timestamps=ts,
            scores=scores,
            truth_curve=truth_curve,
            tree=tree,
            num_frames=n_frames,
            fps=fps,
            best_frame_idx=int(best_idx),
            best_timestamp=best_ts,
            best_score=best_score,
        )

    def cache_features(self, video_path: str, fps: Optional[float] = None) -> Dict[str, bool]:
        """Pre-extract and cache query-independent features for a video.

        Populates the in-memory cache (always) and disk cache (if cache_dir
        was configured).  Useful for corpus pre-processing: run once, then
        fast per-query selection without re-loading expert models.

        Returns:
            Dict indicating which features were cached.
        """
        fps = fps or self.config.fps
        mk = self.engine._mem_key(video_path, fps)
        mem = self._memory_cache.get(mk, {})

        status = {"CLIP": "CLIP" in mem, "OCR": "OCR" in mem,
                  "ASR": "ASR" in mem, "CLAP": False}
        if self.cache:
            disk_status = self.cache.get_cache_status(video_path, fps)
            status = {k: status[k] or disk_status[k] for k in status}

        with VideoProcessor(video_path, fps=fps) as vp:
            frames_arr, timestamps_arr = vp.extract_frames()

        # CLIP
        if self.config.clip.enabled and not status["CLIP"]:
            expert = self.engine._get_expert("CLIP")
            embeddings = expert.extract_embeddings(frames_arr)
            mem["CLIP"] = embeddings
            if self.cache:
                self.cache.save_clip(video_path, fps, embeddings)
            status["CLIP"] = True

        # OCR
        if self.config.ocr.enabled and not status["OCR"]:
            expert = self.engine._get_expert("OCR")
            texts = expert._extract_text_from_frames_batch(list(frames_arr))
            mem["OCR"] = texts
            if self.cache:
                self.cache.save_ocr(video_path, fps, texts)
            status["OCR"] = True

        # ASR
        if self.config.asr.enabled and not status["ASR"]:
            full_audio = AudioProcessor.extract_full_audio(video_path, sample_rate=16000)
            if full_audio is not None:
                expert = self.engine._get_expert("ASR")
                segments = expert._transcribe_audio(full_audio, sample_rate=16000)
                mem["ASR"] = segments
                if self.cache:
                    self.cache.save_asr(video_path, fps, segments)
                status["ASR"] = True

        # CLAP — no query-independent embeddings to cache in current design

        self._memory_cache[mk] = mem
        del frames_arr
        gc.collect()

        return status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _joint_normalize(
        self,
        expert_type: str,
        node_queries: List[Tuple[int, str]],
        batch_scores: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Joint normalization, with ASR short/long split."""
        if expert_type == "ASR":
            threshold = ASRExpert._SEMANTIC_MIN_WORDS
            short_dict = {}
            long_dict = {}
            for _, q in node_queries:
                arr = batch_scores.get(q)
                if arr is None:
                    continue
                if len(q.split()) < threshold:
                    short_dict[q] = arr
                else:
                    long_dict[q] = arr

            normalized = {}
            if short_dict:
                normalized.update(self.normalizer.normalize_joint(short_dict))
            if long_dict:
                normalized.update(self.normalizer.normalize_joint(long_dict))
            return normalized
        else:
            raw_dict = {q: batch_scores[q] for _, q in node_queries if q in batch_scores}
            return self.normalizer.normalize_joint(raw_dict)

    def _compose(
        self, tree: dict, frame_scores: Dict[int, np.ndarray], n_frames: int
    ) -> np.ndarray:
        """Stage 4: compose truth curve from frame scores."""
        method = self.config.composition.method

        if method == "fuzzy_logic_tree":
            if self.config.composition.rescale_range is not None:
                lo, hi = self.config.composition.rescale_range
                truth_curve, _ = self.logic_engine.evaluate_tree_rescaled(
                    tree, frame_scores, lo=lo, hi=hi
                )
            elif self.config.composition.post_restandardize:
                truth_curve, _ = self.logic_engine.evaluate_tree_restandardized(
                    tree, frame_scores
                )
            else:
                truth_curve, _ = self.logic_engine.evaluate_tree(tree, frame_scores)
        elif method == "additive":
            truth_curve = np.mean(list(frame_scores.values()), axis=0)
        elif method == "multiplicative":
            truth_curve = np.prod(list(frame_scores.values()), axis=0)
        else:
            raise ValueError(f"Unknown composition method: {method}")

        return truth_curve


def create_himu_selector(config_name: str = "default", **kwargs) -> HiMuSelector:
    """Factory to create a HiMuSelector with a named configuration.

    Args:
        config_name: One of ``"default"``, ``"visual_only"``, ``"fast"``,
            ``"no_smoothing"``, ``"additive"``.
        **kwargs: Forwarded to :class:`HiMuSelector` constructor
            (e.g. ``device``, ``cache_dir``, ``verbose``).

    Returns:
        Configured :class:`HiMuSelector` instance.
    """
    config_map = {
        "default": DEFAULT_CONFIG,
        "visual_only": VISUAL_ONLY_CONFIG,
        "fast": FAST_CONFIG,
        "no_smoothing": ABLATION_NO_SMOOTHING_CONFIG,
        "additive": ABLATION_ADDITIVE_CONFIG,
    }

    if config_name not in config_map:
        raise ValueError(
            f"Unknown config: {config_name}. Available: {list(config_map.keys())}"
        )

    config = config_map[config_name]
    return HiMuSelector(config=config, **kwargs)
