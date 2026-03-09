"""ASR expert using faster-whisper for full-audio transcription."""

import re
import numpy as np
from typing import List, Optional, Dict, Tuple
import tempfile
import os
import logging

from .base import BaseExpert, get_models_weights_dir

log = logging.getLogger(__name__)


class ASRExpert(BaseExpert):
    """ASR expert using faster-whisper for full-audio transcription.

    Key design:
    - Processes full audio (no frame-aligned chunking)
    - Uses segment-level timestamps from model output
    - Converts segment matches to per-frame scores
    """

    _SEMANTIC_MIN_WORDS = 3
    _FUZZY_THRESHOLD = 0.5
    _ST_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = "turbo",
        device: str = "cuda",
        weights_dir: Optional[str] = None
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for ASRExpert. "
                "Install with: pip install 'himu[asr]'"
            )

        self.model_name = model_name
        self.device = device
        self._st_model = None

        device_type = device
        device_index = 0
        if ":" in device:
            parts = device.split(":")
            device_type = parts[0]
            try:
                device_index = int(parts[1])
            except (ValueError, IndexError):
                device_index = 0

        base_weights_dir = get_models_weights_dir(weights_dir)
        cache_dir = base_weights_dir / "faster_whisper"
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading faster-whisper model: {model_name}")

        self.model = WhisperModel(
            model_name,
            device=device_type,
            device_index=device_index,
            compute_type="float16" if device_type == "cuda" else "int8",
            download_root=str(cache_dir)
        )

    def _transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Tuple[float, float, str]]:
        """Transcribe audio and return segments with timestamps."""
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required for audio I/O. "
                "Install with: pip install soundfile"
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio, sample_rate)

        try:
            segments_iter, info = self.model.transcribe(temp_path, language=None)
            segments = []
            for segment in segments_iter:
                segments.append((segment.start, segment.end, segment.text))
            return segments
        except Exception as e:
            log.warning(f"ASR transcription failed: {e}")
            return []
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def _segments_to_frame_scores(
        self,
        segments: List[Tuple[float, float, str]],
        query: str,
        timestamps: np.ndarray,
        fps: float,
        query_embedding: Optional[np.ndarray] = None,
        segment_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert segment-level transcriptions to per-frame scores.

        Uses two-path scoring:
        - Short queries (< 3 words): exact substring + per-word Levenshtein
        - Long queries (>= 3 words): exact substring + sentence-transformer cosine sim
        """
        num_frames = len(timestamps)
        scores = np.zeros(num_frames, dtype=np.float32)
        frame_duration = 1.0 / fps
        query_lower = query.lower().strip()

        if not segments:
            return scores

        is_short = len(query.split()) < self._SEMANTIC_MIN_WORDS

        segment_scores = []
        for idx, (start, end, text) in enumerate(segments):
            text_lower = text.lower().strip()

            if is_short:
                similarity = self._score_segment_short(
                    query_lower, text_lower, self._FUZZY_THRESHOLD
                )
            else:
                if query_embedding is not None and segment_embeddings is not None:
                    similarity = self._score_segment_long(
                        query_lower, text_lower,
                        query_embedding, segment_embeddings[idx]
                    )
                else:
                    similarity = 1.0 if query_lower in text_lower else 0.0

            segment_scores.append((start, end, similarity))

        for i, t in enumerate(timestamps):
            frame_start = t - frame_duration / 2
            frame_end = t + frame_duration / 2

            max_score = 0.0

            for seg_start, seg_end, seg_score in segment_scores:
                if seg_end >= frame_start and seg_start <= frame_end:
                    overlap_start = max(seg_start, frame_start)
                    overlap_end = min(seg_end, frame_end)
                    overlap_duration = overlap_end - overlap_start

                    overlap_weight = min(1.0, overlap_duration / frame_duration)
                    weighted_score = seg_score * overlap_weight

                    max_score = max(max_score, weighted_score)

            scores[i] = max_score

        return scores

    @staticmethod
    def _text_similarity(s1: str, s2: str) -> float:
        """Compute text similarity using Levenshtein distance."""
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        distance = dp[m][n]
        max_len = max(m, n)

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    @staticmethod
    def _score_segment_short(query_lower: str, text_lower: str,
                             fuzzy_threshold: float) -> float:
        """Score a segment for short queries (< 3 words).

        1. Exact substring match -> 1.0
        2. Per-word Levenshtein: multi-word queries require all words to match,
           single-word queries use best matching word.
        """
        if query_lower in text_lower:
            return 1.0

        words = re.findall(r"[a-z0-9']+", text_lower)
        if not words:
            return 0.0

        query_words = query_lower.split()
        if len(query_words) > 1:
            min_best = 1.0
            for qw in query_words:
                best = max(ASRExpert._text_similarity(qw, w) for w in words)
                min_best = min(min_best, best)
            if min_best >= fuzzy_threshold:
                return min_best

        best = max(ASRExpert._text_similarity(query_lower, w) for w in words)
        return best if best >= fuzzy_threshold else 0.0

    @staticmethod
    def _score_segment_long(query_lower: str, text_lower: str,
                            query_embedding: np.ndarray,
                            segment_embedding: np.ndarray) -> float:
        """Score a segment for long queries (>= 3 words).

        1. Exact substring match -> 1.0
        2. Cosine similarity between embeddings, clipped to [0, 1].
        """
        if query_lower in text_lower:
            return 1.0

        dot = float(np.dot(query_embedding, segment_embedding))
        norm_q = float(np.linalg.norm(query_embedding))
        norm_s = float(np.linalg.norm(segment_embedding))
        if norm_q < 1e-9 or norm_s < 1e-9:
            return 0.0
        cosine = dot / (norm_q * norm_s)
        return max(0.0, cosine)

    def _semantic_similarity(self, query: str, segment_text: str) -> float:
        """Compute similarity using two-path scoring."""
        query_lower = query.lower().strip()
        text_lower = segment_text.lower().strip()

        if not query_lower or not text_lower:
            return 0.0

        if query_lower in text_lower:
            return 1.0

        if len(query.split()) < self._SEMANTIC_MIN_WORDS:
            return self._score_segment_short(query_lower, text_lower, self._FUZZY_THRESHOLD)
        else:
            st_model = self._get_st_model()
            q_emb = st_model.encode(
                [query_lower], batch_size=1,
                show_progress_bar=False, normalize_embeddings=True,
            )[0]
            s_emb = st_model.encode(
                [text_lower], batch_size=1,
                show_progress_bar=False, normalize_embeddings=True,
            )[0]
            return self._score_segment_long(query_lower, text_lower, q_emb, s_emb)

    def _get_st_model(self):
        """Lazy-load sentence-transformer model for long query scoring."""
        if self._st_model is None:
            import logging as _logging
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading sentence-transformer: {self._ST_MODEL_NAME}")
            _report_logger = _logging.getLogger("transformers.modeling_utils")
            _prev_level = _report_logger.level
            _report_logger.setLevel(_logging.ERROR)
            try:
                self._st_model = SentenceTransformer(
                    self._ST_MODEL_NAME, device=self.device
                )
            finally:
                _report_logger.setLevel(_prev_level)
        return self._st_model

    def compute_scores(
        self,
        audio: np.ndarray,
        query: str,
        timestamps: np.ndarray,
        fps: float,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """Compute per-frame ASR match scores."""
        segments = self._transcribe_audio(audio, sample_rate)

        query_embedding = None
        segment_embeddings = None
        if len(query.split()) >= self._SEMANTIC_MIN_WORDS and segments:
            st_model = self._get_st_model()
            segment_texts = [seg[2].lower().strip() for seg in segments]
            segment_embeddings = st_model.encode(
                segment_texts, batch_size=256,
                show_progress_bar=False, normalize_embeddings=True,
            )
            query_embedding = st_model.encode(
                [query.lower().strip()], batch_size=1,
                show_progress_bar=False, normalize_embeddings=True,
            )[0]

        scores = self._segments_to_frame_scores(
            segments, query, timestamps, fps,
            query_embedding=query_embedding,
            segment_embeddings=segment_embeddings,
        )

        return scores

    def compute_batch_scores(
        self,
        audio: np.ndarray,
        queries: List[str],
        timestamps: np.ndarray,
        fps: float,
        sample_rate: int = 16000,
        pre_segments: Optional[List[Tuple[float, float, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """Compute scores for multiple queries. Transcribe once, match multiple."""
        if pre_segments is not None:
            log.info(f"Using {len(pre_segments)} pre-loaded subtitle segments (skipping Whisper)")
            segments = pre_segments
        else:
            segments = self._transcribe_audio(audio, sample_rate)

        long_queries = [q for q in queries if len(q.split()) >= self._SEMANTIC_MIN_WORDS]

        segment_embeddings = None
        long_query_embeddings: Dict[str, np.ndarray] = {}
        if long_queries and segments:
            st_model = self._get_st_model()
            segment_texts = [seg[2].lower().strip() for seg in segments]
            segment_embeddings = st_model.encode(
                segment_texts, batch_size=256,
                show_progress_bar=False, normalize_embeddings=True,
            )
            unique_long = list({q.lower().strip() for q in long_queries})
            long_embs = st_model.encode(
                unique_long, batch_size=64,
                show_progress_bar=False, normalize_embeddings=True,
            )
            long_query_embeddings = dict(zip(unique_long, long_embs))

        results = {}
        for query in queries:
            q_emb = long_query_embeddings.get(query.lower().strip())
            results[query] = self._segments_to_frame_scores(
                segments, query, timestamps, fps,
                query_embedding=q_emb,
                segment_embeddings=segment_embeddings,
            )

        return results
