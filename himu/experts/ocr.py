"""docTR OCR expert for text detection and recognition.

Uses db_resnet50 (detection) + parseq (recognition) via PyTorch backend.
"""

import numpy as np
from typing import List, Optional, Dict
import warnings
import logging

from .base import BaseExpert

log = logging.getLogger(__name__)


class OCRExpert(BaseExpert):
    """docTR OCR for text detection and recognition.

    Uses db_resnet50 for detection and parseq for recognition.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        det_arch: str = "db_resnet50",
        reco_arch: str = "parseq",
        weights_dir: Optional[str] = None,
    ):
        try:
            from doctr.models import ocr_predictor
        except ImportError as e:
            raise ImportError(
                f"Please install python-doctr: {e}\n"
                "pip install 'himu[ocr]'"
            )

        warnings.filterwarnings("ignore")

        import torch
        self.device = torch.device(device)
        self.model = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
        ).to(self.device)

        log.info(f"docTR OCR initialized on {device} (det={det_arch}, reco={reco_arch})")

    def _extract_text_from_frames_batch(self, frames: List[np.ndarray]) -> List[List[tuple]]:
        """Extract text from multiple frames in a single batched docTR call."""
        if not frames:
            return []

        try:
            result = self.model(frames)
        except Exception as e:
            log.warning(f"Batch OCR failed ({len(frames)} frames), falling back to per-frame: {e}")
            return [self._extract_text_from_frame(f) for f in frames]

        batch_texts = []
        for page in result.pages:
            text_results = []
            for block in page.blocks:
                for line in block.lines:
                    words = [w.value for w in line.words if w.value.strip()]
                    confs = [w.confidence for w in line.words if w.value.strip()]
                    if words:
                        line_text = " ".join(words)
                        line_conf = min(confs)
                        text_results.append((line_text, float(line_conf)))
            batch_texts.append(text_results)

        return batch_texts

    def _extract_text_from_frame(self, frame: np.ndarray) -> List[tuple]:
        """Extract all text from a frame using docTR."""
        try:
            result = self.model([frame])

            if not result.pages:
                return []

            text_results = []
            for block in result.pages[0].blocks:
                for line in block.lines:
                    words = [w.value for w in line.words if w.value.strip()]
                    confs = [w.confidence for w in line.words if w.value.strip()]
                    if words:
                        line_text = " ".join(words)
                        line_conf = min(confs)
                        text_results.append((line_text, float(line_conf)))

            return text_results

        except Exception as e:
            log.warning(f"Per-frame OCR failed: {e}")
            return []

    def _match_query_to_texts(self, query: str, text_results: List[tuple]) -> float:
        """Match a query against detected texts from a frame."""
        if not text_results:
            return 0.0

        detected_texts = [text.lower().strip() for text, _ in text_results]
        best_score = 0.0
        for text in detected_texts:
            if query in text:
                return 1.0
            score = self._text_similarity(query, text)
            best_score = max(best_score, score)
        return best_score

    def compute_scores(self, frames: np.ndarray, query: str) -> np.ndarray:
        """
        Run OCR and compute similarity to target text.

        Args:
            frames: RGB frames (num_frames, H, W, 3)
            query: Target text to find

        Returns:
            Scores (num_frames,) in [0, 1]
        """
        num_frames = len(frames)
        scores = np.zeros(num_frames, dtype=np.float32)
        query_lower = query.lower().strip()

        for i, frame in enumerate(frames):
            try:
                text_results = self._extract_text_from_frame(frame)
                scores[i] = self._match_query_to_texts(query_lower, text_results)
            except Exception:
                continue

        return scores

    def compute_batch_scores(
        self,
        frames: np.ndarray,
        queries: List[str],
        cached_texts: Optional[List[Optional[List[tuple]]]] = None
    ) -> Dict[str, np.ndarray]:
        """Batch scoring for multiple queries, with optional cached OCR texts."""
        num_frames = len(frames)

        if cached_texts is not None:
            log.info(f"Using cached OCR texts for {num_frames} frames (skipping docTR)")
            all_texts = cached_texts
        else:
            all_texts = []
            batch_size = 8
            for i in range(0, num_frames, batch_size):
                batch = [frames[j] for j in range(i, min(i + batch_size, num_frames))]
                batch_texts = self._extract_text_from_frames_batch(batch)
                all_texts.extend(batch_texts)

        results = {}
        for query in queries:
            scores = np.zeros(num_frames, dtype=np.float32)
            query_lower = query.lower().strip()
            for i, text_results in enumerate(all_texts):
                if text_results:
                    scores[i] = self._match_query_to_texts(query_lower, text_results)
            results[query] = scores

        return results

    @staticmethod
    def _text_similarity(s1: str, s2: str) -> float:
        """Compute normalized similarity using Levenshtein distance."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        dist = np.zeros((len1 + 1, len2 + 1), dtype=int)
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
                    dist[i - 1][j - 1] + cost
                )

        max_len = max(len1, len2)
        similarity = 1.0 - (dist[len1][len2] / max_len)
        return max(0.0, similarity)
