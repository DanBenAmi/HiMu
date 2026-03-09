"""CLAP expert for audio-text semantic matching."""

import gc
import numpy as np
from typing import List, Optional
import logging

from .base import BaseExpert, get_models_weights_dir

log = logging.getLogger(__name__)


class CLAPExpert(BaseExpert):
    """CLAP (Contrastive Language-Audio Pretraining) expert.

    Uses CLAP model for audio-text semantic matching.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        device: str = "cuda",
        weights_dir: Optional[str] = None
    ):
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CLAPExpert. "
                "Install with: pip install 'himu[clap]'"
            )

        self.device = device
        self.model_name = model_name

        base_weights_dir = get_models_weights_dir(weights_dir)
        clap_cache = base_weights_dir / "clap"
        clap_cache.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading CLAP model: {model_name}")

        self.processor = ClapProcessor.from_pretrained(
            model_name,
            cache_dir=str(clap_cache)
        )
        self.model = ClapModel.from_pretrained(
            model_name,
            cache_dir=str(clap_cache)
        ).to(device)
        self.model.eval()
        self.torch = torch

    def compute_scores(
        self,
        audio_chunks: List[np.ndarray],
        query: str
    ) -> np.ndarray:
        """
        Compute audio-text similarity scores.

        Args:
            audio_chunks: List of audio segments (one per frame), each shape (samples,)
            query: Semantic audio description (e.g., "dog barking", "music playing")

        Returns:
            Scores (num_frames,) in [0, 1]
        """
        num_frames = len(audio_chunks)
        all_scores = []

        with self.torch.no_grad():
            # Encode text query once
            text_inputs = self.processor(
                text=[query],
                return_tensors="pt"
            ).to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            if not self.torch.is_tensor(text_features):
                text_features = text_features.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            batch_size = 16

            for batch_start in range(0, num_frames, batch_size):
                batch_end = min(batch_start + batch_size, num_frames)
                batch_audio = audio_chunks[batch_start:batch_end]

                valid_indices = []
                valid_audio = []
                for idx, audio in enumerate(batch_audio):
                    if audio is not None and len(audio) > 0:
                        valid_indices.append(idx)
                        valid_audio.append(audio)

                if not valid_audio:
                    all_scores.extend([0.0] * (batch_end - batch_start))
                    continue

                try:
                    # Handle parameter name change: v4.x uses 'audios', v5.x uses 'audio'
                    try:
                        audio_inputs = self.processor(
                            audio=valid_audio,
                            sampling_rate=48000,
                            return_tensors="pt"
                        ).to(self.device)
                    except (TypeError, ValueError):
                        audio_inputs = self.processor(
                            audios=valid_audio,
                            sampling_rate=48000,
                            return_tensors="pt"
                        ).to(self.device)

                    audio_features = self.model.get_audio_features(**audio_inputs)
                    if not self.torch.is_tensor(audio_features):
                        audio_features = audio_features.pooler_output
                    audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

                    similarities = (audio_features @ text_features.T).squeeze(-1)
                    similarities_np = similarities.cpu().numpy()

                    batch_scores = [0.0] * (batch_end - batch_start)
                    for i, idx in enumerate(valid_indices):
                        batch_scores[idx] = similarities_np[i]

                    all_scores.extend(batch_scores)

                except Exception as e:
                    log.warning(f"CLAP encoding failed for batch {batch_start}-{batch_end}: {e}")
                    all_scores.extend([0.0] * (batch_end - batch_start))

        scores = np.array(all_scores, dtype=np.float32)

        # Normalize from [-1, 1] to [0, 1]
        scores = (scores + 1.0) / 2.0
        scores = np.clip(scores, 0, 1)

        gc.collect()
        self.torch.cuda.empty_cache()

        return scores
