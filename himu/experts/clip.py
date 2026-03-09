"""OpenCLIP expert for semantic image-text matching."""

import os
import gc
import numpy as np
from typing import List, Optional, Dict
import logging

from .base import BaseExpert, get_models_weights_dir

log = logging.getLogger(__name__)


def _make_frame_dataset(frames: np.ndarray, preprocess):
    """Create a FrameDataset (deferred torch import)."""
    from torch.utils.data import Dataset

    class FrameDataset(Dataset):
        """Dataset wrapper for numpy frames with preprocessing."""

        def __init__(self, frames, preprocess):
            self.frames = frames
            self.preprocess = preprocess

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):
            from PIL import Image
            frame = self.frames[idx]
            image = Image.fromarray(frame)
            return self.preprocess(image)

    return FrameDataset(frames, preprocess)


class OpenCLIPExpert(BaseExpert):
    """OpenCLIP-based expert for semantic image-text matching."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "dfn2b",
        device: str = "cuda",
        weights_dir: Optional[str] = None,
        num_workers: int = 4,
        batch_size: int = 256,
        use_amp: bool = True,
    ):
        try:
            import open_clip
            import torch
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for OpenCLIPExpert. "
                "Install with: pip install 'himu[clip]'"
            )

        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.torch = torch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_amp = use_amp and 'cuda' in device

        # Setup weights directory for OpenCLIP models
        base_weights_dir = get_models_weights_dir(weights_dir)
        openclip_cache_dir = base_weights_dir / "openclip"
        openclip_cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["TORCH_HOME"] = str(openclip_cache_dir)

        log.info(f"Loading OpenCLIP model ({model_name}/{pretrained}) from: {openclip_cache_dir}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
            cache_dir=str(openclip_cache_dir)
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self._embedding_dim = None

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension of this CLIP model."""
        if self._embedding_dim is None:
            if hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
                self._embedding_dim = self.model.text_projection.shape[-1]
            else:
                with self.torch.no_grad():
                    dummy_text = self.tokenizer(["test"]).to(self.device)
                    text_features = self.model.encode_text(dummy_text)
                    self._embedding_dim = text_features.shape[-1]
        return self._embedding_dim

    def compute_scores(self, frames: np.ndarray, query: str, video_path: Optional[str] = None) -> np.ndarray:
        """
        Compute cosine similarity scores between frames and query.

        Args:
            frames: RGB frames (num_frames, H, W, 3)
            query: Semantic description

        Returns:
            Scores (num_frames,) in [0, 1]
        """
        frame_embeddings = self.extract_embeddings(frames)

        with self.torch.no_grad(), self.torch.amp.autocast('cuda', enabled=self.use_amp):
            text_tokens = self.tokenizer([query]).to(self.device)
            text_features = self.model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            frame_embeddings_torch = self.torch.from_numpy(frame_embeddings).to(self.device)
            similarities = (frame_embeddings_torch @ text_features.T).squeeze(-1)
            scores = similarities.cpu().numpy()

        scores = (scores + 1.0) / 2.0

        gc.collect()
        self.torch.cuda.empty_cache()

        return np.clip(scores, 0, 1).astype(np.float32)

    def compute_batch_scores(
        self,
        frames: np.ndarray,
        queries: List[str],
        video_path: Optional[str] = None,
        cached_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Batch scoring for multiple queries.

        Args:
            frames: RGB frames (num_frames, H, W, 3)
            queries: List of semantic descriptions
            video_path: Unused, kept for API compatibility
            cached_embeddings: Optional pre-computed frame embeddings

        Returns:
            Dict mapping query -> scores array (num_frames,) in [0, 1]
        """
        if not queries:
            return {}

        frame_embeddings = cached_embeddings
        if frame_embeddings is None:
            frame_embeddings = self.extract_embeddings(frames)

        with self.torch.no_grad(), self.torch.amp.autocast('cuda', enabled=self.use_amp):
            text_tokens = self.tokenizer(queries).to(self.device)
            text_features = self.model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = self.torch.from_numpy(frame_embeddings).to(self.device)
            similarity_matrix = (image_features @ text_features.T).cpu().numpy()

        scores = {}
        for query_idx, query in enumerate(queries):
            query_scores = (similarity_matrix[:, query_idx] + 1.0) / 2.0
            scores[query] = np.clip(query_scores, 0, 1).astype(np.float32)

        gc.collect()
        self.torch.cuda.empty_cache()

        return scores

    def compute_batch_scores_with_templates(
        self,
        frames: np.ndarray,
        queries: List[str],
        templates: List[str],
        video_path: Optional[str] = None,
        cached_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Batch scoring with template-averaged text embeddings."""
        if not queries:
            return {}

        frame_embeddings = cached_embeddings
        if frame_embeddings is None:
            frame_embeddings = self.extract_embeddings(frames)

        scores = {}
        with self.torch.no_grad(), self.torch.amp.autocast('cuda', enabled=self.use_amp):
            image_features = self.torch.from_numpy(frame_embeddings).to(self.device)

            for query in queries:
                templated = [t.format(query) for t in templates]
                text_tokens = self.tokenizer(templated).to(self.device)
                text_features = self.model.encode_text(text_tokens).float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                avg_feature = text_features.mean(dim=0)
                avg_feature = avg_feature / avg_feature.norm()

                similarities = (image_features @ avg_feature).cpu().numpy()
                scores[query] = np.clip((similarities + 1.0) / 2.0, 0, 1).astype(np.float32)

        gc.collect()
        self.torch.cuda.empty_cache()

        return scores

    def extract_embeddings(self, frames: np.ndarray, video_path: Optional[str] = None) -> np.ndarray:
        """
        Extract L2-normalized image embeddings.

        Args:
            frames: RGB frames (num_frames, H, W, 3)
            video_path: Unused, kept for API compatibility

        Returns:
            Image embeddings (num_frames, embedding_dim)
        """
        all_embeddings = []

        from torch.utils.data import DataLoader
        dataset = _make_frame_dataset(frames, self.preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if 'cuda' in self.device else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        with self.torch.no_grad(), self.torch.amp.autocast('cuda', enabled=self.use_amp):
            for image_batch in dataloader:
                image_batch = image_batch.to(self.device, non_blocking=True)
                image_features = self.model.encode_image(image_batch).float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_embeddings.append(image_features.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)

        gc.collect()
        self.torch.cuda.empty_cache()

        return embeddings
