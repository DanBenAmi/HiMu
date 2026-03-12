"""YOLO-World expert for open-vocabulary object detection."""

import gc
import os
import shutil
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict
import logging

from .base import BaseExpert, get_models_weights_dir

log = logging.getLogger(__name__)


def _patch_ultralytics_select_device():
    """Monkey-patch ultralytics select_device to prevent CUDA_VISIBLE_DEVICES corruption.

    ultralytics' select_device() sets os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    on every predict() call. In Exclusive Process GPU mode this causes CUDA
    to attempt accessing an occupied GPU, resulting in "device busy" errors.

    Instead, we skip the original entirely and return torch.device("cuda:0").
    """
    import ultralytics.utils.torch_utils as torch_utils
    import torch

    _original = torch_utils.select_device
    if getattr(_original, '_patched', False):
        return

    def _safe_select_device(*args, **kwargs):
        return torch.device("cuda:0")

    _safe_select_device._patched = True

    # Patch the canonical location
    torch_utils.select_device = _safe_select_device

    # Patch all submodules that import select_device directly
    import importlib
    for mod_name in [
        'ultralytics.engine.predictor',
        'ultralytics.engine.validator',
        'ultralytics.utils.checks',
        'ultralytics.utils.benchmarks',
    ]:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, 'select_device'):
                mod.select_device = _safe_select_device
        except ImportError:
            pass


class OVDExpert(BaseExpert):
    """YOLO-World for open-vocabulary object detection."""

    def __init__(
        self,
        model_name: str = "yolov8x-worldv2",
        device: str = "cuda",
        weights_dir: Optional[str] = None
    ):
        try:
            from ultralytics import YOLOWorld
            import torch
        except ImportError:
            raise ImportError(
                "ultralytics is required for OVDExpert. "
                "Install with: pip install 'himu[ovd]'"
            )

        self.torch = torch
        self.model_name = model_name

        self.weights_dir = get_models_weights_dir(weights_dir)
        ovd_weights_dir = self.weights_dir / "ovd"
        ovd_weights_dir.mkdir(parents=True, exist_ok=True)

        model_filename = f"{model_name}.pt"
        local_weights_path = ovd_weights_dir / model_filename

        if local_weights_path.exists():
            log.info(f"Loading OVD weights from: {local_weights_path}")
            model_to_load = str(local_weights_path)
        else:
            log.info(f"OVD weights not found in {ovd_weights_dir}, will download and save")
            model_to_load = model_name

        _patch_ultralytics_select_device()

        self.device = device
        log.info(f"YOLO-World using device: {device}")
        self.model = YOLOWorld(model_to_load)
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(source=dummy, verbose=False, device=device)

        if not local_weights_path.exists():
            self._copy_weights_to_local(model_filename, local_weights_path)

    def _copy_weights_to_local(self, model_filename: str, target_path: Path):
        """Copy downloaded weights to our local models_weights directory."""
        possible_sources = [
            Path.home() / ".config" / "Ultralytics" / model_filename,
            Path.cwd() / model_filename,
            Path(model_filename),
        ]

        for source in possible_sources:
            if source.exists():
                try:
                    shutil.copy2(source, target_path)
                    log.info(f"Copied OVD weights to: {target_path}")
                    return
                except Exception as e:
                    log.warning(f"Failed to copy OVD weights: {e}")

        log.warning(f"Could not find downloaded OVD weights to copy to {target_path}")

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations to improve detection robustness."""
        variations = []

        variations.append(query)

        if not query.endswith('s'):
            variations.append(query + 's')
        else:
            if query.endswith('es'):
                variations.append(query[:-2])
            else:
                variations.append(query[:-1])

        words = query.split()
        if len(words) > 1:
            variations.append(words[-1])
            if not words[-1].endswith('s'):
                variations.append(words[-1] + 's')

        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)

        return unique_variations

    def compute_scores(self, frames: np.ndarray, query: str) -> np.ndarray:
        """
        Run YOLO-World detection and return confidence scores.

        Args:
            frames: RGB frames (num_frames, H, W, 3)
            query: Object to detect (e.g., "red car", "person")

        Returns:
            Scores (num_frames,) in [0, 1]
        """
        num_frames = len(frames)
        scores = np.zeros(num_frames, dtype=np.float32)

        query_variations = self._generate_query_variations(query)

        for query_var in query_variations:
            self.model.set_classes([query_var])

            for i, frame in enumerate(frames):
                results = self.model.predict(
                    source=frame,
                    conf=0.001,
                    verbose=False,
                    device=self.device
                )

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        confidences = boxes.conf.cpu().numpy()
                        frame_score = float(np.max(confidences))
                        scores[i] = max(scores[i], frame_score)

        scores = np.clip(scores, 0, 1)

        gc.collect()
        self.torch.cuda.empty_cache()

        return scores

    def compute_batch_scores(
        self,
        frames: np.ndarray,
        queries: List[str],
        batch_size: int = 512
    ) -> Dict[str, np.ndarray]:
        """
        Batch OVD detection across multiple queries with frame batching.

        Sets all class variations at once and runs batch inference.
        """
        num_frames = len(frames)

        if not queries:
            return {}

        # Step 1: Generate all variations and track mapping
        all_classes = []
        class_to_query = {}

        for query in queries:
            variations = self._generate_query_variations(query)
            for var in variations:
                if var not in class_to_query:
                    all_classes.append(var)
                    class_to_query[var] = query

        # Step 2: Set all classes at once
        self.model.set_classes(all_classes)

        # Step 3: Initialize score arrays
        scores = {query: np.zeros(num_frames, dtype=np.float32) for query in queries}

        # Step 4: Process frames in batches
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_frames = frames[batch_start:batch_end]

            results = self.model.predict(
                source=list(batch_frames),
                conf=0.001,
                verbose=False,
                device=self.device
            )

            for local_idx, result in enumerate(results):
                frame_idx = batch_start + local_idx
                if result.boxes is not None and len(result.boxes) > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)

                    for conf, cls_id in zip(confidences, class_ids):
                        if cls_id < len(all_classes):
                            detected_class = all_classes[cls_id]
                            original_query = class_to_query[detected_class]
                            scores[original_query][frame_idx] = max(
                                scores[original_query][frame_idx],
                                float(conf)
                            )

        gc.collect()
        self.torch.cuda.empty_cache()

        return {query: np.clip(s, 0, 1) for query, s in scores.items()}
