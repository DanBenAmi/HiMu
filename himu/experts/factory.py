"""Factory for creating expert instances."""

from typing import Optional, Dict
import logging

from .base import BaseExpert
from .yolo import YOLOExpert
from .ocr import OCRExpert
from .clip import OpenCLIPExpert
from .asr import ASRExpert
from .clap import CLAPExpert

log = logging.getLogger(__name__)


class ExpertFactory:
    """Factory for creating expert instances."""

    VISUAL_EXPERTS = ["YOLO", "CLIP", "OCR"]
    AUDIO_EXPERTS = ["ASR", "CLAP"]

    @staticmethod
    def create_expert(
        expert_type: str,
        device: str = "cuda",
        clip_config: Optional[Dict] = None,
        weights_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> BaseExpert:
        """
        Create an expert instance.

        Args:
            expert_type: One of "OCR", "YOLO", "CLIP", "ASR", "CLAP"
            device: Device for inference
            clip_config: Optional CLIP model configuration dict
            weights_dir: Optional custom directory for model weights
            model_name: Optional model name override (for ASR/CLAP)
        """
        if expert_type == "OCR":
            return OCRExpert(device=device)

        elif expert_type == "YOLO":
            return YOLOExpert(device=device, weights_dir=weights_dir)

        elif expert_type == "CLIP":
            return ExpertFactory._create_clip_expert(device, clip_config, weights_dir)

        elif expert_type == "ASR":
            asr_model = model_name or "turbo"
            return ASRExpert(
                model_name=asr_model,
                device=device,
                weights_dir=weights_dir
            )

        elif expert_type == "CLAP":
            clap_model = model_name or "laion/clap-htsat-fused"
            return CLAPExpert(
                model_name=clap_model,
                device=device,
                weights_dir=weights_dir
            )

        else:
            raise ValueError(
                f"Unknown expert type: {expert_type}. "
                f"Available: {ExpertFactory.VISUAL_EXPERTS + ExpertFactory.AUDIO_EXPERTS}"
            )

    @staticmethod
    def _create_clip_expert(
        device: str,
        clip_config: Optional[Dict] = None,
        weights_dir: Optional[str] = None
    ) -> BaseExpert:
        """Create a CLIP expert based on configuration."""
        from ..clip_models import CLIP_MODEL_PRESETS, DEFAULT_CLIP_MODEL

        if clip_config is None:
            clip_config = {}

        preset_name = clip_config.get("preset", DEFAULT_CLIP_MODEL)

        if preset_name not in CLIP_MODEL_PRESETS:
            raise ValueError(
                f"Unknown CLIP preset: {preset_name}. "
                f"Available: {list(CLIP_MODEL_PRESETS.keys())}"
            )

        preset = CLIP_MODEL_PRESETS[preset_name]

        backend = clip_config.get("backend", preset.backend)
        model_name = clip_config.get("model_name", preset.model_name)
        pretrained = clip_config.get("pretrained", preset.pretrained)
        batch_size = clip_config.get("batch_size", preset.default_batch_size)

        if backend == "openclip":
            return OpenCLIPExpert(
                model_name=model_name,
                pretrained=pretrained,
                device=device,
                weights_dir=weights_dir,
                batch_size=batch_size,
                use_amp=True,
            )
        else:
            raise ValueError(
                f"Unsupported CLIP backend: {backend}. "
                f"Only 'openclip' is supported."
            )
