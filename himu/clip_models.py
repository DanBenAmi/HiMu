"""CLIP model presets registry for the HiMu system."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CLIPModelPreset:
    """Configuration for a CLIP model variant."""

    name: str
    backend: Literal["siglip", "openclip"]
    model_name: str
    pretrained: Optional[str]
    embedding_dim: int
    image_size: int
    description: str
    cache_subdir: str = "clip"
    default_batch_size: int = 256


CLIP_MODEL_PRESETS = {
    "dfn-clip": CLIPModelPreset(
        name="dfn-clip",
        backend="openclip",
        model_name="ViT-L-14",
        pretrained="dfn2b",
        embedding_dim=768,
        image_size=224,
        description="Apple DFN2B ViT-L/14",
        default_batch_size=256,
    ),
    "metaclip": CLIPModelPreset(
        name="metaclip",
        backend="openclip",
        model_name="ViT-B-16",
        pretrained="metaclip_400m",
        embedding_dim=512,
        image_size=224,
        description="Meta MetaCLIP ViT-B/16",
        default_batch_size=256,
    ),
    "openai-clip": CLIPModelPreset(
        name="openai-clip",
        backend="openclip",
        model_name="ViT-B-16",
        pretrained="openai",
        embedding_dim=512,
        image_size=224,
        description="OpenAI CLIP ViT-B/16",
        default_batch_size=256,
    ),
    "dfn5b-clip": CLIPModelPreset(
        name="dfn5b-clip",
        backend="openclip",
        model_name="ViT-H-14-378-quickgelu",
        pretrained="dfn5b",
        embedding_dim=1024,
        image_size=378,
        description="Apple DFN5B ViT-H/14@378 (default)",
        default_batch_size=64,
    ),
    "laion-clip": CLIPModelPreset(
        name="laion-clip",
        backend="openclip",
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        embedding_dim=512,
        image_size=224,
        description="LAION OpenCLIP ViT-B/16",
        default_batch_size=256,
    ),
}

DEFAULT_CLIP_MODEL = "dfn5b-clip"


def get_clip_preset(preset_name: str) -> CLIPModelPreset:
    """Get a CLIP model preset by name."""
    if preset_name not in CLIP_MODEL_PRESETS:
        available = list(CLIP_MODEL_PRESETS.keys())
        raise ValueError(
            f"Unknown CLIP preset '{preset_name}'. Available: {available}"
        )
    return CLIP_MODEL_PRESETS[preset_name]


def list_clip_presets() -> list:
    """List available CLIP model preset names."""
    return list(CLIP_MODEL_PRESETS.keys())
