"""Configuration settings for the HiMu system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpertConfig:
    """Configuration for individual experts."""
    enabled: bool = True
    model_name: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 512  # Batch size for inference (used by OVD)


@dataclass
class SmoothingConfig:
    """Bandwidth-matched smoothing configuration."""
    enabled: bool = True
    visual_sigma: float = 0.5   # seconds - for CLIP, OVD, OCR
    speech_sigma: float = 1.5   # seconds - for ASR/Whisper
    audio_sigma: float = 2.0    # seconds - for CLAP


@dataclass
class CompositionConfig:
    """Logic composition configuration."""
    method: str = "fuzzy_logic_tree"  # "fuzzy_logic_tree", "additive", "multiplicative"
    kappa: float = 2.0  # Decay rate for RIGHT_AFTER operator
    and_mode: str = "product"  # "product", "min", "geometric_mean"
    post_restandardize: bool = False  # Re-standardize OR branches to [0,1] before combining
    rescale_range: Optional[Tuple[float, float]] = (0.5, 1.0)  # Rescale child signals to [lo, hi] before each operator


@dataclass
class SelectionConfig:
    """Frame selection configuration."""
    mode: str = "pass"  # "pass" or "score_ranked"
    min_gap: Optional[int] = None  # Minimum gap between selected frames
    max_frames_per_window: Optional[int] = None  # Max frames in any time window (None=disabled)
    window_seconds: float = 4.0  # Window size in seconds for density constraint
    peak_prominence: float = 0.01  # Prominence ratio for adaptive peak detection (pass only)


@dataclass
class NormalizationConfig:
    """Score normalization configuration."""
    method: str = "robust"  # "robust", "minmax", "zscore"
    delta: float = 1e-6
    gamma: float = 3.0
    joint: bool = True  # Joint normalization across sibling signals from same expert


@dataclass
class HiMuConfig:
    """Complete HiMu pipeline configuration."""

    # Expert configurations
    ovd: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        enabled=True, model_name="yolov8x-worldv2", batch_size=512
    ))
    clip: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        enabled=True
    ))
    ocr: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        enabled=True
    ))
    asr: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        enabled=True, model_name="turbo"  # faster-whisper large-v3-turbo
    ))
    clap: ExpertConfig = field(default_factory=lambda: ExpertConfig(
        enabled=True, model_name="laion/clap-htsat-fused"
    ))

    # Pipeline stages
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    # General settings
    fps: float = 1.0
    device: str = "cuda"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    weights_dir: Optional[str] = None

    # CLIP configuration
    clip_preset: str = "dfn5b-clip"  # CLIP model preset
    clip_templates: Optional[List[str]] = None  # Template averaging (None = disabled)

    # Cache configuration
    cache_dir: Optional[str] = None  # Directory for feature caching

    # Multi-GPU configuration
    gpu_map: Optional[Dict[str, List[int]]] = None  # Expert -> GPU assignments

    def is_expert_enabled(self, expert_type: str) -> bool:
        """Check if an expert type is enabled."""
        expert_map = {
            "OVD": self.ovd,
            "CLIP": self.clip,
            "OCR": self.ocr,
            "ASR": self.asr,
            "CLAP": self.clap,
        }
        expert_config = expert_map.get(expert_type)
        return expert_config.enabled if expert_config else False

    def get_expert_model_name(self, expert_type: str) -> Optional[str]:
        """Get model name for an expert type."""
        expert_map = {
            "OVD": self.ovd,
            "CLIP": self.clip,
            "OCR": self.ocr,
            "ASR": self.asr,
            "CLAP": self.clap,
        }
        expert_config = expert_map.get(expert_type)
        return expert_config.model_name if expert_config else None


# Default configurations for common use cases
DEFAULT_CONFIG = HiMuConfig()

VISUAL_ONLY_CONFIG = HiMuConfig(
    asr=ExpertConfig(enabled=False),
    clap=ExpertConfig(enabled=False),
)

FAST_CONFIG = HiMuConfig(
    ovd=ExpertConfig(enabled=False),
    asr=ExpertConfig(enabled=False),
    clap=ExpertConfig(enabled=False),
    smoothing=SmoothingConfig(enabled=False),
)

ABLATION_NO_SMOOTHING_CONFIG = HiMuConfig(
    smoothing=SmoothingConfig(enabled=False),
)

ABLATION_ADDITIVE_CONFIG = HiMuConfig(
    composition=CompositionConfig(method="additive"),
)
