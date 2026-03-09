"""HiMu: Hierarchical Multimodal Frame Selection for Long Video QA."""

__version__ = "1.0.0"

from .selector import HiMuSelector, create_himu_selector
from .config import HiMuConfig
from ._types import FrameSelectionResult

__all__ = [
    "HiMuSelector",
    "create_himu_selector",
    "HiMuConfig",
    "FrameSelectionResult",
    "__version__",
]
