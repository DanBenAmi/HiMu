"""Multi-GPU configuration: assign experts to specific GPUs.

Uses a sample Video-MME question about space debris.
Run `python assets/videomme/download_video.py` first to get the video.
Requires 2 GPUs (GPU 0 for visual experts, GPU 1 for audio + LLM).
"""

import json
from pathlib import Path

from himu import HiMuSelector, HiMuConfig
from himu.llm import create_llm

# Load a sample question
repo_root = Path(__file__).resolve().parent.parent
sample = json.loads((repo_root / "assets/videomme/347-1.json").read_text())

config = HiMuConfig(
    gpu_map={
        "CLIP": [0],
        "YOLO": [0],
        "OCR":  [1],
        "ASR":  [1],
        "CLAP": [1],
    }
)

llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
selector = HiMuSelector(config=config, llm=llm, device="cuda:0")

result = selector.select_frames(
    video_path=str(repo_root / sample["video_path"]),
    question=sample["question"],
    num_frames=16,
)

print(f"Question: {sample['question']}")
print(f"Selected frames: {result.frame_indices.tolist()}")
