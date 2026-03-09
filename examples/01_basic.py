"""Basic HiMu usage: select question-relevant frames from a video.

Uses a sample Video-MME question about space debris.
Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
from pathlib import Path

from himu import HiMuSelector
from himu.llm import create_llm

# Load a sample question
repo_root = Path(__file__).resolve().parent.parent
sample = json.loads((repo_root / "assets/videomme/347-1.json").read_text())

# Use local Qwen3 for tree parsing (no API key needed)
llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
selector = HiMuSelector(llm=llm, device="cuda:0")

result = selector.select_frames(
    video_path=str(repo_root / sample["video_path"]),
    question=sample["question"],
    num_frames=16,
)

print(f"Question:         {sample['question']}")
print(f"Selected {len(result.frame_indices)} frames")
print(f"Best frame index: {result.best_frame_idx}")
print(f"Best timestamp:   {result.best_timestamp:.1f}s")
print(f"Best score:       {result.best_score:.3f}")
print(f"Frame indices:    {result.frame_indices.tolist()}")
