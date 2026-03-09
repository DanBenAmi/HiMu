"""MCQ usage: pass answer candidates so the LLM builds a richer logic tree.

Uses a sample Video-MME question about space debris.
Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
from pathlib import Path

from himu import HiMuSelector
from himu.llm import create_llm

# Load a sample MCQ question
repo_root = Path(__file__).resolve().parent.parent
sample = json.loads((repo_root / "assets/videomme/347-2.json").read_text())

# Use local Qwen3 for tree parsing
llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
selector = HiMuSelector(llm=llm, device="cuda:0")

result = selector.select_frames(
    video_path=str(repo_root / sample["video_path"]),
    question=sample["question"],
    candidates=sample["candidates"],
    num_frames=16,
)

print(f"Question:   {sample['question']}")
print(f"Candidates: {sample['candidates']}")
print(f"Selected frames: {result.frame_indices.tolist()}")
print(f"Best frame at {result.best_timestamp:.1f}s (score {result.best_score:.3f})")
