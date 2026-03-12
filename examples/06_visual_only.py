"""Visual-only mode: disable audio experts for silent videos or faster inference.

Uses a sample Video-MME question.
Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
from pathlib import Path

from himu import HiMuSelector, create_himu_selector
from himu.llm import create_llm
from _utils import print_tree

# ── Try your own video/question ──────────────────────────────────────
# Set USE_CUSTOM = True and fill in the fields below to use your own input.
USE_CUSTOM = False
CUSTOM_VIDEO_PATH = ""           # e.g. "/path/to/my_video.mp4"
CUSTOM_QUESTION = ""             # e.g. "What happens after the explosion?"
CUSTOM_CANDIDATES = []           # e.g. ["A. Fire", "B. Smoke", "C. Nothing", "D. Rain"]

# ── Load input ───────────────────────────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent

if USE_CUSTOM:
    video_path = CUSTOM_VIDEO_PATH
    question = CUSTOM_QUESTION
    candidates = CUSTOM_CANDIDATES or None
else:
    sample = json.loads((repo_root / "assets/videomme/580-1.json").read_text())
    video_path = str(repo_root / sample["video_path"])
    question = sample["question"]
    candidates = sample["candidates"]

# Use the visual_only preset (no ASR/CLAP experts)
llm = create_llm("qwen3vl", model="Qwen/Qwen3-VL-8B-Instruct", device="cuda:1")
selector = create_himu_selector("visual_only", llm=llm, device="cuda:0")

result = selector.select_frames(
    video_path=video_path,
    question=question,
    candidates=candidates,
    num_frames=16,
)

print(f"Question: {question}")
print()
print("Logic Tree:")
print_tree(result.tree)
print()
print(f"Selected {len(result.frame_indices)} frames (visual-only)")
print(f"Best frame at {result.best_timestamp:.1f}s (score {result.best_score:.3f})")
print(f"Frame indices: {result.frame_indices.tolist()}")
print(f"Timestamps (s): {[f'{t:.1f}' for t in result.timestamps]}")
