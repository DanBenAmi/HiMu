"""Feature caching: extract once, query many times fast.

Uses sample Video-MME questions.
Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
from pathlib import Path

from himu import HiMuSelector
from himu.llm import create_llm
from _utils import print_tree

# ── Try your own video/question ──────────────────────────────────────
# Set USE_CUSTOM = True and fill in the fields below to use your own input.
# In custom mode, a single question is used (caching still demonstrated).
USE_CUSTOM = False
CUSTOM_VIDEO_PATH = ""           # e.g. "/path/to/my_video.mp4"
CUSTOM_QUESTION = ""             # e.g. "What happens after the explosion?"
CUSTOM_CANDIDATES = []           # e.g. ["A. Fire", "B. Smoke", "C. Nothing", "D. Rain"]

# ── Load input ───────────────────────────────────────────────────────
repo_root = Path(__file__).resolve().parent.parent

if USE_CUSTOM:
    video_path = CUSTOM_VIDEO_PATH
    samples = [{
        "question": CUSTOM_QUESTION,
        "candidates": CUSTOM_CANDIDATES or None,
        "answer": None,
    }]
else:
    samples = []
    for qid in ["580-1", "580-2", "580-3"]:
        samples.append(json.loads((repo_root / f"assets/videomme/{qid}.json").read_text()))
    video_path = str(repo_root / samples[0]["video_path"])

# Use local Qwen3-VL for tree parsing
llm = create_llm("qwen3vl", model="Qwen/Qwen3-VL-8B-Instruct", device="cuda:1")
selector = HiMuSelector(llm=llm, device="cuda:0", cache_dir="/tmp/himu_cache")

# First call: extracts and caches CLIP, OCR, ASR features
status = selector.cache_features(video_path)
print(f"Cached: {status}")

# Subsequent queries reuse cached features (only OVD re-runs per query)
for sample in samples:
    result = selector.select_frames(
        video_path=video_path,
        question=sample["question"],
        candidates=sample.get("candidates"),
        num_frames=16,
    )
    print(f"Q: {sample['question']}")
    print("  Logic Tree:")
    print_tree(result.tree, prefix="  ")
    print(f"  Timestamps (s): {[f'{t:.1f}' for t in result.timestamps]}")
    print(f"  Best frame at {result.best_timestamp:.1f}s (score {result.best_score:.3f})")
    if sample.get("answer"):
        print(f"  Answer: {sample['answer']}")
    print()
