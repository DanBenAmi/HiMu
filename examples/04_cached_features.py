"""Feature caching: extract once, query many times fast.

Uses sample Video-MME questions about space debris.
Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
from pathlib import Path

from himu import HiMuSelector
from himu.llm import create_llm

# Load all three sample questions
repo_root = Path(__file__).resolve().parent.parent
samples = []
for qid in ["347-1", "347-2", "347-3"]:
    samples.append(json.loads((repo_root / f"assets/videomme/{qid}.json").read_text()))

video_path = str(repo_root / samples[0]["video_path"])

# Use local Qwen3 for tree parsing
llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
selector = HiMuSelector(llm=llm, device="cuda:0", cache_dir="/tmp/himu_cache")

# First call: extracts and caches CLIP, OCR, ASR features
status = selector.cache_features(video_path)
print(f"Cached: {status}")

# Subsequent queries reuse cached features (only YOLO re-runs per query)
for sample in samples:
    result = selector.select_frames(
        video_path=video_path,
        question=sample["question"],
        candidates=sample["candidates"],
        num_frames=16,
    )
    print(f"Q: {sample['question']}")
    print(f"  Best frame at {result.best_timestamp:.1f}s (score {result.best_score:.3f})")
    print(f"  Answer: {sample['answer']}\n")
