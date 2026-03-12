"""HiMu + Qwen3-VL-8B for video question answering with subtitles.

Pattern: HiMu selects frames, then Qwen3-VL-8B answers using the selected
frames + full subtitles in structured prompt format.

Run `python assets/videomme/download_video.py` first to get the video.
"""

import gc
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from himu import HiMuSelector
from himu.llm import create_llm
from _utils import print_tree

# ── Try your own video/question ──────────────────────────────────────
# Set USE_CUSTOM = True and fill in the fields below to use your own input.
# In custom mode, a single question is used instead of the 3 default ones.
USE_CUSTOM = False
CUSTOM_VIDEO_PATH = ""           # e.g. "/path/to/my_video.mp4"
CUSTOM_QUESTION = ""             # e.g. "What happens after the explosion?"
CUSTOM_CANDIDATES = []           # e.g. ["A. Fire", "B. Smoke", "C. Nothing", "D. Rain"]
CUSTOM_SUBTITLE_PATH = ""       # optional, e.g. "/path/to/subtitles.srt"


# ── Subtitle utilities ────────────────────────────────────────────────

def load_srt(srt_path):
    """Load SRT subtitle file, returning list of (start, end, text) tuples."""
    content = Path(srt_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    for block in blocks:
        lines = block.strip().split("\n")
        ts_line = None
        text_start = 0
        for i, line in enumerate(lines):
            if " --> " in line:
                ts_line = line
                text_start = i + 1
                break
        if not ts_line:
            continue
        match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            ts_line,
        )
        if not match:
            continue
        start = _parse_srt_ts(match.group(1))
        end = _parse_srt_ts(match.group(2))
        text = " ".join(lines[text_start:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            segments.append((start, end, text))
    return segments


def _parse_srt_ts(ts):
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def format_timestamp(seconds):
    """Format seconds as [HH:MM:SS]."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def build_structured_prompt(question, candidates, num_frames, subtitle_segments):
    """Build intro / subtitles block / question block for structured format.

    Returns (intro, subtitles_block, question_block).
    """
    if subtitle_segments:
        intro = (
            f"Below is a video represented by {num_frames} frames "
            f"and its accompanying subtitles.\n\nVideo:"
        )
    else:
        intro = f"Below is a video represented by {num_frames} frames.\n\nVideo:"

    # Subtitles block
    sub_lines = []
    for start, end, text in subtitle_segments:
        sub_lines.append(f"{format_timestamp(start)} - {format_timestamp(end)}: {text}")
    subtitles_block = "\n\nSubtitles:\n" + "\n".join(sub_lines) if sub_lines else ""

    # Question block
    if candidates:
        opts = "\n".join(candidates)
        question_block = (
            f"\n\nQuestion:\n{question}\n{opts}\n"
            "Answer with the option letter only."
        )
    else:
        question_block = f"\n\nQuestion:\n{question}\nAnswer concisely."

    return intro, subtitles_block, question_block


# ── Main ──────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent

    # ── Load input ───────────────────────────────────────────────────
    if USE_CUSTOM:
        video_path = CUSTOM_VIDEO_PATH
        subtitle_path = CUSTOM_SUBTITLE_PATH or None
        samples = [{
            "question": CUSTOM_QUESTION,
            "candidates": CUSTOM_CANDIDATES or [],
            "answer_letter": None,
        }]
    else:
        samples = []
        for qid in ["580-1", "580-2", "580-3"]:
            samples.append(
                json.loads((repo_root / f"assets/videomme/{qid}.json").read_text())
            )
        video_path = str(repo_root / samples[0]["video_path"])
        subtitle_path = str(repo_root / samples[0]["subtitle_path"])

    # Load subtitles
    subtitle_segments = load_srt(subtitle_path) if subtitle_path else []
    if subtitle_segments:
        print(f"Loaded {len(subtitle_segments)} subtitle segments from SRT")

    # ── Step 1: HiMu frame selection for all questions ──
    llm = create_llm("qwen3vl", model="Qwen/Qwen3-VL-8B-Instruct", device="cuda:1")
    selector = HiMuSelector(llm=llm, device="cuda:0")

    selection_results = []
    for sample in samples:
        question = sample["question"]
        candidates = sample.get("candidates") or []

        print(f"Q: {question}")
        result = selector.select_frames(
            video_path=video_path,
            question=question,
            candidates=candidates or None,
            num_frames=16,
        )
        print(f"  HiMu selected {len(result.frame_indices)} frames, "
              f"best at {result.best_timestamp:.1f}s")
        print(f"  Timestamps (s): {[f'{t:.1f}' for t in result.timestamps]}")
        print("  Logic Tree:")
        print_tree(result.tree, prefix="  ")
        selection_results.append(result)

    # Free all HiMu models (experts + LLM) to make room for VLM
    del selector, llm
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 2: Load Qwen3-VL-8B for answering ──
    print("\nLoading Qwen3-VL-8B for VLM answering...")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    vlm_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True
    )
    vlm.eval()
    print("Qwen3-VL-8B loaded.\n")

    # ── Step 3: VLM answering for each question ──
    correct = 0
    for sample, result in zip(samples, selection_results):
        question = sample["question"]
        candidates = sample.get("candidates") or []
        answer_letter = sample.get("answer_letter")

        print(f"Q: {question}")

        # Extract selected frames as PIL images
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        images = []
        for idx in result.frame_indices:
            frame_no = int(idx * video_fps / result.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            if ret:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        # Build structured prompt: intro + frames + subtitles + question
        intro, subs_block, q_block = build_structured_prompt(
            question, candidates,
            len(images), subtitle_segments,
        )

        content = [{"type": "text", "text": intro}]
        for img in images:
            content.append({"type": "image", "image": img})
        if subs_block:
            content.append({"type": "text", "text": subs_block})
        content.append({"type": "text", "text": q_block})

        messages = [{"role": "user", "content": content}]

        # Generate answer
        text = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text], images=images, return_tensors="pt"
        ).to("cuda:0")

        with torch.no_grad():
            output_ids = vlm.generate(
                **inputs, max_new_tokens=10, temperature=0.0, do_sample=False
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        answer = vlm_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        # Check answer
        if answer_letter:
            is_correct = answer_letter in answer
            correct += int(is_correct)
            mark = "\u2713" if is_correct else "\u2717"
            print(f"  VLM answer: {answer}  (GT: {answer_letter})  {mark}\n")
        else:
            print(f"  VLM answer: {answer}\n")

        # Cleanup per-iteration tensors
        del output_ids, generated, inputs, text
        for img in images:
            img.close()
        del images, content, messages
        torch.cuda.empty_cache()
        gc.collect()

    if any(s.get("answer_letter") for s in samples):
        total = sum(1 for s in samples if s.get("answer_letter"))
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")


if __name__ == "__main__":
    main()
