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

    # Question block with MCQ options
    opts = "\n".join(candidates)
    question_block = (
        f"\n\nQuestion:\n{question}\n{opts}\n"
        "Answer with the option letter only."
    )

    return intro, subtitles_block, question_block


# ── Main ──────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Load all three sample questions
    samples = []
    for qid in ["347-1", "347-2", "347-3"]:
        samples.append(
            json.loads((repo_root / f"assets/videomme/{qid}.json").read_text())
        )

    video_path = str(repo_root / samples[0]["video_path"])
    srt_path = str(repo_root / samples[0]["subtitle_path"])

    # Load subtitles once
    subtitle_segments = load_srt(srt_path)
    print(f"Loaded {len(subtitle_segments)} subtitle segments from SRT")

    # ── Step 1: Set up HiMu frame selector ──
    llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
    selector = HiMuSelector(llm=llm, device="cuda:0")

    # ── Step 2: Load Qwen3-VL-8B for answering ──
    print("Loading Qwen3-VL-8B for VLM answering...")
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

    # ── Step 3: Process each question ──
    correct = 0
    for sample in samples:
        print(f"Q: {sample['question']}")

        # HiMu frame selection
        result = selector.select_frames(
            video_path=video_path,
            question=sample["question"],
            candidates=sample["candidates"],
            num_frames=16,
        )
        print(f"  HiMu selected {len(result.frame_indices)} frames, "
              f"best at {result.best_timestamp:.1f}s")

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
            sample["question"], sample["candidates"],
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
        is_correct = sample["answer_letter"] in answer
        correct += int(is_correct)
        mark = "✓" if is_correct else "✗"
        print(f"  VLM answer: {answer}  (GT: {sample['answer_letter']})  {mark}\n")

        # Cleanup
        del output_ids, generated, inputs, text
        for img in images:
            img.close()
        del images, content, messages
        torch.cuda.empty_cache()
        gc.collect()

    print(f"Accuracy: {correct}/{len(samples)} ({100*correct/len(samples):.0f}%)")


if __name__ == "__main__":
    main()
