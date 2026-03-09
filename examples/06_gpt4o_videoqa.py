"""HiMu + GPT-4o for video question answering.

Pattern: HiMu selects frames, then GPT-4o answers from the images.
Requires OPENAI_API_KEY environment variable (or .env file).

Run `python assets/videomme/download_video.py` first to get the video.
"""

import json
import re
import base64
from pathlib import Path

import cv2
from openai import OpenAI

from himu import HiMuSelector
from himu.llm import create_llm


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
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Load .env for OPENAI_API_KEY
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                import os
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    sample = json.loads((repo_root / "assets/videomme/347-3.json").read_text())
    video_path = str(repo_root / sample["video_path"])
    srt_path = str(repo_root / sample["subtitle_path"])

    subtitle_segments = load_srt(srt_path)

    # HiMu frame selection (using local Qwen3 for tree parsing)
    llm = create_llm("qwen3", model="Qwen/Qwen3-8B", device="cuda:1")
    selector = HiMuSelector(llm=llm, device="cuda:0")

    result = selector.select_frames(
        video_path=video_path,
        question=sample["question"],
        candidates=sample["candidates"],
        num_frames=16,
    )
    print(f"Q: {sample['question']}")
    print(f"HiMu selected {len(result.frame_indices)} frames, best at {result.best_timestamp:.1f}s")

    # Extract selected frames as base64
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames_b64 = []
    for idx in result.frame_indices:
        frame_no = int(idx * video_fps / result.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame)
            frames_b64.append(base64.b64encode(buf).decode())
    cap.release()

    # Build structured prompt with subtitles
    sub_lines = []
    for start, end, text in subtitle_segments:
        sub_lines.append(f"{format_timestamp(start)} - {format_timestamp(end)}: {text}")

    intro = (
        f"Below is a video represented by {len(frames_b64)} frames "
        f"and its accompanying subtitles.\n\nVideo:"
    )
    subs_block = "\n\nSubtitles:\n" + "\n".join(sub_lines)
    opts = "\n".join(sample["candidates"])
    q_block = (
        f"\n\nQuestion:\n{sample['question']}\n{opts}\n"
        "Answer with the option letter only."
    )

    # Build OpenAI content array
    content = [{"type": "text", "text": intro}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": subs_block + q_block})

    # Send to GPT-4o
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=10,
    )
    answer = response.choices[0].message.content.strip()
    is_correct = sample["answer_letter"] in answer
    mark = "✓" if is_correct else "✗"
    print(f"GPT-4o answer: {answer}  (GT: {sample['answer_letter']})  {mark}")


if __name__ == "__main__":
    main()
