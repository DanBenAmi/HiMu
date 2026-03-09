"""Download the VideoMME sample video from YouTube.

Usage:
    python download_video.py                                # auto-detects browser cookies
    python download_video.py --cookies-from-browser chrome   # specify browser explicitly
"""

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_URL = "https://www.youtube.com/watch?v=VQKpMmBDtZo"
VIDEO_ID = "VQKpMmBDtZo"

# Browsers to try for cookie extraction (in order of preference)
BROWSERS = ["chrome", "chromium", "firefox", "edge", "opera", "brave", "safari"]


def ensure_yt_dlp():
    """Install yt-dlp if not available."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])


def download_video(cookies_from_browser=None):
    output_dir = Path(__file__).parent
    output_path = output_dir / f"{VIDEO_ID}.mp4"

    if output_path.exists():
        print(f"Video already exists: {output_path}")
        return output_path

    ensure_yt_dlp()
    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / f"{VIDEO_ID}.%(ext)s"),
        "merge_output_format": "mp4",
    }

    browsers_to_try = [cookies_from_browser] if cookies_from_browser else BROWSERS

    for browser in browsers_to_try:
        opts = {**ydl_opts, "cookiesfrombrowser": (browser,)}
        print(f"Downloading {VIDEO_URL} (cookies from {browser}) ...")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([VIDEO_URL])
            print(f"Saved to {output_path}")
            return output_path
        except Exception as e:
            if cookies_from_browser:
                raise
            print(f"  {browser} failed: {e}\n")

    # Last resort: try without cookies
    print(f"Downloading {VIDEO_URL} (no cookies) ...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([VIDEO_URL])
    print(f"Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VideoMME sample video")
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser to extract cookies from (e.g. chrome, firefox). "
        "If not specified, tries common browsers automatically.",
    )
    args = parser.parse_args()
    download_video(cookies_from_browser=args.cookies_from_browser)
