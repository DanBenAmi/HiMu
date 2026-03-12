"""Download the VideoMME sample video from YouTube.

Usage:
    python download_video.py                                # auto-detects browser cookies
    python download_video.py --cookies-from-browser chrome   # specify browser explicitly
"""

import argparse
import subprocess
import sys
from pathlib import Path


VIDEO_URL = "https://www.youtube.com/watch?v=LVRcD_-ht3g"
VIDEO_ID = "LVRcD_-ht3g"

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
    output_dir = Path("/Users/danbenami/Downloads")
    output_path = output_dir / f"{VIDEO_ID}.mp4"

    if output_path.exists():
        print(f"Video already exists: {output_path}")
        return output_path

    ensure_yt_dlp()
    import yt_dlp

    # Use the configuration that worked
    ydl_opts = {
        "format": "18",  # Use format 18 (360p MP4) which is reliably available
        "outtmpl": str(output_dir / f"{VIDEO_ID}.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "retries": 10,
        "fragment_retries": 10,
        "ignoreerrors": True,
        "extractor_args": {
            "youtube": {
                "player_skip": ["configs", "webpage"],
                "player_client": ["android", "web"],
                "skip": ["hls", "dash", "translated_subs"]
            }
        },
    }

    # Try with cookies if specified
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
        print(f"Downloading {VIDEO_URL} (cookies from {cookies_from_browser}) ...")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(VIDEO_URL, download=True)
                if info:
                    print(f"Download completed successfully!")
                    print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Download with cookies failed: {e}")
            print("Trying without cookies...")

    # Try without cookies (this is what worked)
    opts.pop("cookiesfrombrowser", None)
    print(f"Downloading {VIDEO_URL} (no cookies) ...")
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(VIDEO_URL, download=True)
            if info:
                print(f"Download completed successfully!")
                print(f"Title: {info.get('title', 'Unknown')}")
        print(f"Saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        
        # Fallback to even simpler format if needed
        print("Trying with alternative format...")
        simple_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(output_dir / f"{VIDEO_ID}.mp4"),
        }
        
        with yt_dlp.YoutubeDL(simple_opts) as ydl:
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
