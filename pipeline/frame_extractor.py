"""Extract frames from video at regular intervals using FFmpeg."""

import subprocess
import os
import json
from pathlib import Path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list[dict]:
    """
    Extract frames from video at the given FPS rate.

    Returns a list of dicts: [{"path": "frame_0001.jpg", "timestamp": 0.0}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    frames = []
    for f in sorted(Path(output_dir).glob("frame_*.jpg")):
        # Frame number is 1-indexed from ffmpeg
        frame_num = int(f.stem.split("_")[1])
        timestamp = (frame_num - 1) / fps
        frames.append({"path": str(f), "timestamp": timestamp})

    return frames
