"""Add smooth zoom effects to screen recordings based on visual change detection.

For screen recordings, YOLO object detection doesn't work well (it sees the whole
frame as "laptop/tv"). Instead, we detect regions of meaningful visual change
between frames, then apply heavily smoothed zoom with a stability lock.

Key anti-jitter measures:
- Compare frames several seconds apart (not consecutive) to ignore mouse movement
- Require sustained change in the same region before zooming
- Very slow lerp (SMOOTH_FACTOR = 0.02) for buttery transitions
- Large stability threshold — only re-target when change region shifts significantly
- Minimum hold time before any zoom transition starts
"""

import cv2
import numpy as np
import subprocess
import json
import os


# --- Tuning knobs ---
SMOOTH_FACTOR = 0.02          # Very slow position/zoom interpolation
ZOOM_FACTOR = 1.25            # Subtle zoom (less = less jarring)
STABILITY_THRESHOLD = 0.20    # New target must be >20% of diagonal away
HOLD_SECONDS = 1.5            # Hold stable for 1.5s before zooming
COMPARE_GAP_SECONDS = 2.0     # Compare frames 2s apart (skips mouse flicker)
MIN_CHANGE_AREA = 0.005       # Change region must be >0.5% of frame area
ZOOM_HOLD_SECONDS = 4.0       # Stay zoomed for at least 4s before easing out


def _get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    for s in streams:
        if s["codec_type"] == "video":
            fps_parts = s["r_frame_rate"].split("/")
            fps = int(fps_parts[0]) / int(fps_parts[1])
            return {
                "width": int(s["width"]),
                "height": int(s["height"]),
                "fps": fps,
            }
    raise ValueError("No video stream found")


def _detect_change_region(
    old_gray: np.ndarray, new_gray: np.ndarray, frame_area: int
) -> tuple[int, int] | None:
    """
    Detect the centroid of meaningful visual change between two grayscale frames.
    Returns (cx, cy) or None if change is too small / too spread out.
    """
    diff = cv2.absdiff(old_gray, new_gray)
    # High threshold to ignore subtle shifts (mouse hover highlights, etc.)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    # Heavy morphological closing to merge nearby changes into one blob
    kernel = np.ones((31, 31), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only contours that are large enough (ignore cursor-sized changes)
    big = [c for c in contours if cv2.contourArea(c) > frame_area * MIN_CHANGE_AREA]
    if not big:
        return None

    # Merge all big contours into one bounding region and return its center
    all_points = np.vstack(big)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x + w // 2, y + h // 2)


def _lerp(current: float, target: float, factor: float) -> float:
    return current + (target - current) * factor


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def enhance_video(video_path: str, output_path: str, zoom_factor: float = ZOOM_FACTOR) -> str:
    """
    Smooth zoom on screen recordings driven by visual change detection.
    """
    info = _get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    diag = (w ** 2 + h ** 2) ** 0.5
    frame_area = w * h

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    compare_gap = int(fps * COMPARE_GAP_SECONDS)
    hold_frames = int(fps * HOLD_SECONDS)
    zoom_hold_frames = int(fps * ZOOM_HOLD_SECONDS)

    # Ring buffer of recent grayscale frames for gap comparison
    frame_buffer: list[np.ndarray | None] = [None] * (compare_gap + 1)

    # Smooth render state
    smooth_cx = w / 2.0
    smooth_cy = h / 2.0
    smooth_zoom = 1.0

    # Stability lock
    locked_cx = w / 2.0
    locked_cy = h / 2.0
    lock_count = 0          # consecutive detections in the same region
    zoom_active_count = 0   # frames since zoom was activated
    target_zoom = 1.0

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    report_interval = max(1, total_frames // 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf_idx = frame_idx % len(frame_buffer)
        old_idx = (frame_idx - compare_gap) % len(frame_buffer)
        frame_buffer[buf_idx] = gray

        # Only run detection once we have enough history
        old_gray = frame_buffer[old_idx]
        if old_gray is not None and frame_idx >= compare_gap:
            centroid = _detect_change_region(old_gray, gray, frame_area)

            if centroid:
                cx, cy = centroid
                dist = _distance(cx, cy, locked_cx, locked_cy)

                if dist > diag * STABILITY_THRESHOLD:
                    # Big shift — new target, reset lock
                    locked_cx = float(cx)
                    locked_cy = float(cy)
                    lock_count = 1
                    zoom_active_count = 0
                    target_zoom = 1.0  # don't zoom yet
                else:
                    # Same region — build confidence
                    lock_count += 1
                    if lock_count >= hold_frames and target_zoom < zoom_factor:
                        target_zoom = zoom_factor
                        zoom_active_count = 0
            else:
                # No change detected — start easing out after hold period
                if zoom_active_count > zoom_hold_frames:
                    target_zoom = 1.0
                    lock_count = 0

        if target_zoom > 1.0:
            zoom_active_count += 1

        # Smooth interpolation
        smooth_cx = _lerp(smooth_cx, locked_cx, SMOOTH_FACTOR)
        smooth_cy = _lerp(smooth_cy, locked_cy, SMOOTH_FACTOR)
        smooth_zoom = _lerp(smooth_zoom, target_zoom, SMOOTH_FACTOR)

        # Apply zoom
        if smooth_zoom > 1.005:
            crop_w = int(w / smooth_zoom)
            crop_h = int(h / smooth_zoom)
            cx_i, cy_i = int(smooth_cx), int(smooth_cy)
            x1 = max(0, min(cx_i - crop_w // 2, w - crop_w))
            y1 = max(0, min(cy_i - crop_h // 2, h - crop_h))
            cropped = frame[y1: y1 + crop_h, x1: x1 + crop_w]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

        out.write(frame)

        if frame_idx % report_interval == 0:
            pct = int(frame_idx / total_frames * 100)
            print(f"    Enhancing: {pct}% ({frame_idx}/{total_frames} frames)")

        frame_idx += 1

    cap.release()
    out.release()
    _reencode_if_needed(output_path)
    print(f"    Enhancement complete: {frame_idx} frames processed")
    return output_path


def _reencode_if_needed(video_path: str):
    """Re-encode to H.264 if needed for ffmpeg compatibility."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video" and s.get("codec_name") == "h264":
            return
    tmp = video_path + ".tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an", tmp,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    os.replace(tmp, video_path)
