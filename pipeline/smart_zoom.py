"""Vision-guided smart zoom using AI analysis instead of frame differencing.

Instead of unreliable OpenCV frame diff, we use the Vision AI analysis
to identify important UI regions and zoom to them intelligently.
"""

import subprocess
import json
import os


def _get_video_info(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
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
                "duration": float(s.get("duration", 0)),
            }
    raise ValueError("No video stream found")


def enhance_video_with_vision(
    video_path: str,
    output_path: str,
    segments: list[dict],
    zoom_factor: float = 1.5,
) -> str:
    """
    Apply smart zoom using OpenCV for precise control.
    
    FFmpeg zoompan has limitations with complex expressions.
    OpenCV gives us frame-by-frame control for smooth, reliable zoom.
    """
    import cv2
    import numpy as np
    
    info = _get_video_info(video_path)
    w, h, fps = info["width"], info["height"], info["fps"]
    
    # Map zoom regions to screen coordinates (x, y as percentages)
    region_coords = {
        "top-left": (0.25, 0.25),
        "top-center": (0.5, 0.25),
        "top-right": (0.75, 0.25),
        "center-left": (0.25, 0.5),
        "center": (0.5, 0.5),
        "center-right": (0.75, 0.5),
        "bottom-left": (0.25, 0.75),
        "bottom-center": (0.5, 0.75),
        "bottom-right": (0.75, 0.75),
        "full": (0.5, 0.5),
    }
    
    # Build zoom timeline
    zoom_timeline = []
    print(f"  Analyzing {len(segments)} segments for zoom regions...")
    
    for i, seg in enumerate(segments):
        zoom_region = seg.get("zoom_region", "center")
        should_zoom = zoom_region != "full"
        
        cx, cy = region_coords.get(zoom_region, (0.5, 0.5))
        zoom_timeline.append({
            "start": seg["start"],
            "end": seg["end"],
            "center_x": cx,
            "center_y": cy,
            "zoom": zoom_factor if should_zoom else 1.0,
            "region": zoom_region,
        })
        
        status = f"ZOOM to {zoom_region}" if should_zoom else "NO ZOOM (full screen)"
        print(f"    Segment {i+1}: {status}")
    
    # Check if any zoom is needed
    has_zoom = any(z["zoom"] > 1.0 for z in zoom_timeline)
    if not has_zoom:
        print("  No zoom regions specified, skipping zoom enhancement")
        import shutil
        shutil.copy(video_path, output_path)
        return output_path
    
    print(f"  Applying OpenCV-based smart zoom...")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path + ".tmp.mp4", fourcc, fps, (w, h))
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Smooth interpolation state
    smooth_cx = 0.5
    smooth_cy = 0.5
    smooth_zoom = 1.0
    SMOOTH_FACTOR = 0.05  # Smooth transitions
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        # Find current zoom target
        target_cx = 0.5
        target_cy = 0.5
        target_zoom = 1.0
        
        for seg in zoom_timeline:
            if seg["start"] <= current_time <= seg["end"]:
                target_cx = seg["center_x"]
                target_cy = seg["center_y"]
                target_zoom = seg["zoom"]
                break
        
        # Smooth interpolation
        smooth_cx += (target_cx - smooth_cx) * SMOOTH_FACTOR
        smooth_cy += (target_cy - smooth_cy) * SMOOTH_FACTOR
        smooth_zoom += (target_zoom - smooth_zoom) * SMOOTH_FACTOR
        
        # Apply zoom if needed
        if smooth_zoom > 1.01:
            # Calculate crop region
            crop_w = int(w / smooth_zoom)
            crop_h = int(h / smooth_zoom)
            
            cx_px = int(smooth_cx * w)
            cy_px = int(smooth_cy * h)
            
            x1 = max(0, min(cx_px - crop_w // 2, w - crop_w))
            y1 = max(0, min(cy_px - crop_h // 2, h - crop_h))
            
            # Crop and resize
            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        out.write(frame)
        
        if frame_idx % 100 == 0:
            pct = int(frame_idx / total_frames * 100)
            print(f"    Processing: {pct}% ({frame_idx}/{total_frames} frames)")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    # Re-encode with H.264 and add audio
    print(f"  Re-encoding with audio...")
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path + ".tmp.mp4",
        "-i", video_path,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(output_path + ".tmp.mp4")
    
    if result.returncode != 0:
        print(f"  Warning: Re-encoding failed")
        print(f"  Error: {result.stderr[-500:]}")
        import shutil
        shutil.copy(video_path, output_path)
    else:
        print(f"  Smart zoom applied successfully")
    
    return output_path
