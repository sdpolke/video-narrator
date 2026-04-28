"""Compose final explainer video by merging enhanced video with narration audio.

Strategy:
- Each narration segment corresponds to a time range in the source video.
- We cut that video range into a clip and speed-adjust it to match the audio duration.
- After adjusting, we measure the ACTUAL clip durations and place audio to match.
- All clips are concatenated, then the narration track is merged on top.
"""

import subprocess
import os
import json
import wave
import array
import struct

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit
SEGMENT_GAP = 0.3  # silence gap between narration segments (seconds)


# ---------------------------------------------------------------------------
# WAV utilities (handles Orpheus non-standard layout)
# ---------------------------------------------------------------------------

def _find_data_chunk(wav_path: str) -> tuple[int, int, int, int]:
    """Scan WAV for the 'data' chunk — Orpheus inserts a LIST chunk before it."""
    with open(wav_path, "rb") as f:
        content = f.read()
    sample_rate = struct.unpack_from("<I", content, 24)[0]
    channels    = struct.unpack_from("<H", content, 22)[0]
    sampwidth   = struct.unpack_from("<H", content, 34)[0] // 8
    idx = content.find(b"data", 12)
    if idx == -1:
        raise ValueError(f"No data chunk in {wav_path}")
    return idx + 8, sample_rate, channels, sampwidth


def _get_wav_duration(wav_path: str) -> float:
    pcm_start, rate, channels, sampwidth = _find_data_chunk(wav_path)
    pcm_bytes = os.path.getsize(wav_path) - pcm_start
    return (pcm_bytes // (sampwidth * channels)) / float(rate)


def _read_wav_samples(wav_path: str) -> array.array:
    pcm_start, src_rate, src_channels, sampwidth = _find_data_chunk(wav_path)
    with open(wav_path, "rb") as f:
        f.seek(pcm_start)
        raw = f.read()
    samples = array.array("h", raw)
    if src_channels == 2:
        samples = array.array("h", [
            (samples[i] + samples[i + 1]) // 2
            for i in range(0, len(samples), 2)
        ])
    if src_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / src_rate
        new_len = int(len(samples) * ratio)
        samples = array.array("h", [
            samples[min(int(i / ratio), len(samples) - 1)]
            for i in range(new_len)
        ])
    return samples


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(json.loads(result.stdout)["format"]["duration"])


def _run(cmd: list[str], label: str = ""):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error ({label}): {result.stderr[-400:]}")


def _extract_clip(video_path: str, start: float, end: float, out_path: str):
    duration = end - start
    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "copy", "-an",
        out_path,
    ], "extract_clip")


def _speed_adjust_clip(clip_path: str, target_duration: float, out_path: str):
    """Re-encode clip to play in exactly target_duration seconds."""
    clip_duration = _get_video_duration(clip_path)
    if clip_duration <= 0:
        clip_duration = 0.1
    pts_factor = target_duration / clip_duration
    _run([
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vf", f"setpts={pts_factor:.6f}*PTS",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",
        out_path,
    ], "speed_adjust")


def _concat_clips(clip_paths: list[str], out_path: str, tmp_dir: str):
    list_file = os.path.join(tmp_dir, "_concat_list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "copy", "-an",
        out_path,
    ], "concat")
    os.remove(list_file)


# ---------------------------------------------------------------------------
# Narration track builder — uses ACTUAL clip durations for placement
# ---------------------------------------------------------------------------

def _build_narration_track_synced(
    audio_segments: list[dict],
    clip_durations: list[float],
    output_path: str,
) -> float:
    """
    Build narration WAV with audio placed to match actual video clip durations.

    Each audio segment starts at the beginning of its corresponding video clip.
    This ensures audio and video stay in sync regardless of FFmpeg rounding.
    """
    # Calculate where each clip starts in the final concatenated video
    clip_starts = []
    cursor = 0.0
    for dur in clip_durations:
        clip_starts.append(cursor)
        cursor += dur

    total_duration = cursor + SEGMENT_GAP
    total_samples = int(total_duration * SAMPLE_RATE)
    track = array.array("h", [0] * total_samples)

    for i, seg in enumerate(audio_segments):
        if i >= len(clip_starts):
            break
        offset = int(clip_starts[i] * SAMPLE_RATE)
        samples = _read_wav_samples(seg["audio_path"])
        end = min(offset + len(samples), total_samples)
        for j, s in enumerate(samples[: end - offset]):
            track[offset + j] = s

    with wave.open(output_path, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(track.tobytes())

    return total_duration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose(video_path: str, audio_segments: list[dict], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_dir = os.path.dirname(output_path) or "."

    video_duration = _get_video_duration(video_path)

    # --- Compute audio durations ---
    print(f"  Sequencing {len(audio_segments)} audio segments...")
    audio_durations = []
    for seg in audio_segments:
        audio_durations.append(_get_wav_duration(seg["audio_path"]))

    # --- Evenly divide source video across segments ---
    n = len(audio_segments)
    seg_len = video_duration / n if n > 0 else video_duration
    video_ranges = [(i * seg_len, (i + 1) * seg_len) for i in range(n)]

    # --- Cut + speed-adjust each clip to match its audio duration ---
    print(f"  Cutting and adjusting {n} video clips...")
    adjusted_clips = []
    actual_clip_durations = []

    for i, seg in enumerate(audio_segments):
        v_start, v_end = video_ranges[i]
        target_dur = audio_durations[i] + SEGMENT_GAP  # include gap in video clip

        raw_clip = os.path.join(tmp_dir, f"_clip_{i:03d}_raw.mp4")
        adj_clip = os.path.join(tmp_dir, f"_clip_{i:03d}_adj.mp4")

        _extract_clip(video_path, v_start, v_end, raw_clip)
        _speed_adjust_clip(raw_clip, target_dur, adj_clip)
        os.remove(raw_clip)

        # Measure ACTUAL output duration (may differ slightly from target)
        actual_dur = _get_video_duration(adj_clip)
        actual_clip_durations.append(actual_dur)
        adjusted_clips.append(adj_clip)

        print(f"    Clip {i+1:02d}: video [{v_start:.1f}s-{v_end:.1f}s] → "
              f"audio {audio_durations[i]:.1f}s, clip {actual_dur:.1f}s")

    # --- Concatenate all adjusted clips ---
    print(f"  Concatenating clips...")
    concat_path = os.path.join(tmp_dir, "_concat.mp4")
    _concat_clips(adjusted_clips, concat_path, tmp_dir)
    for p in adjusted_clips:
        if os.path.exists(p):
            os.remove(p)

    # --- Build narration WAV synced to actual clip durations ---
    print(f"  Building synced narration track...")
    narration_path = os.path.join(tmp_dir, "_narration_tmp.wav")
    _build_narration_track_synced(audio_segments, actual_clip_durations, narration_path)

    # --- Merge video + audio ---
    print(f"  Merging video + audio...")
    _run([
        "ffmpeg", "-y",
        "-i", concat_path,
        "-i", narration_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ], "final_merge")

    for f in [concat_path, narration_path]:
        if os.path.exists(f):
            os.remove(f)

    final_dur = _get_video_duration(output_path)
    print(f"  Final video: {output_path} ({final_dur:.1f}s)")
    return output_path


def overlay_avatar_hybrid(
    video_path: str,
    avatar_video_path: str | None,
    portrait_path: str,
    output_path: str,
    position: str = "bottom-right",
    size_ratio: float = 0.2,
    margin: int = 20,
    shape: str = "circle",
) -> str:
    """
    Overlay avatar as PiP on the main video.

    - For the duration covered by avatar_video_path: use the talking video.
    - For any remaining duration: freeze on the static portrait image.
    - If avatar_video_path is None: use static portrait for the full duration.

    Args:
        video_path:        Main explainer video (with audio).
        avatar_video_path: Talking-head video from D-ID (may be None or partial).
        portrait_path:     Static fallback portrait image.
        output_path:       Final output path.
        position:          PiP corner position.
        size_ratio:        Avatar size as fraction of main video height.
        margin:            Pixel margin from edges.
        shape:             "circle" or "rectangle".
    """
    main_dur = _get_video_duration(video_path)

    # Get main video dimensions
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
    streams = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)["streams"]
    main_w = main_h = 0
    for s in streams:
        if s["codec_type"] == "video":
            main_w, main_h = int(s["width"]), int(s["height"])
            break

    pip_size = int(main_h * size_ratio)
    positions = {
        "bottom-right": f"main_w-overlay_w-{margin}:main_h-overlay_h-{margin}",
        "bottom-left":  f"{margin}:main_h-overlay_h-{margin}",
        "top-right":    f"main_w-overlay_w-{margin}:{margin}",
        "top-left":     f"{margin}:{margin}",
    }
    pos = positions.get(position, positions["bottom-right"])
    tmp_dir = os.path.dirname(output_path) or "."

    def _circle_filter(input_label: str, out_label: str) -> str:
        # Correct circular mask: scale → split → geq alpha mask → alphamerge → overlay
        # This avoids the green tinge caused by yuva420p chroma bleed on JPEGs
        r = pip_size // 2
        return (
            f"{input_label}scale={pip_size}:{pip_size},format=yuva420p[_sc];"
            f"[_sc]split[_rgb][_a];"
            f"[_a]geq=lum=255:a='if(lt(pow(X-{r},2)+pow(Y-{r},2),pow({r}-1,2)),255,0)'[_mask];"
            f"[_rgb][_mask]alphamerge{out_label}"
        )

    def _rect_filter(input_label: str, out_label: str) -> str:
        return f"{input_label}scale={pip_size}:{pip_size}{out_label}"

    pip_filter = _circle_filter if shape == "circle" else _rect_filter

    avatar_dur = _get_video_duration(avatar_video_path) if avatar_video_path else 0.0
    print(f"  Main video: {main_dur:.1f}s | Avatar video: {avatar_dur:.1f}s")

    if avatar_dur >= main_dur - 0.5:
        # Avatar covers the whole video — simple overlay
        print(f"  Full avatar overlay ({shape}, {position})...")
        filt = pip_filter("[1:v]", "[pip];") + f"[0:v][pip]overlay={pos}:shortest=1"
        _run([
            "ffmpeg", "-y",
            "-i", video_path, "-i", avatar_video_path,
            "-filter_complex", filt,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest",
            output_path,
        ], "overlay_full")

    elif avatar_dur > 0:
        # Partial: talking video for first avatar_dur seconds, static image for the rest
        print(f"  Hybrid overlay: talking avatar for {avatar_dur:.1f}s, static image for remaining {main_dur - avatar_dur:.1f}s...")

        # Part 1: main[0..avatar_dur] + avatar video overlay
        part1 = os.path.join(tmp_dir, "_hybrid_part1.mp4")
        filt1 = pip_filter("[1:v]", "[pip];") + f"[0:v][pip]overlay={pos}:shortest=1"
        _run([
            "ffmpeg", "-y",
            "-i", video_path, "-i", avatar_video_path,
            "-filter_complex", filt1,
            "-t", f"{avatar_dur:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            part1,
        ], "hybrid_part1")

        # Part 2: main[avatar_dur..end] + static portrait overlay
        part2 = os.path.join(tmp_dir, "_hybrid_part2.mp4")
        filt2 = pip_filter("[1:v]", "[pip];") + f"[0:v][pip]overlay={pos}"
        _run([
            "ffmpeg", "-y",
            "-ss", f"{avatar_dur:.3f}", "-i", video_path,
            "-loop", "1", "-i", portrait_path,
            "-filter_complex", filt2,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest",
            part2,
        ], "hybrid_part2")

        # Concatenate part1 + part2
        _concat_clips([part1, part2], output_path, tmp_dir)
        for p in [part1, part2]:
            if os.path.exists(p):
                os.remove(p)

    else:
        # No avatar video at all — static portrait for full duration
        print(f"  Static portrait overlay for full {main_dur:.1f}s...")
        filt = pip_filter("[1:v]", "[pip];") + f"[0:v][pip]overlay={pos}"
        _run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-loop", "1", "-i", portrait_path,
            "-filter_complex", filt,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest",
            output_path,
        ], "overlay_static")

    final_dur = _get_video_duration(output_path)
    print(f"  Avatar overlay complete: {output_path} ({final_dur:.1f}s)")
    return output_path


def overlay_avatar(
    video_path: str,
    avatar_path: str,
    output_path: str,
    position: str = "bottom-right",
    size_ratio: float = 0.2,
    margin: int = 20,
    shape: str = "circle",
) -> str:
    """
    Overlay a talking avatar video as picture-in-picture on the main video.

    Args:
        video_path: Main explainer video (with audio).
        avatar_path: Avatar talking-head video (no audio needed).
        output_path: Final output path.
        position: "bottom-right", "bottom-left", "top-right", "top-left".
        size_ratio: Avatar size as fraction of main video height (0.2 = 20%).
        margin: Pixel margin from edges.
        shape: "circle" for circular PiP, "rectangle" for rectangular.

    Returns:
        Path to the output video with avatar overlay.
    """
    # Get main video dimensions
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    streams = json.loads(result.stdout)["streams"]
    main_w = main_h = 0
    for s in streams:
        if s["codec_type"] == "video":
            main_w, main_h = int(s["width"]), int(s["height"])
            break

    pip_size = int(main_h * size_ratio)

    # Position calculation
    positions = {
        "bottom-right": f"main_w-overlay_w-{margin}:main_h-overlay_h-{margin}",
        "bottom-left":  f"{margin}:main_h-overlay_h-{margin}",
        "top-right":    f"main_w-overlay_w-{margin}:{margin}",
        "top-left":     f"{margin}:{margin}",
    }
    pos = positions.get(position, positions["bottom-right"])

    # Build filter: scale avatar, optionally make circular, overlay
    r = pip_size // 2
    if shape == "circle":
        avatar_filter = (
            f"[1:v]scale={pip_size}:{pip_size},format=yuva420p[_sc];"
            f"[_sc]split[_rgb][_a];"
            f"[_a]geq=lum=255:a='if(lt(pow(X-{r},2)+pow(Y-{r},2),pow({r}-1,2)),255,0)'[_mask];"
            f"[_rgb][_mask]alphamerge[avatar];"
            f"[0:v][avatar]overlay={pos}:shortest=1"
        )
    else:
        avatar_filter = (
            f"[1:v]scale={pip_size}:{pip_size}[avatar];"
            f"[0:v][avatar]overlay={pos}:shortest=1"
        )

    print(f"  Overlaying avatar ({shape}, {position}, {pip_size}px)...")
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", avatar_path,
        "-filter_complex", avatar_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        "-shortest",
        output_path,
    ], "avatar_overlay")

    final_dur = _get_video_duration(output_path)
    print(f"  Avatar overlay complete: {output_path} ({final_dur:.1f}s)")
    return output_path
