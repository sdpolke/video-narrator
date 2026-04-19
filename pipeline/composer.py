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
