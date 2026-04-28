"""
Video Narrator - Turn screen recordings into narrated explainer videos.

Uses Groq APIs for vision analysis, script generation, and TTS.
"""

import argparse
import os
import subprocess
import sys
import shutil
from dotenv import load_dotenv

from pipeline.frame_extractor import extract_frames, get_video_duration
from pipeline.vision_analyzer import analyze_frames
from pipeline.script_generator import generate_script
from pipeline.tts_generator import generate_audio_segments
from pipeline.smart_zoom import enhance_video_with_vision
from pipeline.composer import compose, overlay_avatar, overlay_avatar_hybrid
from pipeline.avatar_generator import generate_avatar_video


def check_dependencies():
    """Verify required external tools are installed."""
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            print(f"Error: '{tool}' not found. Install FFmpeg first.")
            sys.exit(1)


def _build_full_narration(audio_segments: list[dict], output_path: str):
    """Concatenate all audio segment WAVs into a single file for avatar generation.
    Uses raw PCM reading to handle Orpheus corrupt WAV headers."""
    import wave
    import array
    import struct

    SAMPLE_RATE = 24000

    def _read_pcm(wav_path):
        with open(wav_path, "rb") as f:
            content = f.read()
        idx = content.find(b"data", 12)
        if idx == -1:
            return array.array("h")
        pcm_start = idx + 8
        return array.array("h", content[pcm_start:])

    all_samples = array.array("h")
    gap_samples = array.array("h", [0] * int(SAMPLE_RATE * 0.3))  # 0.3s gap

    for i, seg in enumerate(audio_segments):
        samples = _read_pcm(seg["audio_path"])
        all_samples.extend(samples)
        if i < len(audio_segments) - 1:
            all_samples.extend(gap_samples)

    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(all_samples.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Turn a screen recording into a narrated explainer video."
    )
    parser.add_argument("input", help="Path to input video (.mp4 or .mov)")
    parser.add_argument("-o", "--output", default=None, help="Output video path")
    parser.add_argument(
        "--fps", type=float, default=0.5,
        help="Frames per second to extract for analysis (default: 0.5 = 1 frame every 2s)"
    )
    parser.add_argument(
        "--style", choices=["professional", "casual", "tutorial"],
        default="professional", help="Narration style"
    )
    parser.add_argument(
        "--voice", default="diana", help="Orpheus TTS voice (female: autumn/diana/hannah, male: austin/daniel/troy)"
    )
    parser.add_argument(
        "--no-zoom", action="store_true", help="Skip smart zoom enhancement"
    )
    parser.add_argument(
        "--product-name", default=None,
        help="Product/feature name to use in narration (overrides auto-detection)"
    )
    parser.add_argument(
        "--product-description", default=None,
        help="Short description of what the product does (helps VLM identify features)"
    )
    parser.add_argument(
        "--avatar", default=None, metavar="IMAGE",
        help="Path to a portrait image to generate a talking avatar (PiP overlay)"
    )
    parser.add_argument(
        "--avatar-position", default="bottom-right",
        choices=["bottom-right", "bottom-left", "top-right", "top-left"],
        help="Position of the avatar overlay (default: bottom-right)"
    )
    parser.add_argument(
        "--avatar-size", type=float, default=0.2,
        help="Avatar size as fraction of video height (default: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--avatar-shape", default="circle", choices=["circle", "rectangle"],
        help="Avatar overlay shape (default: circle)"
    )
    args = parser.parse_args()

    load_dotenv()
    check_dependencies()

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in (".mp4", ".mov"):
        print(f"Error: Unsupported format '{ext}'. Use .mp4 or .mov")
        sys.exit(1)

    output_path = args.output or f"output/explainer_{os.path.basename(input_path)}"
    if not output_path.endswith(".mp4"):
        output_path += ".mp4"

    temp_dir = "temp"
    frames_dir = os.path.join(temp_dir, "frames")
    audio_dir = os.path.join(temp_dir, "audio")

    duration = get_video_duration(input_path)
    print(f"\n🎬 Input: {input_path} ({duration:.1f}s)")
    print(f"📁 Output: {output_path}\n")

    # Step 1: Extract frames
    print("Step 1/5: Extracting frames...")
    frames = extract_frames(input_path, frames_dir, fps=args.fps)
    print(f"  Extracted {len(frames)} frames\n")

    # Step 2: Analyze frames with Groq Vision
    print("Step 2/5: Analyzing video content with AI vision...")
    segments = analyze_frames(frames, product_context=args.product_description)
    print(f"  Analyzed {len(segments)} segments\n")

    # Step 3: Generate narration script
    print("Step 3/5: Generating narration script...")
    narrated = generate_script(segments, style=args.style, product_name=args.product_name)
    for seg in narrated:
        print(f"  [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['narration'][:80]}...")
    print()

    # Step 4: Generate voiceover audio
    print("Step 4/5: Generating voiceover audio...")
    audio_segments = generate_audio_segments(narrated, audio_dir, voice=args.voice)
    print()

    # Step 5: Enhance video + compose
    video_for_compose = input_path
    if not args.no_zoom:
        print("Step 5/5: Enhancing video with smart zoom + composing final output...")
        enhanced_path = os.path.join(temp_dir, "enhanced.mp4")
        enhance_video_with_vision(input_path, enhanced_path, segments)
        video_for_compose = enhanced_path
    else:
        print("Step 5/5: Composing final output...")

    compose(video_for_compose, audio_segments, output_path)

    # Step 6 (optional): Generate and overlay talking avatar
    if args.avatar:
        print("\nStep 6: Generating talking avatar...")
        if not os.path.exists(args.avatar):
            print(f"  Error: Avatar image not found: {args.avatar}")
            sys.exit(1)

        # Build full narration audio for avatar (concatenate all segments)
        full_narration = os.path.join(temp_dir, "full_narration.wav")
        _build_full_narration(audio_segments, full_narration)

        avatar_video = os.path.join(temp_dir, "avatar.mp4")
        avatar_result, portrait = generate_avatar_video(args.avatar, full_narration, avatar_video)

        # Hybrid overlay: talking video where available, static portrait for the rest
        final_with_avatar = output_path.replace(".mp4", "_with_avatar.mp4")
        overlay_avatar_hybrid(
            output_path,
            avatar_result,       # None if generation fully failed
            portrait,
            final_with_avatar,
            position=args.avatar_position,
            size_ratio=args.avatar_size,
            shape=args.avatar_shape,
        )
        os.replace(final_with_avatar, output_path)

    # Cleanup temp files
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n✅ Done! Explainer video saved to: {output_path}")


if __name__ == "__main__":
    main()
