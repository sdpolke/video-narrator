"""
Video Narrator - Turn screen recordings into narrated explainer videos.

Uses Groq APIs for vision analysis, script generation, and TTS.
"""

import argparse
import os
import sys
import shutil
from dotenv import load_dotenv

from pipeline.frame_extractor import extract_frames, get_video_duration
from pipeline.vision_analyzer import analyze_frames
from pipeline.script_generator import generate_script
from pipeline.tts_generator import generate_audio_segments
from pipeline.video_enhancer import enhance_video
from pipeline.composer import compose


def check_dependencies():
    """Verify required external tools are installed."""
    for tool in ["ffmpeg", "ffprobe"]:
        if not shutil.which(tool):
            print(f"Error: '{tool}' not found. Install FFmpeg first.")
            sys.exit(1)


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
        "--voice", default="troy", help="Orpheus TTS voice name"
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
        enhance_video(input_path, enhanced_path)
        video_for_compose = enhanced_path
    else:
        print("Step 5/5: Composing final output...")

    compose(video_for_compose, audio_segments, output_path)

    # Cleanup temp files
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n✅ Done! Explainer video saved to: {output_path}")


if __name__ == "__main__":
    main()
