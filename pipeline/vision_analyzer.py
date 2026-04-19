"""Analyze video frames using Groq Vision API to identify features and workflows."""

import base64
import os
from groq import Groq


VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq allows max 5 images per request, so we batch frames
BATCH_SIZE = 4


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_analysis_prompt(product_context: str | None) -> str:
    """Build a detailed prompt that focuses on feature identification."""
    context_line = ""
    if product_context:
        context_line = (
            f"\nProduct context provided by the user: {product_context}\n"
            "Use this context to better identify which features and capabilities "
            "are being demonstrated.\n"
        )

    return (
        "You are a product analyst reviewing sequential screenshots from a "
        "software product screen recording. Your goal is to identify the "
        "FEATURES and CAPABILITIES being demonstrated, not just describe UI.\n"
        f"{context_line}\n"
        "For each frame, analyze:\n"
        "1. FEATURE: What product feature or capability is being shown? "
        "(e.g. 'data collection', 'call logging', 'contact management', "
        "'export/reporting', 'filtering/search')\n"
        "2. WORKFLOW STEP: What step in the workflow is this? "
        "(e.g. 'configuring filters', 'reviewing results', 'initiating action')\n"
        "3. VALUE: What value does this feature provide to the user? "
        "(e.g. 'narrows search to relevant companies', 'tracks interaction history')\n"
        "4. TRANSITION: What changed from the previous frame — did the user "
        "switch features, drill into details, or complete an action?\n\n"
        "Focus on WHAT the product does and WHY, not pixel-level UI descriptions. "
        "Identify distinct capabilities even if the UI looks similar between frames."
    )


def _analyze_batch(
    client: Groq,
    frames: list[dict],
    batch_index: int,
    analysis_prompt: str,
) -> str:
    """Send a batch of frames to Groq Vision and get a feature-focused description."""
    content = [{"type": "text", "text": analysis_prompt}]

    for i, frame in enumerate(frames):
        b64 = _encode_image(frame["path"])
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        content.append({
            "type": "text",
            "text": f"Frame {batch_index * BATCH_SIZE + i + 1} (timestamp: {frame['timestamp']:.1f}s):",
        })

    completion = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.3,
        max_completion_tokens=2048,
    )
    return completion.choices[0].message.content


def analyze_frames(
    frames: list[dict],
    product_context: str | None = None,
) -> list[dict]:
    """
    Analyze all extracted frames in batches via Groq Vision.

    Args:
        frames: List of {"path", "timestamp"} from frame extraction.
        product_context: Optional user-provided description of the product.

    Returns list of segments: [{"start": 0.0, "end": 3.0, "description": "..."}, ...]
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    analysis_prompt = _build_analysis_prompt(product_context)
    segments = []

    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i : i + BATCH_SIZE]
        print(f"  Analyzing frames {i + 1}-{i + len(batch)} of {len(frames)}...")
        description = _analyze_batch(client, batch, i // BATCH_SIZE, analysis_prompt)

        start_ts = batch[0]["timestamp"]
        end_ts = batch[-1]["timestamp"] + 1.0

        segments.append({
            "start": start_ts,
            "end": end_ts,
            "description": description,
        })

    return segments
