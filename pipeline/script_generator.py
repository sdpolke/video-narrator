"""Generate narration script from video analysis using Groq LLM.

Two-pass approach:
  1. Summarise the entire video and assign a title to each step.
  2. Write concise narration per step in a polished product-walkthrough style.
"""

import os
import json
from groq import Groq

LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _call_llm(client: Groq, prompt: str, max_tokens: int = 2048) -> str:
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_completion_tokens=max_tokens,
    )
    raw = completion.choices[0].message.content.strip()
    # Strip markdown fences if the model wraps them
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def _build_segment_block(segments: list[dict]) -> str:
    block = ""
    for i, seg in enumerate(segments):
        block += (
            f"Segment {i + 1} ({seg['start']:.1f}s – {seg['end']:.1f}s):\n"
            f"{seg['description']}\n\n"
        )
    return block


STYLE_INSTRUCTIONS = {
    "professional": (
        "Use a confident, benefit-driven SaaS product demo tone. "
        "Focus on what the user achieves, not just what they click. "
        "Start sentences with action verbs. Keep it punchy and outcome-focused."
    ),
    "casual": (
        "Use a friendly, conversational tone like showing a colleague. "
        "Focus on benefits and ease of use. Keep it natural and engaging."
    ),
    "tutorial": (
        "Use an instructional step-by-step tutorial tone. "
        "Focus on clarity and actionable guidance."
    ),
}


def generate_script(
    segments: list[dict],
    style: str = "professional",
    product_name: str | None = None,
) -> list[dict]:
    """
    Generate a narration script with:
      - Segment 0: a 2-sentence video summary (what the product is + what the video covers)
      - Segments 1-N: one concise narration sentence per step

    Args:
        segments: Vision analysis segments.
        style: Narration style.
        product_name: User-provided product name (overrides auto-detection).

    Returns list of {"start", "end", "narration"}.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    segment_block = _build_segment_block(segments)
    tone = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["professional"])

    # ── Pass 1: Summary + step titles ──────────────────────────────
    print("    Pass 1: Generating summary and step outline...")

    product_name_instruction = ""
    if product_name:
        product_name_instruction = (
            f'\n"product_name" MUST be exactly: "{product_name}" (provided by the user).\n'
        )

    outline_prompt = f"""You are analysing a screen recording of a software product.

Here are frame-by-frame descriptions of each segment:

{segment_block}

Produce a JSON object with:
  "product_name": best guess of the product or feature name shown,
  "summary": a 2-sentence overview of what this video demonstrates (what the product does and what the viewer will learn),
  "steps": [
    {{"segment": 1, "title": "short step title (3-6 words)"}},
    ...one per segment
  ]
{product_name_instruction}
Return valid JSON only, no markdown fences."""

    outline_raw = _call_llm(client, outline_prompt)
    outline = json.loads(outline_raw)

    summary_text = outline.get("summary", "")
    steps = outline.get("steps", [])
    step_titles = {s["segment"]: s["title"] for s in steps}

    # ── Pass 2: Narration per step ─────────────────────────────────
    print("    Pass 2: Writing narration for each step...")
    titles_block = "\n".join(
        f"Step {i+1}: {step_titles.get(i+1, 'Untitled')}" for i in range(len(segments))
    )

    narration_prompt = f"""You are writing voiceover narration for a SaaS product explainer video in the style of modern AI-generated product demos (like Trupeer, Loom, Descript).

Video summary: {summary_text}

Steps:
{titles_block}

Detailed segment descriptions:
{segment_block}

Write the narration following this style guide:

TONE & DELIVERY:
- {tone}
- Confident but not salesy — focus on credibility and usability
- High-density information delivery — every sentence introduces value
- Benefit-driven: emphasize outcomes, not just actions
- Use parallel phrasing: "Record your screen... Generate instantly... Edit with AI..."

STRUCTURE:
- Segment 0 (INTRO): Hook with the problem/opportunity, then introduce the solution (2 sentences, ~25 words)
  Example: "Creating product demos takes hours of editing. [Product] turns any screen recording into a polished explainer video — automatically."
- Each step (1 to {len(segments)}): ONE sentence, 12-18 words
  - Start with ACTION VERBS: "Record...", "Generate...", "Customize...", "Export..."
  - Focus on WHAT THE USER ACHIEVES, not just what they click
  - Good: "Generate professional narration in seconds — no recording needed."
  - Bad: "The user clicks the generate button to create audio."
  - Good: "Customize your video with AI-powered editing tools."
  - Bad: "In the editor, various options are available for customization."

CRITICAL RULES:
- Mention the product name ONLY in the intro. After that, use "the platform", "the tool", or just describe the action
- NO filler words or phrases like "simply", "just", "easily" — let the speed speak for itself
- Use em-dashes for emphasis: "Record your screen — that's it."
- Add strategic pauses with ellipses: "Upload your footage... and watch the magic happen."
- Keep transitions tight and modular — each line should work standalone
- Focus on SPEED, QUALITY, SCALE as key benefits

ZOOM DETECTION (NEW):
For each segment, identify if there's a specific UI element or region that should be zoomed/highlighted.
If the segment involves clicking a button, filling a form, selecting an option, or any focused UI interaction,
specify the approximate screen region (as percentages from top-left):
- "top-left" (0-33% x, 0-33% y)
- "top-center" (33-66% x, 0-33% y)
- "top-right" (66-100% x, 0-33% y)
- "center-left" (0-33% x, 33-66% y)
- "center" (33-66% x, 33-66% y)
- "center-right" (66-100% x, 33-66% y)
- "bottom-left" (0-33% x, 66-100% y)
- "bottom-center" (33-66% x, 66-100% y)
- "bottom-right" (66-100% x, 66-100% y)
- "full" (no zoom, show entire screen)

Return a JSON array:
  [{{"segment": 0, "narration": "intro hook + solution", "zoom_region": "full"}},
   {{"segment": 1, "narration": "action-verb driven step", "zoom_region": "center"}},
   {{"segment": 2, "narration": "another step", "zoom_region": "top-right"}},
   ...]

Return valid JSON only, no markdown fences."""

    narration_raw = _call_llm(client, narration_prompt, max_tokens=2048)
    narrations = json.loads(narration_raw)

    # ── Merge into final result ────────────────────────────────────
    result = []

    # Find the intro (segment 0)
    intro_text = summary_text
    intro_zoom = "full"
    for item in narrations:
        if item.get("segment") == 0:
            intro_text = item["narration"]
            intro_zoom = item.get("zoom_region", "full")
            break

    # Intro uses the first segment's video range
    if segments:
        result.append({
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "narration": intro_text,
            "zoom_region": intro_zoom,
        })

    # Steps 1-N
    narration_map = {item["segment"]: item for item in narrations}
    for i, seg in enumerate(segments):
        step_num = i + 1
        item = narration_map.get(step_num, {})
        text = item.get("narration", "")
        zoom_region = item.get("zoom_region", "center")
        if text:
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "narration": text,
                "zoom_region": zoom_region,
            })

    return result
