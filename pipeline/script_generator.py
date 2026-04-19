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
    "professional": "Use a clear, confident, professional product-demo tone.",
    "casual": "Use a friendly, conversational tone like showing a colleague.",
    "tutorial": "Use an instructional step-by-step tutorial tone.",
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

    narration_prompt = f"""You are writing voiceover narration for a product explainer video.

Video summary: {summary_text}

Steps:
{titles_block}

Detailed segment descriptions:
{segment_block}

Write the narration. Rules:
- {tone}
- Segment 0 is the INTRO: read the summary naturally (2 sentences max, ~20 words).
- For each step (1 to {len(segments)}): write exactly ONE sentence, max 15 words.
  Start with the action or purpose, NOT "the user" or "we".
  Example good: "Select saved credentials and sign in to access the dashboard."
  Example bad:  "The user clicks the sign-in button to log in."
  Example bad:  "In the Data Collector Agent, the user navigates to sessions."
- CRITICAL: Mention the product/tool name ONLY ONCE in the intro. NEVER repeat it in the steps.
  After the intro, refer to it as "the platform", "the tool", or just describe the action directly.
- Keep it concise. The video should feel snappy, not drawn out.
- Flow naturally from one step to the next.

Return a JSON array:
  [{{"segment": 0, "narration": "intro summary text"}},
   {{"segment": 1, "narration": "step 1 text"}},
   ...]

Return valid JSON only, no markdown fences."""

    narration_raw = _call_llm(client, narration_prompt, max_tokens=2048)
    narrations = json.loads(narration_raw)

    # ── Merge into final result ────────────────────────────────────
    result = []

    # Find the intro (segment 0)
    intro_text = summary_text
    for item in narrations:
        if item.get("segment") == 0:
            intro_text = item["narration"]
            break

    # Intro uses the first segment's video range
    if segments:
        result.append({
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "narration": intro_text,
        })

    # Steps 1-N
    narration_map = {item["segment"]: item["narration"] for item in narrations}
    for i, seg in enumerate(segments):
        step_num = i + 1
        text = narration_map.get(step_num, "")
        if text:
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "narration": text,
            })

    return result
