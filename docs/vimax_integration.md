# ViMax Integration Design

**Status:** Proposal
**Target project:** `/Users/shrijeetpolke/projects/video-narrator`
**Source project:** `/Users/shrijeetpolke/projects/ViMax` (sibling folder)
**Date:** May 2026

---

## 1. Goals

1. **Phase 1 — Generate avatars from text.** Let users skip the `--avatar <image>` requirement by describing a character instead. ViMax generates a consistent portrait via `CharacterPortraitsGenerator`, which then feeds the existing D-ID talking-avatar flow.
2. **Phase 2 — Assess what else from ViMax can make explainer videos more interactive** (B-roll cutaways, cinematic transitions, character cameos over the original recording, etc.).

The existing `--avatar <path>` flow is preserved. A new `--generate-avatar "<description>"` flag is added.

---

## 2. Integration approach: path-based sibling import (recommended)

### Why path-based over copying

| Concern | Path-based | Copy modules |
|---|---|---|
| Stay in sync with ViMax updates | ✅ automatic | ❌ manual |
| Avoid code duplication | ✅ | ❌ |
| Python version | ⚠️ ViMax requires 3.12+ | ⚠️ same |
| Isolation from ViMax's heavy deps | ⚠️ need a thin adapter | ✅ pick and choose |
| Works out of the box | ✅ | ⚠️ edits needed |

**Decision:** path-based import, but wrapped in a **thin adapter** (`pipeline/vimax_avatar_generator.py`) so video-narrator never directly depends on ViMax internals. If ViMax's API changes, only the adapter needs fixing.

### Python version note

ViMax declares `requires-python = ">=3.12"`. video-narrator targets 3.10+. **Action item:** bump video-narrator to 3.12 for this feature, or guard the import with a version check and raise a clear error if the user tries `--generate-avatar` on 3.10/3.11.

### New dependencies pulled in

ViMax's portrait generator needs:
- `google-genai>=1.47.0` — for Gemini image generation (nanobanana)
- `langchain>=0.3.26` + `langchain-openai` + `pydantic` — used by the character extractor / portrait agent
- `tenacity` — for retry logic
- `pyyaml` — for loading configs

We add only these to `requirements.txt`. We don't need `faiss-cpu`, `scenedetect`, or torch for the avatar use case.

---

## 3. File layout

```
video-narrator/
├── main.py                                   # adds --generate-avatar flag
├── pipeline/
│   ├── avatar_generator.py                   # unchanged (D-ID)
│   └── vimax_avatar_generator.py             # NEW — thin adapter
├── configs/
│   └── vimax_avatar.yaml                     # NEW — image generator + key
├── docs/
│   └── vimax_integration.md                  # this doc
└── requirements.txt                          # adds google-genai, langchain, pyyaml, tenacity
```

No changes to ViMax. It stays at its sibling path; video-narrator locates it via a resolved relative path.

---

## 4. Adapter design

### `pipeline/vimax_avatar_generator.py`

Responsibilities:
1. Resolve the ViMax path (default: `../ViMax`, overridable via `VIMAX_PATH` env var).
2. Insert into `sys.path` before importing.
3. Load a minimal YAML config to build an image generator via `RenderBackend`.
4. Build a `CharacterInScene` Pydantic object from a user description.
5. Call `CharacterPortraitsGenerator.generate_front_portrait()` and save the PNG.
6. Return the saved path — the existing `avatar_generator.generate_avatar_video()` consumes it unchanged.

Only the **front portrait** is needed for D-ID (it's the talking head). Side and back portraits are optional and skipped by default for speed and cost.

### Public API

```python
# pipeline/vimax_avatar_generator.py

def generate_portrait_from_description(
    description: str,
    style: str = "photorealistic",
    output_path: str = "temp/generated_portrait.png",
    config_path: str = "configs/vimax_avatar.yaml",
) -> str:
    """
    Generate a character portrait from a text description using ViMax.
    Returns the path to the saved PNG, ready for avatar_generator.generate_avatar_video().
    """
```

### Config file: `configs/vimax_avatar.yaml`

Reuses ViMax's config schema so `RenderBackend` works unchanged:

```yaml
image_generator:
  class_path: tools.ImageGeneratorNanobananaGoogleAPI
  init_args:
    api_key: ${GOOGLE_API_KEY}          # read from env
  max_requests_per_minute: 10
  max_requests_per_day: 500

# Not used for portraits but RenderBackend expects it — stubbed:
video_generator:
  class_path: tools.ImageGeneratorNanobananaGoogleAPI
  init_args:
    api_key: ${GOOGLE_API_KEY}
```

*(Alternative: build the image generator directly without `RenderBackend`, skipping the video_generator stub. Simpler but bypasses rate limiting.)*

---

## 5. CLI changes (`main.py`)

Add one flag and one mutual-exclusion rule:

```python
group = parser.add_mutually_exclusive_group()
group.add_argument("--avatar", default=None, metavar="IMAGE",
                   help="Path to a portrait image for the talking avatar")
group.add_argument("--generate-avatar", default=None, metavar="DESCRIPTION",
                   help='Generate the portrait from a text description via ViMax '
                        '(e.g. "A 30-year-old male sales coach, short brown hair, blue blazer")')
parser.add_argument("--avatar-style", default="photorealistic",
                    help="Style for generated avatars (photorealistic, cartoon, anime, ...)")
```

In the main flow, **before Step 6**:

```python
avatar_image_path = args.avatar
if args.generate_avatar:
    from pipeline.vimax_avatar_generator import generate_portrait_from_description
    print("🎨 Generating avatar portrait via ViMax...")
    avatar_image_path = generate_portrait_from_description(
        description=args.generate_avatar,
        style=args.avatar_style,
        output_path=os.path.join(temp_dir, "generated_portrait.png"),
    )
    print(f"  Portrait saved: {avatar_image_path}")

if avatar_image_path:
    # existing D-ID flow — unchanged
    ...
```

### Example usage

```bash
# Existing flow (unchanged)
python main.py recording.mp4 --avatar headshot.jpg

# New: generate portrait from text
python main.py recording.mp4 \
  --generate-avatar "A 30-year-old female sales coach, short black hair, navy blazer, friendly smile" \
  --avatar-style photorealistic \
  --product-name "Sales AI Worker"
```

---

## 6. Setup (user-facing)

```bash
# 1. Ensure ViMax is cloned next to video-narrator
cd /Users/shrijeetpolke/projects
git clone https://github.com/HKUDS/ViMax.git   # if not already present

# 2. Upgrade video-narrator's venv to Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add Google AI Studio key to .env
echo "GOOGLE_API_KEY=<your-key>" >> .env
```

If ViMax lives at a non-default path:
```bash
export VIMAX_PATH=/some/other/path/to/ViMax
```

---

## 7. Error handling and guardrails

- **ViMax not found:** raise a clear error telling the user where to clone it or how to set `VIMAX_PATH`.
- **Python < 3.12:** raise early with an actionable message before attempting the ViMax import.
- **No `GOOGLE_API_KEY`:** raise before spending time on frame extraction.
- **Portrait generation fails after retries (tenacity already retries 3×):** fall back gracefully — print a warning and skip avatar overlay (same behavior as current D-ID failure path).

---

## 8. Testing plan

1. **Smoke test** (no D-ID needed): call `generate_portrait_from_description()` directly and verify a PNG is saved.
2. **End-to-end**: run `main.py recording.mp4 --generate-avatar "..."` with a short 5–10s input video.
3. **Regression**: run existing `--avatar headshot.jpg` path to confirm it still works.
4. **Fallback**: run with a bad Google key → verify a clean error surfaces.

---

## 9. Rollout

1. Land the adapter + CLI flag behind the new `--generate-avatar` flag (opt-in, zero risk to existing users).
2. Document in `README.md` under a new "Generated avatars" section.
3. Iterate on style presets (`photorealistic`, `cartoon`, `corporate headshot`) based on real output quality.

---

---

# Part 2: Evaluating ViMax for the existing-video-edit use case

Our use case is **editing existing screen recordings into explainer videos**, not generating video from scratch. So idea2video / novel2movie aren't directly useful. But several ViMax components map well onto this domain and can deliver visible improvements.

Below is a ranked assessment of ViMax's agents and tools by **value-to-effort ratio** for our use case.

## High-value additions (strong wow factor)

### 1. AutoCameo — user-as-character overlays ⭐⭐⭐
**What it is:** ViMax's upcoming AutoCameo feature inserts a user's photo into generated scenes as a consistent character.

**Why it matters for us:** Instead of a static talking head in the corner, we could render the narrator *reacting* to what's on screen — pointing at UI elements, nodding when a feature appears, etc. Paired with our existing PiP overlay, this feels dramatically more present than a D-ID head.

**Effort:** medium-high. AutoCameo is marked "Coming Soon" in ViMax's README, so we'd be on the bleeding edge.

### 2. `StoryboardArtist` + `ScriptEnhancer` for narration ⭐⭐⭐
**What it is:** ViMax's `storyboard_artist.py` and `script_enhancer.py` produce shot-aware, cinematically-structured scripts with pacing and beat design.

**Why it matters for us:** Our current `script_generator.py` calls Groq directly and produces flat narration. ViMax's storyboard artist thinks in scenes and shots, which maps perfectly to "this UI appears → wait → highlight → transition." Even without generating video, the *script* gets dramatically richer: better pacing, emphasis cues, natural pause points, and explicit visual-beat markers.

**Effort:** low-medium. Drop-in replacement at the script generation step. Biggest lift is adapting the output schema.

### 3. `CameraImageGenerator` for transition B-roll ⭐⭐
**What it is:** Generates smooth transition videos between two frames using Veo/Seedance.

**Why it matters for us:** Screen recordings have abrupt jump-cuts when users switch tabs or navigate between screens. ViMax can synthesize a 1–2 second cinematic transition between two captured frames (e.g., a smooth zoom from dashboard → detail view). Drops hard cuts entirely.

**Effort:** medium. Needs a Veo API key and some careful first-frame/last-frame extraction logic. But the visual polish is immediate and obvious.

## Medium-value additions

### 4. `BestImageSelector` + `ReferenceImageSelector` for frame quality ⭐⭐
**What it is:** MLLM-based consistency checker that picks the best of N generated candidates.

**Why it matters for us:** When we do smart zoom or generate supplementary assets, these selectors ensure visual consistency across generated frames. Only relevant once we generate more than just portraits.

**Effort:** low if we already have the image pipeline. Just another call.

### 5. `CharacterExtractor` for multi-presenter explainers ⭐
**What it is:** Parses a script and identifies all characters, their features, and scene membership.

**Why it matters for us:** If a product demo references "the sales rep" and "the customer," we can automatically generate consistent portraits for each and swap them in context (e.g., dialogue-style explainer videos). Niche but cool.

**Effort:** low.

## Low-value for our use case

- **`NovelCompressor`** — we don't work with novels.
- **`EventExtractor` / `SceneExtractor`** — designed for narrative structure in long-form fiction.
- **`GlobalInformationPlanner`** — for coordinating characters across hundreds of shots. Overkill.
- **`Idea2VideoPipeline`** — generates full videos from scratch. Orthogonal to our edit-existing-video use case.

---

## Recommended roadmap

| Phase | What | Wow factor | Complexity |
|---|---|---|---|
| 1 | `--generate-avatar` via `CharacterPortraitsGenerator` (this doc) | 🌟🌟 | Low |
| 2 | Replace `script_generator.py` with `StoryboardArtist` + `ScriptEnhancer` for richer narration scripts | 🌟🌟🌟 | Low-Medium |
| 3 | Cinematic transitions between scene cuts using `CameraImageGenerator` + Veo | 🌟🌟🌟 | Medium |
| 4 | Multi-character generated avatars (dialogue-style explainers) via `CharacterExtractor` + portraits | 🌟🌟 | Medium |
| 5 | AutoCameo — user-as-character reacting to on-screen content | 🌟🌟🌟🌟 | High (blocked on ViMax release) |

**My suggestion:** Ship Phase 1 first (this doc), then prototype Phase 2 since it's low-effort and noticeably upgrades narration quality. Phase 3 is the biggest visible win on the final video but costs the most in API calls (Veo generations aren't cheap).
