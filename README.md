# Video Narrator

Turn silent screen recordings into narrated explainer videos using AI.

**Pipeline:** Extract frames → Groq Vision analysis → Script generation → TTS voiceover → Smart zoom → Final video

## Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- [Groq API key](https://console.groq.com/keys)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

```bash
# Basic usage
python main.py recording.mp4

# Custom output path
python main.py recording.mov -o my_explainer.mp4

# Override the product name (instead of auto-detection)
python main.py recording.mp4 --product-name "Sales AI Worker"

# Give the VLM context about what the product does
python main.py recording.mp4 \
  --product-name "Sales AI Worker" \
  --product-description "A sales agent that collects company data, manages contacts, initiates calls, records transcripts, and suggests follow-up actions"

# Casual narration style with different voice
python main.py recording.mp4 --style casual --voice tara

# Skip smart zoom, faster processing
python main.py recording.mp4 --no-zoom

# Analyze more frames (slower but more detailed)
python main.py recording.mp4 --fps 1.0
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `output/explainer_<input>` | Output video path |
| `--fps` | `0.5` | Frames/sec to extract (0.5 = 1 frame every 2s) |
| `--style` | `professional` | Narration style: professional, casual, tutorial |
| `--voice` | `troy` | Orpheus TTS voice |
| `--no-zoom` | off | Skip smart zoom enhancement |
| `--product-name` | auto-detected | Product name to use in narration |
| `--product-description` | none | Context about the product to help VLM identify features |
