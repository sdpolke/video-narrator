"""Generate voiceover audio from narration script using Groq TTS (Orpheus)."""

import os
import wave
import struct
from groq import Groq

TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "diana"


def _fix_wav_header(wav_path: str):
    """
    Orpheus TTS writes INT32_MAX as nframes in the WAV header (streaming bug).
    This patches the header with the correct sizes derived from the actual file size.
    """
    file_size = os.path.getsize(wav_path)
    # WAV: 4-byte RIFF size at offset 4, 4-byte data chunk size at offset 40
    riff_size = file_size - 8
    data_size = file_size - 44
    with open(wav_path, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<I", riff_size))
        f.seek(40)
        f.write(struct.pack("<I", data_size))


def generate_audio_segments(
    narrated_segments: list[dict],
    output_dir: str,
    voice: str = DEFAULT_VOICE,
) -> list[dict]:
    """
    Generate a WAV audio file for each narrated segment.

    Args:
        narrated_segments: List of {"start", "end", "narration"}.
        output_dir: Directory to save audio files.
        voice: Orpheus voice name.

    Returns:
        List of {"start", "end", "narration", "audio_path"}.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, seg in enumerate(narrated_segments):
        audio_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
        print(f"  Generating audio for segment {i + 1}/{len(narrated_segments)}...")

        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=seg["narration"],
            response_format="wav",
        )
        response.write_to_file(audio_path)
        _fix_wav_header(audio_path)

        results.append({**seg, "audio_path": audio_path})

    return results
