"""Generate a talking avatar video using the D-ID API.

Flow:
  1. Optimize portrait image (resize + compress while keeping transparency)
  2. Upload portrait image  → D-ID /images  → image URL
  3. Upload audio WAV       → D-ID /audios  → audio URL
  4. POST /talks            → job id
  5. Poll GET /talks/{id}   → wait for status "done"
  6. Download result_url    → save to output_path
"""

import os
import time
import requests
from PIL import Image


_BASE_URL = "https://api.d-id.com"
MAX_PORTRAIT_SIZE = 512  # D-ID works well with 512x512, saves bandwidth


def _optimize_portrait(portrait_path: str, output_path: str) -> str:
    """
    Resize and compress portrait image while preserving transparency.
    
    - Resizes to max 512px (D-ID works well with this, saves bandwidth)
    - Keeps alpha channel for PNG
    - Compresses to reduce file size
    
    Returns path to optimized image.
    """
    img = Image.open(portrait_path)
    
    # Resize if larger than MAX_PORTRAIT_SIZE
    if max(img.size) > MAX_PORTRAIT_SIZE:
        ratio = MAX_PORTRAIT_SIZE / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save as PNG with optimization (preserves transparency)
    img.save(output_path, "PNG", optimize=True, compress_level=6)
    
    orig_size = os.path.getsize(portrait_path) / 1024 / 1024
    new_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Optimized portrait: {orig_size:.1f}MB → {new_size:.1f}MB ({img.width}x{img.height})")
    
    return output_path


def _auth_header() -> dict:
    """
    Build the Authorization header for D-ID.

    D-ID dashboard gives a key in the format: base64(email):secret
    The API expects: Authorization: Basic base64(base64(email):secret)
    i.e. base64-encode the entire raw key string as-is.
    """
    import base64
    key = os.environ.get("DID_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DID_API_KEY not set in environment / .env")
    token = base64.b64encode(key.encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _upload_image(portrait_path: str) -> str:
    """Upload a local image to D-ID and return the hosted URL."""
    print("  Uploading portrait to D-ID...")
    headers = _auth_header()
    with open(portrait_path, "rb") as f:
        mime = "image/png" if portrait_path.lower().endswith(".png") else "image/jpeg"
        resp = requests.post(
            f"{_BASE_URL}/images",
            headers=headers,
            files={"image": (os.path.basename(portrait_path), f, mime)},
            timeout=60,
        )
    if not resp.ok:
        raise RuntimeError(f"D-ID image upload failed {resp.status_code}: {resp.text}")
    url = resp.json().get("url")
    if not url:
        raise RuntimeError(f"D-ID image upload returned no URL: {resp.text}")
    print(f"  Portrait uploaded: {url}")
    return url


def _upload_audio(audio_path: str) -> str:
    """Upload a local WAV to D-ID and return the hosted URL."""
    print("  Uploading audio to D-ID...")
    headers = _auth_header()
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{_BASE_URL}/audios",
            headers=headers,
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=120,
        )
    if not resp.ok:
        raise RuntimeError(f"D-ID audio upload failed {resp.status_code}: {resp.text}")
    url = resp.json().get("url")
    if not url:
        raise RuntimeError(f"D-ID audio upload returned no URL: {resp.text}")
    print(f"  Audio uploaded: {url}")
    return url


def _create_talk(image_url: str, audio_url: str) -> str:
    """Submit a /clips job (full-body avatar with hands) and return the clip ID."""
    headers = {**_auth_header(), "Content-Type": "application/json"}
    payload = {
        "presenter_id": "custom",  # Use custom presenter from uploaded image
        "presenter_config": {
            "source_url": image_url,
        },
        "script": {
            "type": "audio",
            "audio_url": audio_url,
        },
        "background": {
            "color": "#00000000",  # Transparent background
        },
    }
    # Try /clips endpoint first (full-body with hands), fallback to /talks if not available
    resp = requests.post(f"{_BASE_URL}/clips", headers=headers, json=payload, timeout=30)
    
    if resp.status_code == 404 or resp.status_code == 403:
        # Clips not available on free tier, fallback to talks (head-only)
        print("  Full-body avatar not available, using head-only mode...")
        payload_talks = {
            "source_url": image_url,
            "script": {
                "type": "audio",
                "audio_url": audio_url,
            },
            "config": {
                "stitch": True,
            },
        }
        resp = requests.post(f"{_BASE_URL}/talks", headers=headers, json=payload_talks, timeout=30)
    
    if not resp.ok:
        raise RuntimeError(f"D-ID video creation failed {resp.status_code}: {resp.text}")
    
    talk_id = resp.json().get("id")
    if not talk_id:
        raise RuntimeError(f"D-ID returned no id: {resp.text}")
    print(f"  Avatar job created: {talk_id}")
    return talk_id


def _poll_talk(talk_id: str, timeout: int = 300) -> str:
    """Poll until status == 'done' and return result_url. Works for both /talks and /clips."""
    headers = _auth_header()
    deadline = time.time() + timeout
    interval = 5
    
    # Try /clips first, fallback to /talks
    endpoint = f"{_BASE_URL}/clips/{talk_id}"
    resp = requests.get(endpoint, headers=headers, timeout=15)
    if resp.status_code == 404:
        endpoint = f"{_BASE_URL}/talks/{talk_id}"
    
    while time.time() < deadline:
        resp = requests.get(endpoint, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"  Status: {status}")
        if status == "done":
            url = data.get("result_url")
            if not url:
                raise RuntimeError("D-ID video done but no result_url in response")
            return url
        if status == "error":
            raise RuntimeError(f"D-ID video failed: {data.get('error', data)}")
        time.sleep(interval)
    raise TimeoutError(f"D-ID video {talk_id} did not complete within {timeout}s")


def _download_video(url: str, output_path: str):
    """Stream-download the result video."""
    print(f"  Downloading avatar video...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def generate_avatar_video(
    portrait_path: str,
    audio_path: str,
    output_path: str,
    still: bool = True,
    enhancer: str = "gfpgan",
):
    """
    Generate a talking avatar video via the D-ID API.

    Returns:
        (avatar_video_path_or_None, portrait_path)
        - avatar_video_path is the downloaded .mp4 if generation succeeded, else None
        - portrait_path is always the original static image (used as fallback)
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        # Optimize portrait before upload (resize + compress, keep transparency)
        tmp_dir = os.path.dirname(output_path) or "."
        optimized_portrait = os.path.join(tmp_dir, "_optimized_portrait.png")
        _optimize_portrait(portrait_path, optimized_portrait)
        
        image_url  = _upload_image(optimized_portrait)
        audio_url  = _upload_audio(audio_path)
        talk_id    = _create_talk(image_url, audio_url)
        result_url = _poll_talk(talk_id)
        _download_video(result_url, output_path)
        
        # Clean up optimized temp file
        if os.path.exists(optimized_portrait):
            os.remove(optimized_portrait)
        
        print(f"  Avatar video saved to: {output_path}")
        return output_path, portrait_path
    except Exception as e:
        print(f"  D-ID avatar generation failed ({e}).")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  Partial avatar video found, will use it for covered duration.")
            return output_path, portrait_path
        print(f"  No avatar video available, will use static portrait for full duration.")
        return None, portrait_path
