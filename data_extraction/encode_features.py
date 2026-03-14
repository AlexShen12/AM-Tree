"""
Phase 2: Multimodal Encoding via ImageBind
==========================================
Processes every clip in VideoMME_clips/ and writes two separate embedding
tensors per clip using ImageBind's visual and audio encoders.

Output layout (per clip, embeddings are NOT fused):
    data/VideoMME_clip_feature/
        <video_id>/
            visual/
                clip_001.pt   # Tensor[1024], float32, CPU
                clip_002.pt
                ...
            audio/
                clip_001.pt   # Tensor[1024], float32, CPU  (absent if no audio track)
                ...

Run from the project root:
    python data_extraction/encode_features.py

Or submit via encode_features.sl for multi-hour Slurm jobs.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
IMAGEBIND_DIR = PROJECT_ROOT / "data_extraction" / "ImageBind"
CLIPS_DIR     = PROJECT_ROOT / "data" / "VideoMME_clips"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "VideoMME_clip_feature"
WEIGHTS_PATH  = PROJECT_ROOT / "data_extraction" / ".checkpoints" / "imagebind_huge.pth"

# ImageBind must be on sys.path before any import from it.
sys.path.insert(0, str(IMAGEBIND_DIR))

from imagebind import data as ib_data                              # noqa: E402
from imagebind.models import imagebind_model                       # noqa: E402
from imagebind.models.imagebind_model import ModalityType          # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> torch.nn.Module:
    """
    Load ImageBind huge from the absolute weights path.

    Bypasses imagebind_huge(pretrained=True)'s hardcoded CWD-relative
    checkpoint lookup so the script can be run from any directory.
    """
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"ImageBind weights not found at {WEIGHTS_PATH}. "
            "Run imagebind_model.imagebind_huge(pretrained=True) once from "
            "data_extraction/ to download them, or adjust WEIGHTS_PATH."
        )

    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    log.info("ImageBind loaded on %s", device)
    return model


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

# A valid 16 kHz 16-bit mono WAV for even 0.1 s of audio is ~3.2 KB.
# Anything smaller is either an empty file or just the 44-byte WAV header
# produced when ffmpeg found no audio stream to demux.
_MIN_AUDIO_BYTES = 1_000


def extract_audio_to_wav(clip_path: Path, wav_path: Path) -> bool:
    """
    Demux the audio stream from *clip_path* into a 16 kHz mono PCM WAV at
    *wav_path* using a single ffmpeg pass.

    Returns False when the clip has no audio stream (indicated by ffmpeg
    producing an empty or near-empty output file) or when ffmpeg errors out.
    Cleans up *wav_path* on failure so callers never see a partial file.

    ffmpeg is preferred over moviepy/VideoFileClip here because -vn demuxes
    the audio bitstream without initialising a video decoder at all, which
    matters when processing tens of thousands of clips.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(clip_path),
            "-vn",                   # no video decoding
            "-acodec", "pcm_s16le", # lossless PCM — required by torchaudio
            "-ar", "16000",          # ImageBind target sample rate
            "-ac", "1",              # mono
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )

    # ffmpeg may return 0 but still write an empty/header-only WAV when the
    # clip contains no audio stream.  The size check catches both that case
    # and genuine errors that somehow return 0.
    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < _MIN_AUDIO_BYTES:
        wav_path.unlink(missing_ok=True)
        if result.returncode != 0:
            log.warning(
                "ffmpeg audio extraction failed for %s:\n%s",
                clip_path.name, result.stderr.strip(),
            )
        return False

    return True


# ---------------------------------------------------------------------------
# Per-clip encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_clip(
    model: torch.nn.Module,
    clip_path: Path,
    device: torch.device,
    tmp_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Run ImageBind on one clip.

    Returns
    -------
    visual_emb : Tensor[1024]  — L2-normalised visual embedding (CPU)
    audio_emb  : Tensor[1024] | None — L2-normalised audio embedding (CPU),
                 or None when the clip has no audio track.

    The two vectors are NEVER summed or concatenated here; all downstream
    fusion decisions are left to the caller.
    """
    inputs: dict[str, torch.Tensor] = {}

    # ── Visual ──────────────────────────────────────────────────────────────
    # load_and_transform_video_data returns [1, S, C, T, H, W].
    # The model averages over S (spatial/temporal crops) internally.
    visual_tensor = ib_data.load_and_transform_video_data(
        [str(clip_path)], device
    )
    inputs[ModalityType.VISION] = visual_tensor

    # ── Audio ────────────────────────────────────────────────────────────────
    wav_path = tmp_dir / f"{clip_path.stem}.wav"
    has_audio = extract_audio_to_wav(clip_path, wav_path)

    if has_audio:
        audio_tensor = ib_data.load_and_transform_audio_data(
            [str(wav_path)], device,
            clips_per_video=3,   # 3 × 2 s clips; works on our 5-10 s segments
        )
        inputs[ModalityType.AUDIO] = audio_tensor
        wav_path.unlink(missing_ok=True)

    # ── Forward pass ─────────────────────────────────────────────────────────
    embeddings = model(inputs)

    visual_emb = embeddings[ModalityType.VISION].squeeze(0).detach().cpu()
    audio_emb  = (
        embeddings[ModalityType.AUDIO].squeeze(0).detach().cpu()
        if has_audio
        else None
    )
    return visual_emb, audio_emb


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    video_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    tmp_dir: Path,
) -> None:
    """Encode all clips for one video, skipping clips that are already done."""
    video_id = video_dir.name
    visual_dir = OUTPUT_DIR / video_id / "visual"
    audio_dir  = OUTPUT_DIR / video_id / "audio"
    visual_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = sorted(video_dir.glob("clip_*.mp4"))
    if not clip_paths:
        log.warning("[%s] No clips found — skipping.", video_id)
        return

    for clip_path in clip_paths:
        clip_stem = clip_path.stem
        visual_out = visual_dir / f"{clip_stem}.pt"

        # Resume: visual is written last, so its presence guarantees all
        # outputs for this clip (including audio) were successfully saved.
        if visual_out.exists():
            continue

        try:
            visual_emb, audio_emb = encode_clip(model, clip_path, device, tmp_dir)
        except Exception as exc:
            log.error("[%s] %s failed: %s", video_id, clip_path.name, exc)
            continue
        finally:
            # Release any cached CUDA allocations between clips.
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Write audio BEFORE visual.  visual_out is the resume sentinel — it
        # must only exist once all outputs for this clip are safely on disk.
        # If the job is killed between these two saves, audio is overwritten
        # on the next run (acceptable), but we never silently miss audio.
        if audio_emb is not None:
            torch.save(audio_emb, str(audio_dir / f"{clip_stem}.pt"))
        else:
            log.debug("[%s] %s has no audio — audio embedding skipped.", video_id, clip_stem)

        torch.save(visual_emb, str(visual_out))  # sentinel — written last


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("Device: %s", device)

    if device.type == "cuda":
        log.info(
            "GPU: %s  |  VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(device)

    video_dirs = sorted(d for d in CLIPS_DIR.iterdir() if d.is_dir())
    log.info("Processing %d video folders under %s", len(video_dirs), CLIPS_DIR)

    # A single shared temp directory avoids repeated FS creation overhead.
    with tempfile.TemporaryDirectory(prefix="ib_audio_") as tmp:
        tmp_dir = Path(tmp)
        for video_dir in tqdm(video_dirs, desc="Videos", unit="video"):
            process_video(video_dir, model, device, tmp_dir)

    log.info("Done. Embeddings saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
