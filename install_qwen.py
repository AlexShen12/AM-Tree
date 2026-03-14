#!/usr/bin/env python3
"""
One-time setup: pins transformers to a version compatible with the installed
torch, installs Qwen2-VL/Audio dependencies, and pre-downloads both model
checkpoints into the HuggingFace cache.

Run on a login node (internet access required):
    python install_qwen.py

Why the pin?  transformers >= 4.50 requires torch >= 2.4, but the venv has
torch 2.1.2.  transformers 4.46.x is the last release that ships full
Qwen2-VL + Qwen2-Audio support and remains compatible with torch 2.1.x.
"""

import os
import subprocess
import sys
import importlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

hf_home = os.environ.get("HF_HOME")
if hf_home:
    os.makedirs(hf_home, exist_ok=True)
    print(f"HuggingFace cache → {hf_home}")

TRANSFORMERS_PIN = "transformers==4.46.3"


def pip_install(*packages: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    print(f"  pip install {' '.join(packages)}")
    subprocess.check_call(cmd)


def check_torch() -> str:
    try:
        import torch
        ver = torch.__version__
        print(f"torch {ver}  (cuda available: {torch.cuda.is_available()})")
        return ver
    except ImportError:
        sys.exit("ERROR: torch is not installed in this environment.")


def install_packages() -> None:
    print("\n=== Installing / pinning packages ===")
    pip_install(TRANSFORMERS_PIN)
    pip_install("qwen-vl-utils")
    pip_install("librosa")
    pip_install("pyarrow")

    import importlib.metadata
    installed = importlib.metadata.version("transformers")
    print(f"  transformers {installed} active")


def download_qwen_vl() -> None:
    print("\n=== Downloading Qwen2-VL-7B-Instruct (~15 GB) ===")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch

    print("  processor...")
    AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    print("  model weights (float16)...")
    Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
    )
    print("  Done.")


def download_qwen_audio() -> None:
    print("\n=== Downloading Qwen2-Audio-7B-Instruct (~15 GB) ===")
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    import torch

    print("  processor...")
    AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    print("  model weights (float16)...")
    Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        torch_dtype=torch.float16,
    )
    print("  Done.")


def main() -> None:
    print("=== Qwen2-VL / Qwen2-Audio environment setup ===\n")

    torch_ver = check_torch()

    major, minor = (int(x) for x in torch_ver.split(".")[:2])
    if (major, minor) >= (2, 4):
        print(
            f"WARNING: torch {torch_ver} >= 2.4 detected. "
            f"You may be able to use a newer transformers. "
            f"Proceeding with {TRANSFORMERS_PIN} anyway."
        )

    install_packages()
    download_qwen_vl()
    download_qwen_audio()

    print("\n=== All done ===")
    print("See README for running the breadth, depth, and QA pipeline.")


if __name__ == "__main__":
    main()
