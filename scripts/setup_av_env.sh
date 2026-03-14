#!/bin/bash
set -euo pipefail

# VideoMME AV pipeline environment setup.
# Run from any directory; script resolves project root from its location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/venv_av}"

echo "=== VideoMME AV pipeline environment setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Venv: $VENV_DIR"

# Create venv if it does not exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --quiet --upgrade pip

# Install torch with CUDA 11.8 support (or cu121 via CUDA_VERSION env var)
CUDA_SUFFIX="${CUDA_VERSION:-cu118}"
echo ""
echo "=== Installing PyTorch ($CUDA_SUFFIX) ==="
pip install --quiet torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_SUFFIX"

# Install AV pipeline requirements
echo ""
echo "=== Installing requirements_av.txt ==="
pip install --quiet -r "$PROJECT_ROOT/requirements_av.txt"

# Download Qwen2-VL and Qwen2-Audio models (install_qwen.py expects to run from project root)
echo ""
echo "=== Downloading Qwen2-VL and Qwen2-Audio models ==="
cd "$PROJECT_ROOT"
python install_qwen.py

echo ""
echo "=== Setup complete ==="
echo "Add OPENAI_API_KEY to .env before running breadth or QA stages."
