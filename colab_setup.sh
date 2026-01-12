#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-/content/football_analysis}"
CACHE_ROOT="${FA_CACHE_ROOT:-}"

if [[ ! -f "$ROOT/requirements-colab.txt" ]]; then
  if [[ -n "$REPO_URL" ]]; then
    if [[ ! -d "$REPO_DIR/.git" ]]; then
      git clone "$REPO_URL" "$REPO_DIR"
    fi
    exec bash "$REPO_DIR/colab_setup.sh"
  fi
  echo "requirements-colab.txt not found in $ROOT. Run inside the repo or set REPO_URL." >&2
  exit 1
fi

python -m pip install --upgrade pip

torch_present=0
if python -c "import torch" >/dev/null 2>&1; then
  torch_present=1
fi

if [[ "${INSTALL_TORCH:-0}" == "1" || "$torch_present" -eq 0 ]]; then
  python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1
fi

python -m pip install -r "$ROOT/requirements-colab.txt"

if [[ -f "$ROOT/src/run.sh" ]]; then
  chmod +x "$ROOT/src/run.sh"
fi
if [[ -f "$ROOT/src/setup.sh" ]]; then
  chmod +x "$ROOT/src/setup.sh"
fi

if [[ -n "$CACHE_ROOT" ]]; then
  mkdir -p "$CACHE_ROOT"/{input_videos,models}
  if compgen -G "$CACHE_ROOT/input_videos/*.mp4" > /dev/null; then
    mkdir -p "$ROOT/src/input_videos"
    cp -n "$CACHE_ROOT"/input_videos/*.mp4 "$ROOT/src/input_videos"/
  fi
  if compgen -G "$CACHE_ROOT/models/*.pt" > /dev/null; then
    mkdir -p "$ROOT/src/models"
    cp -n "$CACHE_ROOT"/models/*.pt "$ROOT/src/models"/
  fi
fi

if [[ "${SKIP_ASSETS:-0}" != "1" ]]; then
  if [[ -f "$ROOT/src/setup.sh" ]]; then
    echo "Downloading models and sample videos..."
    if ! bash "$ROOT/src/setup.sh"; then
      echo "WARNING: setup.sh failed, trying direct downloads..." >&2
      mkdir -p "$ROOT/src/models" "$ROOT/src/input_videos"
      # Try downloading directly with gdown
      gdown -O "$ROOT/src/models/ball_detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V" || echo "Failed to download ball_detection.pt"
      gdown -O "$ROOT/src/models/player_detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q" || echo "Failed to download player_detection.pt"
      gdown -O "$ROOT/src/models/pitch_detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf" || echo "Failed to download pitch_detection.pt"
      gdown -O "$ROOT/src/input_videos/0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF" || echo "Failed to download 0bfacc_0.mp4"
      gdown -O "$ROOT/src/input_videos/121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu" || echo "Failed to download 121364_0.mp4"
    fi
  else
    echo "src/setup.sh not found; skipping sample assets" >&2
  fi
fi

mkdir -p "$ROOT/src/input_videos" "$ROOT/src/output_videos" "$ROOT/src/stubs" "$ROOT/src/models"

# Cache models and videos for faster re-runs
if [[ -n "$CACHE_ROOT" ]]; then
  mkdir -p "$CACHE_ROOT"/{input_videos,models}
  if compgen -G "$ROOT/src/input_videos/*.mp4" > /dev/null; then
    cp -n "$ROOT"/src/input_videos/*.mp4 "$CACHE_ROOT/input_videos"/
  fi
  if compgen -G "$ROOT/src/models/*.pt" > /dev/null; then
    cp -n "$ROOT"/src/models/*.pt "$CACHE_ROOT/models"/
  fi
fi

echo "Available clips:"
ls -1 "$ROOT/src/input_videos" 2>/dev/null || true
echo "Example:"
echo "DEVICE=cuda bash src/run.sh all 0bfacc_0 --fresh"
