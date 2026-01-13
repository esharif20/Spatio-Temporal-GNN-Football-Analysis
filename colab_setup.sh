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

# Function to download from Google Drive with multiple fallback methods
download_gdrive() {
  local file_id="$1"
  local output_path="$2"
  local filename=$(basename "$output_path")

  # Skip if file already exists and is non-empty
  if [[ -f "$output_path" && -s "$output_path" ]]; then
    echo "  $filename already exists, skipping"
    return 0
  fi

  echo "  Downloading $filename..."

  # Method 1: gdown (standard)
  if gdown -O "$output_path" "https://drive.google.com/uc?id=$file_id" 2>/dev/null; then
    if [[ -s "$output_path" ]]; then
      echo "  ✓ $filename downloaded via gdown"
      return 0
    fi
  fi

  # Method 2: gdown with --fuzzy flag
  if gdown --fuzzy -O "$output_path" "https://drive.google.com/file/d/$file_id/view" 2>/dev/null; then
    if [[ -s "$output_path" ]]; then
      echo "  ✓ $filename downloaded via gdown --fuzzy"
      return 0
    fi
  fi

  # Method 3: curl with confirmation bypass
  local confirm_url="https://drive.google.com/uc?export=download&id=$file_id"
  local confirm_code=$(curl -sc /tmp/gcookie "$confirm_url" 2>/dev/null | grep -o 'confirm=[^&]*' | head -1)
  if [[ -n "$confirm_code" ]]; then
    curl -Lb /tmp/gcookie "${confirm_url}&${confirm_code}" -o "$output_path" 2>/dev/null
  else
    curl -L "$confirm_url" -o "$output_path" 2>/dev/null
  fi

  if [[ -s "$output_path" ]]; then
    echo "  ✓ $filename downloaded via curl"
    return 0
  fi

  # Method 4: wget fallback
  wget -q --no-check-certificate "https://drive.google.com/uc?export=download&id=$file_id" -O "$output_path" 2>/dev/null
  if [[ -s "$output_path" ]]; then
    echo "  ✓ $filename downloaded via wget"
    return 0
  fi

  echo "  ✗ Failed to download $filename"
  rm -f "$output_path"
  return 1
}

if [[ "${SKIP_ASSETS:-0}" != "1" ]]; then
  mkdir -p "$ROOT/src/models" "$ROOT/src/input_videos"

  echo "Downloading models..."
  download_gdrive "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V" "$ROOT/src/models/ball_detection.pt"
  download_gdrive "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q" "$ROOT/src/models/player_detection.pt"
  download_gdrive "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf" "$ROOT/src/models/pitch_detection.pt"

  echo "Downloading sample videos..."
  download_gdrive "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF" "$ROOT/src/input_videos/0bfacc_0.mp4"
  download_gdrive "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu" "$ROOT/src/input_videos/121364_0.mp4"
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
