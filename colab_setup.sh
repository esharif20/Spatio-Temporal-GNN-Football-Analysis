#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-/content/football_analysis}"

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
python -m pip install --no-cache-dir -r "$ROOT/requirements-colab.txt"

if [[ -f "$ROOT/src/run.sh" ]]; then
  chmod +x "$ROOT/src/run.sh"
fi
if [[ -f "$ROOT/src/setup.sh" ]]; then
  chmod +x "$ROOT/src/setup.sh"
fi

if [[ "${SKIP_ASSETS:-0}" != "1" ]]; then
  if [[ -f "$ROOT/src/setup.sh" ]]; then
    bash "$ROOT/src/setup.sh" || true
  else
    echo "src/setup.sh not found; skipping sample assets" >&2
  fi
fi

mkdir -p "$ROOT/src/input_videos" "$ROOT/src/output_videos" "$ROOT/src/stubs"

if compgen -G "$ROOT/src/data/*.mp4" > /dev/null; then
  cp -n "$ROOT"/src/data/*.mp4 "$ROOT/src/input_videos"/
fi

echo "Available clips:"
ls -1 "$ROOT/src/input_videos" 2>/dev/null || true
echo "Example:"
echo "DEVICE=cuda bash src/run.sh all 0bfacc_0 --fresh"
