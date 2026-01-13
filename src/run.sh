#!/bin/bash
set -euo pipefail

# Get the directory where the script is located (src/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get repo root (parent of src/)
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

MODE_INPUT=""
VIDEO_INPUT=""
OUT=""
NO_STUB=0
CLEAR_STUB=0
FAST_BALL=0
BALL_SLICE=""
BALL_OVERLAP=""
BALL_SLICER_IOU=""
BALL_SLICER_WORKERS=""
BALL_IMGSZ=""
BALL_CONF=""
BALL_MC_CONF=""
BALL_KALMAN=0
BALL_KALMAN_PREDICT=0
BALL_KALMAN_MAX_GAP=""
BALL_AUTO_AREA=0
NO_BALL_MODEL=0
BALL_TILES=""
BALL_ACQUIRE_CONF=""
BALL_MAX_ASPECT=""
BALL_AREA_MIN=""
BALL_AREA_MAX=""
BALL_MAX_JUMP=""
DET_BATCH=""
VORONOI=0
NO_BALL_PATH=0
BALL_ONLY=0
SHOW_KEYPOINTS=0
VORONOI_OVERLAY=0
NO_RADAR=0
ANALYTICS=0
DEBUG_PITCH=0
PITCH_BACKEND=""

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}
if [[ -z "${DEVICE:-}" ]]; then
  arch=$(uname -m)
  if [[ "$arch" == "arm64" ]]; then
    DEVICE=mps
  else
    DEVICE=cpu
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-video-path|--source_video_path)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --source-video-path" >&2
        exit 1
      fi
      VIDEO_INPUT="$2"
      shift 2
      ;;
    --target-video-path|--target_video_path|--out)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --target-video-path" >&2
        exit 1
      fi
      OUT="$2"
      shift 2
      ;;
    --device)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --device" >&2
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --fresh)
      NO_STUB=1
      CLEAR_STUB=1
      shift
      ;;
    --no-stub)
      NO_STUB=1
      shift
      ;;
    --clear-stub)
      CLEAR_STUB=1
      shift
      ;;
    --fast-ball)
      FAST_BALL=1
      shift
      ;;
    --ball-slice)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-slice" >&2
        exit 1
      fi
      BALL_SLICE="$2"
      shift 2
      ;;
    --ball-overlap)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-overlap" >&2
        exit 1
      fi
      BALL_OVERLAP="$2"
      shift 2
      ;;
    --ball-slicer-iou)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-slicer-iou" >&2
        exit 1
      fi
      BALL_SLICER_IOU="$2"
      shift 2
      ;;
    --ball-slicer-workers)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-slicer-workers" >&2
        exit 1
      fi
      BALL_SLICER_WORKERS="$2"
      shift 2
      ;;
    --ball-imgsz)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-imgsz" >&2
        exit 1
      fi
      BALL_IMGSZ="$2"
      shift 2
      ;;
    --ball-conf)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-conf" >&2
        exit 1
      fi
      BALL_CONF="$2"
      shift 2
      ;;
    --ball-mc-conf)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-mc-conf" >&2
        exit 1
      fi
      BALL_MC_CONF="$2"
      shift 2
      ;;
    --ball-kalman)
      BALL_KALMAN=1
      shift
      ;;
    --ball-kalman-predict)
      BALL_KALMAN_PREDICT=1
      shift
      ;;
    --ball-kalman-max-gap)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-kalman-max-gap" >&2
        exit 1
      fi
      BALL_KALMAN_MAX_GAP="$2"
      shift 2
      ;;
    --ball-auto-area)
      BALL_AUTO_AREA=1
      shift
      ;;
    --ball-tiles)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-tiles" >&2
        exit 1
      fi
      BALL_TILES="$2"
      shift 2
      ;;
    --ball-acquire-conf)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-acquire-conf" >&2
        exit 1
      fi
      BALL_ACQUIRE_CONF="$2"
      shift 2
      ;;
    --ball-max-aspect)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-max-aspect" >&2
        exit 1
      fi
      BALL_MAX_ASPECT="$2"
      shift 2
      ;;
    --ball-area-min)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-area-min" >&2
        exit 1
      fi
      BALL_AREA_MIN="$2"
      shift 2
      ;;
    --ball-area-max)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-area-max" >&2
        exit 1
      fi
      BALL_AREA_MAX="$2"
      shift 2
      ;;
    --ball-max-jump)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --ball-max-jump" >&2
        exit 1
      fi
      BALL_MAX_JUMP="$2"
      shift 2
      ;;
    --det-batch|--det-batch-size)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --det-batch" >&2
        exit 1
      fi
      DET_BATCH="$2"
      shift 2
      ;;
    --no-ball-model)
      NO_BALL_MODEL=1
      shift
      ;;
    --voronoi)
      VORONOI=1
      shift
      ;;
    --no-ball-path)
      NO_BALL_PATH=1
      shift
      ;;
    --ball-only)
      BALL_ONLY=1
      shift
      ;;
    --show-keypoints)
      SHOW_KEYPOINTS=1
      shift
      ;;
    --voronoi-overlay)
      VORONOI_OVERLAY=1
      shift
      ;;
    --no-radar)
      NO_RADAR=1
      shift
      ;;
    --analytics)
      ANALYTICS=1
      shift
      ;;
    --debug-pitch)
      DEBUG_PITCH=1
      shift
      ;;
    --pitch-local)
      PITCH_BACKEND="ultralytics"
      shift
      ;;
    --pitch-backend)
      if [[ -z "${2:-}" ]]; then
        echo "Missing value for --pitch-backend" >&2
        exit 1
      fi
      PITCH_BACKEND="$2"
      shift 2
      ;;
    *)
      if [[ -z "$MODE_INPUT" ]]; then
        MODE_INPUT="$1"
        shift
      elif [[ -z "$VIDEO_INPUT" ]]; then
        VIDEO_INPUT="$1"
        shift
      elif [[ -z "$OUT" ]]; then
        OUT="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

MODE_INPUT=${MODE_INPUT:-all}
VIDEO_INPUT=${VIDEO_INPUT:-Test1}

mode_key=$(lower "$MODE_INPUT")
case "$mode_key" in
  all) mode=ALL ;;
  pitch|pitch_detection) mode=PITCH_DETECTION ;;
  players|player|player_detection) mode=PLAYER_DETECTION ;;
  ball|ball_detection) mode=BALL_DETECTION ;;
  track|tracking|player_tracking) mode=PLAYER_TRACKING ;;
  team|team_classification) mode=TEAM_CLASSIFICATION ;;
  radar|tactical) mode=RADAR ;;
  *)
    echo "Unknown mode: $MODE_INPUT" >&2
    echo "Valid modes: all, pitch, players, ball, track, team, radar" >&2
    exit 1
    ;;
esac

if [[ -f "$VIDEO_INPUT" ]]; then
  video="$VIDEO_INPUT"
else
  # Try multiple locations for the video
  candidate="$SCRIPT_DIR/input_videos/$VIDEO_INPUT"
  if [[ "$candidate" != *.mp4 ]]; then
    candidate="${candidate}.mp4"
  fi
  if [[ -f "$candidate" ]]; then
    video="$candidate"
  else
    # Also try repo root relative path (for backwards compatibility)
    candidate="$REPO_ROOT/src/input_videos/$VIDEO_INPUT"
    if [[ "$candidate" != *.mp4 ]]; then
      candidate="${candidate}.mp4"
    fi
    if [[ -f "$candidate" ]]; then
      video="$candidate"
    else
      echo "Video not found: $VIDEO_INPUT" >&2
      echo "Searched in: $SCRIPT_DIR/input_videos/" >&2
      exit 1
    fi
  fi
fi

if [[ -z "$OUT" ]]; then
  base=$(basename "$video")
  base="${base%.*}"
  OUT="$SCRIPT_DIR/output_videos/${base}/${base}_${mode}.mp4"
fi

# Ensure output directory exists
mkdir -p "$(dirname "$OUT")"

cmd=(python "$SCRIPT_DIR/main.py"
  --source-video-path "$video"
  --target-video-path "$OUT"
  --mode "$mode"
  --device "$DEVICE"
)

if [[ "$FAST_BALL" -eq 1 ]]; then
  cmd+=(--fast-ball)
fi
if [[ -n "$BALL_SLICE" ]]; then
  cmd+=(--ball-slice "$BALL_SLICE")
fi
if [[ -n "$BALL_OVERLAP" ]]; then
  cmd+=(--ball-overlap "$BALL_OVERLAP")
fi
if [[ -n "$BALL_SLICER_IOU" ]]; then
  cmd+=(--ball-slicer-iou "$BALL_SLICER_IOU")
fi
if [[ -n "$BALL_SLICER_WORKERS" ]]; then
  cmd+=(--ball-slicer-workers "$BALL_SLICER_WORKERS")
fi
if [[ -n "$BALL_IMGSZ" ]]; then
  cmd+=(--ball-imgsz "$BALL_IMGSZ")
fi
if [[ -n "$BALL_CONF" ]]; then
  cmd+=(--ball-conf "$BALL_CONF")
fi
if [[ -n "$BALL_MC_CONF" ]]; then
  cmd+=(--ball-mc-conf "$BALL_MC_CONF")
fi
if [[ "$BALL_KALMAN" -eq 1 ]]; then
  cmd+=(--ball-kalman)
fi
if [[ "$BALL_KALMAN_PREDICT" -eq 1 ]]; then
  cmd+=(--ball-kalman-predict)
fi
if [[ -n "$BALL_KALMAN_MAX_GAP" ]]; then
  cmd+=(--ball-kalman-max-gap "$BALL_KALMAN_MAX_GAP")
fi
if [[ "$BALL_AUTO_AREA" -eq 1 ]]; then
  cmd+=(--ball-auto-area)
fi
if [[ -n "$BALL_TILES" ]]; then
  cmd+=(--ball-tiles "$BALL_TILES")
fi
if [[ -n "$BALL_ACQUIRE_CONF" ]]; then
  cmd+=(--ball-acquire-conf "$BALL_ACQUIRE_CONF")
fi
if [[ -n "$BALL_MAX_ASPECT" ]]; then
  cmd+=(--ball-max-aspect "$BALL_MAX_ASPECT")
fi
if [[ -n "$BALL_AREA_MIN" ]]; then
  cmd+=(--ball-area-min "$BALL_AREA_MIN")
fi
if [[ -n "$BALL_AREA_MAX" ]]; then
  cmd+=(--ball-area-max "$BALL_AREA_MAX")
fi
if [[ -n "$BALL_MAX_JUMP" ]]; then
  cmd+=(--ball-max-jump "$BALL_MAX_JUMP")
fi
if [[ "$NO_BALL_MODEL" -eq 1 ]]; then
  cmd+=(--no-ball-model)
fi
if [[ -n "$DET_BATCH" ]]; then
  cmd+=(--det-batch "$DET_BATCH")
fi
if [[ "$NO_STUB" -eq 1 ]]; then
  cmd+=(--no-stub)
fi
if [[ "$CLEAR_STUB" -eq 1 ]]; then
  cmd+=(--clear-stub)
fi
if [[ "$VORONOI" -eq 1 ]]; then
  cmd+=(--voronoi)
fi
if [[ "$NO_BALL_PATH" -eq 1 ]]; then
  cmd+=(--no-ball-path)
fi
if [[ "$BALL_ONLY" -eq 1 ]]; then
  cmd+=(--ball-only)
fi
if [[ "$SHOW_KEYPOINTS" -eq 1 ]]; then
  cmd+=(--show-keypoints)
fi
if [[ "$VORONOI_OVERLAY" -eq 1 ]]; then
  cmd+=(--voronoi-overlay)
fi
if [[ "$NO_RADAR" -eq 1 ]]; then
  cmd+=(--no-radar)
fi
if [[ "$ANALYTICS" -eq 1 ]]; then
  cmd+=(--analytics)
fi
if [[ "$DEBUG_PITCH" -eq 1 ]]; then
  cmd+=(--debug-pitch)
fi
if [[ -n "$PITCH_BACKEND" ]]; then
  cmd+=(--pitch-backend "$PITCH_BACKEND")
fi

"${cmd[@]}"
