# Football Analysis Pipeline

A command-line tool for analyzing broadcast football footage using
computer vision and machine learning.

```
================================================================================

                           PIPELINE ARCHITECTURE

    Input Video
         |
         v
    +------------+     +------------+     +---------------+
    | Detection  | --> | Tracking   | --> | Ball Tracking |
    | (YOLOv8)   |     | (ByteTrack)|     | (Kalman)      |
    +------------+     +------------+     +---------------+
                                                |
                                                v
                            +------------------+     +-------------+
                            | Team Assignment  | --> | Output      |
                            | (SigLIP+KMeans)  |     | Video       |
                            +------------------+     +-------------+

================================================================================
```

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Project Structure](#project-structure)
7. [Models](#models)
8. [Testing](#testing)

--------------------------------------------------------------------------------

## Features

- Multi-object detection using YOLOv8
- Player and referee tracking with ByteTrack
- Ball tracking with Kalman filtering and motion gating
- Team classification using SigLIP embeddings and KMeans clustering
- Role stabilization to prevent goalkeeper/referee flickering
- Configurable caching system for faster iteration
- Apple Silicon (MPS) and CUDA support

--------------------------------------------------------------------------------

## Requirements

- Python 3.11+
- PyTorch 2.x
- 8GB+ RAM (16GB recommended for team classification)

--------------------------------------------------------------------------------

## Installation

```bash
git clone <repo>
cd football_analysis

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

chmod +x src/run.sh
```

Optional: Download Roboflow models and sample videos:

```bash
chmod +x src/setup.sh
./src/setup.sh
```

--------------------------------------------------------------------------------

## Quick Start

```bash
# Run full pipeline
./src/run.sh all Test6 --fresh

# Run specific modes
./src/run.sh ball Test6 --ball-conf 0.12
./src/run.sh team Test6
./src/run.sh players Test6
```

Output: `src/output_videos/<clip>/<clip>_<mode>.mp4`

--------------------------------------------------------------------------------

## Colab Quick Start

Recommended: open `colab.ipynb` in Colab and **Run all**. It includes:

- GPU check (`nvidia-smi`)
- install + asset download via `colab_setup.sh`
- sample clip listing
- optional Drive mount / `gdown` download
- safe run with CPU fallback

By default, Colab's preinstalled `torch` is used to avoid large downloads. To force a pinned CUDA torch install, set `INSTALL_TORCH=True` in the notebook install cell.

Manual setup (if you don't want the notebook):

```bash
!git clone https://github.com/esharif20/Spatio-Temporal-GNN-Football-Analysis.git
%cd football_analysis
!bash colab_setup.sh   # installs Colab deps + pulls sample assets into src/input_videos
!ls -1 src/input_videos | sed -n '1,200p'
```

One-line bootstrap (downloads the script, clones repo, installs deps):

```bash
!REPO_URL=https://github.com/esharif20/Spatio-Temporal-GNN-Football-Analysis.git \
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/esharif20/Spatio-Temporal-GNN-Football-Analysis/main/colab_setup.sh)"
```

Skip sample downloads (if you have your own clips):

```bash
!SKIP_ASSETS=1 bash colab_setup.sh
```

Avoid slow uploads (download via Drive file id):

```bash
!gdown -O src/input_videos/custom.mp4 "https://drive.google.com/uc?id=YOUR_FILE_ID"
!DEVICE=cuda bash src/run.sh all custom --fresh
```

Run with CUDA:

```bash
!DEVICE=cuda bash src/run.sh all 0bfacc_0 --fresh
!DEVICE=cuda bash src/run.sh ball 0bfacc_0 --fresh --ball-conf 0.12 --ball-slice 768 --ball-overlap 128
```

`src/run.sh` accepts either a base clip name (looks in `src/input_videos/<clip>.mp4`) or a full path.

Performance tips:

- Override detection batch size with `--det-batch` (0=auto). Example:

```bash
!DEVICE=cuda bash src/run.sh all 0bfacc_0 --det-batch 64 --fresh
```

- For faster (lower‑accuracy) ball tracking, try `--fast-ball` or increase `--ball-slice` size.

Default outputs are written to:

```
src/output_videos/<clip>/<clip>_<mode>.mp4
```

Example:

```
src/output_videos/0bfacc_0/0bfacc_0_ALL.mp4
```

To save somewhere else:

```bash
!DEVICE=cuda bash src/run.sh ball 0bfacc_0 /content/0bfacc_0_ball.mp4 --fresh
```

Quick preview or download:

```python
from IPython.display import Video
Video('src/output_videos/Test6/Test6_ALL.mp4')

from google.colab import files
files.download('src/output_videos/Test6/Test6_ALL.mp4')
```

--------------------------------------------------------------------------------

## CLI Reference

### Modes

| Mode      | Command    | Description                          |
|-----------|------------|--------------------------------------|
| Full      | `all`      | Complete pipeline (single output)    |
| Pitch     | `pitch`    | Detect pitch keypoints               |
| Players   | `players`  | Detect player bounding boxes         |
| Ball      | `ball`     | Detect and track ball                |
| Tracking  | `track`    | Track players with ByteTrack         |
| Team      | `team`     | Classify teams with SigLIP+KMeans    |

### Ball Tracking Options

| Flag                    | Default | Description                       |
|-------------------------|---------|-----------------------------------|
| `--ball-conf`           | 0.15    | Ball detection confidence         |
| `--ball-slice`          | 640     | Slicer tile size                  |
| `--ball-overlap`        | 96      | Slicer tile overlap               |
| `--ball-kalman`         | off     | Enable Kalman filtering           |
| `--ball-kalman-predict` | off     | Predict during detection gaps     |
| `--ball-kalman-max-gap` | 10      | Max frames to predict through     |
| `--ball-auto-area`      | off     | Auto-tune area ratio gates        |
| `--ball-max-aspect`     | 3.0     | Max aspect ratio for candidates   |
| `--ball-max-jump`       | 8.0     | Max motion jump ratio             |

### Caching

| Flag          | Description                              |
|---------------|------------------------------------------|
| `--fresh`     | Delete stubs and skip cache reads        |
| `--no-stub`   | Skip reading cached detections           |
| `--clear-stub`| Delete cached detections before running  |

### Device Selection

```bash
# Default: mps (Apple Silicon) or cpu
./src/run.sh all Test6

# Override device
DEVICE=cuda ./src/run.sh all Test6
DEVICE=cpu ./src/run.sh all Test6
```

--------------------------------------------------------------------------------

## Project Structure

```
football_analysis/
├── README.md
├── requirements.txt
├── pytest.ini
├── src/
│   ├── main.py              Entry point
│   ├── config.py            Configuration constants
│   ├── run.sh               Shell wrapper
│   ├── cli/                 Argument parsing
│   │   ├── __init__.py
│   │   ├── args.py
│   │   └── parsing.py
│   ├── pipeline/            Pipeline mode implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pitch.py
│   │   ├── players.py
│   │   ├── ball.py
│   │   ├── tracking.py
│   │   ├── team.py
│   │   └── full.py
│   ├── trackers/            Detection and tracking
│   ├── team_assigner/       Team classification
│   ├── utils/               Utilities
│   │   ├── __init__.py
│   │   ├── bbox_utils.py
│   │   ├── video_utils.py
│   │   ├── metrics.py
│   │   ├── cache.py
│   │   └── drawing.py
│   ├── models/              Player model (best.pt)
│   ├── data/                Ball + pitch models
│   ├── input_videos/        Input clips
│   ├── output_videos/       Generated outputs
│   └── stubs/               Cached detections
└── tests/                   Test suite
```

--------------------------------------------------------------------------------

## Models

| Model                          | Path                                 | Purpose            |
|--------------------------------|--------------------------------------|--------------------|
| Player/Referee/GK detector     | `src/models/best.pt`                 | Main detection     |
| Ball detector (optional)       | `src/data/football-ball-detection.pt`| Ball-only model    |
| Pitch detector (optional)      | `src/data/football-pitch-detection.pt`| Pitch keypoints   |

If the ball-only model is missing or `--no-ball-model` is used, the pipeline
falls back to the multi-class model with `--ball-mc-conf` threshold.

--------------------------------------------------------------------------------

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tracker.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

--------------------------------------------------------------------------------

## Notes

- Apple Silicon uses `mps` by default for inference
- Ball-only slicing is slower but improves recall on wide shots
- Output folders are grouped per clip under `src/output_videos/`
- Ball metrics are printed after tracking (observed vs interpolated, jitter, candidate ambiguity)
