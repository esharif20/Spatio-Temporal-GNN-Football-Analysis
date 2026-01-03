# Football Video Analysis Pipeline

YOLO detection → ByteTrack tracking → Kalman ball tracking → Role stabilization → K-means team assignment → Ball possession → Annotated output.

```
Input Video → [Detection] → [Tracking] → [Ball Tracking] → [Team Assignment] → [Possession] → Output Video
                YOLO v8      ByteTrack     Kalman Filter     K-means HSV        Hysteresis
```

---

## Project Structure

```
football_analysis/
├── src/
│   ├── main.py                     # Entry point
│   ├── trackers/
│   │   ├── tracker.py              # YOLO + ByteTrack + visualization
│   │   ├── single_ball_tracker.py  # Kalman filter ball tracking
│   │   └── track_stabilizer.py     # Role locking (majority voting)
│   ├── team_assigner/
│   │   └── team_assigner.py        # K-means clustering on jersey HSV
│   ├── player_ball_assigner/
│   │   └── player_ball_assigner.py # Ball possession (hysteresis-based)
│   ├── utils/
│   │   ├── video_utils.py          # read/save video
│   │   └── bbox_utils.py           # bbox helpers
│   ├── models/
│   │   └── best.pt                 # YOLO weights (195MB)
│   ├── stubs/
│   │   └── track_stubs.pkl         # Cached detections
│   ├── input_videos/               # Your videos here
│   └── output_videos/              # Annotated output
├── tests/                          # 74 tests
│   ├── conftest.py
│   ├── test_tracker.py
│   ├── test_team_assigner.py
│   ├── test_single_ball_tracker.py
│   ├── test_video_utils.py
│   ├── test_bbox_utils.py
│   ├── test_main.py
│   └── run_tests.sh
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Pipeline

1. **Detection** - YOLO v8 (conf=0.15, imgsz=1280)
2. **Tracking** - ByteTrack for players/referees
3. **Ball Tracking** - Kalman filter with lock/unlock state machine
4. **Ball Smoothing** - Cubic spline interpolation + Gaussian smoothing
5. **Role Stabilization** - Majority voting to fix referee flickering
6. **Team Assignment** - K-means clustering on jersey HSV colours
7. **Ball Possession** - Hysteresis-based assignment (resolution-independent)
8. **Visualization** - Modern minimalist annotations

---

## Quick Start

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
cd src
python main.py

# Output → src/output_videos/output_video.mp4
```

---

## Testing

```bash
# Run all tests (74 tests)
./tests/run_tests.sh

# Or manually
python -m pytest tests/ -v
```

| Module | Tests |
|--------|-------|
| bbox_utils | 8 |
| video_utils | 9 |
| tracker | 12 |
| team_assigner | 10 |
| main | 12 |
| single_ball_tracker | 23 |

---

## Key Features

### Kalman Ball Tracker
- Lock/unlock state machine for robust tracking
- Velocity prediction during occlusions
- Confidence-based detection filtering
- Automatic re-acquisition after lost frames

### Role Stabilization
- Majority voting across all frames
- Fixes referee flickering (player ↔ referee misclassification)
- Runs BEFORE team assignment to prevent contamination

### Hysteresis Ball Possession
- Resolution-independent threshold (`player_height * 0.5`)
- 30% closer required to switch possession (prevents flickering)
- Temporal smoothing for stable assignments

### Modern UI
- Team-colored possession glow
- Minimalist chevron indicator
- Semi-transparent possession stats panel
- Anti-aliased rendering throughout

---

## Design Choices

### Why 1280px resolution?
```
640px  → ball = 11px → too small for YOLO
1280px → ball = 22px → detectable ✓
```

### Why conf=0.15?
```
0.10 → NMS overload on crowded frames (2s+ timeout)
0.15 → Fast NMS, still catches all detections ✓
```

### Why single-frame inference on MPS?
```
Batch processing on MPS → frames 3+ empty (Apple Silicon bug)
Single-frame processing → 100% frames correct ✓
```

### Why hysteresis for possession?
```
Without: Possession flickers between nearby players
With: Stable possession, 30% closer required to switch ✓
```

---

## Key Configuration

```python
# src/trackers/tracker.py
Tracker(
    model_path="models/best.pt",
    det_conf_player=0.15,  # Detection confidence
    det_conf_ref=0.15,
    imgsz=1280,            # Match training resolution
    max_det=50,            # Football has ~25 objects max
)

# src/player_ball_assigner/player_ball_assigner.py
PlayerBallAssigner(
    distance_ratio=0.5,    # max_distance = player_height * 0.5
    hysteresis=0.3,        # 30% closer required to switch
)
```

---

## Results

| Metric | Value |
|--------|-------|
| Ball detection | ~80% (before interpolation) |
| Ball coverage | 100% (after interpolation) |
| Player tracking | 100% |
| Referee tracking | ~99% |
| Team assignment | 2 teams auto-detected |

---

## Commands Reference

```bash
# Run pipeline
python main.py

# Run tests
./tests/run_tests.sh
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Clear cache and re-detect
rm src/stubs/track_stubs.pkl
python main.py
```

---

## Dependencies

```txt
numpy==1.26.4
opencv-python==4.9.0.80
torch==2.2.2
ultralytics==8.3.223
supervision==0.18.0
pandas==2.2.0
scikit-learn==1.4.0
scipy
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named cv2` | `pip install opencv-python` |
| `No module named pytest` | `pip install pytest pytest-mock` |
| NMS time limit warning | Already fixed (conf=0.15) |
| Low ball detection | Check imgsz=1280 |
| MPS errors | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Possession flickering | Already fixed (hysteresis) |

---

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack)](https://github.com/roboflow/supervision)
- [Roboflow Dataset](https://roboflow.com/)
