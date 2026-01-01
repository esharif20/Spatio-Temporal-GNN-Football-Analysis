# ⚽ Football Video Analysis

YOLO detection → ByteTrack tracking → K-means team assignment → Annotated video output.

```
Input Video → [Detection] → [Tracking] → [Team Assignment] → Output Video
                 88.4%        100%           2 teams
               ball rate    player frames    auto-assigned
```

---

## 📁 Project Structure

```
football_analysis/
├── src/
│   ├── main.py                 # Entry point
│   ├── trackers/
│   │   └── tracker.py          # YOLO + ByteTrack
│   ├── team_assigner/
│   │   └── team_assigner.py    # K-means clustering
│   ├── utils/
│   │   ├── video_utils.py      # read/save video
│   │   └── bbox_utils.py       # bbox helpers
│   ├── models/
│   │   └── best.pt             # YOLO weights (195MB)
│   ├── stubs/
│   │   └── track_stubs.pkl     # Cached detections
│   ├── input_videos/           # Your videos here
│   └── output_videos/          # Annotated output
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_tracker.py
│   ├── test_team_assigner.py
│   ├── test_video_utils.py
│   ├── test_bbox_utils.py
│   ├── test_main.py
│   └── run_tests.sh            # Test runner
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

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

## 🧪 Testing

```bash
# Run all tests (51 tests)
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

---

## 🎯 Design Choices

### Why 1280px resolution?
```
640px  → ball = 11px → too small for YOLO
1280px → ball = 22px → detectable ✓
```

### Why class-specific confidence?
```python
# Ball is small → YOLO gives low confidence (0.01-0.10)
# Players are large → high confidence (0.70-0.95)

conf = 0.01  # Detect everything
# Then filter:
if class == "ball" and conf >= 0.01: keep
if class == "player" and conf >= 0.10: keep
```

### Why single-frame inference?
```
Batch processing on MPS → frames 3+ empty (bug)
Single-frame processing → 100% frames correct ✓
```

### Why manual filtering over two models?
```
Two YOLO models     → 2x inference time
Manual filtering    → same speed, simpler code ✓
```

---

## ⚙️ Key Configuration

```python
# src/trackers/tracker.py

results = self.model.predict(
    source=frame,
    conf=0.01,      # Low to catch balls
    imgsz=1280,     # Match training resolution
    max_det=50,     # Football has ~25 objects max
    verbose=False
)

# FP16 for speed
self.model.model.half()
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Ball detection | 88.4% (663/750 frames) |
| Player tracking | 100% (750/750 frames) |
| Referee tracking | 99.6% (747/750 frames) |
| Avg players/frame | 20.7 |

---

## 🔧 Commands Reference

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

## 📦 Dependencies

```txt
numpy==1.26.4
opencv-python==4.9.0.80
torch==2.2.2
ultralytics==8.3.223
supervision==0.18.0
pandas==2.2.0
scikit-learn==1.4.0
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named cv2` | `pip install opencv-python` |
| `No module named pytest` | `pip install pytest pytest-mock` |
| Low ball detection | Check `conf=0.01` and `imgsz=1280` |
| MPS errors | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` |

---

## 📚 References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision (ByteTrack)](https://github.com/roboflow/supervision)
- [Roboflow Dataset](https://roboflow.com/)
