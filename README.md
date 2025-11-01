# Spatio-Temporal GNN Football Analysis

End-to-end football video analysis. YOLO handles detection. A spatio-temporal GNN links objects across frames and infers events. The pipeline runs locally or on a Slurm GPU cluster. Results are reproducible with pinned dependencies.

---

## What this project does

- Detect players, ball, and officials with YOLO
- Build per-frame graphs and link tracks across time
- Train or load a spatio-temporal GNN for event inference
- Export annotated videos, CSVs, and summary stats
- Run single videos or whole folders in batch

---

## Requirements

- Python 3.11
- macOS or Linux
- Optional GPU with CUDA on Linux or MPS on Apple Silicon

Pinned Python packages:

```
numpy==1.26.4
opencv-python==4.9.0.80
torch==2.2.2
torchvision==0.17.2
ultralytics==8.3.223
```


---

## Quick start

```bash
# clone
git clone https://github.com/<you>/Spatio-Temporal-GNN-Football-Analysis.git
cd Spatio-Temporal-GNN-Football-Analysis

# create venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install
pip install -r requirements.txt

# place a short clip for testing
mkdir -p input_videos
# add: input_videos/sample.mp4

# run YOLO inference
python src/yolo_inference.py --source input_videos/sample.mp4 --outdir runs/inference
