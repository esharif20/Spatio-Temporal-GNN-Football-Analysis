from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "models" / "best.pt"

if not BEST.exists():
    raise FileNotFoundError(f"Missing weights: {BEST}. Place your trained best.pt there.")

model = YOLO(str(BEST))



res = model.predict("src/input_videos/08fd33_4.mp4", save=True)
print(res[0])
print("==================================================")
for box in res[0].boxes:
    print(box)