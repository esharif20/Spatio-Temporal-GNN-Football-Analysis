from ultralytics import YOLO

model = YOLO("yolov8m")

res = model.predict("input_videos/08fd33_4.mp4", save=True)
print(res[0])
print("==================================================")
for box in res[0].boxes:
    print(box)