self.model = YOLO(model_path)
try:
    self.model.to("mps")
except Exception:
    pass
