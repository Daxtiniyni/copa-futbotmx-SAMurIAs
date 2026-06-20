from ultralytics import YOLO
from pathlib import Path
from project_paths import OUTPUTS_DIR, VIDEO_V1, YOLO_MODEL

MODEL_PATH = YOLO_MODEL
VIDEO_PATH = VIDEO_V1

model = YOLO(MODEL_PATH)

model.predict(
    source=str(VIDEO_PATH),
    conf=0.10,
    iou=0.5,
    imgsz=960,
    device=0,
    save=True,
    project=str(OUTPUTS_DIR / "videos"),
    name="test_V1"
)

print("Listo. Revisa outputs/videos/test_V1")
