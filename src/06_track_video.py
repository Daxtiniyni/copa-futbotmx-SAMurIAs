import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from project_paths import OUTPUTS_DIR, VIDEO_V1, YOLO_MODEL

MODEL_PATH = YOLO_MODEL
VIDEO_PATH = VIDEO_V1

OUT_DIR = OUTPUTS_DIR / "tracking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO = OUT_DIR / "V1_tracking.mp4"
OUT_CSV = OUT_DIR / "V1_tracks.csv"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

records = []

results = model.track(
    source=VIDEO_PATH,
    conf=0.15,
    iou=0.5,
    imgsz=960,
    device=0,
    persist=True,
    stream=True,
    tracker="bytetrack.yaml"
)

frame_id = 0

for result in results:
    frame = result.orig_img.copy()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        names = result.names

        for box, track_id, class_id, conf in zip(boxes, ids, classes, confs):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            class_name = names[class_id]

            records.append({
                "frame": frame_id,
                "time_sec": frame_id / fps,
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(conf),
                "x": float(cx),
                "y": float(cy),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })

            color = (0, 255, 0)
            if class_name == "ball":
                color = (0, 140, 255)
            elif class_name == "goal":
                color = (0, 255, 255)
            elif class_name == "robot":
                color = (255, 0, 255)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{class_name} #{track_id}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    writer.write(frame)
    frame_id += 1

cap.release()
writer.release()

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print("Video guardado:", OUT_VIDEO)
print("CSV guardado:", OUT_CSV)
print("Registros:", len(df))
