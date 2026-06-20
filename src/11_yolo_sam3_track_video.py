import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO, SAM
from project_paths import OUTPUTS_DIR, SAM3_MODEL, VIDEO_V0, YOLO_MODEL

VIDEO_PATH = VIDEO_V0

OUT_DIR = OUTPUTS_DIR / "sam3_pipeline_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_VIDEO = OUT_DIR / "V0_yolo_sam3_tracking.mp4"
OUT_CSV = OUT_DIR / "V0_yolo_sam3_tracks.csv"

yolo = YOLO(YOLO_MODEL)
sam3 = SAM(SAM3_MODEL)

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

results = yolo.track(
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

        # SAM3 refina usando las cajas detectadas por YOLO
        sam_result = sam3(
            frame,
            bboxes=boxes.tolist(),
            verbose=False
        )[0]

        masks = None
        if sam_result.masks is not None:
            masks = sam_result.masks.data.cpu().numpy()

        for i, (box, track_id, class_id, conf) in enumerate(zip(boxes, ids, classes, confs)):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = names[class_id]

            color = (0, 255, 0)
            if class_name == "ball":
                color = (0, 140, 255)
            elif class_name == "goal":
                color = (0, 255, 255)
            elif class_name == "robot":
                color = (255, 0, 255)

            if masks is not None and i < len(masks):
                mask = masks[i]
                mask = cv2.resize(mask.astype("uint8"), (width, height))
                colored = frame.copy()
                colored[mask > 0] = color
                frame = cv2.addWeighted(frame, 0.75, colored, 0.25, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                frame,
                f"{class_name} #{track_id}",
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

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

    cv2.putText(
        frame,
        "YOLO fine-tuned + SAM3 mask refinement + ByteTrack",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
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
