import cv2
from pathlib import Path
from project_paths import DATA_DIR

VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_STEP = 30

video_files = sorted(VIDEOS_DIR.glob("*.MOV"))

for video_path in video_files:

    video_name = video_path.stem

    print(f"Procesando {video_name}")

    out_dir = OUTPUT_DIR / video_name
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_id = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP == 0:

            out_path = out_dir / f"{video_name}_{frame_id:06d}.jpg"

            cv2.imwrite(str(out_path), frame)

            saved += 1

        frame_id += 1

    cap.release()

    print(f"Frames guardados: {saved}")

print("Proceso terminado")
