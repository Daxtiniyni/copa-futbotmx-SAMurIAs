import cv2
from pathlib import Path
from project_paths import DATA_DIR, VIDEO_V0

VIDEO_PATH = VIDEO_V0
OUT_DIR = DATA_DIR / "frames_v0_sample"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 60

cap = cv2.VideoCapture(str(VIDEO_PATH))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, total_frames // N_FRAMES)

frame_id = 0
saved = 0

print("Total frames:", total_frames)
print("Guardando cada:", step)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % step == 0 and saved < N_FRAMES:
        out_path = OUT_DIR / f"V0_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1
        print("Guardado:", out_path.name)

    frame_id += 1

    if saved >= N_FRAMES:
        break

cap.release()

print(f"Listo. Frames guardados: {saved}")
print(f"Carpeta: {OUT_DIR}")
