from pathlib import Path
import shutil
from project_paths import PROJECT_ROOT

INPUT_DIR = PROJECT_ROOT / "data" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "data" / "frames_to_label"

N_PER_VIDEO = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Limpia la carpeta antes de copiar
for old_file in OUTPUT_DIR.glob("*.jpg"):
    old_file.unlink()

video_dirs = sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()])

total_copied = 0

for video_dir in video_dirs:
    frames = sorted(video_dir.glob("*.jpg"))

    if len(frames) == 0:
        print(f"{video_dir.name}: sin frames")
        continue

    step = max(1, len(frames) // N_PER_VIDEO)
    selected = frames[::step][:N_PER_VIDEO]

    for frame_path in selected:
        output_name = f"{video_dir.name}_{frame_path.name}"
        output_path = OUTPUT_DIR / output_name
        shutil.copy2(frame_path, output_path)
        total_copied += 1

    print(f"{video_dir.name}: {len(selected)} frames copiados")

print("--------------------------------")
print(f"Total de frames copiados: {total_copied}")
print(f"Carpeta destino: {OUTPUT_DIR}")
