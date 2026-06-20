import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from project_paths import OUTPUTS_DIR

CSV_PATH = OUTPUTS_DIR / "sam3_pipeline_v0" / "V0_yolo_sam3_tracks.csv"
OUT_DIR = OUTPUTS_DIR / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
robots = df[df["class_name"] == "robot"]
ball = df[df["class_name"] == "ball"]

# Heatmap robots
plt.figure(figsize=(8, 12))
plt.hist2d(robots["x"], robots["y"], bins=60)
plt.gca().invert_yaxis()
plt.title("Mapa de calor de actividad - Robots")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Frecuencia")
plt.savefig(OUT_DIR / "heatmap_robots.png", dpi=200)
plt.close()

# Heatmap pelota
plt.figure(figsize=(8, 12))
plt.hist2d(ball["x"], ball["y"], bins=60)
plt.gca().invert_yaxis()
plt.title("Mapa de calor de actividad - Balón")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Frecuencia")
plt.savefig(OUT_DIR / "heatmap_ball.png", dpi=200)
plt.close()

# Trails por robot
plt.figure(figsize=(8, 12))
for track_id, group in robots.groupby("track_id"):
    group = group.sort_values("frame")
    plt.plot(group["x"], group["y"], linewidth=1, label=f"Robot {track_id}")

plt.gca().invert_yaxis()
plt.title("Trayectorias por robot")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig(OUT_DIR / "robot_trails.png", dpi=200)
plt.close()

print("Visualizaciones guardadas en:", OUT_DIR)
