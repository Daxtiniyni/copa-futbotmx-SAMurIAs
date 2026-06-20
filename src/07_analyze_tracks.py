import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import OUTPUTS_DIR

CSV_PATH = OUTPUTS_DIR / "sam3_pipeline_v0" / "V0_yolo_sam3_tracks.csv"
OUT_DIR = OUTPUTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Leyendo CSV...")
df = pd.read_csv(CSV_PATH)
print("CSV cargado:", df.shape)
print(df["class_name"].value_counts())

robots = df[df["class_name"] == "robot"].copy()
ball = df[df["class_name"] == "ball"].copy()

print("Calculando métricas por robot...")

summary = []

for track_id, group in robots.groupby("track_id"):
    print("Robot:", track_id, "registros:", len(group))

    group = group.sort_values("frame")
    dx = group["x"].diff()
    dy = group["y"].diff()
    dt = group["time_sec"].diff()

    dist = np.sqrt(dx**2 + dy**2).fillna(0)
    speed = dist / dt.replace(0, np.nan)

    summary.append({
        "track_id": int(track_id),
        "frames_detected": len(group),
        "time_visible_sec": round(group["time_sec"].max() - group["time_sec"].min(), 2),
        "distance_px": round(dist.sum(), 2),
        "avg_speed_px_s": round(speed.mean(), 2) if not np.isnan(speed.mean()) else 0,
        "max_speed_px_s": round(speed.max(), 2) if not np.isnan(speed.max()) else 0,
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "robot_metrics_v0.csv", index=False)

print("Calculando posesión rápida...")

merged_rows = []

for frame, b in ball.groupby("frame"):
    r = robots[robots["frame"] == frame]

    if r.empty:
        continue

    bx = b.iloc[0]["x"]
    by = b.iloc[0]["y"]

    distances = np.sqrt((r["x"] - bx) ** 2 + (r["y"] - by) ** 2)
    nearest_idx = distances.idxmin()
    nearest = r.loc[nearest_idx]

    merged_rows.append({
        "frame": int(frame),
        "time_sec": float(b.iloc[0]["time_sec"]),
        "robot_id": int(nearest["track_id"]),
        "dist_to_ball": float(distances.loc[nearest_idx])
    })

possession_df = pd.DataFrame(merged_rows)
possession_df.to_csv(OUT_DIR / "possession_v0.csv", index=False)

if not possession_df.empty:
    possession_summary = possession_df.groupby("robot_id").size().reset_index(name="frames_in_possession")
    possession_summary["seconds"] = (possession_summary["frames_in_possession"] / 30).round(2)
    possession_summary["percent"] = (
        possession_summary["frames_in_possession"] / possession_summary["frames_in_possession"].sum() * 100
    ).round(2)
    possession_summary.to_csv(OUT_DIR / "possession_summary_v0.csv", index=False)
    print(possession_summary)

print("Listo.")
print("Guardado en:", OUT_DIR)
