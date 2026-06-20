import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
from project_paths import PROJECT_ROOT

VIDEO_PATH = PROJECT_ROOT / "outputs" / "final" / "V1_final_narrated.mp4"
TRACKS_CSV = PROJECT_ROOT / "outputs" / "sam3_pipeline" / "V1_yolo_sam3_tracks.csv"
EVENTS_CSV = PROJECT_ROOT / "outputs" / "events" / "V1_events.csv"

tracks = pd.read_csv(TRACKS_CSV)
events = pd.read_csv(EVENTS_CSV)

MAX_TIME = float(tracks["time_sec"].max())

CSS = """
body {
    background: #0b1020;
}

.gradio-container {
    max-width: 1500px !important;
}

#title {
    text-align: center;
    padding: 18px;
    background: linear-gradient(90deg, #111827, #1f2937);
    border-radius: 18px;
    color: white;
}

.stat-card {
    background: #111827;
    border-radius: 16px;
    padding: 18px;
    color: white;
    text-align: center;
    border: 1px solid #374151;
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #38bdf8;
}

.stat-label {
    font-size: 14px;
    color: #d1d5db;
}
"""


def copy_to_temp(path):
    path = Path(path)
    temp_dir = Path(tempfile.mkdtemp())
    new_path = temp_dir / path.name
    shutil.copy2(path, new_path)
    return str(new_path)


def get_current_data(t):
    current = tracks[tracks["time_sec"] <= t].copy()
    current_events = events[events["time_sec"] <= t].copy()
    return current, current_events


def compute_possession(current):
    robots = current[current["class_name"] == "robot"].copy()
    ball = current[current["class_name"] == "ball"].copy()

    rows = []

    for frame in sorted(ball["frame"].unique()):
        b = ball[ball["frame"] == frame]
        r = robots[robots["frame"] == frame]

        if len(b) == 0 or len(r) == 0:
            continue

        bx, by = b.iloc[0][["x", "y"]]

        r = r.copy()
        r["dist_ball"] = np.sqrt((r["x"] - bx) ** 2 + (r["y"] - by) ** 2)
        nearest = r.sort_values("dist_ball").iloc[0]

        rows.append({
            "robot_id": int(nearest["track_id"]),
            "frame": frame,
            "time_sec": frame / 30
        })

    if not rows:
        return pd.DataFrame(columns=["Robot", "Tiempo de control (s)", "% posesión"])

    pos = pd.DataFrame(rows)
    summary = pos.groupby("robot_id").size().reset_index(name="frames")
    summary["Tiempo de control (s)"] = (summary["frames"] / 30).round(2)
    summary["% posesión"] = (summary["frames"] / summary["frames"].sum() * 100).round(1)
    summary = summary.rename(columns={"robot_id": "Robot"})
    return summary[["Robot", "Tiempo de control (s)", "% posesión"]]


def compute_robot_metrics(current):
    robots = current[current["class_name"] == "robot"].copy()
    rows = []

    for robot_id, g in robots.groupby("track_id"):
        g = g.sort_values("frame")
        dx = g["x"].diff()
        dy = g["y"].diff()
        dt = g["time_sec"].diff()

        dist = np.sqrt(dx**2 + dy**2).fillna(0)
        speed = dist / dt.replace(0, np.nan)

        rows.append({
            "Robot": int(robot_id),
            "Distancia px": round(dist.sum(), 1),
            "Vel. promedio px/s": round(speed.mean(), 1) if not np.isnan(speed.mean()) else 0,
            "Vel. máxima px/s": round(speed.max(), 1) if not np.isnan(speed.max()) else 0,
            "Frames visible": len(g)
        })

    return pd.DataFrame(rows)


def event_tables(current_events):
    passes = current_events[current_events["event"] == "change_of_possession"].copy()
    shots = current_events[current_events["event"] == "possible_shot"].copy()
    collisions = current_events[current_events["event"] == "possible_collision"].copy()

    pass_table = passes[["time_sec", "description"]].tail(10) if len(passes) else pd.DataFrame(columns=["time_sec", "description"])
    shot_table = shots[["time_sec", "description"]].tail(10) if len(shots) else pd.DataFrame(columns=["time_sec", "description"])
    collision_table = collisions[["time_sec", "description"]].tail(10) if len(collisions) else pd.DataFrame(columns=["time_sec", "description"])

    return pass_table, shot_table, collision_table


def make_heatmap(current, class_name="robot"):
    sub = current[current["class_name"] == class_name]

    fig = plt.figure(figsize=(6, 8))

    if len(sub) > 0:
        plt.hist2d(sub["x"], sub["y"], bins=50)

    plt.gca().invert_yaxis()
    plt.title(f"Mapa de calor dinámico: {class_name}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Actividad")

    out = Path(tempfile.mkdtemp()) / f"heatmap_{class_name}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(out)


def make_trails(current):
    robots = current[current["class_name"] == "robot"]

    fig = plt.figure(figsize=(6, 8))

    for robot_id, g in robots.groupby("track_id"):
        g = g.sort_values("frame")
        plt.plot(g["x"], g["y"], linewidth=1.5, label=f"Robot {int(robot_id)}")

    plt.gca().invert_yaxis()
    plt.title("Flujo del juego: trayectorias")
    plt.xlabel("X")
    plt.ylabel("Y")

    if len(robots["track_id"].unique()) <= 10:
        plt.legend(fontsize=8)

    out = Path(tempfile.mkdtemp()) / "trails.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return str(out)


def make_commentary(current_events, t):
    recent = current_events[current_events["time_sec"] >= max(0, t - 10)]

    if len(recent) == 0:
        return f"Tiempo {t:.1f}s: El sistema está observando el flujo del partido. Los robots se reposicionan y buscan controlar el balón."

    lines = [f"Tiempo {t:.1f}s — Narrativa automática:"]
    for _, e in recent.tail(5).iterrows():
        lines.append(f"- {e['description']}")

    return "\n".join(lines)


def update_dashboard(t):
    current, current_events = get_current_data(t)

    possession = compute_possession(current)
    robot_metrics = compute_robot_metrics(current)
    pass_table, shot_table, collision_table = event_tables(current_events)

    robot_count = current[current["class_name"] == "robot"]["track_id"].nunique()
    ball_seen = len(current[current["class_name"] == "ball"])
    total_events = len(current_events)
    total_passes = len(current_events[current_events["event"] == "change_of_possession"])

    cards = f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px;">
        <div class="stat-card"><div class="stat-number">{t:.1f}s</div><div class="stat-label">Tiempo analizado</div></div>
        <div class="stat-card"><div class="stat-number">{robot_count}</div><div class="stat-label">Robots rastreados</div></div>
        <div class="stat-card"><div class="stat-number">{total_passes}</div><div class="stat-label">Cambios de posesión</div></div>
        <div class="stat-card"><div class="stat-number">{total_events}</div><div class="stat-label">Eventos detectados</div></div>
    </div>
    """

    heatmap_robots = make_heatmap(current, "robot")
    heatmap_ball = make_heatmap(current, "ball")
    trails = make_trails(current)
    commentary = make_commentary(current_events, t)

    return (
        cards,
        possession,
        pass_table,
        shot_table,
        collision_table,
        robot_metrics,
        heatmap_robots,
        heatmap_ball,
        trails,
        commentary
    )


with gr.Blocks(css=CSS, title="FutBotMX SAM3 Live Dashboard") as demo:

    gr.HTML("""
    <div id="title">
        <h1>FutBotMX SAM3 Match Analyzer</h1>
        <p>Segmentación con SAM3, tracking, posesión, mapas dinámicos y narrativa automática</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            video = gr.Video(
                value=copy_to_temp(VIDEO_PATH),
                label="Partido analizado"
            )

        with gr.Column(scale=1):
            time_slider = gr.Slider(
                minimum=0,
                maximum=MAX_TIME,
                value=0,
                step=1,
                label="Tiempo del partido (segundos)"
            )
            update_btn = gr.Button("Actualizar dashboard")
            stat_cards = gr.HTML()

    with gr.Tab("Vista pública"):
        commentary = gr.Textbox(label="Narración automática", lines=8)
        with gr.Row():
            heatmap_robots = gr.Image(label="Mapa de calor dinámico — Robots")
            heatmap_ball = gr.Image(label="Mapa de calor dinámico — Balón")

    with gr.Tab("Métricas avanzadas"):
        possession_table = gr.Dataframe(label="Análisis de posesión")
        robot_metrics = gr.Dataframe(label="Métricas por robot")
        trails = gr.Image(label="Visualización del flujo del juego — Trails")

    with gr.Tab("Eventos"):
        pass_table = gr.Dataframe(label="Pases / cambios de posesión")
        shot_table = gr.Dataframe(label="Tiros posibles")
        collision_table = gr.Dataframe(label="Colisiones posibles")

    update_btn.click(
        fn=update_dashboard,
        inputs=time_slider,
        outputs=[
            stat_cards,
            possession_table,
            pass_table,
            shot_table,
            collision_table,
            robot_metrics,
            heatmap_robots,
            heatmap_ball,
            trails,
            commentary
        ]
    )

    time_slider.release(
        fn=update_dashboard,
        inputs=time_slider,
        outputs=[
            stat_cards,
            possession_table,
            pass_table,
            shot_table,
            collision_table,
            robot_metrics,
            heatmap_robots,
            heatmap_ball,
            trails,
            commentary
        ]
    )

demo.launch()
