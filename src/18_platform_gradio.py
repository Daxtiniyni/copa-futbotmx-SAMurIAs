import gradio as gr
import shutil
import tempfile
from pathlib import Path
from project_paths import PROJECT_ROOT

FINAL_VIDEO = PROJECT_ROOT / "outputs" / "final" / "V1_final_narrated.mp4"
HEATMAP_ROBOTS = PROJECT_ROOT / "outputs" / "visualizations" / "heatmap_robots.png"
HEATMAP_BALL = PROJECT_ROOT / "outputs" / "visualizations" / "heatmap_ball.png"
TRAILS = PROJECT_ROOT / "outputs" / "visualizations" / "robot_trails.png"
EVENTS = PROJECT_ROOT / "outputs" / "events" / "V1_events.csv"
COMMENTARY = PROJECT_ROOT / "outputs" / "narration" / "V1_commentary.txt"


def copy_to_temp(path):
    path = Path(path)
    temp_dir = Path(tempfile.mkdtemp())
    new_path = temp_dir / path.name
    shutil.copy2(path, new_path)
    return str(new_path)


def run_demo():
    commentary_text = COMMENTARY.read_text(encoding="utf-8") if COMMENTARY.exists() else "Narración no encontrada."

    return (
        copy_to_temp(FINAL_VIDEO),
        copy_to_temp(HEATMAP_ROBOTS),
        copy_to_temp(HEATMAP_BALL),
        copy_to_temp(TRAILS),
        copy_to_temp(EVENTS),
        commentary_text
    )


with gr.Blocks(title="FutBotMX SAM3 Match Analyzer") as demo:
    gr.Markdown("# FutBotMX SAM3 Match Analyzer")
    gr.Markdown("Demo con resultados generados para V1: SAM3 + tracking + eventos + visualizaciones + narración.")

    run_btn = gr.Button("Mostrar análisis V1")

    output_video = gr.Video(label="Video analizado con narración")

    with gr.Tab("Público general"):
        commentary_box = gr.Textbox(label="Narración automática", lines=12)
        heatmap_robots = gr.Image(label="Mapa de calor de robots")
        heatmap_ball = gr.Image(label="Mapa de calor del balón")

    with gr.Tab("Métricas avanzadas"):
        trails = gr.Image(label="Trayectorias por robot")
        events_file = gr.File(label="Eventos detectados CSV")

    run_btn.click(
        fn=run_demo,
        inputs=[],
        outputs=[
            output_video,
            heatmap_robots,
            heatmap_ball,
            trails,
            events_file,
            commentary_box
        ]
    )

demo.launch()
