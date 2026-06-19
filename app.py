from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "sam3"
UPLOAD_DIR = ROOT / "data" / "uploads"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
CALIBRATION_DIR = ROOT / "data" / "calibrations"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

for directory in (OUTPUT_DIR, UPLOAD_DIR, HEATMAP_DIR, CALIBRATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return value or f"analisis-{uuid.uuid4().hex[:8]}"


def read_records(path: Path) -> list[dict]:
    return read_analysis_payload(path)["frames"]


def read_analysis_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"metadata": {}, "frames": []}
    if isinstance(payload, dict):
        return {
            "metadata": payload.get("metadata", {}),
            "frames": payload.get("frames", []),
        }
    return {"metadata": {}, "frames": payload if isinstance(payload, list) else []}


def analysis_files(analysis_id: str) -> tuple[Path, Path]:
    return OUTPUT_DIR / f"{analysis_id}.json", OUTPUT_DIR / f"{analysis_id}.mp4"


def browser_video_path(analysis_id: str) -> Path:
    return OUTPUT_DIR / f"{analysis_id}.web.mp4"


def transcode_for_browser(source: Path, target: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def discover_analyses() -> list[dict]:
    analyses = []
    for json_path in sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        analysis_id = json_path.stem
        if analysis_id.startswith("smoke-"):
            continue
        video_path = OUTPUT_DIR / f"{analysis_id}.mp4"
        if not video_path.exists():
            continue
        payload = read_analysis_payload(json_path)
        analyses.append(
            build_summary(
                analysis_id,
                payload["frames"],
                video_path,
                payload["metadata"],
            )
        )
    return analyses


def build_summary(
    analysis_id: str,
    records: list[dict],
    video_path: Path,
    metadata: dict | None = None,
) -> dict:
    metadata = metadata or {}
    red = 0
    blue = 0
    balls = 0
    ball_frames = 0
    max_robots = 0
    timeline = []
    positions = {"red_team": [], "blue_team": [], "ball": []}
    trajectories: dict[str, dict] = {}
    calibrated = bool(metadata.get("homography"))
    field_size = metadata.get("field_size", [900, 600])

    max_x = 1.0
    max_y = 1.0
    for frame in records:
        frame_red = 0
        frame_blue = 0
        robots = frame.get("robots", [])
        balls_in_frame = frame.get("balls", [])
        for robot in robots:
            team = robot.get("team", "blue_team")
            box = robot.get("box", [0, 0, 0, 0])
            field_position = robot.get("field_position")
            image_position = robot.get("image_position")
            if calibrated and field_position:
                point = [float(field_position[0]), float(field_position[1])]
                positions.setdefault(team, []).append(point)
                max_x = max(max_x, float(field_size[0]))
                max_y = max(max_y, float(field_size[1]))
            elif image_position:
                point = [float(image_position[0]), float(image_position[1])]
                positions.setdefault(team, []).append(point)
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
            elif len(box) == 4:
                x = (float(box[0]) + float(box[2])) / 2
                y = float(box[3])
                point = [x, y]
                positions.setdefault(team, []).append(point)
                max_x = max(max_x, float(box[2]))
                max_y = max(max_y, float(box[3]))
            else:
                point = None
            track_id = robot.get("track_id")
            if track_id is not None and point is not None:
                key = f"{team}:{track_id}"
                trajectories.setdefault(
                    key,
                    {"track_id": track_id, "team": team, "points": []},
                )["points"].append(point)
            if team == "red_team":
                red += 1
                frame_red += 1
            else:
                blue += 1
                frame_blue += 1
        for ball in balls_in_frame:
            box = ball.get("box", [0, 0, 0, 0])
            field_position = ball.get("field_position")
            image_position = ball.get("image_position")
            if calibrated and field_position:
                point = [float(field_position[0]), float(field_position[1])]
                positions["ball"].append(point)
                max_x = max(max_x, float(field_size[0]))
                max_y = max(max_y, float(field_size[1]))
            elif image_position:
                point = [float(image_position[0]), float(image_position[1])]
                positions["ball"].append(point)
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
            elif len(box) == 4:
                x = (float(box[0]) + float(box[2])) / 2
                y = (float(box[1]) + float(box[3])) / 2
                point = [x, y]
                positions["ball"].append(point)
                max_x = max(max_x, float(box[2]))
                max_y = max(max_y, float(box[3]))
            else:
                point = None
            track_id = ball.get("track_id")
            if track_id is not None and point is not None:
                key = f"ball:{track_id}"
                trajectories.setdefault(
                    key,
                    {"track_id": track_id, "team": "ball", "points": []},
                )["points"].append(point)
        balls += len(balls_in_frame)
        if balls_in_frame:
            ball_frames += 1
        max_robots = max(max_robots, len(robots))
        timeline.append(
            {
                "time": round(float(frame.get("time_sec", len(timeline))), 1),
                "red": frame_red,
                "blue": frame_blue,
                "ball": len(balls_in_frame),
            }
        )

    frame_count = len(records)
    duration = timeline[-1]["time"] if timeline else 0
    total_robot_observations = red + blue
    red_share = round(red * 100 / total_robot_observations) if total_robot_observations else 0
    blue_share = 100 - red_share if total_robot_observations else 0
    narrative = build_narrative(
        frame_count,
        duration,
        red,
        blue,
        ball_frames,
        max_robots,
        len([track for track in trajectories.values() if track["team"] != "ball"]),
        calibrated,
    )
    playable_video = browser_video_path(analysis_id)
    if not playable_video.exists():
        playable_video = video_path

    return {
        "id": analysis_id,
        "name": analysis_id.replace("_", " ").replace("-", " ").title(),
        "video_url": f"/media/{playable_video.name}",
        "heatmap_url": f"/api/analyses/{analysis_id}/heatmap",
        "frames": frame_count,
        "duration": duration,
        "red_observations": red,
        "blue_observations": blue,
        "ball_observations": balls,
        "ball_frames": ball_frames,
        "ball_visibility": round(ball_frames * 100 / frame_count) if frame_count else 0,
        "max_robots": max_robots,
        "red_share": red_share,
        "blue_share": blue_share,
        "timeline": timeline,
        "positions": positions,
        "source_size": field_size if calibrated else metadata.get("source_size", [max_x, max_y]),
        "trajectories": list(trajectories.values()),
        "tracking": bool(metadata.get("tracking")) or bool(trajectories),
        "calibrated": calibrated,
        "field_size": field_size,
        "calibration_points": metadata.get("calibration_points"),
        "narrative": narrative,
    }


def build_narrative(
    frame_count: int,
    duration: float,
    red: int,
    blue: int,
    ball_frames: int,
    max_robots: int,
    track_count: int,
    calibrated: bool,
) -> list[str]:
    if not frame_count:
        return ["El análisis aún no contiene detecciones suficientes."]
    dominant = "rojo" if red > blue else "azul" if blue > red else "equilibrado"
    ball_rate = ball_frames / frame_count
    return [
        f"Se analizaron {frame_count} muestras a lo largo de {duration:.1f} segundos.",
        f"El equipo {dominant} tuvo mayor presencia visual en los frames procesados."
        if dominant != "equilibrado"
        else "La presencia visual de ambos equipos fue equilibrada.",
        f"El máximo observado fue de {max_robots} robots simultáneos.",
        f"Se construyeron {track_count} trayectorias de robots con identidad temporal."
        if track_count
        else "Este análisis anterior no contiene identidades temporales.",
        f"El balón apareció en {ball_rate:.0%} de las muestras analizadas.",
        "Las posiciones fueron proyectadas mediante homografía al campo canónico."
        if calibrated
        else "Las posiciones son aproximadas porque el análisis no tiene calibración homográfica.",
    ]


def create_heatmap(
    analysis_id: str,
    records: list[dict],
    metadata: dict | None = None,
) -> Path:
    output_path = HEATMAP_DIR / f"{analysis_id}.png"
    json_path, _ = analysis_files(analysis_id)
    if output_path.exists() and output_path.stat().st_mtime >= json_path.stat().st_mtime:
        return output_path

    summary = build_summary(
        analysis_id,
        records,
        OUTPUT_DIR / f"{analysis_id}.mp4",
        metadata,
    )
    source_w, source_h = summary["source_size"]
    width, height = 960, 540
    field = np.full((height, width, 3), (34, 104, 54), dtype=np.uint8)
    cv2.rectangle(field, (20, 20), (width - 20, height - 20), (225, 239, 225), 3)
    cv2.line(field, (width // 2, 20), (width // 2, height - 20), (225, 239, 225), 2)
    cv2.circle(field, (width // 2, height // 2), 68, (225, 239, 225), 2)

    layers = {
        "red_team": np.zeros((height, width), dtype=np.float32),
        "blue_team": np.zeros((height, width), dtype=np.float32),
    }
    for team, layer in layers.items():
        for x, y in summary["positions"].get(team, []):
            px = int(np.clip(x / source_w * (width - 40) + 20, 20, width - 20))
            py = int(np.clip(y / source_h * (height - 40) + 20, 20, height - 20))
            cv2.circle(layer, (px, py), 22, 1.0, -1)
        layer[:] = cv2.GaussianBlur(layer, (0, 0), 24)
        peak = float(layer.max())
        if peak:
            layer[:] /= peak

    red_overlay = np.zeros_like(field)
    red_overlay[:, :, 2] = (layers["red_team"] * 255).astype(np.uint8)
    blue_overlay = np.zeros_like(field)
    blue_overlay[:, :, 0] = (layers["blue_team"] * 255).astype(np.uint8)
    intensity = np.maximum(layers["red_team"], layers["blue_team"])[..., None]
    combined = red_overlay + blue_overlay
    field = (field * (1 - intensity * 0.65) + combined * intensity * 0.65).astype(np.uint8)

    cv2.putText(field, "Equipo rojo", (34, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 245), 2)
    cv2.putText(field, "Equipo azul", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 130, 55), 2)
    cv2.imwrite(str(output_path), field)
    return output_path


def update_job(job_id: str, **values) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def run_analysis_job(
    job_id: str,
    source_path: Path,
    analysis_id: str,
    sample_fps: float,
    max_width: int,
    calibration_path: Path | None,
) -> None:
    output_json, output_video = analysis_files(analysis_id)
    command = [
        str(ROOT / ".venv-sam3" / "bin" / "python"),
        "-u",
        str(ROOT / "scripts" / "segment_robot_soccer_video.py"),
        str(source_path),
        "--seconds",
        "999999",
        "--sample-fps",
        str(sample_fps),
        "--max-width",
        str(max_width),
        "--out",
        str(output_video),
        "--json-out",
        str(output_json),
    ]
    if calibration_path is not None:
        command.extend(["--calibration", str(calibration_path)])
    update_job(job_id, status="running", message="Cargando SAM 3 y preparando el video")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log_lines = []
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            log_lines.append(line)
            if line.startswith("frame "):
                update_job(job_id, message=line)
        return_code = process.wait()
        if return_code:
            update_job(
                job_id,
                status="failed",
                message="El análisis terminó con error",
                log=log_lines[-20:],
            )
            return
        update_job(job_id, message="Convirtiendo video para reproducción web")
        transcode_for_browser(output_video, browser_video_path(analysis_id))
        update_job(
            job_id,
            status="completed",
            message="Análisis completado",
            analysis_id=analysis_id,
            log=log_lines[-20:],
        )
    except Exception as exc:
        update_job(job_id, status="failed", message=str(exc))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/analyses")
def analyses():
    return jsonify(discover_analyses())


@app.get("/api/analyses/<analysis_id>")
def analysis_detail(analysis_id: str):
    json_path, video_path = analysis_files(analysis_id)
    if not json_path.exists() or not video_path.exists():
        abort(404)
    payload = read_analysis_payload(json_path)
    return jsonify(
        build_summary(
            analysis_id,
            payload["frames"],
            video_path,
            payload["metadata"],
        )
    )


@app.get("/api/analyses/<analysis_id>/heatmap")
def heatmap(analysis_id: str):
    json_path, _ = analysis_files(analysis_id)
    if not json_path.exists():
        abort(404)
    payload = read_analysis_payload(json_path)
    return send_file(
        create_heatmap(analysis_id, payload["frames"], payload["metadata"]),
        mimetype="image/png",
    )


@app.get("/media/<path:filename>")
def media(filename: str):
    path = (OUTPUT_DIR / filename).resolve()
    if OUTPUT_DIR.resolve() not in path.parents or not path.exists():
        abort(404)
    return send_file(path, conditional=True)


@app.post("/api/analyze")
def analyze_video():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "Selecciona un video"}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Formato de video no compatible"}), 400

    requested_name = request.form.get("name") or Path(upload.filename).stem
    analysis_id = slugify(requested_name)
    stored_name = f"{analysis_id}-{uuid.uuid4().hex[:6]}{suffix}"
    source_path = UPLOAD_DIR / secure_filename(stored_name)
    points = None
    raw_calibration = request.form.get("calibration_points", "").strip()
    if raw_calibration:
        try:
            points = json.loads(raw_calibration)
            if not isinstance(points, list) or len(points) != 4:
                raise ValueError
            if any(
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    not isinstance(value, (int, float)) or not 0 <= value <= 1
                    for value in point
                )
                for point in points
            ):
                raise ValueError
            contour = np.asarray(points, dtype=np.float32)
            if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 0.01:
                raise ValueError
        except (json.JSONDecodeError, ValueError, TypeError):
            return jsonify({"error": "La calibración de cancha no es válida"}), 400

    try:
        sample_fps = min(max(float(request.form.get("sample_fps", 1)), 0.2), 5)
        max_width = min(max(int(request.form.get("max_width", 960)), 480), 1920)
    except ValueError:
        return jsonify({"error": "Parámetros de análisis inválidos"}), 400

    upload.save(source_path)
    calibration_path = None
    if points is not None:
        calibration_path = CALIBRATION_DIR / f"{analysis_id}.json"
        calibration_path.write_text(
            json.dumps({"points": points, "normalized": True}, indent=2),
            encoding="utf-8",
        )

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "Análisis en cola",
            "analysis_id": analysis_id,
        }
    threading.Thread(
        target=run_analysis_job,
        args=(
            job_id,
            source_path,
            analysis_id,
            sample_fps,
            max_width,
            calibration_path,
        ),
        daemon=True,
    ).start()
    return jsonify(jobs[job_id]), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
