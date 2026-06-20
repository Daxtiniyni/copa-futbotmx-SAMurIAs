import base64
import html
import os
import time
import random
import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from project_paths import PROJECT_ROOT

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None

# ============================================================
# RUTAS
# ============================================================
VIDEO_PATH = PROJECT_ROOT / "outputs" / "final" / "V0_final_narrated.mp4"
TRACKS_CSV = PROJECT_ROOT / "outputs" / "sam3_pipeline_v0" / "V0_yolo_sam3_tracks.csv"
EVENTS_CSV = PROJECT_ROOT / "outputs" / "events" / "V0_events.csv"
AUDIO_DIR = PROJECT_ROOT / "outputs" / "martinoli_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BACKGROUND_MUSIC = PROJECT_ROOT / "assets" / "cancion.mp3"

FPS = 30
APP_TITLE = "FutBotMX SAM3 Match Analyzer"
SUBTITLE = "Inteligencia Artificial aplicada al Fútbol Robótico"
VIDEO_NAME = "V0 · SAM3 + YOLO11 + ByteTrack + Narración"

# ============================================================
# IDENTIDAD DE JUGADORES
# El dashboard ya NO enseña IDs raros como Robot 12323.
# Si tus tracks reales son otros IDs, el sistema toma los 4 más frecuentes
# y los renombra automáticamente como los jugadores del partido.
# ============================================================
PLAYER_ORDER = [
    {"slot": 1, "name": "Robotiño", "short": "Robot 1 · Robotiño", "team": "Equipo Azul", "color": "#38bdf8"},
    {"slot": 2, "name": "Messibot", "short": "Robot 2 · Messibot", "team": "Equipo Azul", "color": "#2563eb"},
    {"slot": 3, "name": "Cristiano Ronabot", "short": "Robot 3 · Cristiano Ronabot", "team": "Equipo Rojo", "color": "#fb7185"},
    {"slot": 4, "name": "Botbappé", "short": "Robot 4 · Botbappé", "team": "Equipo Rojo", "color": "#ef4444"},
]
TEAM_COLOR = {"Equipo Azul": "#38bdf8", "Equipo Rojo": "#fb7185", "Sin equipo": "#94a3b8"}

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# DATOS
# ============================================================
def _safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=4)
def load_raw_data():
    tracks_df = pd.read_csv(TRACKS_CSV)
    events_df = pd.read_csv(EVENTS_CSV)

    tracks_df = _safe_numeric(tracks_df, ["time_sec", "frame", "track_id", "x", "y", "w", "h"])
    events_df = _safe_numeric(events_df, [
        "time_sec", "robot_id", "from_robot", "to_robot", "robot_1", "robot_2",
        "duration_sec", "ball_speed_px_s", "goal_distance", "distance"
    ])

    tracks_df["time_sec"] = tracks_df.get("time_sec", 0).fillna(0)
    tracks_df["frame"] = tracks_df.get("frame", 0).fillna(0).astype(int)
    tracks_df["track_id"] = tracks_df.get("track_id", -1).fillna(-1).astype(int)
    if "class_name" not in tracks_df.columns:
        tracks_df["class_name"] = "robot"

    events_df["time_sec"] = events_df.get("time_sec", 0).fillna(0)
    if "event" not in events_df.columns:
        events_df["event"] = "event"
    if "description" not in events_df.columns:
        events_df["description"] = "Evento detectado"

    return tracks_df, events_df

raw_tracks, raw_events = load_raw_data()

@st.cache_data(ttl=4)
def build_identity_map(tracks_df: pd.DataFrame):
    robots = tracks_df[tracks_df["class_name"].astype(str).str.lower().eq("robot")].copy()
    if robots.empty:
        return {}, {}

    counts = robots.groupby("track_id").size().sort_values(ascending=False)

    # Preferencia: si existen 1,2,3,4 como tracks principales, usar esos.
    fixed_ids = [tid for tid in [1, 2, 3, 4] if tid in counts.index]
    remaining = [int(tid) for tid in counts.index if int(tid) not in fixed_ids]
    chosen = (fixed_ids + remaining)[:4]

    id_to_player = {}
    player_to_id = {}
    for idx, tid in enumerate(chosen):
        player = PLAYER_ORDER[idx]
        id_to_player[int(tid)] = player
        player_to_id[player["slot"]] = int(tid)
    return id_to_player, player_to_id

ID_TO_PLAYER, PLAYER_TO_ID = build_identity_map(raw_tracks)


def player_for_track(track_id):
    try:
        return ID_TO_PLAYER.get(int(track_id))
    except Exception:
        return None


def player_short(track_id):
    p = player_for_track(track_id)
    return p["short"] if p else "Robot sin asignar"


def player_name(track_id):
    p = player_for_track(track_id)
    return p["name"] if p else "Robot sin asignar"


def player_team(track_id):
    p = player_for_track(track_id)
    return p["team"] if p else "Sin equipo"


def player_slot(track_id):
    p = player_for_track(track_id)
    return p["slot"] if p else None


def sanitize_description(text):
    """Reemplaza IDs crudos por nombres claros."""
    if pd.isna(text):
        return "Evento detectado"
    text = str(text)

    # Reemplazar Robot 123 por Robot N · Nombre, cuando existe mapeo.
    def repl_robot(m):
        tid = int(m.group(1))
        return player_short(tid) if player_for_track(tid) else "robot sin asignar"

    text = re.sub(r"Robot\s+(\d+)", repl_robot, text)
    text = text.replace("Sin equipo", "sin equipo asignado")
    return text

@st.cache_data(ttl=4)
def normalize_data(tracks_df: pd.DataFrame, events_df: pd.DataFrame, id_keys):
    id_to_player = id_keys
    tracks = tracks_df.copy()
    tracks["player_slot"] = tracks["track_id"].map(lambda x: id_to_player.get(int(x), {}).get("slot") if pd.notna(x) else None)
    tracks["robot_name"] = tracks["track_id"].map(lambda x: id_to_player.get(int(x), {}).get("name", "Robot sin asignar") if pd.notna(x) else "Robot sin asignar")
    tracks["robot_label"] = tracks["track_id"].map(lambda x: id_to_player.get(int(x), {}).get("short", "Robot sin asignar") if pd.notna(x) else "Robot sin asignar")
    tracks["team"] = tracks["track_id"].map(lambda x: id_to_player.get(int(x), {}).get("team", "Sin equipo") if pd.notna(x) else "Sin equipo")
    tracks["color"] = tracks["track_id"].map(lambda x: id_to_player.get(int(x), {}).get("color", "#94a3b8") if pd.notna(x) else "#94a3b8")

    events = events_df.copy()
    events["description_clean"] = events["description"].apply(sanitize_description)

    # Cuando los eventos traen columnas de robots, construir una descripción más limpia.
    def event_label(row):
        ev = str(row.get("event", "event"))
        t = float(row.get("time_sec", 0))
        if ev in ["change_of_possession", "interception"] and pd.notna(row.get("from_robot")) and pd.notna(row.get("to_robot")):
            return f"{player_short(row['from_robot'])} pierde la pelota y {player_short(row['to_robot'])} recupera posesión."
        if ev in ["pass", "possible_pass"] and pd.notna(row.get("from_robot")) and pd.notna(row.get("to_robot")):
            return f"Pase de {player_short(row['from_robot'])} para {player_short(row['to_robot'])}."
        if ev in ["possible_collision", "collision"] and pd.notna(row.get("robot_1")) and pd.notna(row.get("robot_2")):
            d = row.get("distance")
            extra = f" Distancia mínima: {float(d):.1f} px." if pd.notna(d) else ""
            return f"Choque entre {player_short(row['robot_1'])} y {player_short(row['robot_2'])}.{extra}"
        if ev in ["possible_shot", "shot"] and pd.notna(row.get("robot_id")):
            sp = row.get("ball_speed_px_s")
            extra = f" Velocidad: {float(sp):.0f} px/s." if pd.notna(sp) else ""
            return f"Tiro de {player_short(row['robot_id'])}.{extra}"
        if ev == "goal" and pd.notna(row.get("robot_id")):
            return f"¡Gol de {player_short(row['robot_id'])}! El balón entra a zona de portería."
        if ev == "possession" and pd.notna(row.get("robot_id")):
            dur = row.get("duration_sec")
            extra = f" durante {float(dur):.1f} s" if pd.notna(dur) else ""
            return f"{player_short(row['robot_id'])} controla el balón{extra}."
        return sanitize_description(row.get("description_clean", row.get("description", "Evento detectado")))

    events["description_clean"] = events.apply(event_label, axis=1)
    return tracks, events

tracks, events = normalize_data(raw_tracks, raw_events, ID_TO_PLAYER)
MAX_TIME = float(tracks["time_sec"].max()) if not tracks.empty else 0.0

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at 5% 0%, #12325b 0%, #07111f 38%, #02040a 100%);
    color:#f8fafc;
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer{visibility:hidden;height:0;}
.block-container{padding:.65rem 1rem 1.2rem 1rem;max-width:1920px;}
.header{
    border:1px solid rgba(56,189,248,.24);border-radius:18px;padding:16px 22px;margin-bottom:14px;
    background:linear-gradient(90deg,rgba(5,15,30,.98),rgba(9,22,42,.98));
    box-shadow:0 0 40px rgba(56,189,248,.10) inset, 0 0 30px rgba(0,0,0,.35);
    display:flex;align-items:center;justify-content:space-between;gap:16px;
}
.header h1{margin:0;font-size:34px;font-weight:950;letter-spacing:.2px}.header p{margin:4px 0 0 0;color:#38bdf8;font-weight:800}.meta{font-size:31px;font-weight:950}.meta span{color:#3b82f6}
.panel{
    border:1px solid rgba(56,189,248,.20);border-radius:18px;background:linear-gradient(180deg,rgba(8,20,38,.97),rgba(4,9,18,.97));
    padding:14px;margin-bottom:12px;box-shadow:0 0 25px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.04);
}
.panel-title{font-size:16px;font-weight:950;color:#38bdf8;margin-bottom:10px;letter-spacing:.2px}
.section-title{font-size:18px;font-weight:950;color:#22c55e;margin:4px 0 10px 0;}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.metric-card{background:linear-gradient(180deg,#0a1a30,#071426);border:1px solid #1e3a5f;border-radius:15px;padding:13px;text-align:center;position:relative;overflow:hidden}
.metric-card:before{content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:linear-gradient(90deg,#38bdf8,#a855f7,#22c55e)}
.metric-number{font-size:28px;font-weight:950;color:#38bdf8}.metric-label{font-size:13px;color:#cbd5e1}.metric-delta{font-size:12px;margin-top:3px;color:#a7f3d0}
video{width:100%!important;max-height:665px!important;object-fit:contain!important;background:#000;border-radius:14px}
.commentary{background:#050b14;border:1px solid #1f3b5c;border-radius:14px;padding:12px;color:#dbeafe;font-size:15px;line-height:1.45;max-height:285px;overflow:auto}
.small-text{color:#cbd5e1;font-size:13px}.live-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;box-shadow:0 0 12px #22c55e;margin-right:8px}.audio-status{font-size:13px;color:#cbd5e1;margin-top:8px}
.table-wrap{overflow:auto;border-radius:13px;border:1px solid #1f3b5c;background:#050b14;max-height:310px}
table.futbot-table{width:100%;border-collapse:collapse;font-size:14px;color:#f8fafc}
table.futbot-table th{position:sticky;top:0;background:#10233d;color:#7dd3fc;text-align:left;padding:10px;font-weight:950;border-bottom:1px solid #2b4b70;z-index:2}
table.futbot-table td{background:#071426;color:#f8fafc;padding:9px;border-bottom:1px solid #152842;vertical-align:middle}
table.futbot-table tr:nth-child(even) td{background:#0b1930}
.name-pill{display:inline-block;padding:4px 9px;border-radius:999px;font-weight:900;color:#020617;border:1px solid rgba(255,255,255,.15)}
.team-pill{display:inline-block;padding:3px 8px;border-radius:999px;font-weight:800;background:#111827;color:#e5e7eb;border:1px solid #334155}
.bar-bg{height:12px;background:#111827;border:1px solid #334155;border-radius:999px;min-width:92px;overflow:hidden}
.bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#38bdf8,#a855f7)}
.delta-up{color:#86efac;font-weight:900}.delta-down{color:#fca5a5;font-weight:900}.delta-flat{color:#cbd5e1;font-weight:900}
.event-chip{display:inline-block;font-size:12px;padding:3px 8px;border-radius:999px;background:#172554;color:#bfdbfe;border:1px solid #1d4ed8;font-weight:800}
.legend-card{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}.legend-item{background:#071426;border:1px solid #1f3b5c;border-radius:12px;padding:7px 10px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
@media(max-width:1100px){.metric-grid{grid-template-columns:repeat(2,1fr)}.header h1{font-size:24px}.meta{font-size:22px}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUDIO Y NARRACIÓN
# ============================================================
INTRO_LINES = [
    "¡Arranca esto, Doctor García! Bienvenidos al laboratorio convertido en estadio: robots autónomos, baterías calientes y algoritmos con ganas de gloria.",
    "¡Se prende la fábrica del fútbol robótico! Sensores calibrados, motores rugiendo y robots listos para partirse las balatas con elegancia científica.",
    "¡Ajusten firmware, carguen baterías y que ruede el balón! Esto parece final del mundo, pero con servomotores y olor a aceite quemado.",
]
FILLERS = [
    "¡No puede ser, Doctor García, alguien revise esos sensores!",
    "¡Ese robot se quedó pensando más que tesis de doctorado!",
    "¡La inteligencia artificial abandonó el chat por tres segundos!",
    "¡A ese algoritmo le urge una actualización emocional!",
    "¡Esto ya huele a aceite caliente y decisión peligrosa!",
]
EVENT_TEMPLATES = {
    "change_of_possession": ["{desc} ¡Le robaron la cartera, el firmware y hasta la garantía extendida!", "{desc} ¡Entró con las balatas por delante y recuperó como si viniera saliendo del taller!"],
    "interception": ["{desc} ¡Intercepción quirúrgica, Doctor García, puro sensor fino!"],
    "pass": ["{desc} ¡Pase con precisión de servomotor recién aceitado!"],
    "possible_collision": ["{desc} ¡Choque de lámina, eso sonó a refaccionaria abierta en domingo!"],
    "collision": ["{desc} ¡Dos máquinas industriales se dieron un abrazo con odio deportivo!"],
    "possible_shot": ["{desc} ¡Le pegó con overclocking, turbo y batería al cien!"],
    "shot": ["{desc} ¡La pelota salió como misil con inteligencia artificial!"],
    "goal": ["{desc} ¡GOOOOOOL robótico! ¡Se fundieron los sensores de la emoción!"],
    "default": ["{desc} ¡Esto se está poniendo sabroso, con aceite, sensores y drama de liguilla!"],
}

@st.cache_resource
def get_eleven_client():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or ElevenLabs is None:
        return None
    return ElevenLabs(api_key=api_key)

def estimate_audio_seconds(texto: str) -> float:
    words = max(1, len(texto.split()))
    return max(5.0, min(18.0, words / 2.45 + 1.5))

@st.cache_data(show_spinner=False)
def generar_audio_martinoli(texto: str, event_key: str) -> str | None:
    client = get_eleven_client()
    if client is None:
        return None
    digest = hashlib.md5((event_key + texto).encode("utf-8")).hexdigest()[:16]
    out_path = AUDIO_DIR / f"martinoli_{digest}.mp3"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "12tdVBKVpQ2NJSeVcpWR")
    audio = client.text_to_speech.convert(
        text=texto,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    return str(out_path)

def reproducir_audio_autoplay(audio_path: str):
    if not audio_path or not Path(audio_path).exists():
        return
    audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")
    st.components.v1.html(
        f"""
        <audio id="martinoli_audio" autoplay controls style="width:100%;height:34px;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        <script>
            const audios = window.parent.document.querySelectorAll('audio');
            audios.forEach((x) => {{ if (x.id !== 'martinoli_audio') {{ try {{ x.pause(); }} catch(e) {{}} }} }});
            const a = document.getElementById('martinoli_audio');
            if (a) {{ a.volume = 1.0; a.play().catch(()=>{{}}); }}
        </script>
        """,
        height=44,
    )

def reproducir_musica_fondo():
    if not BACKGROUND_MUSIC.exists():
        return

    audio_bytes = BACKGROUND_MUSIC.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    st.components.v1.html(
        f"""
        <audio id="background_music" autoplay loop>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>

        <script>
        const bg = document.getElementById("background_music");

        if(bg){{
            bg.volume = 0.04;
            bg.play().catch(()=>{{}});
        }}
        </script>
        """,
        height=0,
    )

def build_narration(new_events: pd.DataFrame, intro=False) -> str:
    if intro or new_events.empty:
        return random.choice(INTRO_LINES)
    lines = []
    for _, e in new_events.sort_values("time_sec").tail(3).iterrows():
        desc = str(e.get("description_clean", "jugada detectada"))
        ev = str(e.get("event", "default"))
        templates = EVENT_TEMPLATES.get(ev, EVENT_TEMPLATES["default"])
        lines.append(random.choice(templates).format(desc=desc))
    if random.random() < 0.55:
        lines.append(random.choice(FILLERS))
    return " ".join(lines)

# ============================================================
# FUNCIONES DE DATOS
# ============================================================
def current_slice(t, window=8.0):
    current = tracks[tracks["time_sec"] <= t].copy()
    previous = tracks[(tracks["time_sec"] > max(0, t - window)) & (tracks["time_sec"] <= t)].copy()
    prev_previous = tracks[(tracks["time_sec"] > max(0, t - 2 * window)) & (tracks["time_sec"] <= max(0, t - window))].copy()
    current_events = events[events["time_sec"] <= t].copy()
    return current, current_events, previous, prev_previous

def only_players(df):
    if df.empty or "player_slot" not in df.columns:
        return df
    return df[df["player_slot"].notna()].copy()

def delta_html(value, decimals=1, suffix=""):
    try:
        v = float(value)
    except Exception:
        return "<span class='delta-flat'>—</span>"
    if abs(v) < 0.05:
        return "<span class='delta-flat'>→ 0</span>"
    if v > 0:
        return f"<span class='delta-up'>▲ {v:.{decimals}f}{suffix}</span>"
    return f"<span class='delta-down'>▼ {abs(v):.{decimals}f}{suffix}</span>"

def bar_html(value, max_value, color="#38bdf8"):
    try:
        pct = 0 if max_value <= 0 else max(0, min(100, float(value) / float(max_value) * 100))
    except Exception:
        pct = 0
    return f"<div class='bar-bg'><div class='bar-fill' style='width:{pct:.1f}%; background:linear-gradient(90deg,{color},#a855f7);'></div></div>"

def name_pill(label, team):
    color = TEAM_COLOR.get(team, "#94a3b8")
    return f"<span class='name-pill' style='background:{color}'>{html.escape(str(label))}</span>"

def team_pill(team):
    color = TEAM_COLOR.get(team, "#94a3b8")
    return f"<span class='team-pill'><span class='dot' style='background:{color}'></span>{html.escape(str(team))}</span>"

def html_metric_table(df, numeric_bar=None, max_values=None, empty_msg="Sin datos todavía"):
    if df is None or df.empty:
        return f"<div class='table-wrap'><table class='futbot-table'><tr><td>{empty_msg}</td></tr></table></div>"
    numeric_bar = numeric_bar or []
    max_values = max_values or {}
    data = df.copy()
    rows = []
    headers = "".join([f"<th>{html.escape(str(c))}</th>" for c in data.columns])
    for _, r in data.iterrows():
        tds = []
        team = r.get("Equipo", "Sin equipo")
        for c in data.columns:
            val = r[c]
            if c == "Robot":
                val_html = name_pill(val, team)
            elif c == "Equipo":
                val_html = team_pill(val)
            elif c.startswith("Δ"):
                val_html = delta_html(val, 1)
            elif c in numeric_bar:
                color = TEAM_COLOR.get(team, "#38bdf8")
                maxv = max_values.get(c, data[c].max() if c in data else 1)
                val_html = f"<b>{val}</b><br>{bar_html(val, maxv, color)}"
            else:
                val_html = html.escape(str(val))
            tds.append(f"<td>{val_html}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return "<div class='table-wrap'><table class='futbot-table'><thead><tr>" + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"

def compute_possession(current):
    robots = only_players(current[current["class_name"].astype(str).str.lower().eq("robot")].copy())
    ball = current[current["class_name"].astype(str).str.lower().eq("ball")].copy()
    rows = []
    for frame in sorted(ball["frame"].dropna().unique()):
        b = ball[ball["frame"] == frame]
        r = robots[robots["frame"] == frame]
        if b.empty or r.empty:
            continue
        bx, by = b.iloc[0][["x", "y"]]
        r = r.copy()
        r["dist_ball"] = np.sqrt((r["x"] - bx) ** 2 + (r["y"] - by) ** 2)
        nearest = r.sort_values("dist_ball").iloc[0]
        rows.append({"Equipo": nearest["team"], "Robot": nearest["robot_label"], "frame": frame})
    if not rows:
        return pd.DataFrame(columns=["Equipo", "Robot", "Tiempo control (s)", "% posesión"])
    pos = pd.DataFrame(rows)
    summary = pos.groupby(["Equipo", "Robot"]).size().reset_index(name="frames")
    summary["Tiempo control (s)"] = (summary["frames"] / FPS).round(1)
    summary["% posesión"] = (summary["frames"] / summary["frames"].sum() * 100).round(1)
    return summary[["Equipo", "Robot", "Tiempo control (s)", "% posesión"]].sort_values("% posesión", ascending=False)

def compute_team_possession(possession_df):
    if possession_df.empty:
        return pd.DataFrame(columns=["Equipo", "Tiempo control (s)", "% posesión"])
    team = possession_df.groupby("Equipo", as_index=False)["Tiempo control (s)"].sum()
    total = team["Tiempo control (s)"].sum()
    team["% posesión"] = (team["Tiempo control (s)"] / total * 100).round(1) if total else 0
    return team.sort_values("% posesión", ascending=False)

def compute_robot_metrics(current, prev_window=None):
    robots = only_players(current[current["class_name"].astype(str).str.lower().eq("robot")].copy())
    rows = []
    for _, player in enumerate(PLAYER_ORDER):
        raw_id = PLAYER_TO_ID.get(player["slot"])
        if raw_id is None:
            continue
        g = robots[robots["track_id"] == raw_id].sort_values("frame")
        if g.empty:
            rows.append({"Equipo": player["team"], "Robot": player["short"], "Distancia px": 0, "Vel. prom px/s": 0, "Vel. máx px/s": 0, "Frames": 0})
            continue
        dist = np.sqrt(g["x"].diff() ** 2 + g["y"].diff() ** 2).fillna(0)
        dt = g["time_sec"].diff().replace(0, np.nan)
        speed = dist / dt
        rows.append({
            "Equipo": player["team"],
            "Robot": player["short"],
            "Distancia px": round(float(dist.sum()), 1),
            "Vel. prom px/s": round(float(speed.mean()), 1) if not np.isnan(speed.mean()) else 0,
            "Vel. máx px/s": round(float(speed.max()), 1) if not np.isnan(speed.max()) else 0,
            "Frames": int(len(g)),
        })
    df = pd.DataFrame(rows)
    if prev_window is not None and not df.empty:
        prev = compute_robot_metrics(prev_window, None)
        for col in ["Distancia px", "Vel. prom px/s", "Vel. máx px/s", "Frames"]:
            prev_map = prev.set_index("Robot")[col].to_dict() if not prev.empty else {}
            df[f"Δ {col}"] = df.apply(lambda r: round(float(r[col]) - float(prev_map.get(r["Robot"], 0)), 1), axis=1)
    return df.sort_values("Distancia px", ascending=False)

def event_type_label(ev):
    labels = {
        "change_of_possession": "Cambio posesión", "interception": "Intercepción", "pass": "Pase",
        "possible_shot": "Tiro", "shot": "Tiro", "possible_collision": "Colisión", "collision": "Colisión",
        "goal": "Gol", "possession": "Posesión"
    }
    return labels.get(str(ev), str(ev))

def format_events(df, limit=12):
    if df.empty:
        return pd.DataFrame(columns=["Tiempo", "Tipo", "Descripción"])
    out = df[["time_sec", "event", "description_clean"]].tail(limit).copy()
    out["Tiempo"] = out["time_sec"].round(1).astype(str) + "s"
    out["Tipo"] = out["event"].apply(lambda x: f"<span class='event-chip'>{html.escape(event_type_label(x))}</span>")
    out["Descripción"] = out["description_clean"].apply(lambda x: html.escape(str(x)))
    return out[["Tiempo", "Tipo", "Descripción"]]

def commentary_html(current_events):
    if current_events.empty:
        return "<div class='commentary'>El sistema observa el partido. Aún no hay eventos destacados.</div>"
    lines = []
    for _, e in current_events.sort_values("time_sec").tail(10).iloc[::-1].iterrows():
        desc = html.escape(str(e.get("description_clean", "")))
        chip = event_type_label(e.get("event", ""))
        lines.append(f"<div><span class='event-chip'>{chip}</span> <b style='color:#38bdf8'>{float(e['time_sec']):.1f}s</b> — {desc}</div>")
    return "<div class='commentary'>" + "<br>".join(lines) + "</div>"

# ============================================================
# VISUALIZACIONES
# ============================================================
def field_limits(current):
    if current.empty or not {"x", "y"}.issubset(current.columns):
        return 0, 1160, 0, 740
    xs = pd.to_numeric(current["x"], errors="coerce").dropna()
    ys = pd.to_numeric(current["y"], errors="coerce").dropna()
    if xs.empty or ys.empty:
        return 0, 1160, 0, 740
    xmin, xmax = max(0, xs.quantile(.01) - 80), xs.quantile(.99) + 80
    ymin, ymax = max(0, ys.quantile(.01) - 80), ys.quantile(.99) + 80
    if xmax - xmin < 300: xmax = xmin + 300
    if ymax - ymin < 200: ymax = ymin + 200
    return float(xmin), float(xmax), float(ymin), float(ymax)

def draw_field(ax, current=None):
    xmin, xmax, ymin, ymax = field_limits(current if current is not None else tracks)
    ax.set_facecolor("#06251d")
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#0f7a4e", alpha=.45, zorder=0))
    ax.plot([xmin+40, xmax-40, xmax-40, xmin+40, xmin+40], [ymin+40, ymin+40, ymax-40, ymax-40, ymin+40], color="white", lw=1.5, alpha=.9)
    ax.plot([(xmin+xmax)/2, (xmin+xmax)/2], [ymin+40, ymax-40], color="white", lw=1.0, alpha=.75)
    ax.add_patch(plt.Circle(((xmin+xmax)/2, (ymin+ymax)/2), max(40, (xmax-xmin)*.07), fill=False, color="white", lw=1.0, alpha=.75))
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

def heatmap_image(current, class_name="robot"):
    sub = current[current["class_name"].astype(str).str.lower().eq(class_name)].copy()
    if class_name == "robot":
        sub = only_players(sub)
    fig, ax = plt.subplots(figsize=(7.2, 4.25), dpi=170)
    draw_field(ax, current)
    if not sub.empty:
        xmin, xmax, ymin, ymax = field_limits(current)
        x = pd.to_numeric(sub["x"], errors="coerce").dropna().to_numpy()
        y = pd.to_numeric(sub["y"], errors="coerce").dropna().to_numpy()
        if len(x) > 2 and len(y) > 2:
            H, xedges, yedges = np.histogram2d(x, y, bins=75, range=[[xmin, xmax], [ymin, ymax]])
            H = np.log1p(H.T)
            ax.imshow(H, extent=[xmin, xmax, ymax, ymin], cmap="turbo", alpha=.88, interpolation="bilinear", aspect="auto", zorder=2)
    ax.set_title("Concentración de movimiento", color="#e0f2fe", fontsize=10, pad=6, fontweight="bold")
    fig.patch.set_facecolor("#07111d")
    fig.tight_layout(pad=.4)
    return fig

def team_trails_image(current):
    robots = only_players(current[current["class_name"].astype(str).str.lower().eq("robot")].copy())
    fig, ax = plt.subplots(figsize=(7.2, 4.25), dpi=170)
    draw_field(ax, current)
    for slot, player in enumerate(PLAYER_ORDER, start=1):
        raw_id = PLAYER_TO_ID.get(player["slot"])
        if raw_id is None:
            continue
        g = robots[robots["track_id"] == raw_id].sort_values("frame").tail(320)
        if g.empty:
            continue
        ax.plot(g["x"], g["y"], lw=2.2, alpha=.92, color=player["color"], label=player["short"], zorder=3)
        last = g.iloc[-1]
        ax.scatter([last["x"]], [last["y"]], s=90, color=player["color"], edgecolor="white", linewidth=1.2, zorder=5)
        ax.text(last["x"], last["y"]-18, player["name"], color="white", fontsize=7, weight="bold", ha="center", zorder=6)
    ax.legend(fontsize=7, loc="upper right", facecolor="#020617", edgecolor="#1f3b5c", labelcolor="white")
    ax.set_title("Trayectorias recientes por jugador", color="#e0f2fe", fontsize=10, pad=6, fontweight="bold")
    fig.patch.set_facecolor("#07111d")
    fig.tight_layout(pad=.4)
    return fig

def live_snapshot_image(current):
    recent = current[current["time_sec"] >= max(0, current["time_sec"].max() - 1.5)] if not current.empty else current
    robots = only_players(recent[recent["class_name"].astype(str).str.lower().eq("robot")].copy())
    ball = recent[recent["class_name"].astype(str).str.lower().eq("ball")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.25), dpi=170)
    draw_field(ax, current)
    for player in PLAYER_ORDER:
        raw_id = PLAYER_TO_ID.get(player["slot"])
        if raw_id is None:
            continue
        g = robots[robots["track_id"] == raw_id].sort_values("frame")
        if g.empty:
            continue
        last = g.iloc[-1]
        ax.scatter([last["x"]], [last["y"]], s=170, color=player["color"], edgecolor="white", linewidth=1.6, zorder=5)
        ax.text(last["x"], last["y"]-24, player["short"], color="white", fontsize=8, weight="bold", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor=player["color"], alpha=.85))
    if not ball.empty:
        b = ball.sort_values("frame").iloc[-1]
        ax.scatter([b["x"]], [b["y"]], s=90, color="#facc15", edgecolor="white", linewidth=1.4, zorder=7)
        ax.text(b["x"], b["y"]+22, "Balón", color="#fde68a", fontsize=8, weight="bold", ha="center", zorder=8)
    ax.set_title("Posición actual en cancha", color="#e0f2fe", fontsize=10, pad=6, fontweight="bold")
    fig.patch.set_facecolor("#07111d")
    fig.tight_layout(pad=.4)
    return fig

def legend_html():
    items = []
    for p in PLAYER_ORDER:
        raw_id = PLAYER_TO_ID.get(p["slot"], "—")
        items.append(f"<div class='legend-item'><span class='dot' style='background:{p['color']}'></span><b>{p['short']}</b><br><span class='small-text'>{p['team']} · Track real: {raw_id}</span></div>")
    return "<div class='legend-card'>" + "".join(items) + "</div>"

# ============================================================
# ESTADO
# ============================================================
def init_state():
    defaults = {
        "t": 0.0, "playing": False, "live_started_at": None, "paused_at": 0.0,
        "last_narrated_time": -1.0, "audio_busy_until": 0.0, "intro_done": False,
        "last_narration_text": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("Controles")
    play_clicked = st.toggle("Reproducir análisis en vivo", value=st.session_state.playing)
    enable_voice = st.toggle("Narración Martinoli con ElevenLabs", value=True)
    refresh_ms = st.slider("Refresco de métricas", 1000, 6000, 2500, 500)
    narration_gap = st.slider("Separación mínima entre audios", 6, 18, 10, 1)
    st.markdown("---")
    st.markdown("### Robots del partido")
    st.markdown(legend_html(), unsafe_allow_html=True)
    if st.button("Reiniciar transmisión"):
        for k in ["t", "paused_at", "last_narrated_time", "audio_busy_until"]:
            st.session_state[k] = 0.0 if k != "last_narrated_time" else -1.0
        st.session_state.live_started_at = None
        st.session_state.intro_done = False
        st.session_state.last_narration_text = ""
        st.rerun()

now = time.time()
if play_clicked and not st.session_state.playing:
    st.session_state.live_started_at = now - st.session_state.paused_at
if not play_clicked and st.session_state.playing:
    st.session_state.paused_at = st.session_state.t
st.session_state.playing = play_clicked

if st.session_state.playing:
    if st.session_state.live_started_at is None:
        st.session_state.live_started_at = now - st.session_state.t
    st.session_state.t = min(MAX_TIME, max(0.0, now - st.session_state.live_started_at))
else:
    st.session_state.t = st.session_state.paused_at

# ============================================================
# HEADER Y VIDEO
# ============================================================
st.markdown(f"""
<div class="header">
    <div><h1>🤖 {APP_TITLE}</h1><p><span class="live-dot"></span>{SUBTITLE}</p></div>
    <div class="meta"><span>∞</span> Meta</div>
</div>
""", unsafe_allow_html=True)

if "background_music_started" not in st.session_state:
    reproducir_musica_fondo()
    st.session_state.background_music_started = True

top_left, top_right = st.columns([1.78, 1.22], gap="medium")
with top_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">🎥 Partido analizado — {VIDEO_NAME}</div>', unsafe_allow_html=True)
    if VIDEO_PATH.exists():
        st.video(str(VIDEO_PATH))
        st.caption("Dale play al video y activa la transmisión en vivo para que las métricas avancen sin reiniciar el reproductor.")
    else:
        st.error(f"No encuentro el video: {VIDEO_PATH}")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FRAGMENTOS DINÁMICOS
# ============================================================
def maybe_fragment(run_every=None):
    if hasattr(st, "fragment"):
        return st.fragment(run_every=run_every)
    def deco(fn):
        return fn
    return deco

@maybe_fragment(run_every="2.5s")
def live_top_panel():
    if st_autorefresh and st.session_state.playing:
        st_autorefresh(interval=refresh_ms, key="top_refresh")
    current, current_events, window_now, window_prev = current_slice(st.session_state.t)
    players_current = only_players(current[current["class_name"].astype(str).str.lower().eq("robot")].copy())
    players_now = only_players(window_now[window_now["class_name"].astype(str).str.lower().eq("robot")].copy())
    players_prev = only_players(window_prev[window_prev["class_name"].astype(str).str.lower().eq("robot")].copy())

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⏱ Control de tiempo</div>', unsafe_allow_html=True)
    t = st.slider("Tiempo del partido", 0.0, MAX_TIME, float(st.session_state.t), step=0.5, key="time_slider")
    if not st.session_state.playing and abs(t - st.session_state.t) > 0.01:
        st.session_state.t = t
        st.session_state.paused_at = t
        current, current_events, window_now, window_prev = current_slice(t)

    active_players = players_current["player_slot"].nunique() if not players_current.empty else 0
    event_now = len(current_events)
    shot_count = len(current_events[current_events["event"].isin(["possible_shot", "shot"])]) if "event" in current_events.columns else 0
    collision_count = len(current_events[current_events["event"].isin(["possible_collision", "collision"])]) if "event" in current_events.columns else 0
    speed_now = 0
    speed_prev = 0
    for df, name in [(players_now, "now"), (players_prev, "prev")]:
        if not df.empty:
            g = df.sort_values(["track_id", "frame"])
            dist = np.sqrt(g.groupby("track_id")["x"].diff() ** 2 + g.groupby("track_id")["y"].diff() ** 2)
            dt = g.groupby("track_id")["time_sec"].diff().replace(0, np.nan)
            sp = (dist / dt).replace([np.inf, -np.inf], np.nan).dropna()
            if name == "now": speed_now = float(sp.mean()) if not sp.empty else 0
            else: speed_prev = float(sp.mean()) if not sp.empty else 0

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="metric-number">{st.session_state.t:.1f}s</div><div class="metric-label">Tiempo analizado</div><div class="metric-delta">de {MAX_TIME:.1f}s</div></div>
        <div class="metric-card"><div class="metric-number">{active_players}/4</div><div class="metric-label">Robots del partido</div><div class="metric-delta">nombres normalizados</div></div>
        <div class="metric-card"><div class="metric-number">{event_now}</div><div class="metric-label">Eventos detectados</div><div class="metric-delta">Tiros: {shot_count} · Choques: {collision_count}</div></div>
        <div class="metric-card"><div class="metric-number">{speed_now:.1f}</div><div class="metric-label">Velocidad media reciente</div><div class="metric-delta">{delta_html(speed_now-speed_prev, 1, ' px/s')}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎙 Narrativa automática</div>', unsafe_allow_html=True)
    st.markdown(commentary_html(current_events), unsafe_allow_html=True)

    now_audio = time.time()
    should_intro = enable_voice and not st.session_state.intro_done and st.session_state.t < 4.0
    new_events = current_events[current_events["time_sec"] > st.session_state.last_narrated_time] if not current_events.empty else pd.DataFrame()
    should_event_audio = enable_voice and st.session_state.playing and not new_events.empty and now_audio >= st.session_state.audio_busy_until and (float(new_events["time_sec"].max()) - max(0, st.session_state.last_narrated_time)) >= 2

    if enable_voice and now_audio < st.session_state.audio_busy_until:
        st.markdown("<div class='audio-status'>🔊 Narración en curso; esperando para no encimar audios.</div>", unsafe_allow_html=True)

    if enable_voice and now_audio >= st.session_state.audio_busy_until and (should_intro or should_event_audio):
        texto = build_narration(new_events, intro=should_intro)
        if texto != st.session_state.last_narration_text:
            event_key = f"intro_{st.session_state.t:.1f}" if should_intro else f"chunk_{float(new_events['time_sec'].max()):.1f}_{len(new_events)}"
            with st.spinner("Generando narración estilo Martinoli..."):
                audio_path = generar_audio_martinoli(texto, event_key)
            if audio_path:
                reproducir_audio_autoplay(audio_path)
                st.session_state.audio_busy_until = now_audio + estimate_audio_seconds(texto) + narration_gap
                st.session_state.last_narration_text = texto
                if should_intro:
                    st.session_state.intro_done = True
                elif not new_events.empty:
                    st.session_state.last_narrated_time = float(new_events["time_sec"].max())
                st.markdown(f"<div class='audio-status'><b>Última narración:</b> {html.escape(texto)}</div>", unsafe_allow_html=True)
            else:
                st.warning("No se generó audio. Revisa ELEVENLABS_API_KEY en .env e instala elevenlabs.")
    st.markdown('</div>', unsafe_allow_html=True)

@maybe_fragment(run_every="3.5s")
def live_lower_dashboard():
    current, current_events, window_now, window_prev = current_slice(st.session_state.t)
    possession = compute_possession(current)
    team_possession = compute_team_possession(possession)
    robot_metrics = compute_robot_metrics(current, window_prev)

    st.markdown('<div class="section-title">🟢 Visualización táctica del partido</div>', unsafe_allow_html=True)
    v0, v1 = st.columns([1, 1], gap="medium")
    with v0:
        st.markdown('<div class="panel"><div class="panel-title">📍 Posición actual de jugadores y balón</div>', unsafe_allow_html=True)
        fig = live_snapshot_image(current); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with v1:
        st.markdown('<div class="panel"><div class="panel-title">🧭 Trayectorias recientes por jugador</div>', unsafe_allow_html=True)
        fig = team_trails_image(current); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Métricas dinámicas</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns([.9, 1.05, 1.45], gap="medium")
    with m1:
        st.markdown('<div class="panel"><div class="panel-title">🏆 Posesión por equipo</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(team_possession, numeric_bar=["% posesión", "Tiempo control (s)"], max_values={"% posesión":100, "Tiempo control (s)": max(1, team_possession["Tiempo control (s)"].max() if not team_possession.empty else 1)}), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="panel"><div class="panel-title">🤖 Posesión por robot</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(possession, numeric_bar=["% posesión", "Tiempo control (s)"], max_values={"% posesión":100, "Tiempo control (s)": max(1, possession["Tiempo control (s)"].max() if not possession.empty else 1)}), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="panel"><div class="panel-title">⚙️ Métricas avanzadas por robot</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(robot_metrics, numeric_bar=["Distancia px", "Vel. prom px/s", "Vel. máx px/s"], max_values={
            "Distancia px": max(1, robot_metrics["Distancia px"].max() if not robot_metrics.empty else 1),
            "Vel. prom px/s": max(1, robot_metrics["Vel. prom px/s"].max() if not robot_metrics.empty else 1),
            "Vel. máx px/s": max(1, robot_metrics["Vel. máx px/s"].max() if not robot_metrics.empty else 1),
        }), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔥 Mapas de calor mejorados</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2, gap="medium")
    with h1:
        st.markdown('<div class="panel"><div class="panel-title">🔥 Heatmap dinámico — Robots principales</div>', unsafe_allow_html=True)
        fig = heatmap_image(current, "robot"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="panel"><div class="panel-title">🟠 Heatmap dinámico — Balón</div>', unsafe_allow_html=True)
        fig = heatmap_image(current, "ball"); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏁 Eventos detectados</div>', unsafe_allow_html=True)
    passes = format_events(current_events[current_events["event"].isin(["change_of_possession", "interception", "pass"])]) if "event" in current_events.columns else pd.DataFrame()
    shots = format_events(current_events[current_events["event"].isin(["possible_shot", "shot", "goal"])]) if "event" in current_events.columns else pd.DataFrame()
    collisions = format_events(current_events[current_events["event"].isin(["possible_collision", "collision"])]) if "event" in current_events.columns else pd.DataFrame()
    e1, e2, e3 = st.columns(3, gap="medium")
    with e1:
        st.markdown('<div class="panel"><div class="panel-title">🔁 Pases / cambios / intercepciones</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(passes), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="panel"><div class="panel-title">🥅 Tiros y goles</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(shots), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with e3:
        st.markdown('<div class="panel"><div class="panel-title">💥 Colisiones posibles</div>', unsafe_allow_html=True)
        st.markdown(html_metric_table(collisions), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with top_right:
    live_top_panel()

live_lower_dashboard()

st.markdown("""
<div class="panel"><div class="panel-title">🧠 Pipeline de IA</div>
<div class="small-text">Video V0 → YOLO11-seg → SAM3 mask refinement → ByteTrack → Normalización de identidades → Métricas temporales → Visualización táctica → Narración dinámica → ElevenLabs</div></div>
""", unsafe_allow_html=True)
