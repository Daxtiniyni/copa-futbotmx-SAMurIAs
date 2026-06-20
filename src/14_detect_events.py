import pandas as pd
import numpy as np
from pathlib import Path
from project_paths import OUTPUTS_DIR

# =========================
# RUTAS
# =========================
CSV_PATH = OUTPUTS_DIR / "sam3_pipeline_v0" / "V0_yolo_sam3_tracks.csv"

OUT_DIR = OUTPUTS_DIR / "events"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EVENTS = OUT_DIR / "V0_events_full.csv"
OUT_JSON = OUT_DIR / "V0_events_full.json"

# =========================
# CONFIGURACIÓN
# =========================
FPS = 30

POSSESSION_DISTANCE = 130
COLLISION_DISTANCE = 105
SHOT_SPEED_THRESHOLD = 350
GOAL_DISTANCE = 180
PASS_MIN_TIME_GAP = 0.15
EVENT_COOLDOWN_SEC = 0.25

# Cambia estos IDs si tus robots principales son otros
ROBOT_ALIAS = {
    1: "Robotiño",
    2: "Messibot",
    3: "Cristiano Ronabot",
    4: "Botbappé",
}

ROBOT_TEAM = {
    1: "Equipo Azul",
    2: "Equipo Azul",
    3: "Equipo Rojo",
    4: "Equipo Rojo",
}


def robot_name(robot_id):
    return ROBOT_ALIAS.get(int(robot_id), f"Robot {int(robot_id)}")


def robot_team(robot_id):
    return ROBOT_TEAM.get(int(robot_id), "Sin equipo")


def add_event(events, time_sec, event, description, **kwargs):
    events.append({
        "time_sec": round(float(time_sec), 2),
        "event": event,
        "description": description,
        **kwargs
    })


def cooldown_ok(last_time, current_time, cooldown=EVENT_COOLDOWN_SEC):
    return last_time is None or (current_time - last_time) >= cooldown


# =========================
# CARGA
# =========================
print("Leyendo CSV...")
df = pd.read_csv(CSV_PATH)
print("CSV cargado:", df.shape)
print(df["class_name"].value_counts())

robots = df[df["class_name"] == "robot"].copy()
ball = df[df["class_name"] == "ball"].copy()
goals = df[df["class_name"] == "goal"].copy()

robots = robots.sort_values(["frame", "track_id"])
ball = ball.sort_values("frame")
goals = goals.sort_values(["frame", "track_id"])

events = []

# =========================
# 1. POSESIÓN FRAME A FRAME
# =========================
print("Calculando posesión...")

possessions = []

for frame in sorted(ball["frame"].unique()):
    b = ball[ball["frame"] == frame]
    r = robots[robots["frame"] == frame]

    if b.empty or r.empty:
        continue

    bx, by = b.iloc[0][["x", "y"]]

    r = r.copy()
    r["dist_ball"] = np.sqrt((r["x"] - bx) ** 2 + (r["y"] - by) ** 2)
    nearest = r.sort_values("dist_ball").iloc[0]

    if nearest["dist_ball"] <= POSSESSION_DISTANCE:
        rid = int(nearest["track_id"])

        possessions.append({
            "frame": int(frame),
            "time_sec": float(b.iloc[0]["time_sec"]),
            "robot_id": rid,
            "robot_name": robot_name(rid),
            "team": robot_team(rid),
            "dist_ball": float(nearest["dist_ball"]),
        })

pos_df = pd.DataFrame(possessions)

# =========================
# 2. POSESIÓN Y PASES
# =========================
print("Detectando posesiones y pases...")

if not pos_df.empty:
    current_robot = None
    current_start = None
    last_change_time = None

    for _, row in pos_df.iterrows():
        rid = int(row["robot_id"])
        t = float(row["time_sec"])

        if current_robot is None:
            current_robot = rid
            current_start = t
            continue

        if rid != current_robot:
            duration = t - current_start

            if duration >= 0.15:
                add_event(
                    events,
                    current_start,
                    "possession",
                    f"{robot_name(current_robot)} del {robot_team(current_robot)} controla el balón durante {duration:.1f} segundos.",
                    robot_id=current_robot,
                    robot_name=robot_name(current_robot),
                    team=robot_team(current_robot),
                    duration_sec=round(duration, 2),
                )

            if cooldown_ok(last_change_time, t, PASS_MIN_TIME_GAP):
                same_team = robot_team(current_robot) == robot_team(rid)

                if same_team:
                    event_name = "pass"
                    desc = (
                        f"Pase detectado: {robot_name(current_robot)} toca para "
                        f"{robot_name(rid)} del {robot_team(rid)}."
                    )
                else:
                    event_name = "interception"
                    desc = (
                        f"Intercepción detectada: {robot_name(rid)} del {robot_team(rid)} "
                        f"le roba la posesión a {robot_name(current_robot)}."
                    )

                add_event(
                    events,
                    t,
                    event_name,
                    desc,
                    from_robot=current_robot,
                    from_robot_name=robot_name(current_robot),
                    to_robot=rid,
                    to_robot_name=robot_name(rid),
                    from_team=robot_team(current_robot),
                    to_team=robot_team(rid),
                )

                last_change_time = t

            current_robot = rid
            current_start = t

    # Última posesión
    if current_robot is not None and current_start is not None:
        last_time = float(pos_df["time_sec"].max())
        duration = last_time - current_start

        if duration >= 0.15:
            add_event(
                events,
                current_start,
                "possession",
                f"{robot_name(current_robot)} del {robot_team(current_robot)} mantiene la posesión durante {duration:.1f} segundos.",
                robot_id=current_robot,
                robot_name=robot_name(current_robot),
                team=robot_team(current_robot),
                duration_sec=round(duration, 2),
            )

# =========================
# 3. VELOCIDAD DEL BALÓN Y TIROS
# =========================
print("Detectando tiros...")

ball = ball.sort_values("frame").copy()
ball["dx"] = ball["x"].diff()
ball["dy"] = ball["y"].diff()
ball["dt"] = ball["time_sec"].diff()
ball["speed"] = np.sqrt(ball["dx"] ** 2 + ball["dy"] ** 2) / ball["dt"].replace(0, np.nan)

last_shot_time = None

for _, row in ball.iterrows():
    if pd.isna(row["speed"]):
        continue

    t = float(row["time_sec"])

    if row["speed"] >= SHOT_SPEED_THRESHOLD and cooldown_ok(last_shot_time, t, 2.0):
        # Buscar robot en posesión más cercano en ese tiempo
        if not pos_df.empty:
            near_pos = pos_df.iloc[(pos_df["time_sec"] - t).abs().argsort()[:1]]
            shooter_id = int(near_pos.iloc[0]["robot_id"])
            shooter = robot_name(shooter_id)
            team = robot_team(shooter_id)
        else:
            shooter_id = -1
            shooter = "un robot no identificado"
            team = "Sin equipo"

        add_event(
            events,
            t,
            "shot",
            f"Tiro detectado: {shooter} del {team} impulsa el balón a alta velocidad.",
            robot_id=shooter_id,
            robot_name=shooter,
            team=team,
            ball_speed_px_s=round(float(row["speed"]), 2),
        )

        last_shot_time = t

# =========================
# 4. GOLES
# =========================
print("Detectando goles...")

last_goal_time = None

# Si hay porterías detectadas, usamos distancia balón-portería.
# Si una portería aparece varias veces, usamos las detecciones por frame.
for frame in sorted(ball["frame"].unique()):
    b = ball[ball["frame"] == frame]
    g = goals[goals["frame"] == frame]

    if b.empty or g.empty:
        continue

    bx, by = b.iloc[0][["x", "y"]]
    t = float(b.iloc[0]["time_sec"])

    g = g.copy()
    g["dist_ball_goal"] = np.sqrt((g["x"] - bx) ** 2 + (g["y"] - by) ** 2)
    nearest_goal = g.sort_values("dist_ball_goal").iloc[0]

    if nearest_goal["dist_ball_goal"] <= GOAL_DISTANCE and cooldown_ok(last_goal_time, t, 5.0):
        if not pos_df.empty:
            near_pos = pos_df.iloc[(pos_df["time_sec"] - t).abs().argsort()[:1]]
            scorer_id = int(near_pos.iloc[0]["robot_id"])
            scorer = robot_name(scorer_id)
            team = robot_team(scorer_id)
        else:
            scorer_id = -1
            scorer = "un robot no identificado"
            team = "Sin equipo"

        add_event(
            events,
            t,
            "goal",
            f"Gol detectado: {scorer} del {team} manda el balón a zona de portería.",
            robot_id=scorer_id,
            robot_name=scorer,
            team=team,
            goal_distance=round(float(nearest_goal["dist_ball_goal"]), 2),
        )

        last_goal_time = t

# =========================
# 5. COLISIONES
# =========================
print("Detectando colisiones...")

last_collision_by_pair = {}

for frame in sorted(robots["frame"].unique()):
    r = robots[robots["frame"] == frame]

    if len(r) < 2:
        continue

    rows = r.to_dict("records")
    t = float(rows[0]["time_sec"])

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            r1 = rows[i]
            r2 = rows[j]

            id1 = int(r1["track_id"])
            id2 = int(r2["track_id"])

            pair = tuple(sorted([id1, id2]))

            dist = np.sqrt((r1["x"] - r2["x"]) ** 2 + (r1["y"] - r2["y"]) ** 2)

            if dist <= COLLISION_DISTANCE:
                last_t = last_collision_by_pair.get(pair)

                if cooldown_ok(last_t, t, 3.0):
                    add_event(
                        events,
                        t,
                        "collision",
                        f"Posible colisión entre {robot_name(id1)} y {robot_name(id2)}. Distancia mínima aproximada: {dist:.1f} px.",
                        robot_1=id1,
                        robot_1_name=robot_name(id1),
                        robot_2=id2,
                        robot_2_name=robot_name(id2),
                        distance=round(float(dist), 2),
                    )

                    last_collision_by_pair[pair] = t

# =========================
# 6. GUARDAR
# =========================
events_df = pd.DataFrame(events)

if not events_df.empty:
    events_df = events_df.sort_values("time_sec").reset_index(drop=True)

    # Eliminar duplicados exactos por seguridad
    events_df = events_df.drop_duplicates(subset=["time_sec", "event", "description"])

events_df.to_csv(OUT_EVENTS, index=False)
events_df.to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)

print("Eventos guardados en:")
print(OUT_EVENTS)
print(OUT_JSON)
print()
print("Conteo de eventos:")
if not events_df.empty:
    print(events_df["event"].value_counts())
    print()
    print(events_df.head(30))
else:
    print("No se detectaron eventos.")
