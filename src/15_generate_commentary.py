import json
import random
from pathlib import Path

import pandas as pd
from project_paths import OUTPUTS_DIR

# ============================================================
# 15_generate_commentary.py
# Genera comentarios cortos sincronizados para un video de 12:56
# ============================================================

# Cambia esta ruta si tu archivo tiene otro nombre.
EVENTS_CSV = OUTPUTS_DIR / "events" / "V0_events_full.csv"

# Si no existe V0_events_full.csv, intenta con V0_events.csv.
FALLBACK_EVENTS_CSV = OUTPUTS_DIR / "events" / "V0_events.csv"

OUT_DIR = OUTPUTS_DIR / "narration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST_CSV = OUT_DIR / "V0_commentary_segments.csv"
OUT_MANIFEST_JSON = OUT_DIR / "V0_commentary_segments.json"
OUT_TXT = OUT_DIR / "V0_commentary_full_text.txt"

# Duración del video: 12 minutos con 56 segundos
VIDEO_DURATION_SEC = 12 * 60 + 56

# Cada cuánto queremos insertar una intervención del narrador.
# 15 segundos da una narración constante sin saturar demasiado.
BLOCK_SECONDS = 15

# Para que el resultado sea reproducible.
random.seed(7)


INTRO_LINES = [
    "¡Arranca la final internacional de fútbol robótico! Robotiño y Messibot van por el Equipo Azul; Cristiano Ronabot y Botbappé defienden al Equipo Rojo. Doctor García, esto huele a aceite quemado y gloria metálica.",
    "¡Se mueve la pelota en esta fábrica de emociones! Los sensores están despiertos, las baterías al cien y los tornillos rezando para no salir volando.",
]

IDLE_LINES = [
    "Hay pausa en la cancha, Doctor García. Mucho algoritmo pensando y poco pistón atacando. Esto parece junta de ingeniería con balón incluido.",
    "El partido respira tantito. Los robots acomodan sensores, revisan firmware y fingen que todo estaba planeado. Ajá, cómo no.",
    "Momento de cálculo puro. Messibot mira, Cristiano Ronabot espera, y Botbappé parece que está descargando una actualización por internet de rancho.",
    "No pasa mucho, pero se siente la tensión. Hay más procesamiento aquí que en una tesis de doctorado entregada a las tres de la mañana.",
    "Los equipos se estudian. Robotiño calibra la ruta, Botbappé huele el peligro y algún mecánico ya está escondiendo las refacciones.",
    "Doctor García, esto está trabado. Los servomotores giran, las baterías sufren y la inteligencia artificial está buscando señal.",
]

POSSESSION_LINES = [
    "{player} trae la pelota pegada al chasis. La cuida como si fuera la última batería del laboratorio.",
    "{player} controla y piensa. Doctor García, ese robot está procesando más que computadora vieja abriendo veinte pestañas.",
    "{player} administra el balón con calma mecánica. Parece que le pusieron aceite premium y firmware de campeón.",
    "{player} se adueña de la jugada. Nadie se la quita; trae los sensores más finos que báscula de nutriólogo.",
]

PASS_LINES = [
    "Buen toque de {from_player} para {to_player}. Circula la pelota, circula el aceite, y el Equipo Rojo empieza a sacar chispas.",
    "La mueve {from_player}, recibe {to_player}. Eso fue pase filtrado con algoritmo de laboratorio y poquito descaro mecánico.",
    "Toca {from_player} para {to_player}. Doctor García, esa triangulación viene con garantía extendida y olor a refacción nueva.",
    "{from_player} encuentra a {to_player}. La pelota viaja como dato en fibra óptica: rápida, precisa y con ganas de hacer daño.",
]

SHOT_LINES = [
    "¡Le pega {player}! El balón sale con turbo, nitro y overclocking. ¡Cuidado, que eso no fue tiro, fue misil con servomotor!",
    "¡Disparo de {player}! La pelota va caliente, Doctor García. Si entra, funden los sensores del estadio.",
    "¡Zapatazo robótico de {player}! Ese balón trae más velocidad que actualización obligatoria cuando tienes prisa.",
    "¡Tiro peligroso! {player} acaba de mandar la pelota con tanta fuerza que el portero ya pidió seguro de gastos mecánicos.",
]

GOAL_LINES = [
    "¡Gooooooool! ¡Gooooooool de {player}! ¡Hijo de su robotina madre! La mandó a guardar con pistones, alma y batería al cien. Doctor García, eso no fue definición, fue una obra de ingeniería emocional.",
    "¡Gooooool! ¡Explota la cancha! {player} mete el balón y manda al portero directo al taller. ¡La inteligencia artificial acaba de escribir poesía con aceite!",
    "¡Golazo metálico! {player} prende los motores, rompe el firmware defensivo y la clava como final de Copa del Mundo en fábrica descontrolada.",
    "¡Gooooooool! {player} hizo temblar las tuercas del estadio. Doctor García, alguien revise las redes porque quedaron oliendo a cortocircuito.",
]

COLLISION_LINES = [
    "¡Choque durísimo! Dos robots se encontraron con las balatas por delante. Doctor García, eso sonó a licuadora cayéndose por las escaleras.",
    "¡Contacto fuerte! Ahí volaron tornillos imaginarios. A ese robot le urge taller, alineación y terapia de sensores.",
    "¡Se dieron con todo! Parece que uno venía con Windows Vista instalado a media jugada. La inteligencia artificial abandonó el chat.",
    "¡Tremendo golpe mecánico! Eso no fue falta, fue accidente industrial con balón. Que alguien traiga aceite y cinta de aislar.",
    "¡Cortocircuito emocional en la cancha! Entraron como maquinaria pesada en oferta de refacciones.",
]

INTERCEPTION_LINES = [
    "¡Robo de pelota! {to_player} le lee el algoritmo a {from_player} y le apaga la jugada como foco viejo.",
    "¡Intercepción! {to_player} aparece como técnico de taller: rápido, preciso y cobrando caro.",
    "Se la quitan a {from_player}. Doctor García, ahí fallaron los sensores o le dio ansiedad al firmware.",
    "¡Cambio de dueño! {to_player} mete la pinza, roba la pelota y deja a {from_player} buscando refacciones.",
]

CLOSING_LINES = [
    "Se nos termina esta locura robótica. Hubo aceite, golpes, algoritmos confundidos y jugadas que van directo al museo de las refacciones. Doctor García, qué final acabamos de vivir.",
    "Últimos segundos. Los motores bajan temperatura, las baterías piden descanso y la afición todavía no entiende si vio fútbol, Fórmula 1 o una fábrica peleándose con una pelota.",
]


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def robot_to_player(robot_id, robot_name=None, team=None):
    """Convierte IDs genéricos en nombres narrativos."""
    if robot_name and isinstance(robot_name, str):
        clean_name = robot_name.strip()
        if clean_name in ["Robotiño", "Messibot", "Cristiano Ronabot", "Botbappé"]:
            return clean_name

    rid = safe_int(robot_id, 0) or 0
    team = str(team or "").lower()

    if "azul" in team:
        return "Robotiño" if rid % 2 else "Messibot"

    if "rojo" in team:
        return "Cristiano Ronabot" if rid % 2 else "Botbappé"

    # Muchos eventos vienen como "Sin equipo". Para la narración los repartimos.
    players = ["Cristiano Ronabot", "Botbappé", "Messibot", "Robotiño"]
    return players[rid % len(players)]


def event_to_sentence(event_row):
    event = str(event_row.get("event", "")).strip().lower()

    robot_id = event_row.get("robot_id")
    robot_name = event_row.get("robot_name")
    team = event_row.get("team")

    player = robot_to_player(robot_id, robot_name, team)

    from_player = robot_to_player(
        event_row.get("from_robot"),
        event_row.get("from_robot_name"),
        event_row.get("from_team"),
    )
    to_player = robot_to_player(
        event_row.get("to_robot"),
        event_row.get("to_robot_name"),
        event_row.get("to_team"),
    )

    if event == "goal":
        return random.choice(GOAL_LINES).format(player=player)

    if event == "shot":
        return random.choice(SHOT_LINES).format(player=player)

    if event == "pass":
        return random.choice(PASS_LINES).format(from_player=from_player, to_player=to_player)

    if event == "possession":
        duration = safe_float(event_row.get("duration_sec"), 0)
        base = random.choice(POSSESSION_LINES).format(player=player)
        if duration >= 20:
            base += " La está durmiendo tanto que el balón ya pidió almohada y cargador."
        return base

    if event == "interception":
        return random.choice(INTERCEPTION_LINES).format(from_player=from_player, to_player=to_player)

    if event == "collision":
        return random.choice(COLLISION_LINES)

    # Compatibilidad con nombres viejos de eventos
    if event == "possible_collision":
        return random.choice(COLLISION_LINES)

    if event == "possible_shot":
        return random.choice(SHOT_LINES).format(player=player)

    if event == "change_of_possession":
        return random.choice(INTERCEPTION_LINES).format(from_player=from_player, to_player=to_player)

    return random.choice(IDLE_LINES)


def build_block_commentary(block_events, block_start, block_end):
    """Genera un comentario corto para el bloque."""
    if block_start == 0:
        return random.choice(INTRO_LINES)

    if block_start >= VIDEO_DURATION_SEC - 30:
        return random.choice(CLOSING_LINES)

    if block_events.empty:
        return random.choice(IDLE_LINES)

    # Prioridad: gol > tiro > pase/intercepción > posesión > choque
    priority = {
        "goal": 1,
        "shot": 2,
        "possible_shot": 2,
        "pass": 3,
        "interception": 3,
        "change_of_possession": 3,
        "possession": 4,
        "collision": 5,
        "possible_collision": 5,
    }

    temp = block_events.copy()
    temp["priority"] = temp["event"].astype(str).str.lower().map(priority).fillna(9)
    temp = temp.sort_values(["priority", "time_sec"])

    selected = temp.head(2)
    sentences = [event_to_sentence(row) for _, row in selected.iterrows()]

    # Agrega ocasionalmente una frase del Doctor García o reacción extra.
    extras = [
        "Doctor García, esto ya parece final de mundo y revisión de taller al mismo tiempo.",
        "La banca pide calma, pero los servomotores ya están en modo telenovela.",
        "Aquí nadie vino a ahorrar batería; vinieron a romper el firmware del rival.",
        "Qué manera de sufrir, señoras y señores. Esto es fútbol, robótica y comedia industrial.",
    ]

    if random.random() < 0.35:
        sentences.append(random.choice(extras))

    return " ".join(sentences)


def main():
    input_csv = EVENTS_CSV
    if not input_csv.exists() and FALLBACK_EVENTS_CSV.exists():
        input_csv = FALLBACK_EVENTS_CSV

    if not input_csv.exists():
        raise FileNotFoundError(
            "No encontré el CSV de eventos. Revisa EVENTS_CSV o FALLBACK_EVENTS_CSV."
        )

    events = pd.read_csv(input_csv)
    events["time_sec"] = pd.to_numeric(events["time_sec"], errors="coerce")
    events = events.dropna(subset=["time_sec"]).sort_values("time_sec")

    segments = []
    segment_id = 0

    for block_start in range(0, VIDEO_DURATION_SEC, BLOCK_SECONDS):
        block_end = min(block_start + BLOCK_SECONDS, VIDEO_DURATION_SEC)

        block_events = events[
            (events["time_sec"] >= block_start)
            & (events["time_sec"] < block_end)
        ]

        text = build_block_commentary(block_events, block_start, block_end)

        # No queremos audios pegados exactamente uno tras otro.
        # Arrancan 1 segundo después del inicio del bloque.
        start_sec = block_start + 1

        segments.append(
            {
                "segment_id": segment_id,
                "start_sec": start_sec,
                "end_sec": block_end,
                "audio_path": str(OUT_DIR / "audio_segments" / f"commentary_{segment_id:03d}.mp3"),
                "text": text,
            }
        )
        segment_id += 1

    df = pd.DataFrame(segments)
    df.to_csv(OUT_MANIFEST_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['start_sec']:06.2f}s] {seg['text']}\n\n")

    print("Narración segmentada generada correctamente.")
    print("CSV:", OUT_MANIFEST_CSV)
    print("JSON:", OUT_MANIFEST_JSON)
    print("TXT:", OUT_TXT)
    print("Segmentos generados:", len(segments))


if __name__ == "__main__":
    main()
