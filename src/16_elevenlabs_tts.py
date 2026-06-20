import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from project_paths import OUTPUTS_DIR

# ============================================================
# 16_elevenlabs_tts.py
# Convierte cada segmento de narración en un MP3 independiente
# ============================================================

load_dotenv()

# Recomendado: guarda tu API KEY en .env:
# ELEVENLABS_API_KEY=tu_api_key
# ELEVENLABS_VOICE_ID=tu_voice_id
API_KEY = os.getenv("ELEVENLABS_API_KEY", "PEGA_AQUI_TU_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

MANIFEST_CSV = OUTPUTS_DIR / "narration" / "V0_commentary_segments.csv"
AUDIO_DIR = OUTPUTS_DIR / "narration" / "audio_segments"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

OUT_MANIFEST_CSV = OUTPUTS_DIR / "narration" / "V0_commentary_segments_with_audio.csv"

MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"

# Pausa pequeña para evitar saturar la API.
PAUSE_BETWEEN_REQUESTS = 0.35


def clean_text(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def main():
    if API_KEY == "PEGA_AQUI_TU_API_KEY":
        raise ValueError(
            "Falta tu API Key. Ponla en el archivo .env como ELEVENLABS_API_KEY=tu_api_key "
            "o reemplaza PEGA_AQUI_TU_API_KEY en este script."
        )

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            "No encontré V0_commentary_segments.csv. Primero ejecuta 15_generate_commentary.py"
        )

    client = ElevenLabs(api_key=API_KEY)
    df = pd.read_csv(MANIFEST_CSV)

    generated_paths = []

    for _, row in df.iterrows():
        segment_id = int(row["segment_id"])
        text = clean_text(row["text"])
        out_audio = AUDIO_DIR / f"commentary_{segment_id:03d}.mp3"

        if out_audio.exists() and out_audio.stat().st_size > 0:
            print(f"Ya existe, se conserva: {out_audio}")
            generated_paths.append(str(out_audio))
            continue

        print(f"Generando audio {segment_id:03d}...")
        print(text)

        audio = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            text=text,
            output_format=OUTPUT_FORMAT,
        )

        with open(out_audio, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        generated_paths.append(str(out_audio))
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    df["audio_path"] = generated_paths
    df.to_csv(OUT_MANIFEST_CSV, index=False, encoding="utf-8-sig")

    print("Audios generados correctamente.")
    print("Manifest actualizado:", OUT_MANIFEST_CSV)
    print("Carpeta de audios:", AUDIO_DIR)


if __name__ == "__main__":
    main()
