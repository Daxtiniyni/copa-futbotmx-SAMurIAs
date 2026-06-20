# vozMartinoli.py

import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "12tdVBKVpQ2NJSeVcpWR")


def generar_audio_martinoli(texto, output_path="audio_martinoli.mp3"):
    audio = client.text_to_speech.convert(
        text=texto,
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_path