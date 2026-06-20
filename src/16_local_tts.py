import pyttsx3
from pathlib import Path
from project_paths import OUTPUTS_DIR

TEXT_PATH = OUTPUTS_DIR / "narration" / "V0_commentary.txt"
OUT_AUDIO = OUTPUTS_DIR / "narration" / "V0_commentary.wav"

text = TEXT_PATH.read_text(encoding="utf-8")

engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.save_to_file(text, str(OUT_AUDIO))
engine.runAndWait()

print("Audio generado:", OUT_AUDIO)
