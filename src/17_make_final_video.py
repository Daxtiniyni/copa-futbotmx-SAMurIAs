from pathlib import Path

import pandas as pd
from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
from project_paths import OUTPUTS_DIR

# ============================================================
# 17_make_final_video.py
# Mezcla el video original con audios sincronizados sin encimarlos
# ============================================================

VIDEO_PATH = OUTPUTS_DIR / "sam3_pipeline_v0" / "V0_yolo_sam3_tracking.mp4"
MANIFEST_CSV = OUTPUTS_DIR / "narration" / "V0_commentary_segments_with_audio.csv"

OUT_DIR = OUTPUTS_DIR / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIDEO = OUT_DIR / "V0_final_narrated.mp4"

# Volumen del audio original del video.
# Si no quieres escuchar el audio original, ponlo en 0.0
ORIGINAL_AUDIO_VOLUME = 0.18

# Volumen de la narración de ElevenLabs.
COMMENTARY_VOLUME = 1.0

# Separación mínima para evitar que un comentario invada el siguiente bloque.
GAP_BEFORE_NEXT_COMMENT = 0.35


def clip_with_start(clip, start):
    """Compatibilidad MoviePy v1/v2."""
    if hasattr(clip, "with_start"):
        return clip.with_start(start)
    return clip.set_start(start)


def clip_with_volume(clip, volume):
    """Compatibilidad MoviePy v1/v2."""
    if hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(volume)
    return clip.volumex(volume)


def clip_subclip(clip, start, end):
    """Compatibilidad MoviePy v1/v2."""
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def video_with_audio(video, audio):
    """Compatibilidad MoviePy v1/v2."""
    if hasattr(video, "with_audio"):
        return video.with_audio(audio)
    return video.set_audio(audio)


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No encontré el video: {VIDEO_PATH}")

    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            "No encontré V0_commentary_segments_with_audio.csv. Primero ejecuta 16_elevenlabs_tts.py"
        )

    video = VideoFileClip(str(VIDEO_PATH))
    df = pd.read_csv(MANIFEST_CSV)
    df = df.sort_values("start_sec").reset_index(drop=True)

    audio_clips = []

    # Agrega audio original bajito, si existe.
    if video.audio is not None and ORIGINAL_AUDIO_VOLUME > 0:
        original_audio = clip_with_volume(video.audio, ORIGINAL_AUDIO_VOLUME)
        audio_clips.append(original_audio)

    for i, row in df.iterrows():
        audio_path = Path(str(row["audio_path"]))
        start_sec = float(row["start_sec"])

        if not audio_path.exists():
            print(f"No existe este audio, se omite: {audio_path}")
            continue

        if start_sec >= video.duration:
            print(f"Audio fuera del video, se omite: {audio_path}")
            continue

        commentary = AudioFileClip(str(audio_path))
        commentary = clip_with_volume(commentary, COMMENTARY_VOLUME)

        # Límite para que no se encime con el siguiente comentario.
        if i < len(df) - 1:
            next_start = float(df.loc[i + 1, "start_sec"])
        else:
            next_start = video.duration

        max_duration = max(0.2, next_start - start_sec - GAP_BEFORE_NEXT_COMMENT)
        max_duration = min(max_duration, video.duration - start_sec)

        if commentary.duration > max_duration:
            commentary = clip_subclip(commentary, 0, max_duration)

        commentary = clip_with_start(commentary, start_sec)
        audio_clips.append(commentary)

    final_audio = CompositeAudioClip(audio_clips)
    final = video_with_audio(video, final_audio)

    final.write_videofile(
        str(OUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        threads=4,
    )

    print("Video final guardado en:")
    print(OUT_VIDEO)

    video.close()
    final.close()

    for clip in audio_clips:
        try:
            clip.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
