import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"


def configured_path(variable: str, default: Path) -> Path:
    value = os.getenv(variable)
    return Path(value).expanduser().resolve() if value else default


YOLO_MODEL = configured_path(
    "FUTBOT_YOLO_MODEL",
    MODELS_DIR / "futbotmx_yolo11_seg_best.pt",
)
SAM3_MODEL = configured_path("FUTBOT_SAM3_MODEL", MODELS_DIR / "sam3.pt")
VIDEO_V0 = configured_path("FUTBOT_VIDEO_V0", DATA_DIR / "videos" / "V0.MOV")
VIDEO_V1 = configured_path("FUTBOT_VIDEO_V1", DATA_DIR / "videos" / "V1.MOV")
