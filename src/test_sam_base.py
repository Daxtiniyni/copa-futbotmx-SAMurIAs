from ultralytics import SAM
from project_paths import MODELS_DIR

model = SAM(MODELS_DIR / "mobile_sam.pt")

print("Mobile SAM cargado correctamente")
