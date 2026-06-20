from ultralytics import SAM
import torch
from project_paths import SAM3_MODEL

SAM3_PATH = SAM3_MODEL

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = SAM(SAM3_PATH)

print("SAM3 cargó correctamente")
