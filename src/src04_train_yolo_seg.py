from multiprocessing import freeze_support
from ultralytics import YOLO
import torch
from project_paths import OUTPUTS_DIR, PROJECT_ROOT

DATASET_YAML = PROJECT_ROOT / "data" / "yolo_seg" / "dataset.yaml"

def main():
    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=DATASET_YAML,
        epochs=80,
        imgsz=960,
        batch=2,
        device=0,
        workers=0,
        project=str(OUTPUTS_DIR / "runs"),
        name="futbotmx_yolo11_seg_v0_dataset",
        patience=20
    )

if __name__ == "__main__":
    freeze_support()
    main()
