from ultralytics import SAM
from project_paths import DATA_DIR, OUTPUTS_DIR, SAM3_MODEL

SAM3_PATH = SAM3_MODEL
IMAGE_PATH = DATA_DIR / "frames" / "V1" / "V1_000000.jpg"

model = SAM(SAM3_PATH)

results = model(
    IMAGE_PATH,
    bboxes=[[780, 1230, 982, 1397]],
    save=True,
    project=str(OUTPUTS_DIR / "sam3_tests"),
    name="bbox_test"
)

print("Listo")
