from huggingface_hub import snapshot_download
from project_paths import MODELS_DIR

snapshot_download(
    repo_id="facebook/sam3",
    local_dir=MODELS_DIR / "sam3_model"
)
