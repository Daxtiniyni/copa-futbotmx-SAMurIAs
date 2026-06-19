from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="ball")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=Path("outputs/sam3/image_prompt.png"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    torch.set_autocast_enabled(False)
    model = build_sam3_image_model(device="cpu", load_from_HF=True).float()
    processor = Sam3Processor(model, device="cpu", confidence_threshold=args.threshold)

    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = output["masks"].detach().cpu()
    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    print(f"prompt={args.prompt!r} detections={len(scores)}")
    for idx, (box, score) in enumerate(zip(boxes, scores)):
        print(f"{idx}: score={float(score):.3f} box={[round(float(x), 1) for x in box]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    if len(masks):
        overlay = torch.zeros((*masks.shape[-2:], 4), dtype=torch.float32)
        combined = masks[:, 0].any(dim=0)
        overlay[..., 0] = 1.0
        overlay[..., 3] = combined.float() * 0.35
        plt.imshow(overlay.numpy())
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out, dpi=160)
    print(f"saved={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
