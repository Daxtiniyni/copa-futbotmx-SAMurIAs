from __future__ import annotations

import argparse
import traceback

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Also try downloading/loading gated Hugging Face weights.",
    )
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"mps_available={torch.backends.mps.is_available()}")

    from sam3.model_builder import (
        build_sam3_image_model,
        build_sam3_predictor,
    )
    from sam3.model.sam3_image_processor import Sam3Processor

    print("sam3 imports ok")

    image_model = build_sam3_image_model(device="cpu", load_from_HF=False)
    print(f"image model builds without weights: {type(image_model).__name__}")
    _ = Sam3Processor(image_model, device="cpu")
    print("image processor builds on cpu")

    if not args.with_weights:
        return 0

    try:
        image_model = build_sam3_image_model(device="cpu", load_from_HF=True)
        print(f"image model loads weights: {type(image_model).__name__}")
    except Exception:
        print("image weights failed; this usually means Hugging Face access/login is missing")
        traceback.print_exc(limit=4)

    try:
        predictor = build_sam3_predictor(
            version="sam3.1",
            device="cpu",
            compile=False,
            warm_up=False,
            use_fa3=False,
            use_rope_real=False,
        )
        print(f"sam3.1 video predictor loads weights: {type(predictor).__name__}")
    except Exception:
        print("sam3.1 weights failed; this usually means Hugging Face access/login is missing")
        traceback.print_exc(limit=4)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
