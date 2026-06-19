from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


COLORS_BGR = {
    "field": (35, 180, 35),
    "red_team": (35, 35, 230),
    "blue_team": (230, 85, 35),
    "ball": (30, 220, 245),
}


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    if not mask.any():
        return
    color_arr = np.array(color, dtype=np.float32)
    frame[mask] = (frame[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha).astype(np.uint8)


def green_field_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (32, 35, 45), (95, 255, 255))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def classify_team(frame: np.ndarray, mask: np.ndarray) -> str:
    if mask.ndim == 3:
        mask = mask.squeeze()
    region = frame[mask.astype(bool)]
    if len(region) == 0:
        return "blue_team"
    hsv = cv2.cvtColor(region.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    red = ((hsv[:, 0] <= 10) | (hsv[:, 0] >= 170)) & (hsv[:, 1] > 55) & (hsv[:, 2] > 45)
    blue = (hsv[:, 0] >= 90) & (hsv[:, 0] <= 135) & (hsv[:, 1] > 45) & (hsv[:, 2] > 45)
    return "red_team" if red.sum() >= blue.sum() else "blue_team"


def run_prompt(processor: Sam3Processor, state: dict, prompt: str):
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = output["masks"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    return masks, boxes, scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/sam3/segmented_preview.mp4"))
    parser.add_argument("--json-out", type=Path, default=Path("outputs/sam3/segmented_preview.json"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--robot-prompt", default="robot")
    parser.add_argument("--ball-prompt", default="ball")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_step = max(1, round(src_fps / args.sample_fps))
    max_source_frame = min(total_frames, int(args.seconds * src_fps))

    scale = min(1.0, args.max_width / src_w)
    out_w = int(round(src_w * scale))
    out_h = int(round(src_h * scale))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.sample_fps,
        (out_w, out_h),
    )

    torch.set_autocast_enabled(False)
    model = build_sam3_image_model(device="cpu", load_from_HF=True).float()
    processor = Sam3Processor(model, device="cpu", confidence_threshold=args.threshold)

    records = []
    source_idx = 0
    written = 0
    while source_idx < max_source_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        base = frame.copy()
        overlay_mask(frame, green_field_mask(base), COLORS_BGR["field"], 0.28)

        pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
        state = processor.set_image(pil)

        frame_record = {
            "output_frame": written,
            "source_frame": source_idx,
            "time_sec": source_idx / src_fps,
            "robots": [],
            "balls": [],
        }

        robot_masks, robot_boxes, robot_scores = run_prompt(processor, state, args.robot_prompt)
        for mask, box, score in zip(robot_masks, robot_boxes, robot_scores):
            team = classify_team(base, mask)
            overlay_mask(frame, mask, COLORS_BGR[team], 0.55)
            frame_record["robots"].append(
                {"team": team, "score": float(score), "box": [float(x) for x in box]}
            )

        ball_masks, ball_boxes, ball_scores = run_prompt(processor, state, args.ball_prompt)
        for mask, box, score in zip(ball_masks, ball_boxes, ball_scores):
            overlay_mask(frame, mask, COLORS_BGR["ball"], 0.75)
            frame_record["balls"].append(
                {"score": float(score), "box": [float(x) for x in box]}
            )

        cv2.putText(
            frame,
            f"t={source_idx / src_fps:.1f}s robots={len(frame_record['robots'])} ball={len(frame_record['balls'])}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        records.append(frame_record)
        print(
            f"frame {written}: source={source_idx} robots={len(frame_record['robots'])} balls={len(frame_record['balls'])}"
        )

        written += 1
        source_idx += frame_step

    cap.release()
    writer.release()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"saved_video={args.out}")
    print(f"saved_json={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
