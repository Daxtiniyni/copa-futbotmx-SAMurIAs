from __future__ import annotations

import argparse
import json
import math
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
FIELD_WIDTH = 900
FIELD_HEIGHT = 600
TRACK_COLORS = {
    "red_team": (35, 35, 230),
    "blue_team": (230, 85, 35),
    "ball": (30, 220, 245),
}


class CentroidTracker:
    def __init__(self, max_distance: float, max_missed: int = 3):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(detections)))
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track["label"] != detection["label"]:
                    continue
                distance = math.dist(track["point"], detection["point"])
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index))

        assignments = {}
        for _, track_id, detection_index in sorted(candidates):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            assignments[detection_index] = track_id
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "label": detection["label"],
                    "point": detection["point"],
                    "missed": 0,
                    "trail": [],
                }
            track = self.tracks[track_id]
            track["point"] = detection["point"]
            track["missed"] = 0
            track["trail"].append(detection["point"])
            track["trail"] = track["trail"][-60:]
            detection["track_id"] = track_id
            detection["trail"] = [list(point) for point in track["trail"]]

        for track_id in unmatched_tracks:
            self.tracks[track_id]["missed"] += 1
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed
        }
        return detections


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


def load_calibration(path: Path | None, width: int, height: int):
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(payload["points"], dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("La calibración debe contener cuatro puntos")
    if payload.get("normalized", False):
        points[:, 0] *= width
        points[:, 1] *= height
    target = np.float32(
        [[0, 0], [FIELD_WIDTH, 0], [FIELD_WIDTH, FIELD_HEIGHT], [0, FIELD_HEIGHT]]
    )
    matrix = cv2.getPerspectiveTransform(points, target)
    return {
        "points": points.tolist(),
        "matrix": matrix,
        "field_size": [FIELD_WIDTH, FIELD_HEIGHT],
    }


def project_point(point: tuple[float, float], calibration) -> list[float] | None:
    if calibration is None:
        return None
    source = np.float32([[point]])
    projected = cv2.perspectiveTransform(source, calibration["matrix"])[0, 0]
    return [float(projected[0]), float(projected[1])]


def draw_trail(frame: np.ndarray, trail: list[list[float]], color: tuple[int, int, int]):
    if len(trail) < 2:
        return
    points = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], False, color, 3, cv2.LINE_AA)


def draw_track_label(
    frame: np.ndarray,
    point: tuple[float, float],
    track_id: int,
    color: tuple[int, int, int],
):
    x, y = int(point[0]), int(point[1])
    cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"ID {track_id}",
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
    parser.add_argument("--calibration", type=Path)
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
    calibration = load_calibration(args.calibration, out_w, out_h)

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
    robot_tracker = CentroidTracker(max_distance=max(out_w, out_h) * 0.18)
    ball_tracker = CentroidTracker(max_distance=max(out_w, out_h) * 0.25, max_missed=5)

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
        robot_detections = []
        for mask, box, score in zip(robot_masks, robot_boxes, robot_scores):
            team = classify_team(base, mask)
            overlay_mask(frame, mask, COLORS_BGR[team], 0.55)
            point = ((float(box[0]) + float(box[2])) / 2, float(box[3]))
            robot_detections.append(
                {
                    "label": team,
                    "team": team,
                    "score": float(score),
                    "box": [float(x) for x in box],
                    "point": point,
                }
            )
        robot_detections = robot_tracker.update(robot_detections)
        for detection in robot_detections:
            draw_trail(frame, detection["trail"], TRACK_COLORS[detection["team"]])
            draw_track_label(
                frame,
                detection["point"],
                detection["track_id"],
                TRACK_COLORS[detection["team"]],
            )
            frame_record["robots"].append(
                {
                    "team": detection["team"],
                    "score": detection["score"],
                    "box": detection["box"],
                    "track_id": detection["track_id"],
                    "image_position": list(detection["point"]),
                    "field_position": project_point(detection["point"], calibration),
                    "trail": detection["trail"],
                }
            )

        ball_masks, ball_boxes, ball_scores = run_prompt(processor, state, args.ball_prompt)
        ball_candidates = []
        for mask, box, score in zip(ball_masks, ball_boxes, ball_scores):
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if width * height > out_w * out_h * 0.08:
                continue
            point = (
                (float(box[0]) + float(box[2])) / 2,
                (float(box[1]) + float(box[3])) / 2,
            )
            ball_candidates.append(
                {
                    "label": "ball",
                    "score": float(score),
                    "box": [float(x) for x in box],
                    "point": point,
                    "mask": mask,
                }
            )
        if ball_candidates:
            ball_candidates = [max(ball_candidates, key=lambda item: item["score"])]
        ball_candidates = ball_tracker.update(ball_candidates)
        for detection in ball_candidates:
            overlay_mask(frame, detection["mask"], COLORS_BGR["ball"], 0.75)
            draw_trail(frame, detection["trail"], TRACK_COLORS["ball"])
            draw_track_label(
                frame,
                detection["point"],
                detection["track_id"],
                TRACK_COLORS["ball"],
            )
            frame_record["balls"].append(
                {
                    "score": detection["score"],
                    "box": detection["box"],
                    "track_id": detection["track_id"],
                    "image_position": list(detection["point"]),
                    "field_position": project_point(detection["point"], calibration),
                    "trail": detection["trail"],
                }
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
    payload = {
        "metadata": {
            "source_fps": src_fps,
            "sample_fps": args.sample_fps,
            "source_size": [out_w, out_h],
            "tracking": True,
            "homography": calibration is not None,
            "field_size": [FIELD_WIDTH, FIELD_HEIGHT],
            "calibration_points": calibration["points"] if calibration else None,
        },
        "frames": records,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_video={args.out}")
    print(f"saved_json={args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
