# scripts/run_kitti_yolo_deepsort.py
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.mot.deepsort.tracker import SimpleTracker, DeepSortTracker, Track


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(weights_path: str) -> YOLO:
    device = get_device()
    print(f"[INFO] Using device: {device} for inference")
    model = YOLO(weights_path)
    model.to(device)
    return model


def run_tracking_on_sequence(
    model: YOLO,
    sequence_dir: str,
    output_dir: str,
    conf_thres: float = 0.3,
    save_video: bool = True,
):
    sequence_path = Path(sequence_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 이미지 순서대로 정렬
    frame_paths = sorted(sequence_path.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No .png frames found in {sequence_path}")

    print(f"[INFO] Found {len(frame_paths)} frames in {sequence_path}")

    # tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
    tracker = DeepSortTracker(
        reid_weights=None,   # 기본 경로 쓰면 됨, 바꾸고 싶으면 절대경로 넣기
        use_cuda=False,      # Mac MPS 환경이면 False, 리눅스+GPU면 True
    )

    # VideoWriter 세팅 (옵션)
    video_writer = None
    if save_video:
        sample_img = cv2.imread(str(frame_paths[0]))
        h, w = sample_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = output_path / "tracked.mp4"
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
        print(f"[INFO] Saving video to: {video_path}")

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Failed to read frame: {frame_path}")
            continue

        # YOLO inference
        results = model(
            frame,
            verbose=False,
        )[0]

        detections: List[Tuple[float, float, float, float, float, int]] = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls in zip(
                boxes_xyxy, confs, classes
            ):
                if conf < conf_thres:
                    continue
                detections.append(
                    (float(x1), float(y1), float(x2), float(y2), float(conf), int(cls))
                )

        # Tracker update
        tracks: List[Track] = tracker.update(detections, frame=frame)

        # 시각화
        vis_frame = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.astype(int)
            color = (0, 255, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"id:{t.track_id} cls:{t.cls} conf:{t.conf:.2f}"
            cv2.putText(
                vis_frame,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # 프레임 저장
        out_frame_path = output_path / f"{idx:06d}.png"
        cv2.imwrite(str(out_frame_path), vis_frame)

        # 비디오에 추가
        if video_writer is not None:
            video_writer.write(vis_frame)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx+1}/{len(frame_paths)} frames")

    if video_writer is not None:
        video_writer.release()
        print("[INFO] Video writer released")

    print(f"[INFO] Tracking completed. Frames saved to: {output_path}")


if __name__ == "__main__":
    # 1) YOLO weight 경로
    weights = "../src/det2d/outputs/yolo_kitti/yolov8n-kitti7/weights/best.pt"

    # 2) KITTI tracking sequence 디렉토리 (예: 시퀀스 0000)
    seq_dir = "../data/kitti/tracking/training/image_02/0000"

    # 3) 결과 저장 위치
    out_dir = "outputs/tracking/kitti_seq0000"

    model = load_model(weights)
    run_tracking_on_sequence(model, seq_dir, out_dir, conf_thres=0.3, save_video=True)