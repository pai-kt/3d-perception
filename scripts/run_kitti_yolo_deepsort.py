# scripts/run_kitti_yolo_deepsort.py
from ultralytics import YOLO
from pathlib import Path
import cv2

from src.mot.deepsort.tracker import DeepSortTracker  # 나중에 구현/랩핑

def run_tracking(sequence_dir: str, output_dir: str):
    model = YOLO("outputs/yolo_kitti/yolov8n-kitti/weights/best.pt")
    tracker = DeepSortTracker()

    seq_path = Path(sequence_dir)
    frames = sorted(seq_path.glob("*.png"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        results = model(img, verbose=False)[0]

        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            dets.append((x1, y1, x2, y2, conf, cls))

        tracks = tracker.update(dets)

        # TODO: tracks 를 img 위에 그려서 저장
        # cv2.imwrite(...)

if __name__ == "__main__":
    run_tracking(
        "data/kitti/tracking/training/image_02/0000",
        "outputs/tracks/0000",
    )