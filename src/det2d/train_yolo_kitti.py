# src/det2d/train_yolo_kitti.py
from ultralytics import YOLO
import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_yolo(data_yaml: str, epochs: int = 3, img_size: int = 640):
    device = get_device()
    print(f"[INFO] Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        device=device,        # 👈 반드시 추가!
        batch=8,              # M3 18GB면 8로 시작 → 부족하면 4
        project="outputs/yolo_kitti",
        name="yolov8n-kitti",
        workers=4,
    )

if __name__ == "__main__":
    train_yolo("../../data/kitti/kitti_yolo.yaml")