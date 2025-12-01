# src/det2d/train_yolo_kitti.py
from ultralytics import YOLO

def train_yolo(data_yaml: str, epochs: int = 50, img_size: int = 640):
    model = YOLO("yolov8n.pt")  # 작은 모델부터
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project="outputs/yolo_kitti",
        name="yolov8n-kitti",
    )

if __name__ == "__main__":
    # data/kitti/kitti_yolo.yaml 이런 식의 설정 파일 작성 필요
    train_yolo("data/kitti/kitti_yolo.yaml")