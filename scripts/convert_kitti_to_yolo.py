# scripts/kitti_to_yolo.py
import os
from pathlib import Path
import cv2


def kitti_to_yolo(
    input_label_dir: str,
    image_dir: str,
    output_label_dir: str,
    class_map: dict,
):
    input_label_dir = Path(input_label_dir)
    image_dir = Path(image_dir)
    output_label_dir = Path(output_label_dir)

    os.makedirs(output_label_dir, exist_ok=True)

    num_files = 0
    num_boxes_total = 0
    num_boxes_kept = 0
    num_boxes_dropped = 0

    for label_file in sorted(input_label_dir.glob("*.txt")):
        img_name = label_file.stem + ".png"
        img_path = image_dir / img_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] image not found for label: {label_file.name}")
            continue

        h, w = img.shape[:2]

        with open(label_file, "r") as f:
            lines = f.readlines()

        yolo_labels = []
        for line in lines:
            splits = line.strip().split()
            if len(splits) < 8:
                # 잘못된 라인
                num_boxes_dropped += 1
                continue

            cls_name = splits[0]
            if cls_name not in class_map:
                # 우리가 쓰지 않는 클래스 (Van, Truck 등) 스킵
                num_boxes_dropped += 1
                continue

            # KITTI: type, trunc, occl, alpha, left, top, right, bottom, ...
            try:
                x1, y1, x2, y2 = map(float, splits[4:8])
            except ValueError:
                num_boxes_dropped += 1
                continue

            num_boxes_total += 1

            # 이미지 범위로 클리핑
            x1 = max(0.0, min(x1, w - 1.0))
            x2 = max(0.0, min(x2, w - 1.0))
            y1 = max(0.0, min(y1, h - 1.0))
            y2 = max(0.0, min(y2, h - 1.0))

            # 폭/높이 체크 (이상한 박스 제거)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                num_boxes_dropped += 1
                continue

            # center / width / height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # 정규화 (0~1)
            cx /= w
            cy /= h
            bw /= w
            bh /= h

            # 여유 있게 범위 클램프
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                num_boxes_dropped += 1
                continue
            if bw <= 0 or bh <= 0 or bw > 1.0 or bh > 1.0:
                num_boxes_dropped += 1
                continue

            cls_id = class_map[cls_name]
            yolo_labels.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )
            num_boxes_kept += 1

        save_path = output_label_dir / label_file.name
        if yolo_labels:
            with open(save_path, "w") as f:
                f.writelines(yolo_labels)
        else:
            # GT 하나도 없으면 비어있는 파일이라도 만들기 (배경)
            save_path.touch()

        num_files += 1

    print(f"[INFO] Processed {num_files} label files")
    print(f"[INFO] Total boxes: {num_boxes_total}")
    print(f"[INFO] Kept boxes : {num_boxes_kept}")
    print(f"[INFO] Dropped    : {num_boxes_dropped}")


if __name__ == "__main__":
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    kitti_to_yolo(
        input_label_dir="../data/kitti/object/training/label_2",
        image_dir="../data/kitti/object/training/image_2",
        output_label_dir="../data/kitti/yolo/labels/train",
        class_map=class_map,
    )