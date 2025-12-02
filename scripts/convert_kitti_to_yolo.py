import os
from pathlib import Path
import cv2

def kitti_to_yolo(input_label_dir, image_dir, output_label_dir, class_map):
    os.makedirs(output_label_dir, exist_ok=True)

    for label_file in sorted(Path(input_label_dir).glob("*.txt")):
        img_name = label_file.stem + ".png"
        img_path = Path(image_dir) / img_name

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] image not found for {label_file.name}")
            continue

        h, w = img.shape[:2]

        with open(label_file, "r") as f:
            lines = f.readlines()

        yolo_labels = []
        for line in lines:
            splits = line.strip().split()
            cls = splits[0]
            if cls not in class_map:
                continue

            # KITTI: left, top, right, bottom (pixel)
            x1, y1, x2, y2 = map(float, splits[4:8])

            # center, width, height (pixel)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            # normalize
            cx /= w
            cy /= h
            bw /= w
            bh /= h

            yolo_labels.append(
                f"{class_map[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

        save_path = Path(output_label_dir) / label_file.name
        with open(save_path, "w") as f:
            f.writelines(yolo_labels)


if __name__ == "__main__":
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    kitti_to_yolo(
        "../data/kitti/object/training/label_2",
        "../data/kitti/object/training/image_2",
        "../data/kitti/yolo/labels/train",
        class_map,
    )