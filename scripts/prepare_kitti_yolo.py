# scripts/prepare_kitti_yolo.py
from pathlib import Path
import os
import shutil

def prepare_kitti_yolo():
    # 원본 이미지 위치
    src_img_dir = Path("../data/kitti/object/training/image_2")

    # YOLO용 이미지/라벨 폴더
    train_img_dir = Path("../data/kitti/yolo/images/train")
    val_img_dir = Path("../data/kitti/yolo/images/val")
    train_lbl_dir = Path("../data/kitti/yolo/labels/train")
    val_lbl_dir = Path("../data/kitti/yolo/labels/val")

    for d in [train_img_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 간단하게: 앞 6000장은 train, 나머지는 val (대충 분리)
    all_imgs = sorted(src_img_dir.glob("*.png"))
    split_idx = int(len(all_imgs) * 0.8)
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]

    def copy_pair(img_paths, dst_img_dir, dst_lbl_dir):
        for img_path in img_paths:
            dst_img = dst_img_dir / img_path.name
            # 이미지 복사 (용량 아깝으면 나중에 symlink로 바꿔도 됨)
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # 라벨 파일 이름은 동일한 .txt
            lbl_name = img_path.stem + ".txt"
            src_lbl = train_lbl_dir / lbl_name
            if src_lbl.exists():
                dst_lbl = dst_lbl_dir / lbl_name
                if not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)

    copy_pair(train_imgs, train_img_dir, train_lbl_dir)
    copy_pair(val_imgs, val_img_dir, val_lbl_dir)

    print("Done! images/train, images/val, labels/train, labels/val 준비 완료.")

if __name__ == "__main__":
    prepare_kitti_yolo()