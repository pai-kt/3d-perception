# src/mot/deepsort/tracker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import itertools
import numpy as np

# 🔽 추가: deep_sort_pytorch import
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEEPSORT_ROOT = PROJECT_ROOT / "external" / "deep_sort_pytorch"
sys.path.append(str(DEEPSORT_ROOT))

from deep_sort.deep_sort import DeepSort  # type: ignore


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    cls: int
    conf: float
    age: int = 0          # 몇 프레임 동안 존재
    time_since_update: int = 0  # 업데이트 안 된 프레임 수


# DeepSORT용 어댑터를 새로 정의
class DeepSortTracker:
    """
    ZQPei/deep_sort_pytorch의 DeepSort 클래스를
    우리가 정의한 Track 인터페이스에 맞게 래핑한 어댑터.
    """

    def __init__(
        self,
        reid_weights: str | None = None,
        max_dist: float = 0.2,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        use_cuda: bool = False,
    ):
        if reid_weights is None:
            # 기본 위치 (필요에 따라 수정)
            reid_weights = str(
                DEEPSORT_ROOT / "deep_sort" / "deep" / "checkpoint" / "ckpt.t7"
            )

        self.deepsort = DeepSort(
            reid_weights,
            max_dist=max_dist,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            use_cuda=use_cuda,
        )

        # detection index → class id, conf 를 기억해두었다가
        # deepsort output과 다시 매칭하기 위해 사용하는 맵
        self._last_det_classes: dict[int, int] = {}
        self._last_det_confs: dict[int, float] = {}
        self._next_det_idx_iter = itertools.count(0)

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float, int]],
        frame,
    ) -> List[Track]:
        """
        Args:
            detections: [(x1, y1, x2, y2, conf, cls), ...]
            frame: BGR numpy image
        Returns:
            List[Track]
        """
        if len(detections) == 0:
            # 빈 detection → 빈 배열로 update 호출 (내부적으로 tracker.predict() 수행)
            empty = np.zeros((0, 4), dtype=np.float32)
            self.deepsort.update(empty, np.array([]), np.array([]), frame)
            return []

        bboxes_xyxy = np.array([d[:4] for d in detections], dtype=np.float32)
        confs = np.array([d[4] for d in detections], dtype=np.float32)
        classes = np.array([d[5] for d in detections], dtype=np.int32)

        # DeepSORT는 xywh 형태 기대하므로 변환 필요
        xywhs = self._xyxy_to_xywh(bboxes_xyxy)

        # detection index → cls/conf 매핑 저장
        self._last_det_classes = {}
        self._last_det_confs = {}
        for idx, (c, s) in enumerate(zip(classes, confs)):
            self._last_det_classes[idx] = int(c)
            self._last_det_confs[idx] = float(s)

        # DeepSORT forward
        # outputs shape: [N, 6] = [x1, y1, x2, y2, cls, track_id]
        outputs, _ = self.deepsort.update(xywhs, confs, classes, frame)

        tracks: List[Track] = []
        if len(outputs) > 0:
            for out in outputs:
                x1, y1, x2, y2, cls_id, track_id = out
                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                tracks.append(
                    Track(
                        track_id=int(track_id),
                        bbox=bbox,
                        cls=int(cls_id),
                        conf=1.0,
                    )
                )

        return tracks

    @staticmethod
    def _xyxy_to_xywh(bboxes_xyxy: np.ndarray) -> np.ndarray:
        xywh = np.zeros_like(bboxes_xyxy)
        xywh[:, 0] = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2.0  # cx
        xywh[:, 1] = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2.0  # cy
        xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]          # w
        xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]          # h
        return xywh


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """두 개의 [x1, y1, x2, y2] 박스 IoU 계산."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class SimpleTracker:
    """
    DeepSORT 자리에 들어갈 수 있는 '인터페이스 스켈레톤' 겸
    아주 단순한 IoU 기반 tracker.

    실제 프로젝트에서는 이 클래스를 DeepSORT 구현으로 교체하면 됨.
    """

    _id_iter = itertools.count(1)

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.tracks: List[Track] = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float, int]],
        frame=None,
    ) -> List[Track]:
        """
        Args:
            detections: [(x1, y1, x2, y2, conf, cls), ...]
            frame: 원본 이미지 (DeepSORT 쓸 때 appearance feature용, 여기선 안 씀)

        Returns:
            현재 프레임에서 유효한 Track 리스트
        """
        # detections: [(x1, y1, x2, y2, conf, cls), ...] -> det_bboxes: [[x1, y1, x2, y2], ...]
        det_bboxes = np.array([d[:4] for d in detections], dtype=np.float32) if detections else np.zeros((0, 4))
        # detections: [(x1, y1, x2, y2, conf, cls), ...] -> det_confs: [conf, ...]
        det_confs = np.array([d[4] for d in detections], dtype=np.float32) if detections else np.zeros((0,))
        # detections: [(x1, y1, x2, y2, conf, cls), ...] -> det_classes: [cls, ...]
        det_classes = np.array([d[5] for d in detections], dtype=np.int32) if detections else np.zeros((0,), dtype=np.int32)

        # 기존 트랙 상태 업데이트
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        assigned_det_indices = set()

        # 1) 기존 트랙과 현재 detection 매칭 (IoU 최대값 기준)
        for track in self.tracks:
            best_iou = 0.0
            best_det_idx = -1
            for det_idx, bbox in enumerate(det_bboxes):
                if det_idx in assigned_det_indices:
                    continue
                iou = iou_xyxy(track.bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx >= 0 and best_iou >= self.iou_threshold:
                # 매칭 성공 → 트랙 업데이트
                track.bbox = det_bboxes[best_det_idx]
                track.conf = float(det_confs[best_det_idx])
                track.cls = int(det_classes[best_det_idx])
                track.time_since_update = 0
                assigned_det_indices.add(best_det_idx)

        # 2) 매칭 안 된 detection → 새 트랙 생성
        for det_idx, bbox in enumerate(det_bboxes):
            if det_idx in assigned_det_indices:
                continue
            new_id = next(self._id_iter)
            self.tracks.append(
                Track(
                    track_id=new_id,
                    bbox=bbox,
                    cls=int(det_classes[det_idx]),
                    conf=float(det_confs[det_idx]),
                )
            )

        # 3) 오래 업데이트 안 된 트랙 제거
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age
        ]

        return self.tracks