"""
Comprehensive 3D Object Detection Metrics & Evaluation Protocol.
Computes AP@0.3, AP@0.5, and AP@0.7 for:
  - All vehicles
  - Moving vehicles (speed > 0.5 m/s)
  - Stationary vehicles (speed <= 0.5 m/s)
"""
from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict, List, Tuple

def compute_bev_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes 2D BEV IoU between two sets of oriented/axis-aligned 3D boxes.
    boxes: [cx, cy, cz, dx, dy, dz, yaw]
    """
    N = len(boxes_a)
    M = len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # 2D Axis-aligned bounding box bounding approximation with yaw factor
    # Min/max bounds in BEV
    a_min_x = boxes_a[:, 0] - boxes_a[:, 3] / 2.0
    a_max_x = boxes_a[:, 0] + boxes_a[:, 3] / 2.0
    a_min_y = boxes_a[:, 1] - boxes_a[:, 4] / 2.0
    a_max_y = boxes_a[:, 1] + boxes_a[:, 4] / 2.0
    area_a = boxes_a[:, 3] * boxes_a[:, 4]

    b_min_x = boxes_b[:, 0] - boxes_b[:, 3] / 2.0
    b_max_x = boxes_b[:, 0] + boxes_b[:, 3] / 2.0
    b_min_y = boxes_b[:, 1] - boxes_b[:, 4] / 2.0
    b_max_y = boxes_b[:, 1] + boxes_b[:, 4] / 2.0
    area_b = boxes_b[:, 3] * boxes_b[:, 4]

    iou_matrix = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        inter_x1 = np.maximum(a_min_x[i], b_min_x)
        inter_x2 = np.minimum(a_max_x[i], b_max_x)
        inter_y1 = np.maximum(a_min_y[i], b_min_y)
        inter_y2 = np.minimum(a_max_y[i], b_max_y)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = area_a[i] + area_b - inter_area
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)

        # Apply heading penalty if angles are heavily misaligned (|delta_yaw| > 45 deg)
        yaw_diff = np.abs(boxes_a[i, 6] - boxes_b[:, 6])
        yaw_diff = np.minimum(yaw_diff, 2 * np.pi - yaw_diff)
        heading_penalty = np.clip(1.0 - yaw_diff / (np.pi / 2.0), 0.5, 1.0)

        iou_matrix[i] = iou * heading_penalty

    return iou_matrix


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Standard 11-point or continuous VOC/COCO Area Under Precision-Recall Curve.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # Compute AUC
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


class DetectionEvaluator:
    def __init__(self, iou_thresholds: List[float] = [0.3, 0.5, 0.7]):
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        self.predictions = []  # list of (pred_boxes, pred_scores)
        self.ground_truths = []  # list of (gt_boxes, gt_is_moving)

    def add_sample(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_is_moving: np.ndarray
    ):
        self.predictions.append((pred_boxes, pred_scores))
        self.ground_truths.append((gt_boxes, gt_is_moving))

    def evaluate(self) -> Dict[str, float]:
        """
        Calculates AP@0.3, AP@0.5, AP@0.7 across all, moving, and stationary objects.
        """
        results = {}
        for subset in ["all", "moving", "stationary"]:
            for iou_th in self.iou_thresholds:
                ap_val = self._eval_subset_iou(subset, iou_th)
                tag = f"AP@{iou_th:.1f}" if subset == "all" else f"AP@{iou_th:.1f}_{subset}"
                results[tag] = ap_val * 100.0  # percentage
        return results

    def _eval_subset_iou(self, subset: str, iou_thresh: float) -> float:
        all_pred_scores = []
        all_match_flags = []
        total_gt_count = 0

        for (pred_boxes, pred_scores), (gt_boxes, gt_is_moving) in zip(self.predictions, self.ground_truths):
            # Filter GT based on subset
            if subset == "moving":
                mask = gt_is_moving == True
                cur_gt_boxes = gt_boxes[mask]
            elif subset == "stationary":
                mask = gt_is_moving == False
                cur_gt_boxes = gt_boxes[mask]
            else:
                cur_gt_boxes = gt_boxes

            total_gt_count += len(cur_gt_boxes)

            if len(pred_boxes) == 0:
                continue

            # Sort predictions by score descending
            order = np.argsort(-pred_scores)
            sorted_boxes = pred_boxes[order]
            sorted_scores = pred_scores[order]

            if len(cur_gt_boxes) == 0:
                for score in sorted_scores:
                    all_pred_scores.append(score)
                    all_match_flags.append(0)
                continue

            # Compute IoU matrix: (N_pred, M_gt)
            iou_mat = compute_bev_iou_matrix(sorted_boxes, cur_gt_boxes)
            gt_matched = np.zeros(len(cur_gt_boxes), dtype=bool)

            for i in range(len(sorted_boxes)):
                all_pred_scores.append(sorted_scores[i])
                best_gt_idx = np.argmax(iou_mat[i])
                best_iou = iou_mat[i, best_gt_idx]

                if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                    all_match_flags.append(1)  # True Positive
                    gt_matched[best_gt_idx] = True
                else:
                    all_match_flags.append(0)  # False Positive

        if total_gt_count == 0 or len(all_pred_scores) == 0:
            return 0.0

        all_pred_scores = np.array(all_pred_scores)
        all_match_flags = np.array(all_match_flags)

        # Sort globally
        global_order = np.argsort(-all_pred_scores)
        tp = all_match_flags[global_order]
        fp = 1 - tp

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / float(total_gt_count)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

        return compute_ap(recalls, precisions)
