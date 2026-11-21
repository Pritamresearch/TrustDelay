"""
Comprehensive 3D / BEV Object Detection Metrics & Evaluation Protocol.
Computes standard VOC/COCO AP@0.3, AP@0.5, and AP@0.7 using exact 2D oriented polygon intersections (Shapely).
Zero heuristic heading penalties or proxy formulas.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Tuple
from shapely.geometry import Polygon

def box_to_polygon(box: np.ndarray) -> Polygon:
    """
    Converts 3D bounding box [cx, cy, cz, dx, dy, dz, yaw] into an oriented 2D BEV Shapely Polygon.
    """
    cx, cy, _, dx, dy, _, yaw = box
    hx, hy = dx / 2.0, dy / 2.0
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts = corners @ R.T + np.array([cx, cy], dtype=np.float64)
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def compute_bev_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes exact oriented 2D BEV IoU matrix between boxes_a (N, 7) and boxes_b (M, 7).
    """
    N = len(boxes_a)
    M = len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    iou_matrix = np.zeros((N, M), dtype=np.float32)
    polys_a = [box_to_polygon(boxes_a[i]) for i in range(N)]
    polys_b = [box_to_polygon(boxes_b[j]) for j in range(M)]

    for i in range(N):
        pa = polys_a[i]
        if not pa.is_valid or pa.area <= 1e-5:
            continue
        for j in range(M):
            pb = polys_b[j]
            if not pb.is_valid or pb.area <= 1e-5:
                continue
            # AABB spatial pre-filter
            if (pa.bounds[2] < pb.bounds[0] or pa.bounds[0] > pb.bounds[2] or
                pa.bounds[3] < pb.bounds[1] or pa.bounds[1] > pb.bounds[3]):
                continue
            try:
                inter = pa.intersection(pb).area
                union = pa.area + pb.area - inter
                if union > 0:
                    iou_matrix[i, j] = float(inter / union)
            except Exception:
                continue

    return iou_matrix

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Continuous Area Under Precision-Recall Curve (standard VOC/COCO integration).
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

class DetectionEvaluator:
    def __init__(self, iou_thresholds: List[float] = [0.3, 0.5, 0.7]):
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        self.predictions: List[Tuple[np.ndarray, np.ndarray]] = []
        self.ground_truths: List[Tuple[np.ndarray, np.ndarray]] = []

    def add_sample(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_is_moving: np.ndarray
    ):
        """
        Record a single frame evaluation sample.
        pred_boxes: (N, 7) [cx, cy, cz, dx, dy, dz, yaw]
        pred_scores: (N,)
        gt_boxes: (M, 7)
        gt_is_moving: (M,) boolean
        """
        self.predictions.append((np.asarray(pred_boxes, dtype=np.float32), np.asarray(pred_scores, dtype=np.float32)))
        self.ground_truths.append((np.asarray(gt_boxes, dtype=np.float32), np.asarray(gt_is_moving, dtype=bool)))

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates AP@0.3, AP@0.5, AP@0.7 across:
          - Overall (all vehicles)
          - Moving vehicles
          - Stationary vehicles
        """
        results = {}
        for subset in ["all", "moving", "stationary"]:
            for iou_th in self.iou_thresholds:
                ap_val = self._eval_subset_iou(subset, iou_th)
                tag = f"AP@{iou_th:.1f}" if subset == "all" else f"AP@{iou_th:.1f}_{subset}"
                results[tag] = ap_val * 100.0
        return results

    def _eval_subset_iou(self, subset: str, iou_thresh: float) -> float:
        all_pred_scores = []
        all_match_flags = []
        total_gt_count = 0

        for (pred_boxes, pred_scores), (gt_boxes, gt_is_moving) in zip(self.predictions, self.ground_truths):
            if subset == "moving":
                mask = (gt_is_moving == True)
                cur_gt_boxes = gt_boxes[mask]
            elif subset == "stationary":
                mask = (gt_is_moving == False)
                cur_gt_boxes = gt_boxes[mask]
            else:
                cur_gt_boxes = gt_boxes

            total_gt_count += len(cur_gt_boxes)

            if len(pred_boxes) == 0:
                continue

            order = np.argsort(-pred_scores)
            sorted_boxes = pred_boxes[order]
            sorted_scores = pred_scores[order]

            if len(cur_gt_boxes) == 0:
                for score in sorted_scores:
                    all_pred_scores.append(float(score))
                    all_match_flags.append(0)
                continue

            iou_mat = compute_bev_iou_matrix(sorted_boxes, cur_gt_boxes)
            gt_matched = np.zeros(len(cur_gt_boxes), dtype=bool)

            for i in range(len(sorted_boxes)):
                all_pred_scores.append(float(sorted_scores[i]))
                best_gt_idx = int(np.argmax(iou_mat[i]))
                best_iou = float(iou_mat[i, best_gt_idx])

                if best_iou >= iou_thresh and not gt_matched[best_gt_idx]:
                    all_match_flags.append(1)
                    gt_matched[best_gt_idx] = True
                else:
                    all_match_flags.append(0)

        if total_gt_count == 0 or len(all_pred_scores) == 0:
            return 0.0

        all_pred_scores = np.array(all_pred_scores, dtype=np.float64)
        all_match_flags = np.array(all_match_flags, dtype=np.float64)

        global_order = np.argsort(-all_pred_scores)
        tp = all_match_flags[global_order]
        fp = 1.0 - tp

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / float(total_gt_count)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

        return compute_ap(recalls, precisions)
