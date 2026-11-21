"""
Unit test for metrics calculation (exact Shapely BEV IoU & AP integration).
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from src.utils.metrics import compute_bev_iou_matrix, compute_ap, DetectionEvaluator

def test_iou_identical_boxes():
    box_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    box_b = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    iou = compute_bev_iou_matrix(box_a, box_b)
    assert np.isclose(iou[0, 0], 1.0), f"Expected 1.0, got {iou[0, 0]}"

def test_iou_disjoint_boxes():
    box_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    box_b = np.array([[20.0, 20.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    iou = compute_bev_iou_matrix(box_a, box_b)
    assert np.isclose(iou[0, 0], 0.0), f"Expected 0.0, got {iou[0, 0]}"

def test_iou_half_overlap():
    # 4x2 boxes shifted by 2m in x:
    # box 1: x in [-2, 2], y in [-1, 1], area = 8
    # box 2: x in [0, 4], y in [-1, 1], area = 8
    # intersection: x in [0, 2], y in [-1, 1], inter = 4
    # union: 8 + 8 - 4 = 12 -> IoU = 4/12 = 1/3
    box_a = np.array([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    box_b = np.array([[2.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    iou = compute_bev_iou_matrix(box_a, box_b)
    assert np.isclose(iou[0, 0], 1.0 / 3.0), f"Expected 0.3333, got {iou[0, 0]}"

def test_evaluator_ap():
    evaluator = DetectionEvaluator(iou_thresholds=[0.5, 0.7])
    # Frame 1: 1 perfect prediction, 1 ground truth
    gt = np.array([[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    pred = np.array([[10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    scores = np.array([0.95])
    moving = np.array([True])
    evaluator.add_sample(pred, scores, gt, moving)
    res = evaluator.evaluate()
    assert np.isclose(res["AP@0.5"], 100.0)
    assert np.isclose(res["AP@0.7"], 100.0)
    print("Metrics tests passed successfully!")

if __name__ == "__main__":
    test_iou_identical_boxes()
    test_iou_disjoint_boxes()
    test_iou_half_overlap()
    test_evaluator_ap()
