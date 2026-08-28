"""
metrics.py — Evaluation metrics for Cerberus (Paper Section 4.2).

Implements the metrics used in the paper:
  - Dice coefficient (semantic segmentation)
  - Panoptic Quality (PQ) for instance segmentation
  - mPQ  (mean PQ — per-image per-class averaging)
  - mPQ+ (aggregated PQ — stats-aggregation then per-class averaging)

Reference:
    Kirillov et al., "Panoptic Segmentation", CVPR 2019.
    Cerberus paper Section 4.2 — Evaluation Metrics.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


# ===================================================================
# Helpers: numpy / torch interop
# ===================================================================
def _to_numpy(x, dtype=None):
    """Convert torch.Tensor → numpy.ndarray, pass-through otherwise."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    if dtype is not None:
        x = np.asarray(x, dtype=dtype)
    return x


def _unique_ids(mask):
    """Return sorted instance IDs in mask (excluding 0 = background)."""
    ids = np.unique(mask)
    return sorted([int(i) for i in ids if i != 0])


def _instances_of_class(inst_map, type_map, cls_id):
    """Extract instances belonging to a specific semantic class.

    If type_map is None, all instances are treated as a single class (cls_id=1).

    Args:
        inst_map: (H, W)  — instance ID map, 0=background
        type_map: (H, W)  — semantic class map, or None
        cls_id:   int     — target class

    Returns:
        per_inst_mask: list of (H, W) binary masks, one per instance of this class
    """
    ids = _unique_ids(inst_map)
    masks = []
    for iid in ids:
        mask_i = (inst_map == iid)
        if type_map is not None:
            # Determine dominant class for this instance
            types_in_inst = type_map[mask_i]
            vals, counts = np.unique(types_in_inst, return_counts=True)
            dominant = vals[np.argmax(counts)]
            if int(dominant) != cls_id:
                continue
        masks.append(mask_i.astype(np.uint8))
    return masks


def _iou(mask_a, mask_b):
    """Intersection-over-Union between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _match_by_iou(pred_masks, true_masks, threshold=0.5):
    """Greedy bipartite matching of predicted ↔ ground-truth instances.

    Algorithm:
      1. Compute IoU matrix between all (pred, true) pairs.
      2. Greedily match highest-IoU pairs until no remaining IoU > threshold.

    Args:
        pred_masks: list of binary masks (predicted)
        true_masks: list of binary masks (ground truth)
        threshold:  minimum IoU for a valid match (0.5 per paper)

    Returns:
        matched_iou_scores: list of IoU values for matched pairs
        tp_count:  number of matched pairs (TP)
        fp_count:  number of unmatched predictions (FP)
        fn_count:  number of unmatched GT instances (FN)
    """
    n_pred = len(pred_masks)
    n_true = len(true_masks)

    if n_pred == 0 and n_true == 0:
        return [], 0, 0, 0
    if n_pred == 0:
        return [], 0, 0, n_true
    if n_true == 0:
        return [], 0, n_pred, 0

    # Build IoU matrix
    iou_mat = np.zeros((n_pred, n_true), dtype=np.float64)
    for i in range(n_pred):
        for j in range(n_true):
            iou_mat[i, j] = _iou(pred_masks[i], true_masks[j])

    # Greedy matching: repeatedly select max IoU > threshold
    matched_ious = []
    matched_pred = set()
    matched_true = set()

    # Sort all (i, j, iou) descending by IoU
    candidates = []
    for i in range(n_pred):
        for j in range(n_true):
            if iou_mat[i, j] > threshold:
                candidates.append((iou_mat[i, j], i, j))
    candidates.sort(key=lambda x: x[0], reverse=True)

    for iou_val, pi, tj in candidates:
        if pi not in matched_pred and tj not in matched_true:
            matched_ious.append(iou_val)
            matched_pred.add(pi)
            matched_true.add(tj)

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_true - tp
    return matched_ious, tp, fp, fn


# ===================================================================
# Public API
# ===================================================================

def dice_score(pred, target, num_classes=None):
    """Dice coefficient (F1-score) for semantic segmentation.

    Paper Section 4.2, Eq:  Dice = 2×|Y ∩ Ŷ| / (|Y| + |Ŷ|)

    Args:
        pred:        (H, W) or (C, H, W) — predicted class labels or one-hot
        target:      (H, W) or (C, H, W) — ground truth
        num_classes: int or None. If None and inputs are (H, W),
                     infers from unique values (ignores 0=background).

    Returns:
        float  —  Dice score averaged over classes (excluding background).
                  For binary segmentation, returns a single scalar.
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    # Collapse one-hot to label maps if needed
    if pred.ndim == 3:
        pred = np.argmax(pred, axis=0)
    if target.ndim == 3:
        target = np.argmax(target, axis=0)

    if num_classes is None:
        classes = sorted(set(np.unique(pred)) | set(np.unique(target)))
        classes = [c for c in classes if c != 0]  # exclude background
        if not classes:
            classes = [1]  # [ASSUMPTION] binary case, only foreground class
    else:
        classes = list(range(1, num_classes))  # [ASSUMPTION] 0 = background

    scores = []
    for c in classes:
        pred_c = (pred == c).astype(np.float64)
        true_c = (target == c).astype(np.float64)
        inter = (pred_c * true_c).sum()
        denom = pred_c.sum() + true_c.sum()
        if denom == 0:
            continue  # skip class not present in either
        scores.append(2.0 * inter / denom)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def panoptic_quality(pred_inst, true_inst, pred_type=None, true_type=None,
                     match_iou=0.5):
    """Panoptic Quality for a single image.

    Paper Section 4.2:
      PQ = Σ_{(y,ŷ)∈TP} IoU(y,ŷ) / (|TP| + 0.5·|FP| + 0.5·|FN|)

    Matching rule: a predicted instance and a GT instance form a TP if
    they belong to the same class AND have IoU > match_iou (default 0.5).

    If pred_type/true_type are None, all instances are treated as
    belonging to a single class (standard instance segmentation).

    Args:
        pred_inst:  (H, W) int  — predicted instance ID map
        true_inst:  (H, W) int  — ground-truth instance ID map
        pred_type:  (H, W) int  — predicted semantic class map (optional)
        true_type:  (H, W) int  — ground-truth semantic class map (optional)
        match_iou:  float       — IoU threshold for matching

    Returns:
        pq:      float  — panoptic quality score
        sq:      float  — segmentation quality (mean IoU of matched pairs)
        rq:      float  — recognition quality  (|TP|/(|TP|+0.5|FP|+0.5|FN|))
        per_class: dict  — {cls_id: {'tp': N, 'fp': N, 'fn': N, 'iou_sum': float}}
    """
    pred_inst = _to_numpy(pred_inst, dtype=np.int32)
    true_inst = _to_numpy(true_inst, dtype=np.int32)
    if pred_type is not None:
        pred_type = _to_numpy(pred_type, dtype=np.int32)
    if true_type is not None:
        true_type = _to_numpy(true_type, dtype=np.int32)

    # Determine which classes are present
    has_type = (pred_type is not None) and (true_type is not None)
    if has_type:
        all_cls = set(np.unique(pred_type)) | set(np.unique(true_type))
        all_cls = {int(c) for c in all_cls if c != 0}
    else:
        all_cls = {1}  # single-class mode

    if not all_cls:
        # No foreground in either map → perfect (empty) score
        return 0.0, 0.0, 0.0, {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    per_class = {}

    for cls_id in sorted(all_cls):
        pred_masks = _instances_of_class(pred_inst, pred_type, cls_id)
        true_masks = _instances_of_class(true_inst, true_type, cls_id)

        matched_ious, tp_c, fp_c, fn_c = _match_by_iou(
            pred_masks, true_masks, match_iou
        )
        iou_sum_c = float(np.sum(matched_ious))

        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c
        total_iou_sum += iou_sum_c

        per_class[cls_id] = {
            "tp": tp_c, "fp": fp_c, "fn": fn_c, "iou_sum": iou_sum_c,
        }

    # Paper Section 4.2, Eq:
    denom = total_tp + 0.5 * total_fp + 0.5 * total_fn
    if denom == 0:
        pq = 0.0
        sq = 0.0
        rq = 0.0
    else:
        sq = total_iou_sum / max(total_tp, 1)
        rq = total_tp / denom
        pq = sq * rq

    return float(pq), float(sq), float(rq), per_class


def mpq_score(pred_list, true_list, pred_type_list=None, true_type_list=None,
              num_classes=None, match_iou=0.5):
    """Mean Panoptic Quality (mPQ) — Paper Section 4.2.

    Algorithm:
      1. For each image, compute PQ per class.
      2. Average PQ across images for each class.
      3. Average across classes.
      If a class is absent from an image (no GT and no pred), that
      (image, class) pair is skipped.

    Args:
        pred_list:       list of (H,W) int arrays — predicted instance ID maps
        true_list:       list of (H,W) int arrays — GT instance ID maps
        pred_type_list:  list of (H,W) int arrays — predicted semantic maps (optional)
        true_type_list:  list of (H,W) int arrays — GT semantic maps (optional)
        num_classes:     int — total number of semantic classes (incl. background)
        match_iou:       float

    Returns:
        float  — mPQ score
    """
    has_type = pred_type_list is not None and true_type_list is not None

    if num_classes is None:
        if has_type:
            all_c = set()
            for pt, tt in zip(pred_type_list, true_type_list):
                all_c |= set(np.unique(pt)) | set(np.unique(tt))
            num_classes = max(int(c) for c in all_c if c != 0) + 1
        else:
            num_classes = 2  # background + 1 class

    # Per-class accumulator: list of PQ values per image
    class_pq = defaultdict(list)  # cls_id → [pq_values]

    for i in range(len(pred_list)):
        pt = pred_type_list[i] if has_type else None
        tt = true_type_list[i] if has_type else None
        pq, _, _, per_cls = panoptic_quality(
            pred_list[i], true_list[i], pt, tt, match_iou
        )

        # Record per-class PQ for this image
        for cls_id in range(1, num_classes):
            info = per_cls.get(cls_id, {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})
            tp_c, fp_c, fn_c = info["tp"], info["fp"], info["fn"]
            denom = tp_c + 0.5 * fp_c + 0.5 * fn_c
            if denom > 0:
                sq_c = info["iou_sum"] / max(tp_c, 1)
                rq_c = tp_c / denom
                pq_c = sq_c * rq_c
                class_pq[cls_id].append(pq_c)
            # else: class absent → skip (do not contribute 0)

    # Average over classes (only those with at least one valid image)
    per_class_avg = []
    for cls_id in range(1, num_classes):
        vals = class_pq.get(cls_id, [])
        if vals:
            per_class_avg.append(float(np.mean(vals)))

    if not per_class_avg:
        return 0.0
    return float(np.mean(per_class_avg))


def mpq_plus_score(pred_list, true_list, pred_type_list=None, true_type_list=None,
                   num_classes=None, match_iou=0.5):
    """Aggregated mPQ (mPQ+) — Paper Section 4.2.

    Algorithm:
      1. Aggregate TP (sum of matched IoUs), FP, FN across ALL images.
      2. Compute PQ per class from aggregated stats.
      3. Average across classes.

    This prevents small classes in few images from being over-weighted,
    as can happen with standard mPQ.

    Args:
        pred_list:       list of (H,W) int arrays
        true_list:       list of (H,W) int arrays
        pred_type_list:  list of (H,W) int arrays (optional)
        true_type_list:  list of (H,W) int arrays (optional)
        num_classes:     int
        match_iou:       float

    Returns:
        float  — mPQ+ score
    """
    has_type = pred_type_list is not None and true_type_list is not None

    if num_classes is None:
        if has_type:
            all_c = set()
            for pt, tt in zip(pred_type_list, true_type_list):
                all_c |= set(np.unique(pt)) | set(np.unique(tt))
            num_classes = max(int(c) for c in all_c if c != 0) + 1
        else:
            num_classes = 2

    # Aggregated stats: {cls_id: {"tp": N, "fp": N, "fn": N, "iou_sum": float}}
    agg = {c: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
           for c in range(1, num_classes)}

    for i in range(len(pred_list)):
        pt = pred_type_list[i] if has_type else None
        tt = true_type_list[i] if has_type else None
        _, _, _, per_cls = panoptic_quality(
            pred_list[i], true_list[i], pt, tt, match_iou
        )
        for cls_id, info in per_cls.items():
            if cls_id in agg:
                agg[cls_id]["tp"] += info["tp"]
                agg[cls_id]["fp"] += info["fp"]
                agg[cls_id]["fn"] += info["fn"]
                agg[cls_id]["iou_sum"] += info["iou_sum"]

    # Compute PQ per class from aggregated stats
    per_class_pq = []
    for cls_id in range(1, num_classes):
        s = agg[cls_id]
        denom = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
        if denom > 0:
            sq = s["iou_sum"] / max(s["tp"], 1)
            rq = s["tp"] / denom
            pq = sq * rq
            per_class_pq.append(pq)
        # else: class never appears → exclude from average

    if not per_class_pq:
        return 0.0
    return float(np.mean(per_class_pq))
