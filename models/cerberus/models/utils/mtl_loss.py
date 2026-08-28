"""
mtl_loss.py — Multi-Task Learning loss module for Cerberus.

Paper reference:
    "Cerberus: A Multi-headed Model for Joint Semantic and Instance Segmentation"
    Section 3.2.2 — Overall Training Objective

    L = Σ_{t∈[1,T]} Σ_{ρ∈D_t}  L_t({Φ, Ψ_t}, x_ρ, y_ρ)        [Paper Eq.]

Replaces the inline loss computation in run_desc.py with a modular,
configurable loss module that reads per-head settings from paramset.yml.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.utils.loss_utils import xentropy_loss, dice_loss, focal_loss


# ---------------------------------------------------------------------------
# All output head names produced by NetDesc.forward (used for dummy matching).
# IMPORTANT: order is not significant — matching is done by head-name string.
# ---------------------------------------------------------------------------
_ALL_TASK_HEADS = [
    "Lumen-INST",
    "Gland-INST",
    "Nuclei-INST",
    "Nuclei-TYPE",
    "Patch-Class",
]

# Heads that carry pixel-wise TYPE (semantic class) information.
_TYPE_HEADS = {"Nuclei-TYPE"}

# Heads whose target is a single class label (Patch-Class).
_CLASSIFICATION_HEADS = {"Patch-Class"}


# ---------------------------------------------------------------------------
# Helper: class-aware weight map  (moved from run_desc.py for self-containment)
# ---------------------------------------------------------------------------
def _build_class_weight_map(target_map, class_weights):
    """Replace each class value in `target_map` with its configured weight.

    Args:
        target_map:  (N, C, H, W) or (N, H, W) tensor of class indices
        class_weights: dict {class_idx: weight}

    Returns:
        weight_map: same shape as target_map, with class values replaced by weights
    """
    wmap = torch.clone(target_map).detach()
    for cls_val, cls_weight in class_weights.items():
        wmap[target_map == cls_val] = cls_weight
    return wmap


# ---------------------------------------------------------------------------
# Helper: build binary (foreground) mask for Dice computation on TYPE heads
# ---------------------------------------------------------------------------
def _build_binary_mask(target_map):
    """Create a binary mask: 1 where annotation exists, 0 elsewhere.

    Paper Sec 3.2.2: for semantic type classification, loss is computed
    only within annotated (foreground) regions.
    """
    binary = torch.clone(target_map).detach()
    binary = (binary > 0).type(torch.float32)
    return binary


# ---------------------------------------------------------------------------
# Helper: extract per-sample dummy flags for a given head
# ---------------------------------------------------------------------------
def _head_flag(dummy_target, head_name):
    """Return a (N,) float tensor: 1.0 for samples that have GT for `head_name`.

    Args:
        dummy_target: np.ndarray (N, K) of str or None
        head_name:    e.g. 'Gland-INST'

    Returns:
        flag: torch.Tensor (N,) on the same device as pred/true
    """
    flag = np.any(dummy_target == head_name, axis=-1).astype(np.float32)
    return torch.from_numpy(flag)


# ===================================================================
# MTLLoss
# ===================================================================
class MTLLoss(nn.Module):
    """Multi-task loss that replaces the inline loss computation in train_step.

    [Paper Sec 3.2.2]  L = Σ_t w_t · L_t  where:
      - w_t is the task-level loss_weight from paramset.yml
      - L_t is the (possibly composite) per-head loss
      - samples without GT for task t are masked out via dummy_target

    Usage (inside train_step):
        loss_fn = MTLLoss(loss_kwargs)   # created once in get_config()
        ...
        total_loss, loss_dict = loss_fn(pred_dict, true_dict, dummy_target)
        total_loss.backward()

    pred_dict  — from NetDesc.forward:  {'Gland-INST': (N,C,H,W), ...}
    true_dict  — from collate_mtl_batch: {'Gland-INST': (N,1,H,W), ...,
                                           'Gland-INST#WEIGHT-MAP': (N,1,H,W), ...}
    dummy_target — np.ndarray (N, K) of str/None
    """

    # Map loss-name strings to actual functions
    _LOSS_FUNC = {
        "ce": xentropy_loss,
        "dice": dice_loss,
        "focal": focal_loss,
    }

    def __init__(self, loss_kwargs):
        """
        Args:
            loss_kwargs: the ``loss_kwargs`` section from paramset.yml, containing:
                - loss_info:    {head_name: {'weight': w, 'loss': {'ce': 1, 'dice': 1}}}
                - class_weight: {head_name: {class_idx: weight}}  (optional)
        """
        super().__init__()
        self.loss_info = loss_kwargs.get("loss_info", {})
        self.class_weights = loss_kwargs.get("class_weight", {})

        # Validate: every key in loss_info should be a known head
        for hname in self.loss_info:
            if hname not in _ALL_TASK_HEADS:
                raise KeyError(
                    f"Unknown task head '{hname}' in loss_info. "
                    f"Known heads: {_ALL_TASK_HEADS}"
                )

    def forward(self, pred_dict, true_dict, dummy_target):
        """Compute the multi-task total loss.

        Args:
            pred_dict:    OrderedDict  {head_name: Tensor (N, C, H, W)}
                          from NetDesc.forward output
            true_dict:    OrderedDict  {head_name: Tensor (N, C, H, W)}
                          from collate_mtl_batch; may contain #WEIGHT-MAP keys
            dummy_target: np.ndarray (N, K)  — each element is a head-name str or None

        Returns:
            total_loss:  scalar torch.Tensor (for .backward())
            loss_dict:   OrderedDict  {head_name: float} for logging/EMA
        """
        total_loss = torch.tensor(0.0, device=self._device(pred_dict))
        loss_dict = OrderedDict()

        # Only compute loss for heads that the model produced predictions for
        for head_name, head_pred in pred_dict.items():
            if head_name not in self.loss_info:
                # Head exists in model output but not configured for loss — skip
                continue

            head_cfg = self.loss_info[head_name]
            head_weight = head_cfg.get("weight", 1.0)
            head_losses = head_cfg.get("loss", {"ce": 1})

            # ---- Get targets and dummy mask -------------------------------
            head_true = true_dict[head_name]
            head_flag = _head_flag(dummy_target, head_name).to(head_pred.device)

            # ---- Pre-processing per head type -----------------------------
            prepped, aux = self._prep_head(
                head_name, head_pred, head_true, true_dict
            )

            # ---- Compute configured losses --------------------------------
            head_total = torch.tensor(0.0, device=head_pred.device)

            for loss_name, loss_coeff in head_losses.items():
                if loss_coeff == 0:
                    continue

                loss_func = self._LOSS_FUNC[loss_name]
                term = self._compute_single_loss(
                    head_name, loss_name, loss_func, prepped, aux, head_flag
                )
                head_total = head_total + term * loss_coeff

            # ---- Apply task weight [Paper Sec 3.2.2] ----------------------
            weighted = head_total * head_weight
            total_loss = total_loss + weighted

            # Log the un-weighted per-head loss (before task weight) for EMA
            loss_dict[head_name + "_loss"] = head_total.detach().cpu().item()

        loss_dict["overall_loss"] = total_loss.detach().cpu().item()
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Pre-processing: normalise shapes, build weight maps, etc.
    # ------------------------------------------------------------------
    def _prep_head(self, head_name, pred, true, true_dict):
        """Head-specific preprocessing.

        Returns:
            prepped: dict with 'pred' and 'true' in NCHW format
            aux:     dict with head-specific auxiliary tensors
                     (weight_map, binary_mask, nr_classes)
        """
        aux = {}

        # ---- Patch-Class: squeeze spatial dims ----------------------------
        if head_name in _CLASSIFICATION_HEADS:
            # Model output: (N, C, 1, 1), target: (N, 1, 1, 1)
            true = torch.squeeze(true)        # (N,)
            pred = torch.squeeze(pred)        # (N, C)
            aux["weight_map"] = torch.ones_like(true).float()  # (N,)
            return {"pred": pred, "true": true}, aux

        # ---- Segmentation heads -------------------------------------------
        nr_classes = pred.shape[1]  # inferred from output channels
        aux["nr_classes"] = nr_classes

        # Retrieve spatial weight map if available
        wmap_key = head_name + "#WEIGHT-MAP"
        if wmap_key in true_dict:
            weight_map = true_dict[wmap_key]         # (N, 1, H, W)
        else:
            weight_map = torch.ones_like(true)       # (N, 1, H, W)

        # ---- TYPE heads: class-weight override + binary mask --------------
        if head_name in _TYPE_HEADS:
            cw = self.class_weights.get(head_name, {})
            if cw:
                # Replace pixel class values with class weights
                weight_map = _build_class_weight_map(true, cw)
            # Binary mask: 1=foreground, 0=background — restricts Dice to
            # annotated regions only
            aux["binary_mask"] = _build_binary_mask(true)

        aux["weight_map"] = weight_map
        return {"pred": pred, "true": true}, aux

    # ------------------------------------------------------------------
    # Loss computation for a single loss term on a single head
    # ------------------------------------------------------------------
    def _compute_single_loss(self, head_name, loss_name, loss_func, prepped, aux, head_flag):
        """
        Returns:
            scalar loss for this (head, loss_function) pair, with dummy samples masked out.
        """
        pred = prepped["pred"]
        true = prepped["true"]
        weight_map = aux.get("weight_map", None)
        binary_mask = aux.get("binary_mask", None)
        nr_classes = aux.get("nr_classes", None)

        # ---- Dice loss ----------------------------------------------------
        if loss_name == "dice":
            # One-hot encode true: (N, 1, H, W) → (N, nr_classes, H, W)
            true_1h = F.one_hot(
                torch.squeeze(true.to(torch.int64), dim=1), num_classes=nr_classes
            )
            true_1h = true_1h.permute(0, 3, 1, 2).float()    # N C H W

            pred_soft = torch.softmax(pred, dim=1)             # N C H W

            # Exclude background class (channel 0)
            true_1h = true_1h[:, 1:, :, :]
            pred_soft = pred_soft[:, 1:, :, :]

            # Compute per-sample Dice (sum over classes, NOT over batch)
            # dice_loss in loss_utils sums over (0,2,3) — we want per-sample
            term = self._dice_per_sample(
                pred_soft, true_1h, mask=binary_mask
            )  # (N,)

            # [Paper Sec 3.2.2] mask out samples without this task
            term = term * head_flag.to(term.device)
            term = term.sum() / (head_flag.sum() + 1.0e-8)
            return term

        # ---- CE / Focal loss ----------------------------------------------
        elif loss_name in ("ce", "focal"):
            # xentropy_loss / focal_loss with reduction='none' returns
            # per-pixel loss: (N, H, W) for segmentation, (N,) for classification
            loss_raw = loss_func(true, pred, reduction=False)

            if head_name in _CLASSIFICATION_HEADS:
                # Patch-Class: loss is (N,)
                if weight_map is not None:
                    loss_raw = loss_raw * weight_map.to(loss_raw.device)
                # [Paper Sec 3.2.2]
                term = torch.sum(loss_raw * head_flag.to(loss_raw.device)) / (
                    head_flag.sum() + 1.0e-8
                )
            else:
                # Segmentation: loss is (N, H, W)
                # Apply spatial weight map (first channel)
                if weight_map is not None and weight_map.ndim >= 3:
                    wmap = weight_map[:, 0]  # (N, H, W)
                    loss_raw = loss_raw * wmap.to(loss_raw.device)
                # Reduce spatial dims → per-sample loss (N,)
                loss_sample = torch.mean(
                    loss_raw.reshape(loss_raw.shape[0], -1), dim=1
                )
                # [Paper Sec 3.2.2] mask out samples without this task
                term = torch.sum(loss_sample * head_flag.to(loss_sample.device)) / (
                    head_flag.sum() + 1.0e-8
                )
            return term

        else:
            raise ValueError(f"Unknown loss type: {loss_name}")

    # ------------------------------------------------------------------
    # Per-sample Dice (avoids aggregating over dummy samples)
    # ------------------------------------------------------------------
    @staticmethod
    def _dice_per_sample(pred, true, smooth=1e-3, mask=None):
        """Compute Dice loss per sample (summed over classes).

        Args:
            pred:  (N, C, H, W)  softmax probabilities
            true:  (N, C, H, W)  one-hot ground truth
            mask:  (N, 1, H, W)  optional binary foreground mask

        Returns:
            loss:  (N,)  — Dice loss per sample
        """
        if mask is not None:
            inse = torch.sum(pred * true * mask, dim=(2, 3))  # (N, C)
            card_l = torch.sum(pred * mask, dim=(2, 3))       # (N, C)
            card_r = torch.sum(true * mask, dim=(2, 3))       # (N, C)
        else:
            inse = torch.sum(pred * true, dim=(2, 3))
            card_l = torch.sum(pred, dim=(2, 3))
            card_r = torch.sum(true, dim=(2, 3))

        loss = 1.0 - (2.0 * inse + smooth) / (card_l + card_r + smooth)
        loss = torch.sum(loss, dim=1)  # sum over classes → (N,)
        return loss

    # ------------------------------------------------------------------
    # Utility: get device from pred_dict
    # ------------------------------------------------------------------
    @staticmethod
    def _device(pred_dict):
        for v in pred_dict.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Convenience: build MTLLoss from paramset.yml path directly
# ---------------------------------------------------------------------------
def build_mtl_loss(paramset_yml_path=None, loss_kwargs=None):
    """Factory for MTLLoss.

    Args:
        paramset_yml_path: path to paramset.yml (reads loss_kwargs section)
        loss_kwargs:       or, pass the loss_kwargs dict directly

    Returns:
        MTLLoss instance
    """
    if loss_kwargs is None and paramset_yml_path is not None:
        import yaml
        with open(paramset_yml_path, "r") as f:
            cfg = yaml.full_load(f)
        loss_kwargs = cfg["loss_kwargs"]
    if loss_kwargs is None:
        raise ValueError("Provide either paramset_yml_path or loss_kwargs")
    return MTLLoss(loss_kwargs)
