"""
train_loader.py — Training data loader for Cerberus multi-task learning.

Paper reference:
    "Cerberus: A Multi-headed Model for Joint Semantic and Instance Segmentation"
    Section 3.2.1 — Training Strategy

The loader implements:
  - TaskDataset: loads samples from multiple datasets (gland, lumen, nuclei, tissue-type)
  - TaskSampler: super-task-aware batch sampling (Paper Sec 3.2.1)
  - collate_mtl_batch: assembles mixed-task batches for train_step consumption
"""

import collections
import csv
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from loader.targets import gen_targets
from misc.utils import cropping_center

# ---------------------------------------------------------------------------
# Paper Sec 3.2.1: All task output heads in Cerberus.
# Order is significant — must match the decoder head names in net_desc.py.
# ---------------------------------------------------------------------------
ALL_TASK_HEADS = [
    "Lumen-INST",
    "Gland-INST",
    "Nuclei-INST",
    "Nuclei-TYPE",
    "Patch-Class",
]

# Map each head to its super-task category (seg vs cls)
_HEAD_TO_SUPER = {
    "Lumen-INST": "seg",
    "Gland-INST": "seg",
    "Nuclei-INST": "seg",
    "Nuclei-TYPE": "seg",
    "Patch-Class": "cls",
}

# ---------------------------------------------------------------------------
# YAML helpers (avoid hard dependency on PyYAML for simple reads)
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml

    def _load_yaml(path):
        with open(path, "r") as f:
            return _yaml.full_load(f)

except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml")


# ---------------------------------------------------------------------------
# Helper: resolve file path with single or multiple extensions
# ---------------------------------------------------------------------------
def _resolve_file_path(dir_path, base_name, ext):
    """Return first existing path matching any extension.

    ``ext`` can be a single string (``'.png'``) or a list
    (``['.png', '.jpg', '.mat']``). Returns full path or None if none found.
    """
    exts = [ext] if isinstance(ext, str) else ext
    for e in exts:
        full = os.path.join(dir_path, base_name + e)
        if os.path.exists(full):
            return full
    return None


def _image_size(img_path):
    """Return (h, w) of an image file, reading only the header when possible.

    Falls back to a full cv2 decode if PIL is unavailable.
    """
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return im.size[1], im.size[0]  # (h, w)
    except Exception:
        img = cv2.imread(img_path)
        if img is None:
            return 0, 0
        return img.shape[0], img.shape[1]


# ---------------------------------------------------------------------------
# CSV split-info helpers
# ---------------------------------------------------------------------------
def _read_csv(path):
    """Read a CSV file and return list of dicts."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _resolve_split_column(rows, split, fold, nr_splits):
    """Find the column in CSV that corresponds to (split, fold).

    The CSV may use different naming conventions:
      - Single 'split' column with values 'train'/'valid'/'test'
      - Per-fold columns like 'fold_0', 'fold_1', 'fold_2'
      - Columns named 'train', 'valid', 'test' directly
    Returns the column name, or None if not found.
    """
    if not rows:
        return None
    headers = list(rows[0].keys())

    # Convention 1: single 'split' column
    if "split" in headers:
        return "split"

    # Convention 1b: 'Split' column with numeric codes (1=train, 2=valid, 3=test)
    if "Split" in headers:
        return "Split"

    # Convention 2: per-fold columns 'fold_0', 'fold_1', ...
    fold_col = f"fold_{fold}"
    if fold_col in headers:
        return fold_col

    # Convention 3: columns named directly as splits
    if split in headers:
        return split

    # Fallback: try any column containing the fold number
    for h in headers:
        if f"fold_{fold}" in h.lower() or (f"_{fold}" in h and "fold" in h.lower()):
            return h

    return None


def _filter_rows(rows, split, fold, nr_splits):
    """Filter CSV rows for the given split and fold."""
    col = _resolve_split_column(rows, split, fold, nr_splits)
    if col is None:
        # No split column found — include all rows as a fallback
        return rows

    if col == "split":
        return [r for r in rows if r.get("split", "").strip().lower() == split]
    elif col == "Split":
        # Cerberus sample-dataset convention: 1=train, 2=valid, 3=test
        code_map = {"train": "1", "valid": "2", "test": "3"}
        want = code_map.get(split, "1")
        return [r for r in rows if str(r.get("Split", "")).strip() == want]
    else:
        return [r for r in rows if r.get(col, "").strip().lower() == split]


def _get_filename(row):
    """Extract filename base (without extension) from a CSV row."""
    if "filename" in row:
        return row["filename"]
    # Fallback: use the first column
    return row[list(row.keys())[0]]


# ---------------------------------------------------------------------------
# Lightweight augmentation wrapper (uses augs.py functions under the hood)
# ---------------------------------------------------------------------------
class ComposeAugmentations:
    """Compose a sequence of augmentation functions.

    Each function signature:  func(img, targets) -> (img, targets)
    where `targets` is an OrderedDict of {head_name: np.ndarray (HWC)}.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, targets):
        for t in self.transforms:
            img, targets = t(img, targets)
        return img, targets


def _flip_horizontal(img, targets):
    """Random horizontal flip (p=0.5)."""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        targets = OrderedDict(
            (k, cv2.flip(v, 1) if v.ndim >= 2 else v) for k, v in targets.items()
        )
    return img, targets


def _flip_vertical(img, targets):
    """Random vertical flip (p=0.5)."""
    if random.random() > 0.5:
        img = cv2.flip(img, 0)
        targets = OrderedDict(
            (k, cv2.flip(v, 0) if v.ndim >= 2 else v) for k, v in targets.items()
        )
    return img, targets


def _rotate90(img, targets):
    """Random 90-degree rotation (p=0.5)."""
    if random.random() > 0.5:
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k)
            targets = OrderedDict(
                (kn, np.rot90(v, k) if v.ndim >= 2 else v)
                for kn, v in targets.items()
            )
    return img, targets


def _brightness_jitter(img, targets, delta=30):
    """Random brightness jitter (uses augs.py style)."""
    if random.random() > 0.5:
        value = random.uniform(-delta, delta)
        img = np.clip(img.astype(np.float32) + value, 0, 255).astype(np.uint8)
    return img, targets


def _contrast_jitter(img, targets, alpha_range=(0.8, 1.2)):
    """Random contrast jitter."""
    if random.random() > 0.5:
        alpha = random.uniform(*alpha_range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip(img.astype(np.float32) * alpha + mean * (1 - alpha), 0, 255).astype(np.uint8)
    return img, targets


def _hue_jitter(img, targets, delta=10):
    """Random hue jitter (RGB space)."""
    if random.random() > 0.5:
        shift = random.uniform(-delta, delta)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img, targets


def _saturation_jitter(img, targets, value_range=(-0.2, 0.2)):
    """Random saturation jitter."""
    if random.random() > 0.5:
        value = 1.0 + random.uniform(*value_range)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.clip(img.astype(np.float32) * value + (gray * (1 - value))[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return img, targets


def _gaussian_blur(img, targets, max_ksize=3):
    """Random Gaussian blur."""
    if random.random() > 0.5:
        ksize = random.randrange(0, max_ksize) * 2 + 1
        if ksize >= 3:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img, targets


# Default augmentation pipeline for segmentation images (spatial + colour)
SEG_AUGMENTATIONS = ComposeAugmentations(
    [
        _flip_horizontal,
        _flip_vertical,
        _rotate90,
        _brightness_jitter,
        _contrast_jitter,
        _hue_jitter,
        _saturation_jitter,
        _gaussian_blur,
    ]
)

# Default augmentation pipeline for classification images (spatial + colour, no target maps)
CLS_AUGMENTATIONS = ComposeAugmentations(
    [
        _flip_horizontal,
        _flip_vertical,
        _rotate90,
        _brightness_jitter,
        _contrast_jitter,
        _hue_jitter,
        _saturation_jitter,
        _gaussian_blur,
    ]
)


# ---------------------------------------------------------------------------
# TaskDataset
# ---------------------------------------------------------------------------
class TaskDataset(Dataset):
    """Multi-task dataset that aggregates samples from all sub-datasets.

    Paper Sec 3.2.1: segmentation tasks (input 448x448) and tissue-type
    classification (input 144x144) are treated as different super tasks.

    For each sample, returns a dict:
        {
            'img':       np.ndarray (H, W, C), uint8
            'task_name': str,  e.g. 'Gland', 'Nuclei', 'Lumen', 'Patch-Class'
            'targets':   OrderedDict of {head_name: np.ndarray (H, W, C)}
            'has_flags': list of head-name strings that have real GT
        }
    """

    def __init__(
        self,
        dataset_yml_path,
        paramset_yml_path,
        split="train",
        fold=0,
        augment=True,
    ):
        """
        Args:
            dataset_yml_path:  path to dataset.yml
            paramset_yml_path: path to models/paramset.yml
            split:             'train', 'valid', or 'test'
            fold:              cross-validation fold index (0-based)
            augment:           whether to apply data augmentation
        """
        self.split = split
        self.fold = fold
        self.augment = augment

        # ---- Load configuration -------------------------------------------
        dataset_cfg = _load_yaml(dataset_yml_path)
        paramset_cfg = _load_yaml(paramset_yml_path)

        self.input_shape = paramset_cfg["dataset_kwargs"]["input_shape"]       # 448
        self.output_shape = paramset_cfg["dataset_kwargs"]["output_shape"]     # 448
        self.class_input_shape = paramset_cfg["dataset_kwargs"]["class_input_shape"]  # 144
        self.target_code_map = paramset_cfg["dataset_kwargs"]["req_target_code"]

        # ---- Build sample lists -------------------------------------------
        # self.samples:       list of sample dicts (one per image/annotation)
        # self.task_indices:  dict {dataset_name: list of indices into self.samples}
        # self.task_to_heads: dict {dataset_name: list of head names produced}
        self.samples = []
        self.task_indices = collections.defaultdict(list)
        self.task_to_heads = {}

        for dataset_name, cfg in dataset_cfg.items():
            if dataset_name == "tissue-type":
                self._load_cls_dataset(dataset_name, cfg)
            elif "ann_info" in cfg:
                self._load_seg_dataset(dataset_name, cfg)
            # Skip datasets without ann_info and not tissue-type (e.g. metadata-only entries)

        self.num_tasks = len(self.task_indices)

    # ------------------------------------------------------------------
    # Private: segmentation dataset loading
    # ------------------------------------------------------------------
    def _load_seg_dataset(self, name, cfg):
        """Load a segmentation dataset (gland / lumen / nuclei)."""
        rows = _read_csv(cfg["split_info"])
        rows = _filter_rows(rows, self.split, self.fold, cfg.get("nr_splits", 1))

        img_dir = cfg["img_dir"]
        img_ext = cfg["img_ext"]
        ann_info = cfg["ann_info"]

        # Determine which heads this dataset produces
        heads = []
        for head_name in ALL_TASK_HEADS:
            task_prefix = head_name.split("-")[0]  # 'Gland', 'Lumen', 'Nuclei'
            if task_prefix == name.capitalize():
                heads.append(head_name)
        self.task_to_heads[name] = heads

        for row in rows:
            fname = _get_filename(row)
            img_path = _resolve_file_path(img_dir, fname, img_ext)
            if img_path is None:
                continue

            # Paper requirement: keep only images with side >= input_shape (448);
            # smaller images are skipped entirely (they cannot be cropped).
            h, w = _image_size(img_path)
            if h < self.input_shape or w < self.input_shape:
                continue

            # Collect annotation paths
            ann_paths = {}
            for ann_key, ann_info_item in ann_info.items():
                ann_suffix = ann_info_item.get("ann_suffix", "")
                ann_path = os.path.join(
                    ann_info_item["ann_dir"], fname + ann_suffix + ann_info_item["ann_ext"]
                )
                if os.path.exists(ann_path):
                    ann_paths[ann_key] = ann_path

            if not ann_paths:
                continue

            sample = {
                "type": "seg",
                "dataset": name,
                "img_path": img_path,
                "ann_paths": ann_paths,
                "ann_info": ann_info,
            }
            idx = len(self.samples)
            self.samples.append(sample)
            self.task_indices[name].append(idx)

    # ------------------------------------------------------------------
    # Private: classification dataset loading
    # ------------------------------------------------------------------
    def _load_cls_dataset(self, name, cfg):
        """Load the tissue-type classification dataset.

        Tissue-type images are pre-cropped patches; the label comes from
        the split CSV or from the image's parent directory name.
        """
        rows = _read_csv(cfg["split_info"])
        rows = _filter_rows(rows, self.split, self.fold, cfg.get("nr_splits", 1))

        img_dir = cfg["img_dir"]
        img_ext = cfg["img_ext"]
        type_names = cfg.get("type_names", [])

        self.task_to_heads[name] = ["Patch-Class"]

        for row in rows:
            fname = _get_filename(row)
            # Look in flat dir first, then in class subdirectories
            img_path = _resolve_file_path(img_dir, fname, img_ext)
            label = None
            if img_path is None and os.path.isdir(img_dir):
                for sub in sorted(os.listdir(img_dir)):
                    sub_path = _resolve_file_path(
                        os.path.join(img_dir, sub), fname, img_ext
                    )
                    if sub_path is not None:
                        img_path = sub_path
                        try:
                            label = type_names.index(sub)
                        except ValueError:
                            label = 0
                        break
            if img_path is None:
                continue

            # Paper requirement: keep only images with side >= class_input_shape
            # (144) for classification; smaller ones are skipped.
            h, w = _image_size(img_path)
            if h < self.class_input_shape or w < self.class_input_shape:
                continue

            # Label: from CSV 'label' column, else from subdirectory (or 0)
            if label is None:
                if "label" in row:
                    label = int(row["label"])
                else:
                    parent = os.path.basename(os.path.dirname(img_path))
                    try:
                        label = type_names.index(parent)
                    except ValueError:
                        label = 0

            sample = {
                "type": "cls",
                "dataset": name,
                "img_path": img_path,
                "label": label,
                "num_classes": len(type_names),
            }
            idx = len(self.samples)
            self.samples.append(sample)
            self.task_indices[name].append(idx)

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample["type"] == "seg":
            return self._get_seg_item(sample)
        else:
            return self._get_cls_item(sample)

    # ------------------------------------------------------------------
    # Segmentation sample
    # ------------------------------------------------------------------
    def _get_seg_item(self, sample):
        """Load a segmentation sample: read image + .mat annotation,
        generate target maps via gen_targets(), apply augmentations.
        """
        # ---- Read image ---------------------------------------------------
        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---- Read .mat annotation -----------------------------------------
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError(
                "scipy is required for reading .mat annotations. "
                "Install with: pip install scipy"
            )

        ann_channels = []
        channel_names = []
        for ann_key, ann_path in sample["ann_paths"].items():
            mat = sio.loadmat(ann_path)
            # Find the data key (skip MATLAB metadata keys starting with '__')
            data_keys = sorted([k for k in mat if not k.startswith("__")])
            if not data_keys:
                raise ValueError(f"No data found in {ann_path}")

            # Separate pixel-level maps (H,W) from metadata (N,*)
            pixel_maps = {}
            metadata = {}
            for dk in data_keys:
                v = mat[dk]
                if v.ndim >= 2 and v.shape[0] > 10 and v.shape[1] > 10:
                    pixel_maps[dk] = v
                else:
                    metadata[dk] = v

            # Build channel array: match channel_code to pixel map keys
            ch_code = sample["ann_info"][ann_key].get("channel_code", [])
            ch_maps = []
            for ch_name in ch_code:
                # Try matching by key name (case-insensitive)
                matched = None
                for pk, pv in pixel_maps.items():
                    if ch_name.lower() in pk.lower():
                        matched = pv
                        break
                if matched is not None:
                    ch_maps.append(matched)
                    channel_names.append(ch_name)
                elif ch_name == 'TYPE' and 'class' in metadata:
                    # Derive TYPE pixel map from per-instance class labels
                    inst_map = list(pixel_maps.values())[0] if pixel_maps else None
                    if inst_map is not None:
                        inst_ids = metadata.get('id', None)
                        classes = metadata['class'].flatten().astype(np.int32)
                        if inst_ids is not None:
                            inst_ids = inst_ids.flatten().astype(np.int32)
                        type_map = np.zeros_like(inst_map, dtype=np.int32)
                        # Map instance ID -> class label
                        unique_insts = np.unique(inst_map)
                        for idx, inst_id in enumerate(unique_insts):
                            if inst_id == 0:
                                continue
                            if inst_ids is not None:
                                mask = inst_ids == inst_id
                                if mask.any():
                                    cls = classes[mask][0]
                                    type_map[inst_map == inst_id] = cls
                        ch_maps.append(type_map)
                        channel_names.append(ch_name)
                    else:
                        ch_maps.append(np.zeros_like(list(pixel_maps.values())[0]))
                        channel_names.append(ch_name)
                else:
                    # Fill with zeros if channel not found
                    if pixel_maps:
                        ref = list(pixel_maps.values())[0]
                        ch_maps.append(np.zeros_like(ref))
                    else:
                        ch_maps.append(np.zeros((self.input_shape, self.input_shape)))
                    channel_names.append(ch_name)

            ann_channels = [m[..., np.newaxis] if m.ndim == 2 else m for m in ch_maps]

        # Stack channels -> HxWxC  (ann_channels already (H,W,1) each)
        if len(ann_channels) > 1:
            ann = np.concatenate(ann_channels, axis=-1)
        else:
            ann = ann_channels[0][..., np.newaxis] if ann_channels[0].ndim == 2 else ann_channels[0]

        # ---- Crop image + annotation to input shape -----------------------
        # Paper Sec 3.2.1: training uses random crop of input_shape (448),
        # aligned between image and annotation; validation/test use center crop.
        h, w = img.shape[:2]
        crop_h = crop_w = self.input_shape  # 448
        ann_h, ann_w = ann.shape[:2]
        aligned = (ann_h == h and ann_w == w)

        if self.augment:
            y0 = random.randint(0, h - crop_h)
            x0 = random.randint(0, w - crop_w)
        else:
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
        img = img[y0:y0 + crop_h, x0:x0 + crop_w]
        if aligned:
            # Same offset for annotation — keeps image & target aligned.
            ann = ann[y0:y0 + crop_h, x0:x0 + crop_w]
        else:
            # Annotation resolution differs from the image: fall back to a
            # separate center crop so the annotation stays valid.
            ann = cropping_center(ann, [crop_h, crop_w])

        # ---- Build channel_to_target mapping for this dataset --------------
        dataset_name = sample["dataset"]
        task_prefix = dataset_name.capitalize()  # Gland / Lumen / Nuclei

        # Paper Sec 3.2.1: channel_to_target uses short channel names
        #   (e.g., 'INST', 'TYPE'), NOT head names (e.g., 'Gland-INST').
        #   gen_targets matches against channel=channel_names.
        channel_to_target = {}
        for head_name, tg_code in self.target_code_map.items():
            prefix = head_name.split("-")[0]
            if prefix == task_prefix:
                ch_type = head_name.split("-", 1)[1]  # 'INST' or 'TYPE'
                channel_to_target[ch_type] = tg_code

        # ---- Generate targets via gen_targets -----------------------------
        # gen_unet_weight_map=False: skip the O(n^2) distance-transform weight
        # map (the main data-loading bottleneck). The 3-channel bg/inner/contour
        # targets are still produced, so pretrained weights stay aligned.
        targets, has_flags = gen_targets(
            ann,
            channel=channel_names,
            channel_to_target=channel_to_target,
            crop_shape=[self.output_shape, self.output_shape],
            task_mode="seg",
            gen_unet_weight_map=False,
        )
        # gen_targets returns SHORT keys/flags (e.g. 'INST', 'TYPE', 'INST#WEIGHT-MAP').
        # Convert them to FULL head names (e.g. 'Gland-INST', 'Nuclei-TYPE') to
        # match what collate_mtl_batch / train_step / NetDesc.forward expect.
        full_targets = OrderedDict()
        for k, v in targets.items():
            full_targets[task_prefix + "-" + k] = v
        full_flags = [
            None if f is None else task_prefix + "-" + f for f in has_flags
        ]

        # ---- Augmentation -------------------------------------------------
        if self.augment:
            img, full_targets = SEG_AUGMENTATIONS(img, full_targets)

        return {
            "img": img,             # HxWxC uint8
            "task_name": task_prefix,
            "targets": full_targets,     # OrderedDict of {head_name: HxW (or HxWxC)}
            "has_flags": full_flags, # list of head-name strings / None
        }

    # ------------------------------------------------------------------
    # Classification sample (tissue-type -> Patch-Class)
    # ------------------------------------------------------------------
    def _get_cls_item(self, sample):
        """Load a tissue-type classification sample."""
        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = sample["label"]

        # Resize/crop to class input shape (144x144 per paper).
        # Training: random crop; validation/test: center crop.
        crop_size = self.class_input_shape
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            # Fallback for images smaller than 144 (normally filtered at load)
            new_h = max(h, crop_size)
            new_w = max(w, crop_size)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = img.shape[:2]
        if self.augment:
            y0 = random.randint(0, h - crop_size)
            x0 = random.randint(0, w - crop_size)
            img = img[y0:y0 + crop_size, x0:x0 + crop_size]
        else:
            img = cropping_center(img, [crop_size, crop_size])

        # Targets: Patch-Class is a scalar label
        targets = OrderedDict()
        targets["Patch-Class"] = np.array(label, dtype=np.int32)
        has_flags = ["Patch-Class"]

        # ---- Augmentation -------------------------------------------------
        if self.augment:
            img, targets = CLS_AUGMENTATIONS(img, targets)

        return {
            "img": img,
            "task_name": "Patch-Class",
            "targets": targets,
            "has_flags": has_flags,
        }


# ---------------------------------------------------------------------------
# TaskSampler
# ---------------------------------------------------------------------------
class TaskSampler(Sampler):
    """Sampler implementing the Cerberus multi-task sampling strategy.

    Paper Sec 3.2.1:
      "segmentation tasks (input dimensions 448×448) and tissue type
       classification (input dimensions 144×144) are treated as different
       super tasks, which are selected with probabilities of 0.7 and 0.3
       respectively."

      "A mixed batch contains patches from multiple tasks, whereas a fixed
       batch contains data from a single task."

      "In all experiments, we set p_t = 1/T."

      "Once a task is selected, a patch is randomly chosen from the
       corresponding dataset."

    The sampler produces indices into TaskDataset. Each batch:
      1) Paper Sec 3.2.1: super task is selected (seg p=0.7, cls p=0.3)
      2) Within the super task, each specific task is chosen uniformly (p_t=1/T)
      3) A random sample from the chosen task is added to the batch
      4) Repeat until batch is full; all samples share the same input size
    """

    def __init__(self, dataset, batch_size, mixed=True):
        """
        Args:
            dataset:    TaskDataset instance
            batch_size: number of samples per batch
            mixed:      if True (default), batches mix multiple tasks of same super-task;
                        if False, each batch contains only one task (fixed batch).
        """
        if not isinstance(dataset, TaskDataset):
            raise TypeError("TaskSampler requires a TaskDataset instance")

        self.dataset = dataset
        self.batch_size = batch_size
        self.mixed = mixed

        # Group task indices by super-task (skip tasks with no samples)
        self.seg_tasks = {}
        self.cls_tasks = {}

        for task_name, indices in dataset.task_indices.items():
            if len(indices) == 0:
                continue  # skip tasks with no data (e.g., empty lumen)
            sup = _HEAD_TO_SUPER.get(task_name, None)
            if sup is None:
                heads = dataset.task_to_heads.get(task_name, [])
                sup = _HEAD_TO_SUPER.get(heads[0], "seg") if heads else "seg"
            if sup == "seg":
                self.seg_tasks[task_name] = indices
            else:
                self.cls_tasks[task_name] = indices

        # Compute number of batches per epoch (approximate)
        total = len(dataset)
        self.num_batches = max(1, total // batch_size)

    def __iter__(self):
        indices = []

        for _ in range(self.num_batches):
            # Paper Sec 3.2.1: super task selection (p_seg=0.7, p_cls=0.3)
            # Fall back if one pool is empty
            if self.seg_tasks and self.cls_tasks:
                use_seg = random.random() < 0.7
            elif self.seg_tasks:
                use_seg = True
            elif self.cls_tasks:
                use_seg = False
            else:
                continue

            task_pool = self.seg_tasks if use_seg else self.cls_tasks
            task_names = list(task_pool.keys())

            if self.mixed:
                for _ in range(self.batch_size):
                    tname = random.choice(task_names)
                    t_indices = task_pool[tname]
                    if t_indices:
                        indices.append(random.choice(t_indices))
            else:
                # Fixed batch — all samples from a single task
                tname = random.choice(task_names)
                t_indices = task_pool[tname]
                if len(t_indices) >= self.batch_size:
                    indices.extend(random.sample(t_indices, self.batch_size))
                elif t_indices:
                    indices.extend(random.choices(t_indices, k=self.batch_size))

        return iter(indices)

    def __len__(self):
        return self.num_batches * self.batch_size


# ---------------------------------------------------------------------------
# collate_mtl_batch
# ---------------------------------------------------------------------------
def collate_mtl_batch(batch_list):
    """Collate function: list of TaskDataset.__getitem__ dicts → batch dict.

    Produces the exact format expected by `train_step()` in run_desc.py:

        {
            'img':           torch.Tensor (B, H, W, C) – NHWC, float32
            'dummy_target':  np.ndarray (B, num_heads)  – head-name strings or None
            'Lumen-INST':    torch.Tensor (B, H, W, C)  – NHWC, float32 (zero-filled if no GT)
            'Gland-INST':    ...
            'Nuclei-INST':   ...
            'Nuclei-TYPE':   ...
            'Patch-Class':   torch.Tensor (B, 1, 1, 1) – NHWC, float32 (zero-filled if no GT)
            ...plus any WEIGHT-MAP heads...
        }

    Key invariant: ALL heads in ALL_TASK_HEADS (and any WEIGHT-MAP heads
    appearing in the batch) MUST be present as keys, even if the batch has
    no samples with GT for that head. Missing heads are filled with zeros
    and marked as None in dummy_target.
    """
    batch_size = len(batch_list)
    num_heads = len(ALL_TASK_HEADS)

    # ---- 1. Collect images, pad to max spatial dims -----------------------
    img_list = []
    for s in batch_list:
        img = s["img"]
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img_list.append(img)

    max_h = max(im.shape[0] for im in img_list)
    max_w = max(im.shape[1] for im in img_list)

    padded_imgs = []
    for im in img_list:
        ph = max_h - im.shape[0]
        pw = max_w - im.shape[1]
        if ph > 0 or pw > 0:
            im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="constant")
        padded_imgs.append(im)
    img_batch = np.stack(padded_imgs, axis=0)  # B H W C

    # ---- 2. Build dummy_target -------------------------------------------
    # Shape: (B, num_heads). Each entry is a head-name str or None.
    dummy_target = np.full((batch_size, num_heads), None, dtype=object)
    for i, s in enumerate(batch_list):
        has_flags = s["has_flags"]
        for j, hname in enumerate(ALL_TASK_HEADS):
            if hname in has_flags:
                dummy_target[i, j] = hname

    # ---- 3. Collect all unique target keys across the batch ---------------
    all_target_keys = set()
    for s in batch_list:
        all_target_keys.update(s["targets"].keys())

    # Make sure all mandatory heads are present
    all_target_keys.update(ALL_TASK_HEADS)

    # ---- 4. Build target tensors per key --------------------------------
    batch_dict = {
        "img": torch.from_numpy(img_batch.copy()).float(),
        "dummy_target": dummy_target,
    }

    for key in sorted(all_target_keys):
        # Gather targets for this key across all samples
        gathered = []
        have_any = False
        ref_shape = None  # (H, W, C) of the first valid target

        for s in batch_list:
            t = s["targets"].get(key, None)
            if t is not None:
                have_any = True
                # Ensure target is at least 2D (H, W) with channel dim
                if isinstance(t, (int, np.integer)):
                    t = np.array([t], dtype=np.float32)
                if t.ndim == 0:
                    t = np.array([t.item()], dtype=np.float32)
                if t.ndim == 1:
                    t = t[np.newaxis, :]  # 1 x C
                if t.ndim == 2:
                    t = t[..., np.newaxis]  # H x W x 1
                if ref_shape is None:
                    ref_shape = t.shape
                gathered.append(t.astype(np.float32))
            else:
                gathered.append(None)

        if not have_any:
            # All samples missing this key — fill with zeros
            # Use img spatial dims for seg heads, 1x1 for Patch-Class
            if key == "Patch-Class":
                default_shape = (1, 1, 1)
            else:
                default_shape = (max_h, max_w, 1)

            batch_tensor = np.zeros((batch_size,) + default_shape, dtype=np.float32)
        else:
            # Determine output spatial shape — use max across all gathered
            out_h = max((t.shape[0] for t in gathered if t is not None), default=1)
            out_w = max((t.shape[1] for t in gathered if t is not None), default=1)
            out_c = max((t.shape[2] for t in gathered if t is not None), default=1)

            padded = []
            for t in gathered:
                if t is None:
                    padded.append(np.zeros((out_h, out_w, out_c), dtype=np.float32))
                else:
                    ph = out_h - t.shape[0]
                    pw = out_w - t.shape[1]
                    pc = out_c - t.shape[2]
                    if ph > 0 or pw > 0 or pc > 0:
                        t = np.pad(t, ((0, ph), (0, pw), (0, pc)), mode="constant")
                    padded.append(t.astype(np.float32))

            batch_tensor = np.stack(padded, axis=0)

        batch_dict[key] = torch.from_numpy(batch_tensor.copy()).float()

    return batch_dict


# ---------------------------------------------------------------------------
# Convenience: build DataLoader directly
# ---------------------------------------------------------------------------
def build_train_dataloader(
    dataset_yml_path,
    paramset_yml_path,
    split="train",
    fold=0,
    batch_size=12,
    mixed=True,
    augment=True,
    num_workers=0,
):
    """Build a DataLoader for Cerberus training/validation.

    Args:
        dataset_yml_path:  path to dataset.yml
        paramset_yml_path: path to paramset.yml
        split:             'train' / 'valid' / 'test'
        fold:              cross-validation fold (0-based)
        batch_size:        batch size
        mixed:             if True, use mixed-task batches (Paper Sec 3.2.1)
        augment:           enable data augmentation
        num_workers:       DataLoader worker processes

    Returns:
        torch.utils.data.DataLoader with collate_mtl_batch
    """
    dataset = TaskDataset(
        dataset_yml_path=dataset_yml_path,
        paramset_yml_path=paramset_yml_path,
        split=split,
        fold=fold,
        augment=augment,
    )

    sampler = TaskSampler(dataset, batch_size=batch_size, mixed=mixed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_mtl_batch,
        drop_last=True,
        pin_memory=True,
    )
    return loader
