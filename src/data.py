"""Dataset loading, transforms and dataloaders.

We pull the datasets through the HuggingFace `datasets` hub so the team does not
need a Kaggle account, and so a fresh EC2 instance can reproduce everything with a
single `pip install -r requirements.txt`.

Supported datasets (see `config.DATASETS`):
  * "pneumonia"   — binary chest X-ray (Normal / Pneumonia), pediatric.   [multiclass]
  * "ham10000"    — 7-class dermoscopy skin lesions.                       [multiclass]
  * "chestxray14" — NIH ChestX-ray14, ~112k images, 14 findings.          [multilabel]

For multilabel datasets the target is a float multi-hot vector of length `num_classes`;
for multiclass it is an int class index. `cfg.is_multilabel` switches downstream logic.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DATASETS, IMAGENET_MEAN, IMAGENET_STD, DATA_DIR, TrainConfig


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    """ImageNet-style normalisation; light augmentation for training only.

    Medical-image augmentation must stay conservative — aggressive colour jitter or
    large rotations can destroy diagnostic features. We use small rotations, a mild
    zoom/crop and horizontal flip (anatomically harmless for both chest films and
    dermoscopy patches).
    """
    if train:
        return transforms.Compose([
            transforms.Lambda(_to_rgb),
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _to_rgb(img):
    """Chest X-rays are often single-channel; ImageNet backbones expect 3 channels."""
    return img.convert("RGB")


# --------------------------------------------------------------------------- #
# Label encoding helpers
# --------------------------------------------------------------------------- #
class LabelEncoder:
    """Turns a dataset's raw label value into a model target.

    Handles, in order of how the HF datasets we use store labels:
      * an int / ClassLabel               -> int index            (multiclass)
      * a string class name               -> int index            (multiclass)
      * a pipe-delimited string "A|B|C"   -> multi-hot float vec   (multilabel)
      * a list of ints (ClassLabel seq)   -> multi-hot float vec   (multilabel)
      * a list of strings                 -> multi-hot float vec   (multilabel)
    `keep_idx` optionally remaps from the dataset's raw class indices to our (possibly
    smaller) class list — used to drop NIH's "No Finding" pseudo-class.
    """
    def __init__(self, class_names: list[str], multilabel: bool,
                 raw_names: Optional[list[str]] = None):
        self.class_names = class_names
        self.multilabel = multilabel
        self.name2idx = {n: i for i, n in enumerate(class_names)}
        # map from raw dataset index -> our index (or None to drop)
        if raw_names is not None:
            self.raw2ours = {ri: self.name2idx.get(n) for ri, n in enumerate(raw_names)}
        else:
            self.raw2ours = None

    def _ours_idx(self, raw) -> Optional[int]:
        if isinstance(raw, str):
            return self.name2idx.get(raw)
        ri = int(raw)
        if self.raw2ours is not None:
            return self.raw2ours.get(ri)
        return ri if 0 <= ri < len(self.class_names) else None

    def encode(self, raw):
        if self.multilabel:
            vec = np.zeros(len(self.class_names), dtype=np.float32)
            items = raw
            if isinstance(raw, str):
                items = [p for p in raw.replace(",", "|").split("|") if p.strip()]
            elif not isinstance(raw, (list, tuple, np.ndarray)):
                items = [raw]
            for it in items:
                oi = self._ours_idx(it.strip() if isinstance(it, str) else it)
                if oi is not None:
                    vec[oi] = 1.0
            return vec
        # multiclass
        oi = self._ours_idx(raw if not isinstance(raw, (list, tuple)) else raw[0])
        if oi is None:
            raise ValueError(f"Could not map label {raw!r} to a class index.")
        return oi


# --------------------------------------------------------------------------- #
# Dataset wrapper around a HuggingFace split
# --------------------------------------------------------------------------- #
class HFImageDataset(Dataset):
    def __init__(self, hf_split, image_col: str, label_col: str,
                 encoder: LabelEncoder, tfm: transforms.Compose):
        self.ds = hf_split
        self.image_col = image_col
        self.label_col = label_col
        self.enc = encoder
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row[self.image_col]
        target = self.enc.encode(row[self.label_col])
        if self.enc.multilabel:
            return self.tfm(img), torch.from_numpy(target)
        return self.tfm(img), int(target)

    @property
    def label_matrix(self) -> np.ndarray:
        """(N,) int array for multiclass, or (N, C) float multi-hot for multilabel."""
        col = self.ds[self.label_col]
        if self.enc.multilabel:
            return np.stack([self.enc.encode(v) for v in col])
        return np.array([self.enc.encode(v) for v in col])

    # kept for backward-compat with older eval code
    @property
    def targets(self) -> np.ndarray:
        return self.label_matrix

    def _label_to_int(self, raw) -> int:
        return int(self.enc.encode(raw))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _resolve_classes(spec: dict, raw_features):
    """Figure out our class list + the dataset's raw class names (if any).

    For NIH we read the names off the `Sequence(ClassLabel)` feature and drop the
    configured `drop_classes`; if that fails we fall back to the hardcoded list.
    """
    raw_names = None
    try:
        feat = raw_features[spec["label_col"]]
        # Sequence(ClassLabel) -> feat.feature.names ; ClassLabel -> feat.names
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            raw_names = list(feat.feature.names)
        elif hasattr(feat, "names"):
            raw_names = list(feat.names)
    except Exception:
        raw_names = None

    drop = set(spec.get("drop_classes", []))
    if raw_names:
        classes = [n for n in raw_names if n not in drop]
    else:
        classes = list(spec["classes"])
    return classes, raw_names


def get_datasets(cfg: TrainConfig):
    """Return (train_ds, val_ds, test_ds) as torch Datasets, plus they share `cfg`.

    Handles datasets that ship with a 'validation' split and those that don't (we
    carve a slice off training using `cfg.val_split`). For very small built-in val
    splits we also re-carve from train when `cfg.min_val_size` is larger.
    """
    from datasets import load_dataset

    spec = DATASETS[cfg.dataset]
    multilabel = spec.get("task") == "multilabel"
    seed_everything(cfg.seed)

    raw = load_dataset(spec["hf_id"], cache_dir=str(DATA_DIR))
    img_col, lab_col = spec["image_col"], spec["label_col"]

    # class names: prefer reading them off the dataset's features
    any_split = raw[next(iter(raw.keys()))]
    class_names, raw_names = _resolve_classes(spec, any_split.features)
    encoder = LabelEncoder(class_names, multilabel, raw_names=raw_names)
    cfg._resolved_classes = class_names  # cached so TrainConfig.class_names works post-load

    splits = set(raw.keys())
    train_split = raw["train"]
    test_split = raw["test"] if "test" in splits else None
    val_split = raw["validation"] if "validation" in splits else (raw["valid"] if "valid" in splits else None)

    # carve / re-carve a validation split off train when needed
    need_carve = val_split is None or len(val_split) < cfg.min_val_size
    if need_carve:
        n_val = max(cfg.min_val_size, int(len(train_split) * cfg.val_split))
        n_val = min(n_val, len(train_split) // 5)   # never take more than 20% of train
        shuffled = train_split.shuffle(seed=cfg.seed)
        new_val = shuffled.select(range(n_val))
        train_split = shuffled.select(range(n_val, len(shuffled)))
        # if there was a tiny built-in val split, fold it into train (don't waste data)
        if val_split is not None and len(val_split) > 0:
            from datasets import concatenate_datasets
            try:
                train_split = concatenate_datasets([train_split, val_split])
            except Exception:
                pass
        val_split = new_val

    if test_split is None:
        n = len(val_split) // 2
        test_split = val_split.select(range(n))
        val_split = val_split.select(range(n, len(val_split)))

    tf_train = build_transforms(cfg.image_size, train=True)
    tf_eval = build_transforms(cfg.image_size, train=False)
    train_ds = HFImageDataset(train_split, img_col, lab_col, encoder, tf_train)
    val_ds = HFImageDataset(val_split, img_col, lab_col, encoder, tf_eval)
    test_ds = HFImageDataset(test_split, img_col, lab_col, encoder, tf_eval)
    return train_ds, val_ds, test_ds


def get_dataloaders(cfg: TrainConfig):
    train_ds, val_ds, test_ds = get_datasets(cfg)
    pin = torch.cuda.is_available()
    common = dict(num_workers=cfg.num_workers, pin_memory=pin, persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def loss_weights(cfg: TrainConfig, train_ds: HFImageDataset):
    """Class imbalance handling.

    multiclass  -> inverse-frequency `weight` tensor for CrossEntropyLoss (or None)
    multilabel  -> per-class `pos_weight` tensor for BCEWithLogitsLoss (neg/pos ratio,
                   clipped so the ~0.2%-prevalence Hernia class doesn't explode the loss)
    """
    if not cfg.use_class_weights:
        return None
    M = train_ds.label_matrix
    if cfg.is_multilabel:
        pos = M.sum(axis=0).astype(np.float64)
        neg = M.shape[0] - pos
        pos[pos == 0] = 1.0
        pw = np.clip(neg / pos, 1.0, 20.0)
        return torch.tensor(pw, dtype=torch.float32)
    counts = np.bincount(M, minlength=cfg.num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


# backward-compat alias used by older code paths
def class_weights(cfg: TrainConfig, train_ds: HFImageDataset):
    return loss_weights(cfg, train_ds)
