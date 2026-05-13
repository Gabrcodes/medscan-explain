"""Dataset loading, transforms and dataloaders.

We pull the datasets through the HuggingFace `datasets` hub so the team does not
need a Kaggle account, and so a fresh EC2 instance can reproduce everything with a
single `pip install -r requirements.txt`.

The two supported datasets (see `config.DATASETS`):
  * "pneumonia" — binary chest X-ray (Normal / Pneumonia), pediatric.
  * "ham10000"  — 7-class dermoscopy skin lesions (stretch goal).
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
# Dataset wrapper around a HuggingFace split
# --------------------------------------------------------------------------- #
class HFImageDataset(Dataset):
    def __init__(self, hf_split, image_col: str, label_col: str,
                 label_names: list[str], tfm: transforms.Compose):
        self.ds = hf_split
        self.image_col = image_col
        self.label_col = label_col
        self.tfm = tfm
        # The label column may already be ints (ClassLabel) or strings (e.g. HAM10000 `dx`).
        self._str2idx = {name: i for i, name in enumerate(label_names)}

    def __len__(self) -> int:
        return len(self.ds)

    def _label_to_int(self, raw) -> int:
        if isinstance(raw, str):
            return self._str2idx[raw]
        return int(raw)

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = row[self.image_col]
        label = self._label_to_int(row[self.label_col])
        return self.tfm(img), label

    @property
    def targets(self) -> np.ndarray:
        col = self.ds[self.label_col]
        return np.array([self._label_to_int(v) for v in col])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_datasets(cfg: TrainConfig):
    """Return (train_ds, val_ds, test_ds) as torch Datasets.

    Handles datasets that ship with a 'validation' split and those that don't
    (we carve a stratified-ish slice off the training split using `cfg.val_split`).
    """
    from datasets import load_dataset, ClassLabel  # local import keeps `import src.config` cheap

    spec = DATASETS[cfg.dataset]
    seed_everything(cfg.seed)

    raw = load_dataset(spec["hf_id"], cache_dir=str(DATA_DIR))
    label_names = spec["classes"]
    img_col, lab_col = spec["image_col"], spec["label_col"]

    splits = set(raw.keys())
    train_split = raw["train"]
    test_split = raw["test"] if "test" in splits else None

    if "validation" in splits:
        val_split = raw["validation"]
    elif "valid" in splits:
        val_split = raw["valid"]
    else:
        # carve a val set off train
        n_val = max(1, int(len(train_split) * cfg.val_split))
        shuffled = train_split.shuffle(seed=cfg.seed)
        val_split = shuffled.select(range(n_val))
        train_split = shuffled.select(range(n_val, len(shuffled)))

    if test_split is None:
        # rare; fall back to splitting val in half
        n = len(val_split) // 2
        test_split = val_split.select(range(n))
        val_split = val_split.select(range(n, len(val_split)))

    tf_train = build_transforms(cfg.image_size, train=True)
    tf_eval = build_transforms(cfg.image_size, train=False)

    train_ds = HFImageDataset(train_split, img_col, lab_col, label_names, tf_train)
    val_ds = HFImageDataset(val_split, img_col, lab_col, label_names, tf_eval)
    test_ds = HFImageDataset(test_split, img_col, lab_col, label_names, tf_eval)
    return train_ds, val_ds, test_ds


def get_dataloaders(cfg: TrainConfig):
    train_ds, val_ds, test_ds = get_datasets(cfg)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def class_weights(cfg: TrainConfig, train_ds: HFImageDataset) -> Optional[torch.Tensor]:
    """Inverse-frequency weights for CrossEntropyLoss (the pneumonia set is ~3:1)."""
    if not cfg.use_class_weights:
        return None
    targets = train_ds.targets
    counts = np.bincount(targets, minlength=cfg.num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)
