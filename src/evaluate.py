"""Evaluation: full metric suite on the test split + comparison against two baselines.

Produces (in outputs/):
  * <tag>_confusion_matrix.png
  * <tag>_roc.png                 (binary datasets only)
  * <tag>_metrics.json            (accuracy, precision/recall/F1 per class + macro,
                                   ROC-AUC, baseline numbers, complexity notes)

Baselines (course requirement: "compare against at least one baseline"):
  1. Majority-class predictor — the trivial floor.
  2. Logistic regression on down-sampled, flattened pixels — a real but weak ML model,
     shows how much the deep features actually buy us.

Run:
    python -m src.evaluate --dataset pneumonia --checkpoint checkpoints/pneumonia_efficientnet_b0_best.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
)
from tqdm import tqdm

from .config import TrainConfig, OUTPUT_DIR
from .data import get_dataloaders, build_transforms
from .model import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.to(device).eval()
    all_logits, all_targets = [], []
    for imgs, targets in tqdm(loader, desc="test", leave=False):
        logits = model(imgs.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(1)
    return probs, preds, targets


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def majority_baseline(train_targets: np.ndarray, test_targets: np.ndarray):
    maj = np.bincount(train_targets).argmax()
    preds = np.full_like(test_targets, maj)
    return {
        "name": "majority-class",
        "accuracy": float(accuracy_score(test_targets, preds)),
        "macro_f1": float(f1_score(test_targets, preds, average="macro")),
    }


def logreg_pixel_baseline(cfg: TrainConfig, train_loader, test_loader):
    """Logistic regression on 32x32 grayscale flattened pixels.

    We re-read images at a tiny resolution to keep this fast; it deliberately throws
    away most spatial structure, which is exactly the point of the comparison.
    """
    small_tfm = build_transforms(32, train=False)

    def to_xy(loader):
        xs, ys = [], []
        # iterate the underlying dataset so we can re-transform raw PIL images cheaply
        ds = loader.dataset
        for i in range(len(ds)):
            row = ds.ds[i]
            img = row[ds.image_col]
            x = small_tfm(img).mean(0).reshape(-1).numpy()   # gray, flattened
            xs.append(x)
            ys.append(ds._label_to_int(row[ds.label_col]))
        return np.stack(xs), np.array(ys)

    Xtr, ytr = to_xy(train_loader)
    Xte, yte = to_xy(test_loader)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="auto")
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xte)
    return {
        "name": "logreg-on-32x32-pixels",
        "accuracy": float(accuracy_score(yte, preds)),
        "macro_f1": float(f1_score(yte, preds, average="macro")),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_confusion(cm: np.ndarray, class_names, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix · {tag}")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_confusion_matrix.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def _plot_roc(targets, probs, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(targets, probs[:, 1])
    auc = roc_auc_score(targets, probs[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC · {tag}"); ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_roc.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")
    return float(auc)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def evaluate(checkpoint: str, run_baselines: bool = True) -> dict:
    device = _device()
    model, cfg, ckpt = load_checkpoint(checkpoint)
    tag = f"{cfg.dataset}_{cfg.backbone}"
    class_names = cfg.class_names

    train_loader, _, test_loader = get_dataloaders(cfg)
    probs, preds, targets = collect_predictions(model, test_loader, device)

    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(targets, preds)
    _plot_confusion(cm, class_names, tag)

    auc = None
    if cfg.num_classes == 2:
        auc = _plot_roc(targets, probs, tag)
    else:
        try:
            auc = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc = None

    results = {
        "checkpoint": checkpoint,
        "dataset": cfg.dataset,
        "backbone": cfg.backbone,
        "train_seconds": ckpt.get("train_seconds"),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "roc_auc": auc,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "complexity_notes": {
            "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "inference": "O(1) forward pass per image; ~constant memory ~ batch_size * activ. maps",
            "training": "O(epochs * dataset_size) forward+backward passes",
        },
    }

    if run_baselines:
        train_targets = np.array([train_loader.dataset._label_to_int(v)
                                  for v in train_loader.dataset.ds[train_loader.dataset.label_col]])
        results["baselines"] = [
            majority_baseline(train_targets, targets),
            logreg_pixel_baseline(cfg, train_loader, test_loader),
        ]

    out = OUTPUT_DIR / f"{tag}_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n=== {tag} ===")
    print(f"accuracy   {results['accuracy']:.4f}")
    print(f"macro F1   {results['macro_f1']:.4f}")
    if auc is not None:
        print(f"ROC-AUC    {auc:.4f}")
    if run_baselines:
        for b in results["baselines"]:
            print(f"baseline   {b['name']:<26} acc={b['accuracy']:.3f}  macroF1={b['macro_f1']:.3f}")
    print(f"\nfull metrics -> {out}")
    return results


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MedScan checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--no-baselines", action="store_true")
    # --dataset is kept for symmetry / sanity but the real config comes from the checkpoint
    p.add_argument("--dataset", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(args.checkpoint, run_baselines=not args.no_baselines)
