"""Evaluation: full metric suite on the test split + comparison against baselines.

multiclass (pneumonia, ham10000):
  accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC, confusion matrix, ROC plot.
  Baselines: majority-class, logistic-regression on 32x32 flattened pixels.

multilabel (chestxray14):
  per-class ROC-AUC + mean, per-class average precision (mAP), micro/macro-F1 at 0.5,
  a per-class AUC bar chart, and ROC curves for a few representative findings.
  Baselines: prevalence predictor (AUC 0.5 floor), per-class logreg on 32x32 pixels
  (fitted on a subsample of the training images for speed).

Run:
    python -m src.evaluate --checkpoint checkpoints/chestxray14_convnext_base_best.pt
    python -m src.evaluate --checkpoint checkpoints/pneumonia_efficientnet_b0_best.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, roc_curve,
)
from tqdm import tqdm

from .config import TrainConfig, OUTPUT_DIR
from .data import get_dataloaders, build_transforms
from .model import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    if ckpt.get("class_names"):
        cfg._resolved_classes = list(ckpt["class_names"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def collect_predictions(cfg, model, loader, device, tta: bool = False):
    model.to(device).eval()
    all_logits, all_targets = [], []
    for imgs, targets in tqdm(loader, desc="test", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        if tta:
            logits = logits + model(torch.flip(imgs, dims=[3]))
            logits = logits / 2.0
        all_logits.append(logits.float().cpu())
        all_targets.append(targets)
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    if cfg.is_multilabel:
        probs = torch.sigmoid(logits).numpy()
    else:
        probs = torch.softmax(logits, dim=1).numpy()
    return probs, targets


# --------------------------------------------------------------------------- #
# multilabel metrics + plots
# --------------------------------------------------------------------------- #
def _per_class_table(targets, probs, class_names):
    rows = {}
    for c, name in enumerate(class_names):
        y, p = targets[:, c], probs[:, c]
        pred = (p >= 0.5).astype(int)
        entry = {"prevalence": float(y.mean()),
                 "f1@0.5": float(f1_score(y, pred, zero_division=0))}
        if y.min() != y.max():
            entry["roc_auc"] = float(roc_auc_score(y, p))
            entry["avg_precision"] = float(average_precision_score(y, p))
        else:
            entry["roc_auc"] = None
            entry["avg_precision"] = None
        rows[name] = entry
    return rows


def _plot_perclass_auc(per_class: dict, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in per_class if per_class[n]["roc_auc"] is not None]
    aucs = [per_class[n]["roc_auc"] for n in names]
    order = np.argsort(aucs)
    names = [names[i] for i in order]; aucs = [aucs[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, aucs)
    ax.axvline(0.5, color="gray", linestyle="--")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("ROC-AUC"); ax.set_title(f"Per-class ROC-AUC · {tag}")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_perclass_auc.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def _plot_roc_subset(targets, probs, class_names, tag: str, k: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [c for c in range(len(class_names)) if targets[:, c].min() != targets[:, c].max()]
    aucs = [(c, roc_auc_score(targets[:, c], probs[:, c])) for c in valid]
    aucs.sort(key=lambda t: -t[1])
    pick = aucs[:k]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c, a in pick:
        fpr, tpr, _ = roc_curve(targets[:, c], probs[:, c])
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC {a:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC — top findings · {tag}"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_roc.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# multiclass plots
# --------------------------------------------------------------------------- #
def _plot_confusion(cm, class_names, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix · {tag}")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_confusion_matrix.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")


def _plot_roc_binary(targets, probs, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(targets, probs[:, 1]); auc = roc_auc_score(targets, probs[:, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}"); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC · {tag}"); ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / f"{tag}_roc.png"; fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  wrote {out}")
    return float(auc)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _images_as_pixels(ds, n=None, size=32):
    tfm = build_transforms(size, train=False)
    idxs = range(len(ds)) if n is None else range(min(n, len(ds)))
    X = []
    for i in idxs:
        img = ds.ds[int(i)][ds.image_col]
        X.append(tfm(img).mean(0).reshape(-1).numpy())
    return np.stack(X)


def baselines_multiclass(cfg, train_ds, test_ds, train_loader):
    yte = test_ds.label_matrix
    ytr = train_ds.label_matrix
    maj = np.bincount(ytr).argmax()
    out = [{"name": "majority-class",
            "accuracy": float(accuracy_score(yte, np.full_like(yte, maj))),
            "macro_f1": float(f1_score(yte, np.full_like(yte, maj), average="macro"))}]
    Xtr = _images_as_pixels(train_ds); Xte = _images_as_pixels(test_ds)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    pr = clf.predict(Xte)
    out.append({"name": "logreg-on-32x32-pixels",
                "accuracy": float(accuracy_score(yte, pr)),
                "macro_f1": float(f1_score(yte, pr, average="macro"))})
    return out


def baselines_multilabel(cfg, train_ds, test_ds, subsample=12000):
    ytr = train_ds.label_matrix
    yte = test_ds.label_matrix
    # 1) prevalence predictor: constant per-class score == train prevalence -> AUC ~0.5
    prev = ytr.mean(axis=0, keepdims=True).repeat(len(yte), axis=0)
    out = [{"name": "prevalence-predictor",
            "mean_roc_auc": _safe_mean_auc(yte, prev),
            "macro_f1@0.5": float(f1_score(yte, (prev >= 0.5).astype(int), average="macro", zero_division=0))}]
    # 2) per-class logreg on 32x32 pixels, fitted on a training subsample for speed
    rng = np.random.default_rng(cfg.seed)
    sub = rng.choice(len(train_ds), size=min(subsample, len(train_ds)), replace=False)
    Xtr = _images_as_pixels(_Subset(train_ds, sub))
    ytr_sub = ytr[sub]
    Xte = _images_as_pixels(test_ds, n=min(8000, len(test_ds)))
    yte_sub = yte[:Xte.shape[0]]
    scores = np.zeros_like(yte_sub, dtype=float)
    for c in range(ytr_sub.shape[1]):
        if ytr_sub[:, c].min() == ytr_sub[:, c].max():
            scores[:, c] = ytr_sub[:, c].mean()
            continue
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr_sub[:, c])
        scores[:, c] = lr.predict_proba(Xte)[:, 1]
    out.append({"name": "per-class-logreg-on-32x32-pixels",
                "mean_roc_auc": _safe_mean_auc(yte_sub, scores),
                "macro_f1@0.5": float(f1_score(yte_sub, (scores >= 0.5).astype(int),
                                               average="macro", zero_division=0))})
    return out


def _safe_mean_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        if y[:, c].min() != y[:, c].max():
            aucs.append(roc_auc_score(y[:, c], p[:, c]))
    return float(np.mean(aucs)) if aucs else None


class _Subset:
    """Minimal duck-typed view so `_images_as_pixels` works on a subset of indices."""
    def __init__(self, ds, idxs):
        self.ds = ds.ds.select([int(i) for i in idxs])
        self.image_col = ds.image_col
    def __len__(self): return len(self.ds)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def evaluate(checkpoint: str, run_baselines: bool = True, tta: bool = False) -> dict:
    device = _device()
    model, cfg, ckpt = load_checkpoint(checkpoint)
    tag = f"{cfg.dataset}_{cfg.backbone}"
    class_names = cfg.class_names

    train_loader, _, test_loader = get_dataloaders(cfg)
    probs, targets = collect_predictions(cfg, model, test_loader, device, tta=tta)

    results = {"checkpoint": checkpoint, "dataset": cfg.dataset, "task": cfg.task,
               "backbone": cfg.backbone, "train_seconds": ckpt.get("train_seconds"),
               "tta": tta, "n_test": int(targets.shape[0]),
               "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}

    if cfg.is_multilabel:
        mean_auc = _safe_mean_auc(targets, probs)
        per_class = _per_class_table(targets, probs, class_names)
        preds05 = (probs >= 0.5).astype(int)
        results.update({
            "mean_roc_auc": mean_auc,
            "micro_f1@0.5": float(f1_score(targets, preds05, average="micro", zero_division=0)),
            "macro_f1@0.5": float(f1_score(targets, preds05, average="macro", zero_division=0)),
            "mean_avg_precision": float(np.mean([v["avg_precision"] for v in per_class.values()
                                                 if v["avg_precision"] is not None])),
            "per_class": per_class,
            "complexity_notes": {"inference": "O(1) forward pass/image",
                                 "training": "O(epochs * dataset_size)"},
        })
        _plot_perclass_auc(per_class, tag)
        _plot_roc_subset(targets, probs, class_names, tag)
        if run_baselines:
            results["baselines"] = baselines_multilabel(cfg, train_loader.dataset, test_loader.dataset)
        print(f"\n=== {tag} ({cfg.task}) ===")
        print(f"mean ROC-AUC      {mean_auc:.4f}   (over {sum(1 for v in per_class.values() if v['roc_auc'])} classes)")
        print(f"mean AvgPrecision {results['mean_avg_precision']:.4f}")
        print(f"micro-F1@0.5      {results['micro_f1@0.5']:.4f}   macro-F1@0.5 {results['macro_f1@0.5']:.4f}")
        worst = sorted(((v['roc_auc'], k) for k, v in per_class.items() if v['roc_auc']))[:3]
        best = sorted(((v['roc_auc'], k) for k, v in per_class.items() if v['roc_auc']))[-3:]
        print("best classes:  " + ", ".join(f"{k} {a:.3f}" for a, k in reversed(best)))
        print("worst classes: " + ", ".join(f"{k} {a:.3f}" for a, k in worst))
        if run_baselines:
            for b in results["baselines"]:
                print(f"baseline   {b['name']:<32} meanAUC={b['mean_roc_auc']:.3f}")
    else:
        preds = probs.argmax(1)
        report = classification_report(targets, preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(targets, preds)
        _plot_confusion(cm, class_names, tag)
        if cfg.num_classes == 2:
            auc = _plot_roc_binary(targets, probs, tag)
        else:
            auc = _try(lambda: float(roc_auc_score(targets, probs, multi_class="ovr", average="macro")))
        results.update({"accuracy": float(accuracy_score(targets, preds)),
                        "macro_f1": float(f1_score(targets, preds, average="macro")),
                        "roc_auc": auc, "per_class": report, "confusion_matrix": cm.tolist(),
                        "complexity_notes": {"params_millions": results["params_millions"],
                                             "inference": "O(1) forward pass/image",
                                             "training": "O(epochs * dataset_size)"}})
        if run_baselines:
            results["baselines"] = baselines_multiclass(cfg, train_loader.dataset, test_loader.dataset, train_loader)
        print(f"\n=== {tag} ({cfg.task}) ===")
        print(f"accuracy   {results['accuracy']:.4f}")
        print(f"macro F1   {results['macro_f1']:.4f}")
        if auc is not None:
            print(f"ROC-AUC    {auc:.4f}")
        if run_baselines:
            for b in results["baselines"]:
                print(f"baseline   {b['name']:<26} acc={b['accuracy']:.3f}  macroF1={b['macro_f1']:.3f}")

    out = OUTPUT_DIR / f"{tag}_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nfull metrics -> {out}")
    return results


def _try(fn):
    try:
        return fn()
    except Exception:
        return None


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MedScan checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--tta", action="store_true", help="test-time augmentation (hflip averaging)")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    evaluate(a.checkpoint, run_baselines=not a.no_baselines, tta=a.tta)
