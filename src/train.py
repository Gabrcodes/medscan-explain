"""Training loop with two-phase fine-tuning, class-imbalance handling, mixed precision,
checkpointing, optional early stopping, and loss/metric curve logging.

Works for both task types:
  * multiclass  (pneumonia, ham10000)   — softmax + (class-weighted) cross-entropy;
                                           selection metric = validation macro-F1.
  * multilabel  (chestxray14)           — sigmoid + BCE-with-logits (per-class pos_weight);
                                           selection metric = validation macro ROC-AUC.

Run:
    python -m src.train --dataset pneumonia   --backbone efficientnet_b0     --epochs 10
    python -m src.train --dataset chestxray14 --backbone convnext_base       --epochs 12 --image-size 224 --batch-size 96
    python -m src.train --dataset chestxray14 --backbone vit_base_patch16_224 --epochs 12 --image-size 224 --batch-size 96

Outputs:
    checkpoints/<dataset>_<backbone>_best.pt   — best weights + cfg + history
    outputs/<dataset>_<backbone>_curves.png    — loss + selection-metric curves
    outputs/<dataset>_<backbone>_history.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from .config import TrainConfig, OUTPUT_DIR
from .data import get_dataloaders, loss_weights, seed_everything
from .model import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _multilabel_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    """Macro ROC-AUC over classes that have both positives and negatives present."""
    aucs = []
    for c in range(targets.shape[1]):
        y = targets[:, c]
        if y.min() == y.max():       # degenerate column in this batch/split
            continue
        try:
            aucs.append(roc_auc_score(y, probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def run_epoch(cfg, model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_logits, all_targets = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets.float() if cfg.is_multilabel else targets)
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits)
    targs = torch.cat(all_targets).numpy()
    avg_loss = total_loss / len(loader.dataset)

    if cfg.is_multilabel:
        probs = torch.sigmoid(logits).numpy()
        auc = _multilabel_auc(targs, probs)
        preds = (probs >= 0.5).astype(int)
        # subset ("exact match") accuracy is brutal on 14 labels; report it + macro-F1
        exact = float((preds == targs).all(axis=1).mean())
        f1 = float(f1_score(targs, preds, average="macro", zero_division=0))
        return {"loss": avg_loss, "auc": auc, "f1": f1, "exact_acc": exact, "select": auc}
    else:
        preds = logits.argmax(1).numpy()
        acc = float((preds == targs).mean())
        f1 = float(f1_score(targs, preds, average="macro"))
        return {"loss": avg_loss, "acc": acc, "f1": f1, "select": f1}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = _device()
    print(f"[MedScan] device={device}  dataset={cfg.dataset}  task={cfg.task}  backbone={cfg.backbone}")

    train_loader, val_loader, _ = get_dataloaders(cfg)   # this also resolves cfg.class_names
    print(f"[MedScan] classes ({cfg.num_classes}): {cfg.class_names}")
    print(f"[MedScan] split sizes: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    w = loss_weights(cfg, train_loader.dataset)
    if w is not None:
        w = w.to(device)
        print(f"[MedScan] {'pos_weight' if cfg.is_multilabel else 'class weights'}: "
              f"{np.round(w.cpu().numpy(), 3).tolist()}")

    model = build_model(cfg).to(device)
    if cfg.is_multilabel:
        criterion = nn.BCEWithLogitsLoss(pos_weight=w)
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(model.param_groups(cfg.lr, cfg.backbone_lr_mult),
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    sel_name = "val_auc" if cfg.is_multilabel else "val_f1"
    history: dict[str, list] = {}
    best_sel, best_state, epochs_since_improve = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        freeze = epoch <= cfg.freeze_backbone_epochs
        model.set_backbone_trainable(not freeze)

        tr = run_epoch(cfg, model, train_loader, criterion, device, optimizer, scaler)
        va = run_epoch(cfg, model, val_loader, criterion, device)
        scheduler.step()

        for prefix, m in (("train", tr), ("val", va)):
            for k, v in m.items():
                history.setdefault(f"{prefix}_{k}", []).append(v)

        if cfg.is_multilabel:
            line = (f"  epoch {epoch:2d}/{cfg.epochs}  "
                    f"train loss {tr['loss']:.4f} auc {tr['auc']:.3f} | "
                    f"val loss {va['loss']:.4f} auc {va['auc']:.3f} f1 {va['f1']:.3f}")
        else:
            line = (f"  epoch {epoch:2d}/{cfg.epochs}  "
                    f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
                    f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f}")
        print(line + ("  (frozen backbone)" if freeze else ""))

        if va["select"] > best_sel:
            best_sel = va["select"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if cfg.early_stop_patience and epochs_since_improve >= cfg.early_stop_patience:
                print(f"[MedScan] early stop at epoch {epoch} (no {sel_name} improvement for "
                      f"{cfg.early_stop_patience} epochs)")
                break

    elapsed = time.time() - t0
    print(f"[MedScan] done in {elapsed/60:.1f} min  best {sel_name} = {best_sel:.4f}")

    ckpt_path = cfg.checkpoint_path("best")
    torch.save({"state_dict": best_state, "config": _serialisable_cfg(cfg), "history": history,
                "best_select": best_sel, "select_metric": sel_name, "train_seconds": elapsed,
                "class_names": cfg.class_names, "task": cfg.task}, ckpt_path)
    print(f"[MedScan] saved checkpoint -> {ckpt_path}")

    _plot_curves(history, cfg)
    (OUTPUT_DIR / f"{cfg.dataset}_{cfg.backbone}_history.json").write_text(json.dumps(history, indent=2))
    return history


def _serialisable_cfg(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d.pop("_resolved_classes", None)   # not a constructor-friendly value; class_names saved separately
    return d


def _plot_curves(history: dict, cfg: TrainConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    if cfg.is_multilabel:
        axes[1].plot(epochs, history["train_auc"], label="train macro-AUC")
        axes[1].plot(epochs, history["val_auc"], label="val macro-AUC")
        axes[1].plot(epochs, history["val_f1"], label="val macro-F1", linestyle="--")
        axes[1].set_title("Macro ROC-AUC / F1")
    else:
        axes[1].plot(epochs, history["train_acc"], label="train acc")
        axes[1].plot(epochs, history["val_acc"], label="val acc")
        axes[1].plot(epochs, history["val_f1"], label="val macro-F1", linestyle="--")
        axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(f"{cfg.dataset} · {cfg.backbone}")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{cfg.dataset}_{cfg.backbone}_curves.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[MedScan] saved curves -> {out}")


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the MedScan image classifier.")
    p.add_argument("--dataset", default="pneumonia", choices=["pneumonia", "ham10000", "chestxray14"])
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--early-stop", type=int, default=0, help="patience in epochs; 0 = off")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    a = p.parse_args()
    return TrainConfig(
        dataset=a.dataset, backbone=a.backbone, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, image_size=a.image_size, num_workers=a.num_workers,
        freeze_backbone_epochs=a.freeze_epochs, early_stop_patience=a.early_stop,
        amp=not a.no_amp, use_class_weights=not a.no_class_weights,
    )


if __name__ == "__main__":
    train(_parse_args())
