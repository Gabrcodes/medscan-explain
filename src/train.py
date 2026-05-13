"""Training loop with two-phase fine-tuning, class-weighted loss, mixed precision,
checkpointing and loss/accuracy curve logging.

Run:
    python -m src.train --dataset pneumonia --backbone efficientnet_b0 --epochs 10
    python -m src.train --dataset ham10000  --backbone resnet50        --epochs 15

Outputs:
    checkpoints/<dataset>_<backbone>_best.pt   — best-val-F1 weights + cfg + history
    outputs/<dataset>_<backbone>_curves.png    — train/val loss & accuracy curves
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import TrainConfig, OUTPUT_DIR
from .data import get_dataloaders, get_datasets, class_weights, seed_everything
from .model import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    """One pass over `loader`. Training if `optimizer` is given, else eval."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_preds, all_targets = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, targets in tqdm(loader, leave=False, desc="train" if is_train else "eval"):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, targets)
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    targs = np.concatenate(all_targets)
    avg_loss = total_loss / len(loader.dataset)
    acc = float((preds == targs).mean())
    f1 = float(f1_score(targs, preds, average="macro"))
    return {"loss": avg_loss, "acc": acc, "f1": f1}


def train(cfg: TrainConfig) -> dict:
    seed_everything(cfg.seed)
    device = _device()
    print(f"[MedScan] device={device}  dataset={cfg.dataset}  backbone={cfg.backbone}")

    train_loader, val_loader, _ = get_dataloaders(cfg)
    train_ds, _, _ = get_datasets(cfg)   # for class weights (cheap: builds wrappers, not new downloads)
    cw = class_weights(cfg, train_ds)
    if cw is not None:
        cw = cw.to(device)
        print(f"[MedScan] class weights: {cw.tolist()}")

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr, cfg.backbone_lr_mult), weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history = {k: [] for k in ("train_loss", "train_acc", "train_f1",
                               "val_loss", "val_acc", "val_f1")}
    best_val_f1, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # phase 1: frozen backbone for the first few epochs
        freeze = epoch <= cfg.freeze_backbone_epochs
        model.set_backbone_trainable(not freeze)

        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, cfg.amp)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in (("train", tr), ("val", va)):
            history[f"{k}_loss"].append(v["loss"])
            history[f"{k}_acc"].append(v["acc"])
            history[f"{k}_f1"].append(v["f1"])

        tag = "(frozen backbone)" if freeze else ""
        print(f"  epoch {epoch:2d}/{cfg.epochs}  "
              f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f} {tag}")

        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"[MedScan] done in {elapsed/60:.1f} min  best val macro-F1 = {best_val_f1:.4f}")

    # --- persist ----------------------------------------------------------
    ckpt_path = cfg.checkpoint_path("best")
    torch.save(
        {"state_dict": best_state, "config": asdict(cfg), "history": history,
         "best_val_f1": best_val_f1, "train_seconds": elapsed},
        ckpt_path,
    )
    print(f"[MedScan] saved checkpoint -> {ckpt_path}")

    _plot_curves(history, cfg)
    (OUTPUT_DIR / f"{cfg.dataset}_{cfg.backbone}_history.json").write_text(json.dumps(history, indent=2))
    return history


def _plot_curves(history: dict, cfg: TrainConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["val_f1"], label="val macro-F1", linestyle="--")
    axes[1].set_title("Accuracy / F1"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.suptitle(f"{cfg.dataset} · {cfg.backbone}")
    fig.tight_layout()
    out = OUTPUT_DIR / f"{cfg.dataset}_{cfg.backbone}_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[MedScan] saved curves -> {out}")


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the MedScan image classifier.")
    p.add_argument("--dataset", default="pneumonia", choices=["pneumonia", "ham10000"])
    p.add_argument("--backbone", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    a = p.parse_args()
    return TrainConfig(
        dataset=a.dataset, backbone=a.backbone, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, image_size=a.image_size, num_workers=a.num_workers,
        amp=not a.no_amp, use_class_weights=not a.no_class_weights,
    )


if __name__ == "__main__":
    train(_parse_args())
