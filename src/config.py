"""Central configuration: paths, dataset specs, default hyper-parameters.

Keeping every magic number here makes the project reproducible (course requirement)
and keeps the rest of the code free of scattered constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"          # plots, metric dumps, Grad-CAM images
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
FAISS_DIR = ROOT / "faiss_index"

for _d in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, FAISS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Datasets ----------------------------------------------------------------
# Each entry: HuggingFace dataset id, the image column, the label column, and the
# human-readable class names (index order must match the integer labels).
DATASETS = {
    "pneumonia": {
        # Chest X-Ray Images (Pneumonia) — pediatric, binary.
        "hf_id": "hf-vision/chest-xray-pneumonia",
        "image_col": "image",
        "label_col": "label",
        "classes": ["NORMAL", "PNEUMONIA"],
        "in_channels": 3,           # we convert grayscale X-rays to 3-ch for ImageNet backbones
        "notes": "Pediatric patients (Guangzhou Women and Children's Medical Center). "
                 "Single-source — do not assume adult generalisation.",
    },
    "ham10000": {
        # HAM10000 dermoscopy — 7-class skin lesion (stretch goal).
        "hf_id": "marmal88/skin_cancer",   # mirror of HAM10000 with a `dx` label
        "image_col": "image",
        "label_col": "dx",
        "classes": ["actinic_keratoses", "basal_cell_carcinoma", "benign_keratosis-like_lesions",
                    "dermatofibroma", "melanocytic_Nevi", "melanoma", "vascular_lesions"],
        "in_channels": 3,
        "notes": "Known under-representation of darker (Fitzpatrick V-VI) skin tones.",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    dataset: str = "pneumonia"
    backbone: str = "efficientnet_b0"      # any timm model name; "resnet50" also tested
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4                       # head LR; backbone gets lr * backbone_lr_mult
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    freeze_backbone_epochs: int = 1        # warm up the head before unfreezing
    val_split: float = 0.15                # carved from train when no val split exists
    num_workers: int = 4
    seed: int = 351
    use_class_weights: bool = True         # pneumonia set is imbalanced (~3:1)
    amp: bool = True                       # mixed precision on GPU

    @property
    def num_classes(self) -> int:
        return len(DATASETS[self.dataset]["classes"])

    @property
    def class_names(self) -> list[str]:
        return list(DATASETS[self.dataset]["classes"])

    def checkpoint_path(self, tag: str = "best") -> Path:
        return CHECKPOINT_DIR / f"{self.dataset}_{self.backbone}_{tag}.pt"


# --- Bedrock -----------------------------------------------------------------
@dataclass
class BedrockConfig:
    region: str = "us-east-1"
    # Inference profile / model ids — adjust to whatever you've been granted access to.
    chat_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    max_tokens: int = 1200
    temperature: float = 0.2
    top_k_passages: int = 4               # RAG: how many KB snippets to retrieve
