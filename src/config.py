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
# Each entry: HuggingFace dataset id, the image/label columns, the human-readable
# class names (index order must match the integer labels), and the `task`:
#   "multiclass"  — exactly one label per image (softmax + cross-entropy)
#   "multilabel"  — zero or more labels per image (sigmoid + binary cross-entropy)
DATASETS = {
    "pneumonia": {
        # Chest X-Ray Images (Pneumonia) — pediatric, binary.
        "hf_id": "hf-vision/chest-xray-pneumonia",
        "image_col": "image",
        "label_col": "label",
        "task": "multiclass",
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
        "task": "multiclass",
        "classes": ["actinic_keratoses", "basal_cell_carcinoma", "benign_keratosis-like_lesions",
                    "dermatofibroma", "melanocytic_Nevi", "melanoma", "vascular_lesions"],
        "in_channels": 3,
        "notes": "Known under-representation of darker (Fitzpatrick V-VI) skin tones.",
    },
    "chestxray14": {
        # NIH ChestX-ray14 — ~112k frontal chest X-rays, 14 findings, MULTI-LABEL.
        # HF mirror exposes `labels` as a Sequence(ClassLabel) including "No Finding";
        # data.py derives the 14 disease classes from the dataset features and treats
        # "No Finding" as the all-zeros vector.
        "hf_id": "alkzar90/NIH-Chest-X-ray-dataset",
        "image_col": "image",
        "label_col": "labels",
        "task": "multilabel",
        "drop_classes": ["No Finding"],   # not a disease label; absence of all others
        # Canonical 14 NIH findings — used as a fallback if feature names can't be read.
        "classes": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
                    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
                    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"],
        "in_channels": 3,
        "notes": "Labels mined from radiology reports with NLP (noisy ~90% accuracy). "
                 "Image-level labels only, no bounding boxes for most. Strong class "
                 "imbalance (Hernia ~0.2%, Infiltration ~18%). Single-institution (NIH CC). "
                 "Patient overlap between images — splits are patient-disjoint here.",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    dataset: str = "pneumonia"
    backbone: str = "efficientnet_b0"      # any timm model name (resnet50, convnext_base, vit_base_patch16_224, ...)
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4                       # head LR; backbone gets lr * backbone_lr_mult
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05          # multiclass only
    freeze_backbone_epochs: int = 1        # warm up the head before unfreezing
    val_split: float = 0.1                 # fraction of train carved off for validation
    min_val_size: int = 500                # re-carve if a built-in val split is smaller than this
    num_workers: int = 4
    seed: int = 351
    use_class_weights: bool = True         # class-weighted CE / BCE pos_weight for imbalance
    amp: bool = True                       # mixed precision on GPU
    early_stop_patience: int = 0           # 0 = off; else stop if no val improvement for N epochs
    tta: bool = False                      # test-time augmentation at eval (hflip average)

    # filled in by data.get_datasets() once the dataset's real class list is known
    _resolved_classes: list[str] = field(default=None, repr=False)

    @property
    def task(self) -> str:
        return DATASETS[self.dataset].get("task", "multiclass")

    @property
    def is_multilabel(self) -> bool:
        return self.task == "multilabel"

    @property
    def class_names(self) -> list[str]:
        if self._resolved_classes:
            return list(self._resolved_classes)
        return list(DATASETS[self.dataset]["classes"])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def checkpoint_path(self, tag: str = "best") -> Path:
        return CHECKPOINT_DIR / f"{self.dataset}_{self.backbone}_{tag}.pt"


# --- Bedrock -----------------------------------------------------------------
@dataclass
class BedrockConfig:
    region: str = "us-east-1"
    # Inference profile / model ids — adjust to whatever you've been granted access to.
    # NB: Claude 4.x on Bedrock is only invokable via a cross-region *inference profile*
    # id (the "us." prefix), not the bare model id. Titan embeddings use the bare id.
    chat_model_id: str = "us.anthropic.claude-sonnet-4-6"
    embed_model_id: str = "amazon.titan-embed-text-v2:0"
    max_tokens: int = 1200
    temperature: float = 0.2
    top_k_passages: int = 4               # RAG: how many KB snippets to retrieve
