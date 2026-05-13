# MedScan + Explain

> **CSE351 — Introduction to Artificial Intelligence · Spring 2026 · Team Project**
> An explainable medical-imaging assistant: a fine-tuned deep network classifies a chest
> X-ray, **Grad-CAM** shows *where* it looked, and an **LLM agent (Claude on Amazon
> Bedrock) with retrieval-augmented generation** drafts a structured, **explicitly
> non-diagnostic** preliminary note grounded in a curated clinical knowledge base.

**This is a teaching/demo system — not a medical device, and not for clinical use.**

---

## Why this problem matters

Radiology departments are chronically over-loaded; in many regions a specialist read
takes days. ML triage tools can help prioritise urgent films, but clinicians (rightly)
distrust black-box classifiers. MedScan pairs a competitive classifier with **two
complementary layers of transparency** — visual (Grad-CAM heat-maps) and textual (a
cited, hedged natural-language note) — to study how far explainability narrows that trust
gap, and where it does not.

## AI techniques used (multiple — see the rubric's "Model Selection" bonus)

| Layer | Technique |
|---|---|
| Image classifier | **Deep learning** — transfer learning with EfficientNet-B0, **ConvNeXt-Base**, **ViT-Base** (ImageNet → fine-tune) |
| Explainability | **Grad-CAM** (gradient-weighted class activation mapping), per-finding for the multi-label model |
| Report generator | **AI agent + tool use + RAG** — Claude (Sonnet 4.6) on Amazon Bedrock, Titan-embeddings + FAISS over a curated clinical knowledge base |
| Baselines | **Classic ML** — majority-class and logistic-regression-on-pixels, for honest comparison |

The project also exercises **Vision Transformers / foundation models** explicitly (ViT-Base
fine-tuning) and **multi-label classification** (independent sigmoid heads + BCE loss).

## Datasets (pulled at runtime via the HuggingFace `datasets` library — no Kaggle login)

| Dataset | Task | Size | Notes |
|---|---|---|---|
| **Chest X-Ray Pneumonia** (Kermany et al. 2018) | binary: Normal vs Pneumonia | ~5.9k images | **pediatric, single-hospital** — discussed at length in the Responsible-AI section |
| **NIH ChestX-ray14** (Wang et al. 2017) | **multi-label**: 14 findings (Atelectasis, Cardiomegaly, Effusion, …, Hernia) | ~112k images | labels NLP-mined from reports (~90% accurate); strong class imbalance; image-level labels only |
| HAM10000 (optional / stretch) | 7-class skin-lesion | ~10k images | known under-representation of darker skin tones |

## Results so far

| Model | Dataset | Test metric | Baselines |
|---|---|---|---|
| EfficientNet-B0 | Chest X-Ray Pneumonia | **91.4% acc · 0.907 macro-F1 · 0.956 ROC-AUC** (624-image test set) | logreg-on-pixels 75.8% acc · majority-class 62.5% acc |
| ConvNeXt-Base | NIH ChestX-ray14 (multi-label) | mean ROC-AUC over 14 findings — *training in progress* (≈0.84 at epoch 5) | prevalence predictor ≈0.50 mean AUC |
| ViT-Base | NIH ChestX-ray14 (multi-label) | *queued* | — |

Trained on an AWS EC2 GPU instance (NVIDIA T4 for pneumonia, NVIDIA L40S for the larger
models). Wall-clock training time and a back-of-envelope CO₂ estimate are reported in the
written report (the figures are saved in each checkpoint as `train_seconds`).

---

## Repo layout

```
.
├── README.md
├── requirements.txt
├── data/                       # (gitignored) downloaded datasets / HF cache
├── checkpoints/                # (gitignored) saved model weights
├── docs/
│   └── aws_setup.md            # launch a GPU EC2 instance, request Bedrock model access
├── notebooks/
│   └── 01_train_classifier.ipynb
├── scripts/
│   └── run_on_instance.sh      # one-command bootstrap + train on a fresh DLAMI box
├── src/
│   ├── config.py               # all paths, dataset specs, hyper-parameters (single source of truth)
│   ├── data.py                 # dataset download, transforms, dataloaders, multi-/single-label encoding
│   ├── model.py                # timm backbone + classifier head, two-phase fine-tuning, Grad-CAM target layer
│   ├── train.py                # training loop (CE / BCE), class-imbalance handling, AMP, early stopping, curves
│   ├── evaluate.py             # metrics + plots: confusion matrix / ROC (multiclass), per-class AUC / mAP (multilabel), baselines
│   ├── gradcam.py              # Grad-CAM overlays + plain-language region descriptions; per-finding for multi-label
│   ├── bedrock_report.py       # Bedrock client + Titan-embeddings + FAISS RAG + Claude prompts (single- and multi-label)
│   └── knowledge_base/         # curated, cited clinical reference snippets (pneumonia, NIH findings, responsible-AI)
├── app/
│   └── streamlit_app.py        # demo UI: upload image → label(s), confidence, Grad-CAM, generated note
└── reports/                    # written report outline, contribution log
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# --- train the classifier (GPU strongly recommended — see docs/aws_setup.md) ---
python -m src.train --dataset pneumonia   --backbone efficientnet_b0      --epochs 10
python -m src.train --dataset chestxray14 --backbone convnext_base        --epochs 10 --batch-size 128 --image-size 224
python -m src.train --dataset chestxray14 --backbone vit_base_patch16_224 --epochs 10 --batch-size 128 --image-size 224

# --- evaluate (metrics, plots, baselines) ---
python -m src.evaluate --checkpoint checkpoints/pneumonia_efficientnet_b0_best.pt
python -m src.evaluate --checkpoint checkpoints/chestxray14_convnext_base_best.pt --tta

# --- configure AWS / Bedrock once (see docs/aws_setup.md) ---
aws configure                                              # or: aws login  /  an instance role
#   then request model access to "Anthropic Claude" + "Amazon Titan Embeddings" in the Bedrock console
python -m src.bedrock_report                                # smoke-test the RAG report path

# --- run the demo ---
streamlit run app/streamlit_app.py
```

On a fresh AWS Deep Learning AMI GPU box, `bash scripts/run_on_instance.sh chestxray14 convnext_base 10`
installs deps and runs train + eval end to end.

## Responsible AI (summary — full section in `reports/`)

- **Bias & fairness** — the pneumonia set is pediatric-only and single-source; NIH ChestX-ray14
  is single-institution with NLP-mined (~90%-accurate) labels and heavy class imbalance. We
  never claim clinical-grade performance, report per-finding (not just aggregate) AUC, and slice
  by available metadata. Shortcut-learning risk (laterality markers, scanner artefacts) is discussed.
- **Privacy** — only public, de-identified images; the demo holds uploads in memory and never
  persists them; no patient identifiers are ever sent to Bedrock.
- **Sustainability** — we fine-tune (not pre-train) modest backbones; GPU-hours and an estimated
  CO₂ figure are reported.
- **Misuse risk** — the system could be mistaken for a diagnostic tool; mitigations: prominent
  disclaimers everywhere, confidence shown as *uncalibrated*, hard refusal language in the LLM
  prompt, urgent-finding flagging (e.g. pneumothorax), and no clinical-action outputs.

## Team & contributions

See `reports/contribution_log.md`. Per the course guidelines, every member is expected to be
able to explain any part of the system.

## Academic integrity

Original work for CSE351 Spring 2026. An AI coding assistant was used for parts of the
implementation; AI-assisted sections are annotated in-code, and the team understands and can
explain all submitted code. The core model design, analysis, and report are the team's own.
