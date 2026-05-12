# MedScan + Explain

> CSE351 — Introduction to Artificial Intelligence · Spring 2026 · Team Project

An explainable medical-imaging assistant that **(1)** classifies medical images with a
fine-tuned deep CNN, **(2)** explains *why* it predicted what it did with Grad-CAM
heat-maps, and **(3)** uses an LLM agent on **AWS Bedrock** (Claude) with a
retrieval-augmented (RAG) clinical knowledge base to draft a structured, **clearly
non-diagnostic** "preliminary findings" note.

It is built as a teaching/demo system — **not a medical device** and **not for clinical use.**

---

## Why this problem matters

Radiology and dermatology departments are chronically over-loaded; in many regions
patients wait days or weeks for a specialist read. ML triage tools can help prioritise
urgent cases, but "black-box" classifiers are (rightly) distrusted by clinicians. This
project pairs a competitive classifier with **two layers of transparency** — visual
(Grad-CAM) and textual (a cited, hedged natural-language explanation) — to study how
far explainability can close that trust gap, and what its limits are.

## AI techniques used (multiple — see grading "Model Selection")

| Layer | Technique |
|---|---|
| Image classifier | **Deep Learning** — transfer learning with EfficientNet-B0 / ResNet-50 (ImageNet → fine-tune) |
| Explainability | Grad-CAM (gradient-weighted class activation mapping) |
| Report generator | **AI Agent + Tool use + RAG** — Claude on Amazon Bedrock, Titan-embeddings + FAISS over curated clinical references |
| Baseline | **Classic ML** — logistic regression / majority-class baseline for comparison |

## Datasets

| Dataset | Task | Size | Notes |
|---|---|---|---|
| Chest X-Ray Pneumonia | binary: Normal vs Pneumonia | ~5.8k images | **pediatric patients only** — discussed in Responsible AI |
| HAM10000 (stretch goal) | 7-class skin-lesion classification | ~10k dermoscopy images | known under-representation of darker skin tones |

Datasets are pulled at runtime via the HuggingFace `datasets` library (no Kaggle login needed) — see `src/data.py`.

---

## Repo layout

```
aiproj/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                       # (gitignored) downloaded datasets / cache
├── checkpoints/                # (gitignored) saved model weights
├── docs/
│   └── aws_setup.md            # launch a GPU EC2 instance, configure Bedrock
├── notebooks/
│   └── 01_train_classifier.ipynb
├── src/
│   ├── config.py               # paths & hyper-parameters
│   ├── data.py                 # dataset download, transforms, dataloaders
│   ├── model.py                # backbone + classifier head
│   ├── train.py                # training loop, checkpointing, loss curves
│   ├── evaluate.py             # metrics, confusion matrix, ROC, baseline
│   ├── gradcam.py              # Grad-CAM overlays + "attention-in-lung" sanity metric
│   ├── bedrock_report.py       # Bedrock client + RAG + report prompt
│   └── knowledge_base/         # curated, cited clinical reference snippets
├── app/
│   └── streamlit_app.py        # upload image → label, confidence, heat-map, report
└── reports/                    # written report (PDF), slides, contribution log
```

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. train the classifier (GPU strongly recommended — see docs/aws_setup.md)
python -m src.train --dataset pneumonia --backbone efficientnet_b0 --epochs 10

# 3. evaluate + plots
python -m src.evaluate --dataset pneumonia --checkpoint checkpoints/pneumonia_best.pt

# 4. configure AWS / Bedrock (one-time)
aws configure                                          # or use an instance role
#  -> request model access to Anthropic Claude + Amazon Titan Embeddings in the Bedrock console

# 5. run the demo
streamlit run app/streamlit_app.py
```

## Responsible AI (summary — full section in `reports/`)

- **Bias & fairness** — the pneumonia set is pediatric-only and single-source; the model must not be assumed to generalise to adults or other hospitals. HAM10000 skin-tone skew is reported. We slice metrics by available metadata.
- **Privacy** — only public, de-identified images are used; uploaded images in the demo are processed in memory and never persisted.
- **Sustainability** — we fine-tune (not pre-train) a small backbone; we log GPU-hours and estimate CO₂ with `codecarbon`-style figures in the report.
- **Misuse risk** — the system could be mistaken for a diagnostic tool; mitigations: prominent disclaimers, confidence display, hard refusal language in the LLM prompt, and no patient-identifying inputs.

## Team & contributions

See `reports/contribution_log.md`. Every member is expected to be able to explain any part of the system (per the course guidelines).

## Academic integrity

Original work for CSE351 Spring 2026. AI coding assistants were used for parts of the
implementation; AI-assisted sections are annotated in-code with `# AI-assisted` comments,
and the team understands and can explain all submitted code. Core model design, analysis,
and report are the team's own.
