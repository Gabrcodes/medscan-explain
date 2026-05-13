# Written Report — outline (target 4–8 pages PDF)

Map of which section satisfies which grading criterion / guideline requirement.
Fill each section in once experiments are run; keep figures from `outputs/`.

1. **Title, team, contribution log pointer** — see `contribution_log.md`.

2. **Problem statement & motivation** *(Originality & Problem Relevance — 10%)*
   - Radiology/dermatology backlog; why a *transparent* triage aid; who is affected.
   - Why AI helps and where it must not be trusted.
   - Societal/ethical framing (sets up the Responsible AI section).

3. **Data** *(Implementation 15% · feeds Responsible AI)*
   - Chest X-Ray Pneumonia: size, class balance (~3:1), **pediatric-only / single-source**.
   - HAM10000 (if completed): 7 classes, skin-tone skew.
   - Splits, preprocessing, augmentation choices and why they're conservative.

4. **Method / model design & justification** *(Model Selection & Tuning — 20%)*
   - Why transfer learning (small data, compute, overfitting) vs. from-scratch.
   - Backbone choice: EfficientNet-B0 vs. ResNet-50 — params, FLOPs, results table.
   - Architecture diagram (backbone → pooled features → dropout → linear head).
   - Two-phase fine-tuning (freeze → unfreeze), class-weighted CE + label smoothing, AdamW,
     cosine LR, mixed precision. Justify every hyper-parameter (point to `src/config.py`).
   - **Alternatives considered**: ViT, training from scratch, focal loss, oversampling — why not (yet).
   - Pseudocode / flowchart of the full pipeline (classifier → Grad-CAM → RAG → Claude).

5. **Explainability** *(part of Model Selection & Technical Understanding)*
   - How Grad-CAM works (gradient-weighted activation maps); which layer we hook and why.
   - The "attention-in-region" sanity description; example overlays (good + a failure case).

6. **The LLM-agent / RAG layer** *(Model Selection — multi-technique bonus)*
   - Why a second, *textual* explanation layer on top of the visual one.
   - Knowledge base construction (curated, cited snippets); Titan embeddings + FAISS retrieval.
   - Prompt design: the guardrails (non-diagnostic, cite-only-retrieved, surface uncertainty).
   - Why Bedrock/Claude (managed, no key handling on GPU box, strong instruction-following).

7. **Experiments & results** *(Model Selection & Technical Understanding — 40% combined)*
   - Classifier: accuracy, precision/recall/F1 per class + macro, confusion matrix, ROC-AUC.
   - Train/val loss & accuracy curves; convergence behaviour; generalisation gap.
   - **Baselines**: majority-class and logistic-regression-on-pixels — quantify the deep-feature gain.
   - Backbone comparison table (EfficientNet-B0 vs ResNet-50).
   - Qualitative LLM outputs: 2–3 example notes (incl. one low-confidence case). No standard
     automatic metric fits a guardrailed note; we report a short rubric self-assessment
     (faithfulness to retrieved snippets, presence of disclaimer, no fabricated facts).
   - Time & space complexity discussion; training wall-clock + instance type.

8. **Responsible AI** *(required dedicated section — guideline 2.5)*
   - **Bias & fairness**: pediatric-only data; single-hospital shortcut-learning risk; HAM10000
     skin-tone skew; we slice metrics by available metadata; we never claim clinical performance.
   - **Privacy**: only public de-identified images; demo holds uploads in memory, no persistence;
     no patient identifiers are ever sent to Bedrock.
   - **Sustainability**: fine-tune (not pre-train) a small backbone; report GPU-hours and a
     back-of-envelope CO₂ estimate (formula in `docs/aws_setup.md`).
   - **Misuse risk**: could be mistaken for a diagnostic device → disclaimers everywhere,
     confidence shown as *uncalibrated*, hard refusal language in the prompt, no clinical actions.

9. **Limitations & failure cases**
   - Dataset shift; uncalibrated probabilities; Grad-CAM coarseness; LLM may over-hedge or
     mis-summarise; offline fallback is not a real embedding model.

10. **Future work** — calibration (temperature scaling), more diverse data, segmentation-grade
    explainability, multi-view fusion, human-in-the-loop evaluation with clinicians.

11. **Conclusion**

12. **References** — datasets (Kermany et al. 2018 chest X-ray; Tschandl et al. 2018 HAM10000),
    Grad-CAM (Selvaraju et al. 2017), EfficientNet (Tan & Le 2019), plus the KB sources.

13. **Appendix** — full hyper-parameters (`src/config.py`), extra plots, prompt text.
