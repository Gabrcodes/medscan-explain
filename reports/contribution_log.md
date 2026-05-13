# Contribution Log — MedScan + Explain

> Course requirement: each member states their role and the tasks they completed.
> All members are expected to understand the *entire* project and answer questions on any part.

| Member | Role | Tasks completed |
|---|---|---|
| _name_ | Data & training lead | dataset pipeline (`src/data.py`), training loop & tuning (`src/train.py`, `src/config.py`), backbone comparison runs |
| _name_ | Evaluation & explainability | metrics + baselines (`src/evaluate.py`), Grad-CAM (`src/gradcam.py`), result figures, failure-case analysis |
| _name_ | LLM agent & RAG | knowledge base curation (`src/knowledge_base/`), Bedrock + retrieval layer (`src/bedrock_report.py`), prompt design & guardrails |
| _name_ | Demo, report & Responsible AI | Streamlit app (`app/streamlit_app.py`), AWS setup (`docs/aws_setup.md`), written report, slides, Responsible AI section |

_Adjust rows for a team of 3 or 5. If 5, split "Demo" and "Report/Responsible AI" into two roles._

## Meeting / milestone notes
- _date_ — project chosen: MedScan + Explain (DL classifier + Grad-CAM + Bedrock RAG report).
- _date_ — repo scaffolded; data pipeline + model + training + eval + Grad-CAM + RAG layer + Streamlit app implemented.
- _date_ — TODO: run training on EC2 GPU, fill results, write report, build slides, record demo video.

## AI-assistance disclosure
Parts of the implementation were written with an AI coding assistant. AI-assisted code is
annotated in-source. The team has reviewed and understands all submitted code; the model
design, analysis, and report are the team's own work.
