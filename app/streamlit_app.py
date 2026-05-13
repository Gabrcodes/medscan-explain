"""MedScan + Explain — demo UI.

Upload a chest X-ray (or a dermoscopy patch) and see:
  1. the classifier's prediction + per-class probabilities,
  2. a Grad-CAM heat-map overlay ("where the model looked"),
  3. a RAG-grounded, explicitly non-diagnostic preliminary note from Claude on Bedrock
     (falls back to an offline template if AWS isn't configured).

Run:  streamlit run app/streamlit_app.py

This app never stores uploaded images — they are held in memory for the request only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `streamlit run app/streamlit_app.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
from PIL import Image

from src.config import TrainConfig, BedrockConfig, CHECKPOINT_DIR
from src.bedrock_report import generate_report, ClinicalKB, make_bedrock_client

st.set_page_config(page_title="MedScan + Explain", page_icon="🩻", layout="wide")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _load_model(checkpoint_path: str):
    import torch
    from src.model import build_model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    if ckpt.get("class_names"):
        cfg._resolved_classes = list(ckpt["class_names"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


@st.cache_resource(show_spinner=False)
def _load_kb(region: str):
    bcfg = BedrockConfig(region=region)
    client = make_bedrock_client(bcfg)
    return ClinicalKB(bcfg, client=client).build(), bcfg, (client is not None)


def _list_checkpoints() -> list[str]:
    return sorted(str(p) for p in CHECKPOINT_DIR.glob("*_best.pt"))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
st.sidebar.caption("MedScan + Explain · CSE351 Spring 2026 demo")

ckpts = _list_checkpoints()
if ckpts:
    checkpoint_path = st.sidebar.selectbox("Model checkpoint", ckpts)
else:
    checkpoint_path = st.sidebar.text_input(
        "Path to a trained checkpoint (.pt)",
        value=str(CHECKPOINT_DIR / "pneumonia_efficientnet_b0_best.pt"),
    )
    st.sidebar.warning("No checkpoint found in `checkpoints/`. Train one with "
                       "`python -m src.train --dataset pneumonia` first.")

region = st.sidebar.text_input("AWS region (Bedrock)", value="us-east-1")
age_group = st.sidebar.selectbox("Patient age group (synthetic metadata)",
                                 ["unspecified", "pediatric", "adult", "older adult"])
view = st.sidebar.text_input("Image view / note (synthetic)", value="")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Not a medical device.** Outputs are statistical pattern matches for an "
    "educational course project, not diagnoses. Always consult a qualified clinician."
)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🩻 MedScan + Explain")
st.markdown(
    "Classify a medical image, see **why** via Grad-CAM, and read an LLM-generated, "
    "retrieval-grounded **preliminary note** — explicitly non-diagnostic."
)

uploaded = st.file_uploader("Upload an image (chest X-ray or dermoscopy patch)",
                            type=["png", "jpg", "jpeg", "bmp"])

if uploaded is None:
    st.stop()

image = Image.open(uploaded).convert("RGB")

try:
    model, cfg, ckpt = _load_model(checkpoint_path)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model checkpoint: {e}")
    st.stop()

from src.gradcam import gradcam_overlay, describe_cam, top_findings
from src.bedrock_report import generate_multilabel_report

multilabel = getattr(cfg, "is_multilabel", False)
patient_meta = {}
if age_group != "unspecified":
    patient_meta["age_group"] = age_group
if view.strip():
    patient_meta["note"] = view.strip()

if multilabel:
    threshold = st.slider("Finding threshold (probability above which a finding is highlighted)",
                          0.05, 0.95, 0.5, 0.05)
    with st.spinner("Running classifier + Grad-CAM…"):
        # one forward pass gives all 14 scores; we take the top finding's heat-map for the main view
        overlay, cam, pred_idx, probs = gradcam_overlay(model, cfg, image)
        finds = top_findings(probs, cfg.class_names, threshold=threshold, top_k=4)
        cam_descriptions = {}
        cam_imgs = {}
        for idx, name, _ in finds[:3]:
            ov, cm, _, _ = gradcam_overlay(model, cfg, image, target_class=idx)
            cam_imgs[name] = ov
            cam_descriptions[name] = describe_cam(cm, name)

    col_img, col_cam = st.columns(2)
    with col_img:
        st.subheader("Input"); st.image(image, use_column_width=True)
    with col_cam:
        top_name = finds[0][1]
        st.subheader(f"Grad-CAM — top finding: {top_name}")
        st.image(cam_imgs.get(top_name, overlay), use_column_width=True,
                 caption="Coarse attention heat-map (not a lesion boundary).")

    st.subheader("Findings (independent per-finding scores — several can be flagged)")
    import pandas as pd
    df = pd.DataFrame({"finding": cfg.class_names,
                       "score": [float(p) for p in probs]}).sort_values("score", ascending=False)
    st.dataframe(df.style.format({"score": "{:.1%}"}), use_container_width=True, hide_index=True)
    highlighted = [n for _, n, p in finds]
    st.markdown("**Highlighted:** " + ", ".join(f"`{n}` ({p:.0%})" for _, n, p in finds))
    if len(cam_imgs) > 1:
        cols = st.columns(len(cam_imgs))
        for c, (name, ov) in zip(cols, cam_imgs.items()):
            with c:
                st.image(ov, use_column_width=True, caption=name)
                st.caption(cam_descriptions[name])

    st.subheader("Preliminary note (RAG · Claude on Bedrock)")
    with st.spinner("Retrieving clinical references and drafting the note…"):
        kb, bcfg, online = _load_kb(region)
        result = generate_multilabel_report(
            dataset=cfg.dataset, class_names=cfg.class_names, probs=probs,
            cam_descriptions=cam_descriptions, threshold=threshold, top_k=4,
            patient_meta=patient_meta, bedrock_cfg=bcfg, kb=kb,
        )
else:
    col_img, col_cam = st.columns(2)
    with col_img:
        st.subheader("Input"); st.image(image, use_column_width=True)
    with st.spinner("Running classifier + Grad-CAM…"):
        overlay, cam, pred_idx, probs = gradcam_overlay(model, cfg, image)
        cam_text = describe_cam(cam, cfg.class_names[pred_idx])
    with col_cam:
        st.subheader("Grad-CAM — where the model looked")
        st.image(overlay, use_column_width=True, caption="Coarse attention heat-map (not a lesion boundary).")
    st.subheader("Prediction")
    st.markdown(f"**{cfg.class_names[pred_idx]}**  ·  model confidence (uncalibrated softmax): "
                f"**{probs[pred_idx]:.1%}**")
    st.bar_chart({name: float(p) for name, p in zip(cfg.class_names, probs)})
    st.caption(cam_text)
    st.subheader("Preliminary note (RAG · Claude on Bedrock)")
    with st.spinner("Retrieving clinical references and drafting the note…"):
        kb, bcfg, online = _load_kb(region)
        result = generate_report(
            dataset=cfg.dataset, class_names=cfg.class_names, pred_idx=pred_idx,
            probs=probs, cam_description=cam_text, patient_meta=patient_meta,
            bedrock_cfg=bcfg, kb=kb,
        )

if not online:
    st.warning("AWS Bedrock not configured in this environment — showing the offline "
               "template note. Configure credentials and request model access to see the "
               "Claude-generated version.")

st.markdown(result["report"])

with st.expander("Provenance / debug"):
    st.write("**Backend:**", result["backend"])
    st.write("**Retrieved KB snippets:**", result["retrieved"])
    st.write("**Checkpoint selection metric:**", ckpt.get("select_metric"), "=", ckpt.get("best_select", ckpt.get("best_val_f1")))
    st.write("**Dataset notes:**")
    from src.config import DATASETS
    st.caption(DATASETS[cfg.dataset]["notes"])

st.markdown("---")
st.caption("MedScan + Explain is a CSE351 course project. It is not a medical device and "
           "must not be used for diagnosis or treatment decisions.")
