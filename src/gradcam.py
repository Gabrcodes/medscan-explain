"""Grad-CAM explainability utilities.

Grad-CAM (Selvaraju et al., 2017) weights the final conv feature maps by the gradient
of the target class score, giving a coarse spatial heat-map of "where the model looked."
This satisfies the guidelines' interpretability requirement (alongside the LLM's textual
explanation) and feeds a short natural-language region description into the Bedrock report.

Public API:
  * gradcam_overlay(model, cfg, pil_image, target_class=None)
        -> (overlay_rgb_uint8, raw_cam_float01, predicted_class, prob_vector)
  * describe_cam(cam, class_name) -> short text like
        "Model attention concentrated in the lower-right lung field."
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .config import TrainConfig, IMAGENET_MEAN, IMAGENET_STD
from .data import build_transforms
from .model import MedScanClassifier


def _prep(pil_image, image_size: int) -> torch.Tensor:
    tfm = build_transforms(image_size, train=False)
    return tfm(pil_image).unsqueeze(0)


def gradcam_overlay(model: MedScanClassifier, cfg: TrainConfig, pil_image,
                    target_class: Optional[int] = None, device: Optional[torch.device] = None):
    """Return a Grad-CAM overlay for `pil_image`.

    Uses the `pytorch-grad-cam` package if available (robust, well-tested); otherwise
    falls back to a small hand-rolled Grad-CAM so the demo still works in a bare env.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_tensor = _prep(pil_image, cfg.image_size).to(device)

    # forward once for the prediction / probabilities
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax()) if target_class is None else int(target_class)

    target_layer = model.gradcam_target_layer()

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        cam_engine = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam_engine(input_tensor=input_tensor,
                                   targets=[ClassifierOutputTarget(pred)])[0]
        rgb = _denormalise(input_tensor[0].cpu())
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        return overlay, grayscale_cam, pred, probs
    except Exception:
        cam = _manual_gradcam(model, input_tensor, target_layer, pred)
        rgb = _denormalise(input_tensor[0].cpu())
        overlay = _overlay(rgb, cam)
        return overlay, cam, pred, probs


# --------------------------------------------------------------------------- #
# Fallback implementation (no external deps beyond torch/numpy)
# --------------------------------------------------------------------------- #
def _manual_gradcam(model, input_tensor, target_layer, target_class) -> np.ndarray:
    activations, gradients = {}, {}

    def fwd_hook(_m, _i, o):
        activations["v"] = o.detach()

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[0, target_class].backward()
        acts = activations["v"][0]            # (C, H, W)
        grads = gradients["v"][0]             # (C, H, W)
        weights = grads.mean(dim=(1, 2))      # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        return cam.cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def _denormalise(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def _overlay(rgb01: np.ndarray, cam01: np.ndarray) -> np.ndarray:
    # simple jet-ish colour map without bringing in matplotlib at call time
    import matplotlib.cm as cm
    heat = cm.get_cmap("jet")(cam01)[..., :3]
    blended = 0.45 * heat + 0.55 * rgb01
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Turn a CAM into a one-line region description for the LLM report
# --------------------------------------------------------------------------- #
def describe_cam(cam: np.ndarray, class_name: str, threshold: float = 0.6) -> str:
    """Coarse, honest description of where attention concentrates.

    Splits the image into a 3x3 grid, finds the cell(s) above `threshold` of the max,
    and names them in plain words. Deliberately vague — this is a coarse localisation,
    not a segmentation, and the wording reflects that.
    """
    h, w = cam.shape
    rows = np.array_split(np.arange(h), 3)
    cols = np.array_split(np.arange(w), 3)
    row_names = ["upper", "middle", "lower"]
    col_names = ["left", "centre", "right"]

    grid = np.zeros((3, 3))
    for i, rr in enumerate(rows):
        for j, cc in enumerate(cols):
            grid[i, j] = cam[np.ix_(rr, cc)].mean()
    grid_norm = grid / (grid.max() + 1e-8)
    hot = np.argwhere(grid_norm >= threshold)
    if len(hot) == 0 or len(hot) >= 7:
        spread = "diffusely across the image" if len(hot) >= 7 else "weakly, with no clear focus"
        return (f"Grad-CAM shows model attention distributed {spread}; "
                f"localisation for the '{class_name}' prediction is low-confidence.")
    parts = []
    for i, j in hot[:3]:
        parts.append(f"{row_names[i]}-{col_names[j]}")
    where = ", ".join(parts)
    return (f"Grad-CAM indicates model attention concentrated in the {where} "
            f"region(s) of the image for the '{class_name}' prediction. "
            f"This is a coarse heat-map, not a precise lesion boundary.")
