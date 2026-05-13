# Sample chest X-rays for testing

A handful of representative images pulled from the **held-out test splits** of the
datasets we trained on. Upload any of these at <https://ai.gabr.online> to exercise the
classifier + Grad-CAM + Bedrock report pipeline end to end.

| File | Source (test split) | Known label(s) |
|---|---|---|
| `pneumonia_NORMAL.png` | Chest X-Ray Pneumonia (Kermany 2018) | Normal |
| `pneumonia_PNEUMONIA.png` | Chest X-Ray Pneumonia (Kermany 2018) | Pneumonia |
| `nih_No_Finding.png` | NIH ChestX-ray14 | No Finding |
| `nih_Cardiomegaly.png` | NIH ChestX-ray14 | Cardiomegaly, Infiltration, Mass, Nodule |
| `nih_Effusion.png` | NIH ChestX-ray14 | Effusion, Emphysema, Infiltration, Pneumothorax |
| `nih_Pneumothorax.png` | NIH ChestX-ray14 | Emphysema, Pneumothorax |
| `nih_Atelectasis.png` | NIH ChestX-ray14 | Atelectasis, Cardiomegaly, Emphysema, Mass, Pneumothorax |
| `nih_Emphysema.png` | NIH ChestX-ray14 | Emphysema, Pneumothorax |
| `nih_Mass.png` | NIH ChestX-ray14 | Infiltration, Mass |
| `nih_Edema.png` | NIH ChestX-ray14 | Cardiomegaly, Edema, Effusion |

> The NIH images are inherently **multi-label** — many real radiographs show more than
> one finding at once. That is exactly what the ConvNeXt-Base / ViT-Base models are
> trained to surface, and what makes the demo interesting.

**Licenses** — Chest X-Ray Pneumonia is CC BY 4.0 (Kermany et al. 2018); NIH ChestX-ray14
is in the public domain (NIH Clinical Center release, Wang et al. 2017). Both permit
redistribution.

**Reminder:** these are test-set leaks by construction — the model has *not* seen these
exact images during training, so its predictions on them are a fair check, but it has
seen images from the same source / patient population. Treat the outputs as
demonstrations, not clinical evidence.
