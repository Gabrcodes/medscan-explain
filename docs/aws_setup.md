# AWS setup — GPU training + Bedrock

This project uses AWS for two independent things:

1. **A GPU EC2 instance** to fine-tune the image classifier (`src/train.py`).
2. **Amazon Bedrock** (Claude + Titan Embeddings) for the RAG "preliminary note" layer.

You can do (2) from your laptop and only spin up (1) when you actually train. Tear the
GPU instance down when you're done — that's the expensive part.

---

## 0. One-time: install + configure the AWS CLI (local machine)

```powershell
# Windows (this repo's machine)
winget install -e --id Amazon.AWSCLI
aws --version
aws configure        # paste an access key/secret with EC2 + Bedrock permissions, region us-east-1
```

If you'd rather not put long-lived keys on disk, use `aws configure sso` or an IAM role.

---

## 1. Request Bedrock model access (do this first — approval can take minutes)

1. AWS Console → **Amazon Bedrock** → *Model access* → **Manage model access**.
2. Enable at least:
   - **Anthropic — Claude 3.5 Sonnet** (or whichever Claude you've been granted),
   - **Amazon — Titan Text Embeddings V2**.
3. Wait until status shows *Access granted*.
4. Note the exact **model IDs / inference profile IDs** and put them in
   `src/config.py → BedrockConfig` if they differ from the defaults
   (`anthropic.claude-3-5-sonnet-20240620-v1:0`, `amazon.titan-embed-text-v2:0`).

Quick check from the CLI:

```bash
aws bedrock list-foundation-models --region us-east-1 \
  --query "modelSummaries[?contains(modelId,'claude') || contains(modelId,'titan-embed')].modelId"
```

Smoke-test the RAG layer locally (no GPU needed):

```bash
pip install boto3 faiss-cpu numpy
python -m src.bedrock_report          # should print backend: bedrock:anthropic.claude-...
```

If credentials aren't set, it prints `backend: offline-template` and still works — handy
for developing the UI without burning tokens.

---

## 2. Launch a GPU instance for training

Recommended instance types (cheapest first; all handle EfficientNet-B0 / ResNet-50 fine):

| Instance | GPU | Notes |
|---|---|---|
| `g4dn.xlarge` | 1× T4 16 GB | cheapest; ~10–25 min/epoch on these datasets |
| `g5.xlarge` | 1× A10G 24 GB | ~2–3× faster than T4 |

Use the **Deep Learning AMI (PyTorch, Ubuntu)** so CUDA/PyTorch are pre-installed.

### Option A — Console
EC2 → Launch instance → search AMI "Deep Learning OSS PyTorch" → type `g4dn.xlarge` →
key pair → allow SSH (port 22) from your IP → 100 GB gp3 root volume → Launch.

### Option B — CLI (fill in the blanks)

```bash
# find the latest DLAMI id in your region
AMI=$(aws ssm get-parameter --region us-east-1 \
  --name /aws/service/deeplearning/ami/x86_64/oss-pytorch-2.3-ubuntu-22.04/recommended/image_id \
  --query Parameter.Value --output text)

aws ec2 run-instances --region us-east-1 \
  --image-id "$AMI" --instance-type g4dn.xlarge \
  --key-name YOUR_KEYPAIR --security-group-ids sg-XXXX \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=medscan-train}]'
```

> **g4dn/g5 need vCPU quota.** If launch fails with a quota error, request an increase
> for "Running On-Demand G and VT instances" in Service Quotas (often instant for small sizes).

---

## 3. Train on the instance

```bash
ssh -i YOUR_KEYPAIR.pem ubuntu@<public-ip>

git clone <your-github-repo-url> medscan && cd medscan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# datasets stream from HuggingFace — no Kaggle login needed
python -m src.train --dataset pneumonia --backbone efficientnet_b0 --epochs 10
python -m src.train --dataset pneumonia --backbone resnet50        --epochs 10   # comparison backbone
python -m src.evaluate --checkpoint checkpoints/pneumonia_efficientnet_b0_best.pt

# (stretch goal) skin lesions
python -m src.train --dataset ham10000 --backbone efficientnet_b0 --epochs 15
```

Copy the artifacts back so the demo runs anywhere:

```bash
# from your laptop
scp -i YOUR_KEYPAIR.pem -r ubuntu@<public-ip>:~/medscan/checkpoints ./checkpoints
scp -i YOUR_KEYPAIR.pem -r ubuntu@<public-ip>:~/medscan/outputs     ./outputs
```

Log the **wall-clock training time and instance type** (printed by `src/train.py` and
saved in the checkpoint as `train_seconds`) — you need it for the Responsible-AI
"sustainability / CO₂" paragraph. Rough estimate:
`kgCO2 ≈ gpu_hours × instance_power_kW × grid_intensity_kgCO2_per_kWh`
(e.g. a T4 ≈ 0.07 kW; us-east-1 grid ≈ 0.38 kgCO₂/kWh).

---

## 4. ⚠️ Shut it down

```bash
aws ec2 stop-instances  --region us-east-1 --instance-ids i-XXXX   # keeps the disk, stops GPU billing
aws ec2 terminate-instances --region us-east-1 --instance-ids i-XXXX   # deletes everything
```

GPU instances bill per second while *running*; a forgotten `g5.xlarge` is ~$1/hr.
Bedrock is pay-per-token with no idle cost, so you can leave that alone.
