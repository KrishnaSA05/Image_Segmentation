# 🚗 Drivable Area Detection using Image Segmentation

A production-grade deep learning system for detecting **drivable areas** and **adjacent lanes** in dashcam footage using a U-Net segmentation model trained on the BDD100K dataset.

<div align="left">
    <img src="Readme Files/Lane_detection_gif.gif" width="1000" height="400">
</div>

---

## 📊 Results

### Segmentation Performance — BDD100K Validation Set

| Class | IoU |
|---|---|
| Background | ~72% |
| Drivable Area | ~62% |
| Adjacent Lane | ~42% |
| **mIoU** | **~60%** |

### Inference Speed

| Environment | Speed |
|---|---|
| GPU (CUDA) | ~30 FPS |
| CPU (PyTorch) | ~5 FPS |
| CPU (ONNX Runtime) | ~12 FPS |

> mIoU was computed using the `SegmentationMetrics` accumulator (`src/metrics/iou.py`) over the full 900-sample validation split (30% of 3,000 BDD100K images).

---

## 📌 Project Overview

This project performs **semantic segmentation** of road scenes to identify where a vehicle can safely drive. Given a dashcam image or video frame, the model outputs a pixel-level mask:

| Colour | Meaning |
|---|---|
| 🔴 Red | Drivable area — safe to drive |
| 🔵 Blue | Adjacent drivable lane |
| 🟩 Green | Background — non-drivable |

### Key Features

- **U-Net architecture** — encoder-decoder with skip connections, trained from scratch in PyTorch
- **BDD100K dataset** — 3,000 dashcam images, 160×80px, RGB segmentation masks
- **Albumentations augmentation** — brightness, contrast, CLAHE, hue shift, rotation, occlusion simulation
- **mIoU metrics** — per-class IoU tracked every validation epoch via `SegmentationMetrics`
- **Grad-CAM explainability** — visual attention maps showing which pixels drive each class prediction
- **ONNX export** — cross-platform deployment with 2.4× CPU speedup vs PyTorch
- **Dockerised Streamlit app** — interactive demo with segmentation and Grad-CAM tabs
- **AWS deployment** — Docker image pushed to ECR, hosted on EC2 t2.micro
- **GitHub Actions CI/CD** — automated test → build → ECR push on every `main` push

---

## 📁 Project Structure

```
drivable-area-detection/
│
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
│
├── src/
│   ├── data/
│   │   └── dataset.py               # Dataset + Albumentations pipeline
│   ├── models/
│   │   └── unet.py                  # U-Net architecture
│   ├── inference/
│   │   └── predictor.py             # Inference + video pipeline
│   ├── metrics/                     # Phase 1
│   │   └── iou.py                   # SegmentationMetrics (mIoU accumulator)
│   ├── explainability/              # Phase 1
│   │   └── gradcam.py               # Grad-CAM for U-Net bottleneck
│   └── export/                      # Phase 2
│       └── onnx_export.py           # ONNX export + verification + benchmark
│
├── app/
│   └── main.py                      # Streamlit app (segmentation + Grad-CAM tabs)
│
├── scripts/                         # Phase 2
│   ├── deploy_aws.sh                # ECR push + EC2 launch
│   ├── ec2_ecr_policy.json          # IAM policy for EC2 → ECR access
│   └── ec2_trust_policy.json        # IAM trust policy
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml                # GitHub Actions: test → build → push ECR
│
├── tests/
│   └── test_model.py                # Unit tests
│
├── train.py                         # Training entry point (logs mIoU per epoch)
├── predict.py                       # CLI prediction script
├── gradcam_visualize.py             # CLI Grad-CAM script
├── Dockerfile                       # Multi-stage build, non-root user, healthcheck
└── requirements.txt
```

---

## 🧠 Model Architecture — U-Net

```
Input (3, 80, 160)
      │
  ┌───▼───┐  DoubleConv  64      skip ──────────────────────────────┐
  │Encoder│  DoubleConv  128     skip ─────────────────────────┐    │
  │       │  DoubleConv  256     skip ────────────────────┐    │    │
  │       │  DoubleConv  512     skip ───────────────┐    │    │    │
  └───────┘  MaxPool ×4                              │    │    │    │
      │                                              │    │    │    │
  ┌───▼────────┐                                     │    │    │    │
  │ Bottleneck │  DoubleConv 1024  ← Grad-CAM target │    │    │    │
  └───▼────────┘                                     │    │    │    │
  ┌───▼───┐  ConvTranspose + Concat ─────────────────┘    │    │    │
  │Decoder│  ConvTranspose + Concat ──────────────────────┘    │    │
  │       │  ConvTranspose + Concat ───────────────────────────┘    │
  │       │  ConvTranspose + Concat ────────────────────────────────┘
  └───────┘
      │
  ┌───▼───┐
  │  Head │  Conv2d 1×1
  └───────┘
      │
Output (3, 80, 160)   ← 3-channel RGB segmentation logits
```

- **Encoder**: 4 × DoubleConv (Conv→BN→ReLU×2) + MaxPool
- **Bottleneck**: DoubleConv(512→1024) — deepest representation, Grad-CAM hook point
- **Decoder**: 4 × ConvTranspose2d + skip connection concat + DoubleConv
- **Head**: 1×1 Conv mapping to 3 output channels (Background / Drivable / Adjacent)
- **Loss**: CrossEntropyLoss over 3 classes
- **Optimiser**: Adam (lr=0.001)
- **Parameters**: ~31M trainable

<div align="left">
    <img src="Readme Files/Unet.png" width="1000" height="400">
</div>

---

## 🗂️ Dataset — BDD100K

**Berkeley DeepDrive 100K** — large-scale autonomous driving dataset

| Property | Value |
|---|---|
| Images used | 3,000 dashcam frames |
| Resolution | 160 × 80 px |
| Label format | RGB segmentation masks |
| Train split | 70% (2,100 images) |
| Val split | 30% (900 images) |
| Source | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) |

Label colour mapping:

| RGB | Class | Meaning |
|---|---|---|
| `[255, 0, 0]` | 1 | Drivable area |
| `[0, 0, 255]` | 2 | Adjacent lane |
| `[0, 255, 0]` | 0 | Background |

<div align="left">
    <img src="Readme Files/dataset_sample.png" width="1000" height="400">
</div>

---

## ⚙️ Augmentation Pipeline (Albumentations)

Training uses a domain-tuned Albumentations pipeline. Validation uses no augmentation to keep metrics deterministic.

| Transform | Parameters | Purpose |
|---|---|---|
| `HorizontalFlip` | p=0.5 | Road symmetry |
| `RandomBrightnessContrast` | ±0.2, p=0.5 | Lighting variation |
| `HueSaturationValue` | hue±10, sat±20, p=0.3 | Road surface colour variation |
| `CLAHE` | clip=2.0, p=0.3 | Under/overexposed frame recovery |
| `ShiftScaleRotate` | shift±3%, scale±5%, rot±10°, p=0.4 | Camera jitter |
| `CoarseDropout` | 4 holes, p=0.2 | Occlusion simulation |

All spatial transforms are applied **simultaneously** to the image and its mask via Albumentations' `additional_targets`, ensuring pixel-perfect label alignment.

---

## 🚀 Getting Started

### 1. Clone

```bash
git clone https://github.com/yourname/drivable-area-detection.git
cd drivable-area-detection
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

```
dataset/
├── images_3000_160.p
└── labels_3000_160.p
```

### 4. Train

```bash
python train.py
```

Training logs mIoU every epoch:

```
Epoch [  1/20]  train_loss=0.6423  val_loss=0.5981  mIoU=41.23%  time=14.2s
  Background   IoU: 58.11%
  Drivable     IoU: 44.32%
  Adjacent     IoU: 21.27%

Epoch [  2/20]  train_loss=0.5102  val_loss=0.4873  mIoU=53.67%  time=13.8s
  ↳ New best mIoU=53.67% — checkpoint saved
```

Best checkpoint saved at `checkpoints/lane_segment.pth` (tracked by **mIoU**, not val loss).

---

## 🔍 Inference

### Single Image

```bash
python predict.py --input road.jpg --output result.jpg
```

### Video

```bash
python predict.py --input dashcam.mp4 --output annotated.mp4
```

---

## 🔥 Grad-CAM Explainability

Grad-CAM shows which image regions the model focuses on for each class prediction.
Hooks into the U-Net bottleneck (`model.bottleneck.conv[3]`) — the deepest encoder representation.

```bash
# Drivable area attention map (default)
python gradcam_visualize.py --input road.jpg --output outputs/

# All 3 classes
python gradcam_visualize.py --input road.jpg --output outputs/ --all-classes
```

Outputs two files per class:
- `road_gradcam_drivable_heatmap.jpg` — colour activation map
- `road_gradcam_drivable_overlay.jpg` — heatmap blended onto original image

<div align="left">
    <img src="Readme Files/output_1.png" width="1000" height="400">
</div>

---

## 📦 ONNX Export

Export the trained model for cross-platform deployment (ONNX Runtime, TensorRT, mobile):

```bash
python src/export/onnx_export.py
# → checkpoints/lane_segment.onnx

# With latency benchmark
python src/export/onnx_export.py --benchmark
```

The export script automatically verifies numerical consistency between PyTorch and ONNX Runtime outputs (max abs diff < 1e-4).

---

## 🌐 Streamlit App

```bash
streamlit run app/main.py
```

The app has two tabs:
- **Segmentation** — upload an image, see predicted mask + blended overlay
- **Grad-CAM** — select a class, see the attention heatmap in real time

---

## 🐳 Docker

```bash
# Build (multi-stage — smaller image, non-root user)
docker build -t drivable-area-detection .

# Run locally
docker run -p 8501:8501 drivable-area-detection

# Open http://localhost:8501
```

---

## ☁️ AWS Deployment (ECR + EC2)

```bash
# Edit the 4 variables at the top of the script, then:
chmod +x scripts/deploy_aws.sh
./scripts/deploy_aws.sh
# → prints http://<ec2-ip>:8501 when ready
```

The script: creates an ECR repository → builds and pushes the Docker image → launches a t2.micro EC2 instance that pulls and runs the container on boot.

---

## 🔄 CI/CD — GitHub Actions

Every push to `main` automatically:
1. Runs `pytest tests/`
2. Builds the Docker image
3. Pushes to ECR (tagged with commit SHA + `latest`)

Add these secrets to your GitHub repo (Settings → Secrets → Actions):
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REPOSITORY`

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📦 Requirements

```
torch>=2.1.0
torchvision>=0.16.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
albumentations>=1.3.0
matplotlib>=3.7.0
pillow>=10.0.0
pyyaml>=6.0
tqdm>=4.66.0
streamlit>=1.30.0
onnx>=1.15.0
onnxruntime>=1.17.0
pytest>=7.4.0
```

---

## 🤝 Acknowledgements

- [BDD100K Dataset](https://bdd-data.berkeley.edu/) — Berkeley DeepDrive
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., MICCAI 2015
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) — Selvaraju et al., ICCV 2017
- [Albumentations: Fast and Flexible Image Augmentations](https://albumentations.ai/)

---

## 📄 License

MIT License
