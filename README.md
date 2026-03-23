# 🚗 Drivable Area Detection

A deep learning project that detects **drivable areas** and **adjacent lanes** in road images and videos using a U-Net segmentation model trained on the BDD100K dataset.

<div align="Left">
    <img src="Readme Files\Lane_detection_gif.gif" width="1000" height="400">
</div>
---

## 📌 Project Overview

This project uses a **U-Net encoder-decoder architecture** to perform semantic segmentation of road scenes. Given a dashcam image or video frame, the model predicts:

| Colour in Output | Meaning |
|---|---|
| 🔴 Red | Drivable area (safe to drive) |
| 🔵 Blue | Adjacent drivable lane |
| ⚫ Black (masked out) | Background / non-drivable |

The model was trained on **3,000 images** from the [BDD100K dataset](https://bdd-data.berkeley.edu/), resized to `160×80` pixels. Labels are RGB-coded segmentation masks.

---

## 📁 Project Structure

```
drivable_area_detection/
│
├── configs/
│   └── config.yaml              # All hyperparameters and paths
│
├── src/
│   ├── data/
│   │   └── dataset.py           # CustomDataset + DataLoader builder
│   ├── models/
│   │   └── unet.py              # U-Net architecture definition
│   ├── inference/
│   │   └── predictor.py         # Prediction + blending pipeline
│   └── utils/
│       ├── helpers.py           # load_config, save/load checkpoint, etc.
│       └── logger.py            # Logging setup
│
├── app/
│   └── main.py                  # FastAPI web server (REST API)
│
├── tests/
│   └── test_model.py            # Unit tests
│
├── train.py                     # Training entry point
├── predict.py                   # CLI prediction script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker container definition
└── README.md                    # This file
```

---

## 🧠 Model Architecture — U-Net

The model follows the classic **U-Net** encoder-decoder structure:

```
Input (3, H, W)
     │
  [Encoder]
  Conv → MaxPool × 4 levels     ← feature extraction + downsampling
     │
  [Bottleneck]
  Conv Block                    ← deepest representation
     │
  [Decoder]
  ConvTranspose → Concat × 4    ← upsampling + skip connections
     │
Output (3, H, W)                ← 3-channel RGB segmentation mask
```

- **Input channels**: 3 (RGB image)
- **Output channels**: 3 (RGB segmentation mask)
- **Loss function**: `CrossEntropyLoss` with soft RGB label targets
- **Optimizer**: `Adam` (lr = 0.001)
- **Epochs**: 50

<div align="Left">
    <img src="Readme Files/Unet.png" width="1000" height="400">
</div>
---

## 🗂️ Dataset

**BDD100K** — Berkeley DeepDrive 100K dataset

- **Images**: 3,000 dashcam images, resized to `160×80`
- **Labels**: RGB segmentation masks (same size)
  - `[255, 0, 0]` = Drivable area
  - `[0, 0, 255]` = Adjacent lane
  - `[0, 255, 0]` = Background
- **Split**: 70% train / 30% validation
- **Format**: Pickle files (`images3000_160.p`, `labels3000_160.p`)

To download the full dataset, visit: https://bdd-data.berkeley.edu/

<div align="Left">
    <img src="Readme Files/dataset_sample.png" width="1000" height="400">
</div>

---
## ⚙️ Configuration

All settings are stored in `configs/config.yaml`:

```yaml
data:
  images_path: "dataset/images3000_160.p"
  labels_path: "dataset/labels3000_160.p"
  image_height: 80
  image_width: 160
  train_split: 0.7
  batch_size: 16

model:
  in_channels: 3
  out_channels: 3

training:
  epochs: 50
  learning_rate: 0.001
  save_best: true

paths:
  model_checkpoint: "checkpoints/lanesegment.pth"

inference:
  device: "cuda"   # or "cpu"
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/drivable_area_detection.git
cd drivable_area_detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Place your pickle files inside a `dataset/` folder:

```
dataset/
├── images3000_160.p
└── labels3000_160.p
```

### 4. Train the Model

```bash
python train.py
# or with custom config:
python train.py --config configs/config.yaml
```

Training logs are printed per epoch:
```
Epoch [  1/50] train=0.6423  val=0.5981  (12.3s)
Epoch [  2/50] train=0.5102  val=0.4873  (11.8s)
  -> checkpoint saved (val=0.4873)
...
```

The best model checkpoint is saved at `checkpoints/lane_segment.pth`.

---

## 🔍 Running Prediction

### On a Single Image

```bash
python predict.py --input road.jpg --output result.jpg
```

This saves two files:
- `result.jpg` — original image blended with the lane mask
- `result_mask.jpg` — raw lane mask only

### On a Video

```bash
python predict.py --input dashcam.mp4 --output annotated.mp4
```

---

## 🧪 Prediction Pipeline (Technical Detail)

The prediction follows the **exact same pipeline** as the original notebook:

```python
# 1. Load image and convert BGR → RGB (cv2 loads BGR by default)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 2. Run forward pass — get raw 3-channel output
im   = transform(image_rgb).unsqueeze(0).to(device)
pred = model(im)
test = pred.cpu().detach().numpy()    # shape: (3, H, W)

# 3. Split channels + apply binary threshold at 40
r, g, b = rgb_channel(test, thresholding=True, thresh=40)

# 4. Build lane mask — drop green (background)
blank      = np.zeros_like(r).astype(np.uint8)
lane_image = np.dstack((r, blank, b))            # Red=drivable, Blue=adjacent
lane_image = cv2.resize(lane_image, (w, h))

# 5. Blend with original image
result = cv2.addWeighted(lane_image.astype(np.uint8), 0.4,
                         image_rgb, 1.3, 0)
```

> **Important**: The model outputs raw pixel values — not probabilities. The `rgb_channel()` function with `thresh=40` extracts meaningful predictions from these raw values. This is why softmax/argmax must NOT be used.

---

## 🌐 REST API (FastAPI)

You can also run the model as a web service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then send a POST request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@road.jpg" \
     --output result.jpg
```

---

## 🐳 Docker

### Build

```bash
docker build -t drivable-area-detection .
```

### Run

```bash
docker run -p 8000:8000 drivable-area-detection
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Loss (final) | ~0.28 |
| Validation Loss (final) | ~0.31 |
| Inference Speed (GPU) | ~30 FPS |
| Inference Speed (CPU) | ~5 FPS |

**Sample Output:**

The red region shows the detected drivable area, and blue shows adjacent drivable lanes — both overlaid on the original dashcam image.

<div align="Left">
    <img src="Readme Files/output_1.png" width="1000" height="400">
</div>

Furthermore, The mask and original image is blended using cv2.bitwise_and technique.

<div align="Left">
    <img src="Readme Files/output_2.png" width="1000" height="400">
</div>

Testing on sample video:

<div align="Left">
    <img src="Readme Files\output_3.gif" width="1000" height="400">
</div>
---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
fastapi>=0.100.0
uvicorn>=0.23.0
pytest>=7.4.0
```

---

## 🤝 Acknowledgements

- [BDD100K Dataset](https://bdd-data.berkeley.edu/) — Berkeley DeepDrive
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al.
- Original notebook implementation by Krishna Sanjay Ambekar

---

## 📄 License

This project is licensed under the MIT License.
