# 🚗 Drivable Area Detection

Semantic segmentation of drivable areas using a custom **U-Net** architecture
trained on the **BDD100K** dataset.

| Output Colour | Meaning         |
|---------------|-----------------|
| 🟥 Red         | Drivable area   |
| 🟦 Blue        | Adjacent lane   |
| 🟩 Green       | Background      |

---

## 🗂️ Project Structure

```
drivable_area_detection/
├── app/
│   └── main.py              ← Streamlit web demo
├── src/
│   ├── data/
│   │   └── dataset.py       ← Data loading & augmentation
│   ├── models/
│   │   └── unet.py          ← U-Net architecture
│   ├── inference/
│   │   └── predictor.py     ← Inference pipeline + video support
│   └── utils/
│       ├── logger.py        ← Centralised logging (console + file)
│       └── helpers.py       ← Shared utilities
├── configs/
│   └── config.yaml          ← All hyperparameters (no hardcoding)
├── tests/
│   └── test_model.py        ← 10 pytest unit tests
├── train.py                 ← Training entry-point
├── predict.py               ← CLI inference script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset pickle files inside dataset/
mkdir dataset
# Copy: images_3000_160.p  and  labels_3000_160.p  into dataset/

# 3. Train the model
python train.py

# 4. Predict on a single image
python predict.py --input road.jpg --output result.jpg

# 5. Predict on a video
python predict.py --input footage.mp4 --output annotated.mp4 --video

# 6. Launch the Streamlit live demo
streamlit run app/main.py
```

---

## 🐳 Docker

```bash
docker build -t drivable-area .
docker run -p 8501:8501 drivable-area
# Open browser → http://localhost:8501
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📊 Model Architecture (U-Net)

```
Input (3, 80, 160)
    │
    ├─ Encoder
    │   DoubleConv  3  →  64
    │   DoubleConv  64 → 128
    │   DoubleConv 128 → 256
    │   DoubleConv 256 → 512
    │
    ├─ Bottleneck
    │   DoubleConv 512 → 1024
    │
    └─ Decoder (with skip connections)
        ConvTranspose + DoubleConv 1024 → 512
        ConvTranspose + DoubleConv  512 → 256
        ConvTranspose + DoubleConv  256 → 128
        ConvTranspose + DoubleConv  128 →  64
            │
            └─ 1×1 Conv → Output (3, 80, 160)
```

---

## 🔑 Production Features

| Feature                        | Location                  |
|-------------------------------|---------------------------|
| Centralised logging (file+console) | `src/utils/logger.py` |
| Error handling with logger    | All modules               |
| Config-driven (no magic numbers) | `configs/config.yaml`  |
| Unit tests                    | `tests/test_model.py`     |
| Live demo UI                  | `app/main.py`             |
| Docker containerisation       | `Dockerfile`              |
| CLI inference                 | `predict.py`              |
| Video inference               | `src/inference/predictor.py` |

---

## 📁 Dataset

- **Source:** [BDD100K](https://bdd-data.berkeley.edu/)
- **Size used:** 3,000 images + labels (160×80 px)
- **Augmentation:** Horizontal flip (doubles dataset to 6,000 samples)
- **Label encoding:** Green = background | Red = drivable | Blue = adjacent lane
