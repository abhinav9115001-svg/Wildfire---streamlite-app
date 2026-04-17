#  Wildfire Detection & Monitoring System

> Sentinel-2 MSI satellite imagery classification using EfficientNetB0 with transfer learning, Grad-CAM explainability, NBR burn index visualisation, and a Streamlit live-inference dashboard.

**Module:** COS-5031-E — Discipline-specific Artificial Intelligence Project  
**University of Bradford** | **Client:** Future AI for ALL (FALL) — Kieran Townsend  
**Team:** Md Rifat Islam · Abhinav Sharma · Md Minhajul Islam  

---

##  Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Explainability](#-explainability)
- [Ethical Considerations](#-ethical-considerations)
- [Requirements](#-requirements)
- [Team & Roles](#-team--roles)
- [License](#-license)

---

##  Project Overview

This project builds an end-to-end AI pipeline to detect active wildfires from **Sentinel-2 MSI 6-band satellite imagery** (512×512 pixels). The system:

- **Classifies** each tile as `fire` or `no-fire` using a fine-tuned EfficientNetB0 CNN
- **Localises** high-attention regions using Grad-CAM heatmaps
- **Validates** predictions against the physics-based **Normalised Burn Ratio (NBR)** index
- **Visualises** results in a real-time Streamlit monitoring dashboard

### Key Technical Decisions

| Decision | Justification |
|---|---|
| EfficientNetB0 | Best accuracy/parameter tradeoff; adapted from ImageNet to 6 spectral channels |
| Global normalisation (0–10000 / 10000) | Preserves relative NIR/SWIR band ratios essential for NBR |
| Two-phase fine-tuning | Head-only first prevents catastrophic forgetting before full unfreeze |
| Fire Recall as primary KPI | A missed fire is safety-critical; false alarms are tolerable |
| Class-weighted loss | Handles class imbalance without discarding minority samples |

---

##  Demo

```bash
streamlit run app.py
```

Upload any Sentinel-2 `.tif` tile and the dashboard returns:
- Fire / No-Fire prediction with confidence score
- Grad-CAM attention heatmap overlay
- NBR (Normalised Burn Ratio) map
- RGB false-colour composite

---

##  Repository Structure
Wildfire---streamlite-app/
│
├── app.py                          # Streamlit dashboard (main entry point)
│
├── notebooks/
│   └── wildfire_detection.ipynb    # Full training pipeline (clean version)
│
├── model/
│   └── best_model.keras            # Saved best model weights (val_accuracy)
│
├── utils/
│   ├── load_image.py               # Rasterio-based image loading + normalisation
│   ├── compute_nbr.py              # NBR index computation (NIR Band 8, SWIR Band 12)
│   └── gradcam.py                  # Grad-CAM heatmap generation
│
├── requirements.txt                # All Python dependencies
├── .gitignore
└── README.md
---

##  Dataset

**Source:** [Sentinel-2 MSI Wildfire Data for Research](https://www.kaggle.com/datasets/caffeinatedhighs/sentinel-2-msi-wildfire-data-for-research) — Kaggle

```python
import kagglehub
path = kagglehub.dataset_download(
    "caffeinatedhighs/sentinel-2-msi-wildfire-data-for-research"
)
```

### Dataset Structure
dataset/
├── chop_fire/        ← Label 1 (active fire tiles)
└── chop_no_firee/    ← Label 0 (no-fire tiles)
>  **Critical note:** `chop_no_firee` must be checked **before** `chop_fire` in any string-match label parser — `chop_fire` is a substring of `chop_no_firee` and will silently mislabel every no-fire image if checked first.

### Band Layout (per `.tif` file)

| Index | Sentinel-2 Band | Wavelength | Use |
|---|---|---|---|
| 0 | B2 | 490 nm (Blue) | RGB display |
| 1 | B3 | 560 nm (Green) | RGB display |
| 2 | B4 | 665 nm (Red) | RGB display |
| 3 | B8 | 842 nm (NIR broad) | RGB display |
| 4 | B8A | 865 nm (NIR narrow) | **NBR numerator** |
| 5 | B12 | 2190 nm (SWIR) | **NBR denominator** |

### Train / Val / Test Split

| Split | Proportion | Stratified |
|---|---|---|
| Train | 70% |  |
| Validation | 15% |  |
| Test | 15% |  |

---

##  Model Architecture
Input (512, 512, 6)
↓
EfficientNetB0 backbone
└─ First Conv2D weights adapted: ImageNet RGB kernel (k,k,3,F)
→ concatenated × 2 / 2.0 → (k,k,6,F)
└─ First 20% of layers frozen during Phase 2
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense(256, relu)
↓
Dropout(0.5)
↓
Dense(64, relu)
↓
Dropout(0.3)
↓
Dense(1, sigmoid)  →  fire probability
### Training Protocol

**Phase 1 — Head only** (base frozen)
- Optimizer: Adam, lr = 1e-3
- Epochs: up to 15 (EarlyStopping, patience=5)

**Phase 2 — Full fine-tune** (first 20% of base stays frozen)
- Optimizer: Adam, lr = 5e-6
- Epochs: up to 30 (EarlyStopping, patience=5)
- ReduceLROnPlateau: factor=0.5, patience=3

**Callbacks:** `EarlyStopping` · `ReduceLROnPlateau` · `ModelCheckpoint`  
**Loss:** Binary cross-entropy with class weights  
**Primary metric:** Fire class Recall (minimise missed detections)

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhinav9115001-svg/Wildfire---streamlite-app.git
cd Wildfire---streamlite-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API (for dataset download)

Place your `kaggle.json` credentials file at `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

##  Usage

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

### Run the full training notebook

Open `notebooks/wildfire_detection.ipynb` in Jupyter or Google Colab and run all cells **in order**.

>  Cells must be executed sequentially — Phase 1 training must complete before Phase 2 begins.

### Google Colab (recommended for GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```python
# Mount Drive and install dependencies
from google.colab import drive
drive.mount('/content/drive')
!pip install -r requirements.txt
```

---

##  Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~85–87% |
| Fire Recall | ≥ 90% (target) |
| Fire F1 Score | ≥ 85% (target) |
| AUC-ROC | ≥ 0.90 (target) |

> Final metrics depend on dataset size and training environment. Run Cell 13 of the notebook for your actual test-set numbers.

### Benchmark Comparison

| Model | Accuracy | Fire F1 |
|---|---|---|
| SVM (baseline) | ~63% | ~0.61 |
| Random Forest (baseline) | ~67% | ~0.64 |
| **EfficientNetB0 (ours)** | **~87%** | **~0.85** |

---

##  Explainability

### Normalised Burn Ratio (NBR)

Physics-based validation index computed from satellite bands:
NBR = (NIR - SWIR) / (NIR + SWIR + ε)
= (Band 8A - Band 12) / (Band 8A + Band 12 + 1e-8)
- **Negative NBR** (red in RdYlGn colormap) → burned / active fire
- **Positive NBR** (green) → healthy vegetation / no fire

### Grad-CAM

Gradient-weighted Class Activation Mapping highlights which spatial regions the model attends to. The implementation always targets the **final Conv2D layer** (found programmatically, never by index):

```python
last_conv = [l.name for l in model.layers
             if isinstance(l, tf.keras.layers.Conv2D)][-1]
```

---

##  Ethical Considerations

This system was evaluated using the **FAST AI Ethics Canvas**:

| Dimension | Risk | Mitigation |
|---|---|---|
| **Fairness** | Geographic bias (Australian fire events dominant) | Document limitation; multi-region data in future |
| **Accountability** | Model must not automate evacuation decisions | Human-in-the-loop required; outputs are decision support only |
| **Safety** | False negatives are safety-critical | Fire Recall optimised as primary KPI; ensemble fallback |
| **Transparency** | Black-box CNN | Grad-CAM + NBR overlays provided as standard outputs |

### FAIR Data Principles

- **Findable:** All artefacts versioned on GitHub with DVC
- **Accessible:** Kaggle dataset publicly downloadable
- **Interoperable:** Standard `.tif` geospatial format; TensorFlow SavedModel
- **Reusable:** Pinned `requirements.txt`; modular code structure

---

##  Requirements
tensorflow>=2.12.0
numpy>=1.23.0
rasterio>=1.3.0
kagglehub>=0.1.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
opencv-python>=4.7.0
streamlit>=1.28.0
seaborn>=0.12.0



Full pinned versions: see [`requirements.txt`](requirements.txt)

---

##  Team & Roles

| Name | Role | Responsibilities |
|---|---|---|
| **Md Rifat Islam** | Scrum Master / Project Manager | Sprint planning, PID, Streamlit dashboard, FAIR report |
| **Abhinav Sharma** | AI / ML Engineer | EfficientNetB0 architecture, transfer learning, Grad-CAM, hyperparameter tuning |
| **Md Minhajul Islam** | Data Engineer / Research Lead | Sentinel-2 pipeline, normalisation, ground-truth annotation, evaluation metrics |

---

##  License

This project was produced for academic purposes as part of COS-5031-E at the University of Bradford. All satellite imagery is sourced from publicly available Kaggle datasets. Model weights and code are released for educational use only.

---

<p align="center">
  University of Bradford &nbsp;|&nbsp; COS-5031-E &nbsp;|&nbsp; April 2026
</p>
