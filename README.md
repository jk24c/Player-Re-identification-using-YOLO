# Player Re-Identification using YOLO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Compatible-red)

---

## 📋 Table of Contents

1. [Requirements](#requirements)  
2. [Approach & Methodology](#approach--methodology)  
   1. [Pipeline Design](#pipeline-design)  
   2. [Key Techniques](#key-techniques)  
   3. [Challenges & Solutions](#challenges--solutions)  
3. [Updated `requirements.txt`](#updated-requirementstxt)  
4. [Kaggle-Specific Setup](#kaggle-specific-setup)  
   1. [Enable GPU](#1-enable-gpu)   
   2. [Install Dependencies](#2-install-dependencies)  
    

---

## Requirements

- **OS**: Linux / macOS / Windows  
- **Python**: 3.8 or above  
- **GPU (optional but recommended)**: NVIDIA T4 / P100 / RTX series  
- **CUDA**: 11.x  

---

## Approach & Methodology

### Pipeline Design

1. **Detection**  
   - Utilizes a fine‑tuned YOLOv8 model (players & ball)  
   - Outputs bounding boxes & confidences  

2. **Tracking**  
   - **DeepSORT** core:  
     - **Kalman Filter** → predicts object motion  
     - **OSNet Embedder** → 512‑D appearance features  

3. **Filtering**  
   - Discard boxes below area threshold  
   - Enforce aspect ratio ∈ [0.3, 3.0]  

---

### Key Techniques

- **Re‑Identification (Re‑ID)**  
  - OSNet produces 512‑dim embeddings  
  - Matches appearance across frames  

- **Track Management**  
  | Parameter   | Value | Description                                      |
  |-------------|-------|--------------------------------------------------|
  | `n_init`    | 5     | Frames to confirm a new track                    |
  | `max_age`   | 90    | Frames to retain a track during occlusion        |

- **Performance**  
  - Achieves ∼8 FPS on NVIDIA T4 (real‑time optimized)  

---

### Challenges & Solutions

| Challenge                 | Impact                                   | Solution                                            |
|---------------------------|------------------------------------------|-----------------------------------------------------|
| Similar team uniforms     | ID switches between players             | ↑ Motion weight in Kalman updates                   |
| Occlusions (e.g., posts)  | Tracks lost during overlaps             | Extend `max_age` to 90 frames                       |
| False positives (refs/ball)| Spurious detections as players          | Strict filter → only `class_id == 2` (players)      |

---

## Updated `requirements.txt`

```text
ultralytics==8.0.0
opencv-python-headless==4.5.5.64
deep-sort-realtime==1.3.0
gdown==4.7.1
torchreid @ git+https://github.com/KaiyangZhou/deep-person-reid.git
```
## 4. Kaggle-Specific Setup

Follow these steps to configure and run the player re-identification pipeline on a Kaggle notebook.

---

### 4.1 Enable GPU

> **⚠️ Manual Step**  
> In your Kaggle Notebook: **Settings → Accelerator → GPU (P100)**

---

### 4.2 Install Dependencies

```text
!pip install -q \
    ultralytics==8.0.0 \
    opencv-python-headless==4.5.5.64 \
    deep-sort-realtime==1.3.0 \
    gdown==4.7.1 \
    git+https://github.com/KaiyangZhou/deep-person-reid.git

