# RSNA Chest X-Ray Analysis
 
Three deep learning approaches to pneumonia detection on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) dataset — classification, object detection, and semantic segmentation — all trained on the same chest X-ray data.
 
---

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![GPU](https://img.shields.io/badge/GPU-RTX3050-green)

## Approaches
 
| Approach | File | Model | Task | Output |
|---|---|---|---|---|
| Classification | `classification/classification.py` | VGG16 (Transfer Learning) | Does pneumonia exist? | Normal / Pneumonia label |
| Detection | `detection/detection.py` | Faster RCNN ResNet50 FPN | Where is the pneumonia? | Bounding box coordinates |
| Segmentation | `segmentation/segmentation.py` | U-Net (from scratch) | Which pixels are pneumonia? | Pixel-wise binary mask |
 
---
 
## Dataset
 
**RSNA Pneumonia Detection Challenge**  
Download from Kaggle: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
 
After downloading, place files as:
 
```
rsna-pneumonia-detection-challenge/
├── stage_2_train_images/     ← .dcm chest X-ray files
├── stage_2_test_images/      ← .dcm test files
└── stage_2_train_labels.csv  ← bounding box annotations
```
 
Update the `TRAIN_DIR`, `TEST_DIR`, `CSV_PATH` variables in each script to match your local path.
 
---
 
## Repository Structure
 
```
rsna-chest-xray-analysis/
├── classification/
│   └── classification.py
├── detection/
│   └── detection.py
├── segmentation/
│   └── segmentation.py
└── README.md
```
 
---
 
## Setup
 
```bash
pip install torch torchvision pydicom pandas numpy pillow scikit-learn matplotlib seaborn
```
 
---
 
## Run
 
```bash
# Classification
python classification/classification.py
 
# Detection
python detection/detection.py
 
# Segmentation
python segmentation/segmentation.py
```
 
---
 
## Model Details
 
### 1. Classification — VGG16
 
| Setting | Value |
|---|---|
| Base model | VGG16 pretrained on ImageNet |
| Frozen layers | All feature layers except last 4 |
| Custom head | Linear(4096→256) → ReLU → Dropout(0.4) → Linear(256→2) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.2) |
| Image size | 224×224 |
| Batch size | 4 |
| Epochs | 10 |
| Augmentations | HorizontalFlip, Rotation(10°), ColorJitter, RandomCrop |
 
### 2. Detection — Faster RCNN
 
| Setting | Value |
|---|---|
| Base model | Faster RCNN ResNet50 FPN pretrained |
| Frozen layers | Backbone (ResNet50) |
| Custom head | FastRCNNPredictor (2 classes: background, pneumonia) |
| Loss | Faster RCNN internal (cls + bbox + rpn) |
| Optimizer | SGD (lr=0.005, momentum=0.9, weight_decay=0.0005) |
| Scheduler | StepLR (step=3, gamma=0.1) |
| Image size | 224×224 |
| Batch size | 4 |
| Epochs | 10 |
| Augmentations | HorizontalFlip, Rotation(10°), ColorJitter, RandomCrop |
 
### 3. Segmentation — U-Net
 
| Setting | Value |
|---|---|
| Architecture | U-Net built from scratch |
| Encoder | 4 blocks: 64→128→256→512 channels |
| Bottleneck | 512→1024 channels |
| Decoder | 4 blocks with skip connections: 1024→512→256→128→64 |
| Loss | BCEWithLogitsLoss + DiceLoss |
| Optimizer | Adam (lr=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Image size | 256×256 |
| Batch size | 8 |
| Epochs | 10 |
| Input channels | 1 (grayscale) |
| Augmentations | HorizontalFlip, Normalize |
 
---
 
## Results
 
### Classification
 
| Metric | Value |
|---|---|
| Best Val Accuracy | __%  |
| Test Accuracy | __%  |
| Precision | __ |
| Recall | __ |
| F1 Score | __ |
 
### Detection
 
| Metric | Value |
|---|---|
| Best Train Loss | 27.7919|
| Best Val Loss |27.2433|
 
### Segmentation
 
| Metric | Value |
|---|---|
| Best Val Loss | 1.0096|
| Best Val Dice Score |0.5481|
| Train Dice Score |0.5013 |
 
> Fill in values after training. Add your output plots below.
 
---
 
## Output Plots
 
### Classification
| Training Curves | Confusion Matrix | Sample Predictions |
|---|---|---|
| `chesx_ray_training.png` | `confusion_matrix.png` | `Predicted.png` |
 
### Detection
| Training Curves |
|---|
| `detection_training.png` |

### Training
<img width="640" height="480" alt="detection_training" src="https://github.com/user-attachments/assets/b496d74e-51b8-4849-9c5f-55ad95dcd824" />

### Segmentation
| Training Curves | Predictions |
|---|---|
| `segmentation_training.png` | `segmentation_predictions.png` |

### Training

<img width="1200" height="400" alt="segmentation_training" src="https://github.com/user-attachments/assets/6066294b-fa69-441f-9b9c-61a092a7a765" />

### Predictions
<img width="1200" height="900" alt="segmentation_predictions" src="https://github.com/user-attachments/assets/ca34f471-1f57-4ca8-89e5-4b383b69b229" />

 
---
 
## Why Three Approaches?
 
| Question | Approach |
|---|---|
| Does this patient have pneumonia? | Classification — fast, yes/no answer |
| Roughly where in the lung is it? | Detection — draws a box around the region |
| Exactly which pixels are affected? | Segmentation — precise pixel-level boundary |
 
Each approach answers a different clinical question. Classification is fastest and simplest. Detection adds localization. Segmentation gives the most precise output but requires the most compute.
 
---
 
## Comparison
 
| Aspect | Classification | Detection | Segmentation |
|---|---|---|---|
| Model | VGG16 | Faster RCNN | U-Net |
| Pretrained | Yes (ImageNet) | Yes (COCO) | No (from scratch) |
| Output | Label | Bounding box | Pixel mask |
| Annotation used | Target (0/1) | x, y, w, h boxes | x, y, w, h → converted to mask |
| Spatial precision | None | Approximate | Exact |
| Training speed | Fast | Medium | Medium |
| Params | ~138M (frozen) | ~41M (frozen) | ~31M (all trainable) |


## Author
**Vikas Reddy**
- GitHub: [@vikasreddy11](https://github.com/vikasreddy11)
- LinkedIn: [Vikas Reddy](https://www.linkedin.com/in/vikas-reddy-veeramreddy-26057138a)


