# Chest X-Ray Classification using Transfer Learning (VGG16)

Image classification using the **fine-tuning method** on chest X-ray images.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![GPU](https://img.shields.io/badge/GPU-RTX3050-green)

---

## Project Overview

This project applies transfer learning using the VGG16 convolutional neural network to classify chest X-ray images into two categories: Normal and Pneumonia. The model is fine-tuned on a medical imaging dataset and evaluated using multiple performance metrics including accuracy, precision, recall, and F1 score.
---

# Dataset

The model is trained on a **Chest X-ray dataset** consisting of two classes:

| Class | Description |
|------|-------------|
| Normal | Healthy chest X-ray images |
| Pneumonia | Chest X-ray images with pneumonia infection |

---

# Files

| File | Description |
|-----|-------------|
| `chest_x_ray.py` | Implementation of VGG16 fine-tuning for chest X-ray classification |

---

# Fine-Tuning Strategy

Transfer learning is implemented by **fine-tuning the VGG16 network**.

Steps used in the project:

1. Load pretrained **VGG16** from `torchvision`.
2. Freeze early convolution layers to preserve general image features.
3. Unfreeze the final layers for task-specific learning.
4. Replace the final classifier layer for **binary classification**.
5. Train the modified network using the chest X-ray dataset.

---

# Model Architecture

### VGG16 Fine-Tuning
```
Input Image (224×224)
        ↓
VGG16 Convolution Backbone
        ↓
Flatten (25088)
        ↓
Linear (25088 → 256)
        ↓
      ReLU
        ↓
Dropout(0.4)
        ↓
Linear (256 → 2)
```

# Evaluation Metrics

Model performance is evaluated using:

| Metric | Description |
|------|-------------|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions |
| Recall | Ability to detect pneumonia cases |
| F1 Score | Balance between precision and recall |

---

# Results

| Metric | Score |
|------|------|
| Accuracy | 97.96% |
| Precision |0.9885 |
| Recall |0.9839 |
| F1 Score | 0.9862|

## Training curves
- The graph shows accuracy increasing and loss decreasing over 10 epochs.
- Train and Val curves are close together which means no overfitting.
<img width="1200" height="400" alt="chesx_ray_training" src="https://github.com/user-attachments/assets/796828a3-f362-4297-ad8f-1a97462bdfdc" />


## Confusion Matrix
- The diagonal cells represent correct predictions.
- Off-diagonal values indicate misclassifications.
- The matrix shows how well the model distinguishes between Normal and Pneumonia cases.
<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/64fb5769-6ec7-42ed-837f-35b04dfeb73d" />



## Predictions
- Green title indicates a correct prediction.
- Red title indicates an incorrect prediction.
<img width="1600" height="300" alt="Predicted" src="https://github.com/user-attachments/assets/2bc7e626-16af-4fd4-b253-bf0802eccbdb" />



## How to Run

### Install dependencies
```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

### Run code
```bash
python chest_x_ray.py
```

## Author
**Vikas Reddy**
- GitHub: [@vikasreddy11](https://github.com/vikasreddy11)
- LinkedIn: [Vikas Reddy](https://www.linkedin.com/in/vikas-reddy-veeramreddy-26057138a)


