# Chest X-Ray Classification using Transfer Learning (VGG16)

Image classification using the **fine-tuning method** on chest X-ray images.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![GPU](https://img.shields.io/badge/GPU-RTX3050-green)

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
<img width="1200" height="400" alt="chesx_ray_training" src="https://github.com/user-attachments/assets/efa3094d-2d36-4c1a-95da-b8ca40b646de" />


## Confusion Matrix
- The diagonal cells represent correct predictions.
- Off-diagonal values indicate misclassifications.
- The matrix shows how well the model distinguishes between Normal and Pneumonia cases.
<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/13d0a32a-b7af-4d8f-98f4-7e0e885bd871" />


## Predictions
- Green title = correct prediction
- Red title = wrong prediction
<img width="1600" height="300" alt="Predicted" src="https://github.com/user-attachments/assets/5628d1f0-6318-4e09-952d-f5f1a62459ff" />


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


