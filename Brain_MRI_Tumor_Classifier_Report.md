# Brain MRI Tumor Classifier — Technical Report

**Project:** Brain MRI Tumor Classification Web Application
**Stack:** Python, Flask, PyTorch, ResNet-18/50, Pillow, SQLite, Gunicorn
**Purpose:** Educational and Research Use Only

---

## 1. Dataset Overview

**Dataset Name:** Brain Tumor MRI Dataset
**Source:** Kaggle — [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
**Format:** JPEG images organized into class-named folders
**Total Images:** ~7,023

### Dataset Split

| Split    | Images | Percentage |
|----------|--------|------------|
| Training | 5,712  | ~81.3%     |
| Testing  | 1,311  | ~18.7%     |
| **Total**| **7,023** | **100%** |

---

## 2. Tumor Types and Class Distribution

### Training Set

| Class           | Images | % of Training |
|-----------------|--------|---------------|
| Glioma          | 1,321  | 23.1%         |
| Meningioma      | 1,339  | 23.4%         |
| No Tumor        | 1,595  | 27.9%         |
| Pituitary Tumor | 1,457  | 25.5%         |
| **Total**       | **5,712** | **100%**   |

### Testing Set

| Class           | Images | % of Testing |
|-----------------|--------|--------------|
| Glioma          | 300    | 22.9%        |
| Meningioma      | 306    | 23.3%        |
| No Tumor        | 405    | 30.9%        |
| Pituitary Tumor | 300    | 22.9%        |
| **Total**       | **1,311** | **100%**  |

### Tumor Type Descriptions

| Class | Description |
|-------|-------------|
| **Glioma** | A tumor arising from glial cells (supporting cells) in the brain or spinal cord. One of the most common and aggressive primary brain tumors. |
| **Meningioma** | A tumor forming on the meninges — the membranes surrounding the brain and spinal cord. Usually slow-growing and often benign. |
| **Pituitary Tumor** | A tumor in the pituitary gland located at the base of the brain. Most are non-cancerous but can affect hormonal function. |
| **No Tumor** | Healthy brain MRI scan with no detectable tumor region. Included as a negative class for realistic classification. |

---

## 3. Model Architecture

### Backbone: ResNet-18 (Fast Mode) / ResNet-50 (Full Mode)

This project uses **transfer learning** on a ResNet family backbone pretrained on ImageNet.

#### What is ResNet?

ResNet (Residual Network) is a deep convolutional neural network architecture introduced by Microsoft Research in 2015. Its key innovation is the **residual (skip) connection** — a shortcut that lets the gradient flow directly through layers during backpropagation, solving the vanishing gradient problem that plagued very deep networks.

```
Input → [Conv Block] → [+ skip] → [Conv Block] → [+ skip] → ... → FC Layer → Output
              ↑__________________________|
                    Residual Connection
```

#### ResNet-18 vs ResNet-50

| Property         | ResNet-18         | ResNet-50          |
|------------------|-------------------|--------------------|
| Layers           | 18                | 50                 |
| Parameters       | ~11.7 million     | ~25.6 million      |
| FC input size    | 512               | 2048               |
| Speed            | Faster (CPU-friendly) | Slower (GPU preferred) |
| Use case in app  | `--fast` / default deployed | Full training mode |

#### How ResNet is Used in This Project

1. **Pretrained weights loaded:** The ResNet backbone is initialized with ImageNet weights (trained on 1.2 million images across 1,000 classes).

2. **Final layer replaced:** The original 1,000-class fully-connected output layer is removed and replaced with a new `nn.Linear(in_features, 4)` layer — one output per tumor class.

   ```python
   in_features = model.fc.in_features   # 512 for ResNet-18, 2048 for ResNet-50
   model.fc = nn.Linear(in_features, 4) # 4 classes
   ```

3. **Fine-tuned end-to-end:** All layers are trained (not frozen) on the brain MRI dataset, allowing the network to adapt its low-level feature detectors from general image patterns to MRI-specific ones.

4. **Architecture auto-detection:** At inference time, the app detects which backbone was used by inspecting the FC weight shape from the saved `.pth` file — no manual configuration needed.

---

## 4. Preprocessing Pipeline

Preprocessing is applied to every image before it is fed into the model, both during training and inference.

### Steps (in order)

| Step | Operation | Details |
|------|-----------|---------|
| 1 | **Resize** | All images resized to 224×224 px (160×160 in fast mode) |
| 2 | **Convert to Tensor** | Pixel values scaled from [0, 255] → [0.0, 1.0] |
| 3 | **Normalize** | Per-channel normalization using ImageNet statistics |

### ImageNet Normalization Values

```
Mean:  [0.485, 0.456, 0.406]   (R, G, B)
Std:   [0.229, 0.224, 0.225]   (R, G, B)
```

**Why ImageNet normalization?**
The backbone was pretrained on ImageNet using these statistics. Applying the same normalization at fine-tuning and inference ensures the feature activations are in the same numerical range the network learned from, improving convergence and accuracy.

### Image Handling at Upload

- All uploaded images are converted to **RGB** (handles grayscale MRI and RGBA PNGs)
- Supported formats: JPG, PNG, BMP, TIFF
- Maximum file size: 16 MB
- Images saved with a UUID filename to prevent collisions

---

## 5. Data Augmentation

Augmentation is applied **only during training**, not at inference. It artificially increases dataset diversity to reduce overfitting.

### Augmentation Techniques Used

| Technique | Parameters | Purpose |
|-----------|------------|---------|
| **Random Horizontal Flip** | p = 0.5 | MRI scans can be mirrored; teaches position invariance |
| **Random Rotation** | ±10 degrees | Handles slight head tilt variations in real scans |

```python
transforms.RandomHorizontalFlip()
transforms.RandomRotation(10)
```

### Why These Specific Augmentations?

- **Horizontal flip:** Brain anatomy is roughly symmetric. Flipping a scan is medically plausible and doubles effective data.
- **Rotation ±10°:** Scans are not always perfectly aligned. Small rotations simulate real-world variation without distorting the anatomy enough to confuse the label.
- **No color jitter or brightness shifts:** MRI pixel intensity carries diagnostic meaning, so aggressive color augmentation would be inappropriate.

---

## 6. Training Configuration

| Parameter         | Default Value     | Fast Mode Value  |
|-------------------|-------------------|------------------|
| Backbone          | ResNet-50         | ResNet-18        |
| Image size        | 224 × 224 px      | 160 × 160 px     |
| Epochs            | 12                | 12               |
| Batch size        | 16                | 8                |
| Learning rate     | 1e-4              | 1e-4             |
| Optimizer         | Adam              | Adam             |
| Loss function     | CrossEntropyLoss  | CrossEntropyLoss |
| LR scheduler      | ReduceLROnPlateau (factor=0.5, patience=2) | same |
| DataLoader workers| 2 (0 on CPU/Windows) | 0             |
| Best model saving | Yes (highest val accuracy) | Yes        |

### Optimizer: Adam
Adam (Adaptive Moment Estimation) adapts the learning rate per parameter. It works well for transfer learning because different layers may need different effective learning rates.

### Scheduler: ReduceLROnPlateau
Monitors validation accuracy. If accuracy does not improve for 2 consecutive epochs, the learning rate is halved. This prevents overshooting the optimal weights late in training.

### Loss Function: CrossEntropyLoss
Standard multi-class classification loss. Combines log-softmax and negative log-likelihood. Penalizes confident wrong predictions more heavily.

---

## 7. Inference Pipeline

When a user uploads an MRI scan, the following steps happen:

1. **Image loaded** and converted to RGB
2. **Preprocessing applied** (resize → tensor → normalize)
3. **Forward pass** through ResNet — produces 4 logits
4. **Softmax** converts logits to probabilities (sum = 100%)
5. **Argmax** selects the predicted class
6. **Confidence** reported as the highest probability percentage
7. **Gradient saliency** computed (backward pass on top class score)
8. Three visualizations generated:
   - **Red heatmap** — gradient magnitude overlaid on original MRI
   - **Orange tumor spot** — warm glow highlighting the suspected region
   - **Annotation image** — bounding box with "TUMOR REGION DETECTED" label

---

## 8. Visualization Methods

### Gradient Saliency (Heatmap)

A simple but effective technique:
1. Run forward pass, get top class score
2. Backpropagate the score through the network
3. Take the absolute value of input gradients
4. Average across color channels → scalar heat map per pixel
5. Normalize to [0, 1] and resize to original image dimensions
6. Pixels with high gradient = regions that most influenced the prediction

```
heat = |∂score/∂input|  averaged over channels, normalized
```

### Orange Tumor Spot Overlay

Built on the same gradient map:
- Regions with heat > 0.30 are blended with an orange-to-yellow colormap
- Alpha ramps smoothly from 0 (at threshold 0.30) to 0.78 (at 0.60+)
- Color: orange (255, 140, 0) → yellow (255, 220, 0) based on heat intensity
- Soft gamma (α^1.5) applied for smooth, non-harsh edges

### Bounding Box Annotation

- Saliency mask thresholded at 0.55
- Bounding box fitted to the rows/columns where mask is active
- Dashed red box drawn with L-shaped corner ticks
- Red banner label: "TUMOR REGION DETECTED"
- Coordinate readout shown below the box

---

## 9. Web Application Stack

| Component | Technology |
|-----------|------------|
| Web framework | Flask 2.3+ |
| Model inference | PyTorch 2.0+, torchvision |
| Image processing | Pillow, NumPy |
| Database | SQLite (via built-in `sqlite3`) |
| Production server | Gunicorn |
| Frontend | HTML/CSS/Vanilla JS |

### Monitoring Dashboard
Every prediction is logged to SQLite with:
- Timestamp, predicted class, confidence
- Tumor detected (boolean), tumor area percentage
- Image URL, heatmap URL

Dashboard available at `/monitor` with bar chart and CSV export.

---

## 10. Limitations

- Model is trained on a curated Kaggle dataset; real-world MRI scans vary significantly in resolution, contrast, and acquisition protocol
- Gradient saliency is an approximation — it shows where the model is sensitive, not a true anatomical segmentation
- No clinical validation has been performed
- For educational and research use only — not a medical device

---

*Report generated from source code analysis of the Brain MRI Tumor Classifier project.*
*GitHub: https://github.com/Aaryamallik03/Brain_tumor_detection*
