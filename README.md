# 🧠 Brain MRI Tumor Classifier

### Deep Learning-Based Multi-Class MRI Image Classification Using ResNet-50 Transfer Learning

An educational deep learning project that classifies brain MRI images into four dataset categories using a **ResNet-50 model, PyTorch, and Flask**. The application provides an interface for image uploads, prediction results, prediction history, and exploratory saliency visualizations.

> **Disclaimer:** This project is intended for educational and research demonstrations only. It is not a medical device and must not be used for diagnosis, treatment decisions, or ruling out medical conditions.

## 📸 Screenshots


### Home Page
<img width="1137" height="651" alt="image" src="https://github.com/user-attachments/assets/00ed52cc-2356-489a-ac32-37eed752abf5" />


### MRI Image Upload
<img width="570" height="268" alt="image" src="https://github.com/user-attachments/assets/315cd369-67c8-4cb5-bb62-b20586791784" />
<img width="612" height="266" alt="image" src="https://github.com/user-attachments/assets/ee071cf5-0d6b-498e-92f1-54fb5f53ff0c" />
<img width="609" height="293" alt="image" src="https://github.com/user-attachments/assets/1c68df02-5a66-461f-a0a3-912af1426e71" />


### Prediction Results
<img width="569" height="311" alt="image" src="https://github.com/user-attachments/assets/67a54c65-d044-43b1-98d6-f5ec49a8a68d" />


### Monitoring Dashboard
<img width="289" height="404" alt="image" src="https://github.com/user-attachments/assets/93a8703f-4727-4df3-8dce-a79b80fe691f" />


## ✨ Features

- **Four-class classification:** Predicts one of four dataset labels: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
- **Transfer learning:** Uses an ImageNet-pretrained ResNet-50 backbone adapted for four output classes.
- **MRI image preprocessing:** Resizes images to 224 × 224 pixels, converts them to RGB, and applies ImageNet normalization.
- **Prediction interface:** Upload an image and view the predicted class and associated confidence scores.
- **Saliency visualization:** Generates gradient-based visualizations to explore which image regions influence model predictions.
- **Prediction monitoring:** Provides a dashboard and prediction history for inspection.
- **Local data storage:** Uses SQLite for prediction history, with CSV export described in the project report.
- **Command-line workflow:** Supports an alternative prediction workflow outside the web interface.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | Torchvision, Pillow |
| Model Architecture | ResNet-50 |
| Web Framework | Flask |
| Database | SQLite |
| Data Visualization | Matplotlib |
| Machine Learning Utilities | scikit-learn |
| Report Generation | ReportLab |

## 🧠 Model Architecture

The classifier uses **ResNet-50**, initialized with ImageNet-pretrained weights. Its original classification layer is replaced with a fully connected layer that maps the backbone's 2,048-dimensional feature representation to four output classes.

### Model Workflow

1. Upload a brain MRI image.
2. Convert the image to RGB and resize it to 224 × 224 pixels.
3. Normalize the image using ImageNet statistics.
4. Pass the preprocessed image through the ResNet-50 model.
5. Obtain scores for the four dataset classes.
6. Display the predicted class and confidence values.
7. Generate an exploratory saliency visualization when supported by the application.

```text
Brain MRI Image
       |
       v
Image Preprocessing
Resize + RGB + Normalization
       |
       v
ResNet-50 Backbone
       |
       v
Four-Class Output Layer
       |
       v
Prediction and Confidence Scores
       |
       v
Web Interface and Prediction History
```

## 📂 Dataset

The project report identifies the [Brain Tumor MRI Dataset by Masoud Nickparvar on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

The report records the following dataset distribution:

| Class | Training Folder | Testing Folder | Total |
|---|---:|---:|---:|
| Glioma | 1,321 | 300 | 1,621 |
| Meningioma | 1,339 | 306 | 1,645 |
| Pituitary Tumor | 1,457 | 300 | 1,757 |
| No Tumor | 1,595 | 405 | 2,000 |
| **Total** | **5,712** | **1,311** | **7,023** |

These counts are transcribed from the project report and should be confirmed against the exact dataset version used.

## ⚙️ Installation and Setup

### Prerequisites

- Python installed
- Git
- pip
- The project source code and trained model checkpoint

### 1. Clone the repository

```bash
git clone https://github.com/Aaryamallik03/Brain_tumor_detection.git
cd Brain_tumor_detection
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If dependencies are not listed, install the packages required by the actual application and model implementation.

### 4. Verify the configuration

Ensure that the trained model checkpoint is available at the path expected by the inference code. Check that the required templates, static assets, and any other application files are present.

### 5. Run the application

If `app.py` is the Flask application entry point:

```bash
python app.py
```

Open the local URL printed by Flask in your browser, typically `http://127.0.0.1:5000`.

> The commands above assume the repository's entry point and directory structure match the project report. Adjust them if the current repository uses different paths or startup settings.

## 📊 Evaluation Status

The original project report mentions an accuracy of approximately 97%. However, a review of the training script identified a potential evaluation issue: the default stratified workflow combines images from the original Training and Testing folders before creating a new split.

Consequently, the reported figure **is not presented as verified held-out test accuracy**. Final performance metrics need to be regenerated using a corrected evaluation protocol.

Recommended evaluation metrics include:

- Test accuracy
- Macro precision, recall, and F1-score
- Per-class precision, recall, and F1-score
- Confusion matrix

The original Testing folder should remain untouched during training, validation, checkpoint selection, and other model-development decisions.

## 🔍 Explainability and Limitations

The application includes gradient-based saliency visualizations intended to explore image regions that influence a model score.

These visualizations are exploratory. They are not validated tumor segmentation masks, and any bounding-box-style overlay derived from a saliency map must not be interpreted as a confirmed anatomical tumor boundary.

Other limitations include:

- The dataset may not represent MRI scans from different hospitals, scanners, or patient populations.
- The model predicts dataset labels rather than a complete clinical diagnosis.
- Confidence scores should not be interpreted as clinical certainty.
- Duplicate or near-duplicate images may affect evaluation results.
- Reliable generalization requires a corrected evaluation protocol and, ideally, external validation.

## 🚀 Future Improvements

- Correct the training, validation, and test partitioning strategy.
- Regenerate final evaluation metrics and the confusion matrix.
- Add reproducible experiment configurations, random seeds, and training logs.
- Explore Grad-CAM for model visualization.
- Investigate calibrated confidence estimates.
- Evaluate the model on an independent dataset.
- Explore validated segmentation methods if suitable labelled data are available.

## 🎓 Project Information

**Project:** Brain MRI Tumor Classifier  
**Institution:** National Institute of Technology Meghalaya  
**Department:** Computer Science and Engineering  
**Author:** Aarya Mallik  
**Supervisor:** Dr. Ngangbam Herojit Singh

## 📚 References

1. He, K., Zhang, X., Ren, S., and Sun, J. *Deep Residual Learning for Image Recognition.* CVPR, 2016. [DOI](https://doi.org/10.1109/CVPR.2016.90)
2. Simonyan, K., Vedaldi, A., and Zisserman, A. *Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.* [arXiv](https://arxiv.org/abs/1312.6034)
3. Selvaraju, R. R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV, 2017.
4. Nickparvar, M. *Brain Tumor MRI Dataset.* [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
5. [PyTorch ResNet-50 Documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
6. [Flask Documentation](https://flask.palletsprojects.com/)

---

*Built as an educational deep learning project exploring medical image classification and web application integration.*
