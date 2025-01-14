# CytologIA Cell Classification

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference](#inference)
- [Results Visualization](#results-visualization)

## 🔍 Overview
This project implements a deep learning-based cell classification system using YOLOv8. It's designed to detect and classify different types of cells in microscopic images, with particular focus on cytological samples.

## ✨ Features
- YOLOv8-based cell detection and classification
- Comprehensive training pipeline with optimized parameters
- Inference tools for single images and batch processing
- Advanced data augmentation techniques
- Detailed visualization tools for training metrics and results
- Submission generation for competitions

## 📁 Project Structure
```
cytologia_project/
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── inference.py
│   ├── utils.py
│   └── visualize.py
├── requirements.txt
└── main.py
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cytologia-classification.git
cd cytologia-classification
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables to avoid OpenMP runtime issues:
```python
# Add to the start of your script
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

## 💻 Usage

### Training
```bash
python main.py --mode train --dataset_path /path/to/dataset
```

### Single Image Prediction
```bash
python main.py --mode predict --model_path path/to/best.pt --image_path path/to/image.jpg
```

### Generate Submission.csv
```bash
python main.py --mode predict --model_path path/to/best.pt --test_csv path/to/test.csv --test_dir path/to/test/images
```

## 📊 Dataset
The project expects data in the following structure:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```


## 🚀 Model Training
Training configuration is defined in `src/train.py`. Key parameters include:

```python
model.train(
    epochs=80,
    batch=8,
    imgsz=800,
    patience=10,
    # ... other parameters
)
```

## 🔮 Inference
The project supports:
- Single image inference with visualization
- Batch processing for multiple images
- Competition submission generation
- Confidence threshold customization

## 📈 Results Visualization
Available visualization tools:
- Training metrics plots
- Confusion matrix
- Prediction grid visualization
- Class distribution analysis

## Example Inference
```python
python main.py --mode predict --model_path best.pt --image_path inference.jpg
```
<div style="display: flex; justify-content: space-between;"> <img src="https://github.com/user-attachments/assets/654081b6-6d53-4492-aa2d-ebb186558808" width="48%" /> <img src="https://github.com/user-attachments/assets/1e158714-ed77-4e66-867d-675b15e85311" width="48%" /> </div>
