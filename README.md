# 🌼 CNN Flower Image Classification

This repository contains Jupyter Notebooks demonstrating the use of Convolutional Neural Networks (CNNs) for flower image classification, covering essential techniques like data augmentation and transfer learning using TensorFlow/Keras.

## 📁 Repository Contents

### 1. `cnn_flower_image_classification_data_augmentations.ipynb`
- Implements a CNN from scratch for flower image classification.
- Applies various **data augmentation techniques** such as rotation, zooming, flipping, and brightness variation to enhance generalization.
- Visualizes augmented images and training performance.

### 2. `cnn_transfer_learning.ipynb`
- Demonstrates **transfer learning** using a pre-trained model (like VGG16, InceptionV3, or MobileNet).
- Fine-tunes the model for the flower classification task.
- Shows improved performance with fewer training epochs.

### 3. (Duplicate or updated version of notebook 2)

## 🧠 Techniques Used
- CNN architecture design and training
- Data preprocessing and augmentation with `ImageDataGenerator`
- Transfer learning with pre-trained models from `keras.applications`
- Model evaluation with training/validation accuracy and loss plots

## 📊 Dataset
The models are trained on a flower classification dataset (e.g., Oxford 102 Flowers or a subset), which is loaded from a local or cloud-based directory. Instructions for dataset usage can be added depending on your dataset location and format.

## 🚀 Getting Started

### Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, Jupyter

```bash
pip install tensorflow matplotlib numpy
```

### Run Notebooks
Launch Jupyter Notebook and open the files:
```bash
jupyter notebook
```

## 📈 Results
- The data-augmented CNN achieves reasonable performance but may overfit on small datasets.
- Transfer learning significantly boosts accuracy and reduces training time.

## 📚 References
- [TensorFlow Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

