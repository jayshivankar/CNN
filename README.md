# CIFAR-10 Image Classification with Convolutional Neural Networks (CNN)

This project demonstrates image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It includes model training, evaluation, and visualization of performance metrics.

---

## 📦 Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of:
- 60,000 32×32 color images
- 10 classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- Split into 50,000 training images and 10,000 test images

---

## 🧰 Technologies & Libraries Used

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 🧪 Model Workflow

1. **Data Loading**: CIFAR-10 dataset loaded using `tensorflow.keras.datasets`
2. **Preprocessing**:
   - Normalizing image data
   - Reshaping label arrays
3. **Model Architecture**:
   - Stacked convolutional layers
   - MaxPooling layers
   - Fully connected Dense layers
4. **Training**:
   - Model compiled using `sparse_categorical_crossentropy` loss
   - Trained and validated using accuracy metrics
5. **Evaluation**:
   - Confusion matrix
   - Accuracy plots
   - Sample predictions with true vs predicted labels

---

## 📊 Results

The notebook includes:
- Visual comparison of sample images
- Training vs validation accuracy/loss graphs
- Confusion matrix to evaluate classification performance

---

## 🚀 Getting Started

### Prerequisites

Install the necessary Python packages:

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

### Running the Notebook

Clone this repository and run the notebook:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
jupyter notebook
```

Open the `.ipynb` file and run the cells.

---

## 📁 File Structure

```
.
├── cifar10_cnn_classification.ipynb  # Jupyter Notebook with all code and visualizations
└── README.md                         # Project documentation
```

---

