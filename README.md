# GitHub Repository Description

**Dog vs Cat Classifier (CNN)**  
A Convolutional Neural Network (CNN) model that classifies images as either dogs or cats. This project demonstrates image preprocessing, CNN architecture building, model training, and prediction using TensorFlow/Keras. Achieves ~83% validation accuracy after 25 epochs.

---

# README.md

# Dog vs Cat Classifier using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Convolutional Neural Network (CNN) model that classifies whether an image contains a dog or a cat.

## Features

- Image preprocessing with data augmentation
- Custom CNN architecture with:
  - Convolutional layers
  - Max pooling layers
  - Fully connected layers
- Model training with validation
- Single image prediction capability
- Achieves ~83% validation accuracy

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dog-cat-classifier.git
   cd dog-cat-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset should be organized as follows:
```
Dog_Classifier/
│
├── training_set/
│   ├── cats/
│   └── dogs/
│
└── test_set/
    ├── cats/
    └── dogs/
```

## Usage

1. Run the Jupyter notebook `cat_or_dog-CNN.ipynb` to:
   - Preprocess the images
   - Build and train the CNN model
   - Evaluate on test data
   - Make single predictions

2. For single prediction:
   - Place your image in `single_prediction/` folder
   - Rename it to `cat_or_dog.jpg`
   - Run the prediction cell in the notebook

## Model Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPool2D(pool_size=2, strides=2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPool2D(pool_size=2, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## Training Results

After 25 epochs:
- Training accuracy: ~90%
- Validation accuracy: ~83%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow/Keras documentation
- Andrew Ng's Deep Learning Specialization
