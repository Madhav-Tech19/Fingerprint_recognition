# Fingerprint Recognition using CNN

This repository contains a Jupyter Notebook that demonstrates a **fingerprint recognition system** built using Convolutional Neural Networks (CNNs). The project focuses on training and evaluating models to classify fingerprints based on `SubjectID` and `FingerName` categories.

## Project Overview
This project is designed to:
1. Process a raw fingerprint dataset into two categories:
   - `SubjectID`
   - `FingerName`
2. Build and train two CNN-based models for classification on these categories.
3. Evaluate the models using a randomly selected fingerprint and ensure predictions match for both `SubjectID` and `FingerName`.

## Features
- **Data Preprocessing**: Splits and processes the raw dataset into usable subsets.
- **Model Training**: Implements two CNN models for classification.
- **Evaluation**: Tests models on unseen data and evaluates prediction accuracy.

## Dependencies
This project relies on the following Python libraries:
- **Data Manipulation**: `numpy`, `os`, `random`, `itertools`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `sklearn`
- **Deep Learning**: `keras` (TensorFlow backend)
- **Image Processing**: `cv2`
- **Data Sourcing**: `kagglehub`

Install dependencies using pip:
```bash
pip install numpy matplotlib seaborn scikit-learn keras opencv-python kagglehub
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fingerprint-recognition.git
   ```
2. Navigate to the repository:
   ```bash
   cd fingerprint-recognition
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Fingerprint_recognition.ipynb
   ```
4. Follow the notebook cells to preprocess the dataset, train the models, and evaluate the predictions.

## Dataset
- The dataset used in this project is sourced using `kagglehub`. Ensure you have access to the dataset by setting up your Kaggle API credentials.
- Modify the paths in the notebook to point to your local dataset directory.
