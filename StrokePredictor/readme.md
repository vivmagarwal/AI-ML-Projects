# Stroke Prediction Model - Machine Learning Project

## Introduction
Welcome to my first machine learning project where I've developed a stroke prediction model using various Python libraries like Pandas, Scikit-Learn, Seaborn, and others. This project is aimed at beginners in data science and machine learning, providing a practical example of how to approach a real-world problem with these tools.

### Project Overview
The goal of this project is to predict the likelihood of a stroke based on several health indicators and demographic information. This repository contains all the code and datasets used for training and evaluating the model.

As my first venture into machine learning, this project is primarily focused on learning and experimentation. The model and approaches used here are not fully optimized but serve as a great starting point for understanding the basics of machine learning in Python.

## Repository Contents
- `stroke_train_set.csv`: The training dataset.
- `stroke_test_set_nogt.csv`: The test dataset without ground truth labels.
- `stroke_prediction_submission.csv`: Sample submission file with predictions.
- `Stroke_Prediction.ipynb`: Jupyter notebook containing all the code and analysis.

## Getting Started
To get started with this project, you'll need to have Python installed on your system along with the following libraries:
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Scikit-Learn
- XGBoost
- Imbalanced-Learn

Absolutely, I can help compile a short installation and getting started guide for setting up a machine learning environment on your MacBook Pro M1 Laptop. Here's a comprehensive guide based on the steps you've followed:

---

## Machine Learning Environment Setup Guide for MacBook Pro M1

#### Installing Miniconda

First, you need to install Miniconda, a smaller version of Anaconda that includes only Conda, Python, the packages they depend on, and a small number of other useful packages.

1. **Download Miniconda**:
   Open the Terminal and run:
   ```bash
   curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
   ```

2. **Install Miniconda**:
   Execute the downloaded script:
   ```bash
   sh Miniconda3-latest-MacOSX-arm64.sh
   ```
   Follow the on-screen instructions to complete the installation.

#### Creating a New Conda Environment

Creating a separate environment for your machine learning projects is a good practice to avoid dependency conflicts.

1. **Create a Conda environment**:
   Name your environment `mlenv` and specify the Python version. Here, we use Python 3.10.13:
   ```bash
   conda create -n mlenv python=3.10.13
   ```

2. **Activate the environment**:
   Activate the newly created environment:
   ```bash
   conda activate mlenv
   ```

#### Setting Up VSCode

If you are using Visual Studio Code (VSCode) as your Integrated Development Environment (IDE), you should select the `mlenv` environment as your Python kernel.

1. **Open VSCode**:
   Launch Visual Studio Code.

2. **Select the Python Kernel**:
   In the bottom-right corner, click on the Python version, and then select `mlenv (Python)` as your active kernel.

#### Installing PyTorch

Install PyTorch, a popular deep learning library.

1. **Install PyTorch**:
   With `mlenv` activated, run:
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

#### Installing TensorFlow

TensorFlow is another major library for machine learning and deep learning.

1. **Install TensorFlow dependencies**:
   ```bash
   conda install -c apple tensorflow-deps
   ```

2. **Install TensorFlow and TensorFlow Metal** (for GPU support):
   ```bash
   pip install tensorflow-macos
   pip install tensorflow-metal
   ```

#### Other Essential Installations

Install other essential Python libraries for data science and machine learning.

1. **Install Numpy**:
   ```bash
   conda install numpy
   ```

2. **Install Matplotlib**:
   ```bash
   conda install matplotlib
   ```

3. **Install Pandas**:
   ```bash
   conda install pandas
   ```

4. **Install Scikit-learn**:
   ```bash
   conda install scikit-learn
   ```

5. **Install XGBoost**:
   ```bash
   conda install -c conda-forge xgboost
   ```

6. **Install Imbalanced-learn**:
   ```bash
   conda install -c conda-forge imbalanced-learn
   ```

### Verifying the Installation

After completing the installations, you should verify that each library has been installed correctly. Open a Python interpreter in your terminal or create a new Python file in VSCode, and run the following commands:

```python
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import xgboost as xgb
import imblearn

print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Matplotlib version: {plt.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"Imbalanced-learn version: {imblearn.__version__}")
```

This will print the versions of all the installed libraries, confirming their successful installation.


## Project Workflow
The project follows these key steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and removing duplicates.
2. **Exploratory Data Analysis (EDA)**: Understanding data distributions and correlations between different variables.
3. **Feature Selection**: Choosing the most relevant features for the model.
4. **Data Resampling**: Balancing the dataset using SMOTE.
5. **Model Training**: Training a Support Vector Classifier (SVC) model.
6. **Model Evaluation**: Assessing the model's performance using metrics like accuracy, precision, recall, and F1 score.
7. **Making Predictions**: Using the trained model to make predictions on new data.

## Code Overview
- The code is organized in a Jupyter Notebook for ease of understanding and visualization.
- Each step of the process, from data loading to making predictions, is clearly documented with code and comments.
- Visualizations are used extensively to provide insights into the data and model's performance.

## Challenges and Learnings
- Handling imbalanced datasets: Learned about techniques like SMOTE for oversampling.
- Feature selection: Understanding how to choose the most impactful features for a model.
- Model evaluation: Gained experience in interpreting various performance metrics.

## Conclusion
This project, while not highly optimized, is a great starting point for anyone new to machine learning. It covers fundamental aspects of a machine learning pipeline, from data preprocessing to model evaluation.

## Future Work
- Experiment with different models and hyperparameters to improve performance.
- Explore more advanced techniques for feature engineering and selection.
- Implement model deployment strategies for real-world use.

---

**Note:** This project is a learning endeavor and is open to suggestions and improvements. Feel free to fork, star, and contribute to this repository!