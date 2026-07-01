# Machine Learning from Scratch — KNN & AdaBoost

This project implements two binary classification algorithms from scratch. **K-Nearest Neighbors (KNN)** and **AdaBoost with a weighted linear weak learner**. The models classify wine samples as either high quality (`+1`) or low quality (`-1`) using physicochemical features from the Wine Quality dataset.

The goal of the project was to build the core machine learning logic manually using only **NumPy** and **Pandas**, without using libraries such as scikit-learn for the classifiers.

## Project Overview

The project contains two independent classifiers:

- **K-Nearest Neighbors (`knn.py`)**
  - Loads CSV feature and label data.
  - Standardizes features with z-score normalization.
  - Computes Euclidean distance from each test point to all training points.
  - Predicts using majority vote among the nearest neighbors.
  - Handles ties by choosing the closest neighbor among tied labels.

- **AdaBoost (`boosting.py`)**
  - Implements AdaBoost over multiple boosting rounds.
  - Uses a weighted linear classifier as the weak learner.
  - Computes weighted positive and negative class centroids.
  - Builds a decision boundary halfway between the weighted centroids.
  - Updates sample weights after each round based on whether each point was misclassified.
  - Combines weak learners using their learned alpha weights.

## Dataset

The project uses the Wine Quality dataset for binary classification.

`wine_X.csv` contains 11 numeric wine features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

`wine_y.csv` contains the binary labels:

- `+1` = high-quality wine
- `-1` = low-quality wine

The included training dataset has **1,279 samples** and **11 features**.

## Technologies Used

- Python
- NumPy
- Pandas
- CSV data processing
- From-scratch machine learning implementation

## Files

```text
.
├── knn.py          # K-Nearest Neighbors classifier
├── boosting.py     # AdaBoost classifier with weighted linear weak learners
├── wine_X.csv      # Training feature matrix
├── wine_y.csv      # Training labels
└── README.md
```

## How the KNN Classifier Works

The KNN model stores the full training set during training. During prediction, it calculates the Euclidean distance from each test sample to every training sample, sorts the neighbors by distance and predicts the class with the most votes among the `k` nearest points.

Because KNN is distance-based, the code standardizes every feature using the training-set mean and standard deviation before making predictions. This prevents large-scale features from dominating the distance calculation.

## How the AdaBoost Classifier Works

The boosting model starts with equal weights for every training example. For each boosting round, it computes weighted centroids for the positive and negative classes, creates a linear decision rule between those centroids, calculates the weighted error and assigns the weak learner an alpha value.

Misclassified samples receive larger weights for the next round, while correctly classified samples receive smaller weights. Final predictions are made by taking the sign of the weighted sum of all weak learner predictions.

## Running the Project

Install the required Python packages:

```bash
pip install numpy pandas
```

The assignment autograder calls each file through a `run()` function with this signature:

```python
def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
```

Example usage from another Python script or an interactive shell:

```python
import knn

knn.run(
    "wine_X.csv",
    "wine_y.csv",
    "wine_X.csv",
    "knn_predictions.csv"
)
```

```python
import boosting

boosting.run(
    "wine_X.csv",
    "wine_y.csv",
    "wine_X.csv",
    "boosting_predictions.csv"
)
```

Each prediction file is written with one integer prediction per line and no header:

```text
1
-1
-1
1
```

## What I Implemented

- Implemented data loading from CSV files using Pandas.
- Added z-score feature normalization for training and test data.
- Built a KNN classifier from scratch using NumPy distance calculations.
- Implemented KNN voting and tie-breaking behavior.
- Built an AdaBoost classifier from scratch using weighted training examples.
- Implemented weighted class centroids, weak learner error calculation, alpha weighting, sample-weight updates and final ensemble prediction.
- Wrote prediction output in the exact format required by the autograder.

## Notes

This project was completed to experiment with machine learning focused on understanding the mechanics behind classification algorithms. The implementation intentionally avoids external machine learning libraries so the core KNN and AdaBoost logic is visible in the source code.
