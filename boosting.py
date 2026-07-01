#!/usr/bin/env python3
"""
CMPSC 165B - Machine Learning
Homework 3, Problem 2: Boosting Classifier
"""

import numpy as np
import pandas as pd


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # TODO: Implement
    X = pd.read_csv(X_path, header=0).values.astype(float)
    y = None
    if y_path is not None:
        y = pd.read_csv(y_path, header=0).values.ravel().astype(float)
    return X, y



def preprocess_data(X_train, X_test):
    """Preprocess training and test data."""
    # TODO: Implement
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X_train_stan = (X_train - feat_mean) / feat_std
    X_test_stan = (X_test - feat_mean) / feat_std
    return X_train_stan, X_test_stan
    #raise NotImplementedError


class BoostingClassifier:
    """AdaBoost Classifier with weighted linear classifier as weak learner."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        # TODO: Implement
        numberEpochs = getattr(self, 'T', 100)
        m = len(y)
        weights = np.ones(m) / m
        self.learner_exemplar_postive = []
        self.learner_exemplar_negative = []
        self.alphas = []

        for _ in range(numberEpochs):
            postive_mask = (y == 1)
            negative_mask = (y == -1)
            w_postive = weights[postive_mask]
            w_negative = weights[negative_mask]
            exemplar_postive = (w_postive @ X[postive_mask]) / w_postive.sum()
            exemplar_negative = (w_negative @ X[negative_mask]) / w_negative.sum()
            midpoint = (exemplar_postive + exemplar_negative) / 2
            direction = exemplar_postive - exemplar_negative
            scores = (X - midpoint) @ direction
            y_pred = np.where(scores > 0, 1, -1)

            misclassified = (y_pred != y)
            epsilon = np.clip(weights[misclassified].sum(), 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)

            weights[misclassified] *= np.exp(alpha)
            weights[~misclassified] *= np.exp(-alpha)
            weights /= weights.sum()

            self.learner_exemplar_postive.append(exemplar_postive)
            self.learner_exemplar_negative.append(exemplar_negative)
            self.alphas.append(alpha)
    def predict(self, X):
        """Predict labels for input samples."""
        # TODO: Implement
        scores = np.zeros(len(X))
        for alpha, exemplar_postive, exemplar_negative in zip(self.alphas, self.learner_exemplar_postive, self.learner_exemplar_negative):
            midpoint = (exemplar_postive + exemplar_negative) / 2
            direction = exemplar_postive - exemplar_negative
            h = np.where((X - midpoint) @ direction > 0, 1, -1)
            scores += alpha * h
        return np.where(scores > 0, 1, -1)


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # TODO: Implement
    return np.mean(y_true == y_pred)


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # TODO: Implement
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test, _ = load_data(test_data_file)
    X_train_stan, X_test_stan = preprocess_data(X_train, X_test)
    clf = BoostingClassifier()
    clf.train(X_train_stan, y_train)
    y_pred = clf.predict(X_test_stan)
    pd.Series(y_pred.astype(int)).to_csv(pred_file, index=False, header=False)
##