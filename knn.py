#!/usr/bin/env python3
"""
CMPSC 165B - Machine Learning
Homework 3, Problem 1: K-Nearest Neighbors
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
    #raise NotImplementedError


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


class KNNClassifier:
    """K-Nearest Neighbors Classifier."""

    def train(self, X, y):
        """Fit the classifier to training data."""
        # TODO: Implement
        self.k = getattr(self, 'k', 1)
        self.X_train = X
        self.y_train = y
        #raise NotImplementedError

    def predict(self, X):
        """Predict labels for input samples."""
        # TODO: Implement
        predictions = []
        for x_i in X:
            diffs = self.X_train - x_i
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            order = np.lexsort((self.y_train, dists))
            k_indices = order[:self.k]
            k_labels = self.y_train[k_indices]
            k_dists = dists[k_indices]
            positive_votes = np.sum(k_labels == 1) #
            negative_votes = np.sum(k_labels == -1)
            if negative_votes > positive_votes:
                predictions.append(-1)
            elif positive_votes > negative_votes:
                predictions.append(1)
            else:
                closest_label = k_labels[np.argmin(k_dists)]
                predictions.append(int(closest_label))
        return np.array(predictions)
        #raise NotImplementedError


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # TODO: Implement
    return np.mean(y_true == y_pred)
    #raise NotImplementedError


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # TODO: Implement
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)
    X_test, _ = load_data(test_data_file)
    X_train_stan, X_test_stan = preprocess_data(X_train, X_test)
    clf = KNNClassifier()
    clf.train(X_train_stan, y_train)
    y_pred = clf.predict(X_test_stan)
    pd.Series(y_pred.astype(int)).to_csv(pred_file, index=False, header=False)
    #raise NotImplementedError
##