"""This module contains functions to fit classification models for HW 2 in AE 498 Computational Systems Engineering.

"""

import numpy as np


def fit_knn(X, y, n_neighbors):
    """Function to fit a KNN classifier to a given dataset.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor.
    y : array_like
        Output/response data (n elements), where each element is an observation.
    n_neighbors : int
        Number of k neighbors to use for each prediction.

    Returns
    -------
    y_predicted : array_like
        Predicted responses for input data (n elements).
    error : float
        Error rate for the input data, calculated as 1/n_obs * number of incorrect predictions.

    Notes
    -----
    The test functions assume the k nearest neighbors of observation x include x.

    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n = X.shape[0]
    k = int(n_neighbors)

    sq_norms = np.sum(X**2, axis=1)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    d2 = np.maximum(d2, 0.0)
    dist = np.sqrt(d2)

    nn_idx = np.argsort(dist, axis=1, kind="stable")[:, :k]

    classes, y_int = np.unique(y, return_inverse=True)
    neighbor_labels = y_int[nn_idx]

    counts = np.zeros((n, len(classes)), dtype=int)
    for j in range(k):
        np.add.at(counts, (np.arange(n), neighbor_labels[:, j]), 1)

    y_predicted = classes[np.argmax(counts, axis=1)]
    y_predicted = y_predicted.astype(y.dtype)

    error = float(np.mean(y_predicted != y))

    return error, y_predicted
