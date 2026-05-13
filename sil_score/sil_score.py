"""
sil_score.metrics

Utilities for exact and approximate silhouette scoring.

The exact mode delegates to scikit-learn's silhouette_samples.
The approximate mode uses Euclidean distances to cluster centroids, so it is
faster but not identical to the classical silhouette definition.
"""
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances


def sil_samples(X, labels, approximation: bool = False, centers=None) -> np.ndarray:
    """
    Compute silhouette scores for each point in the dataset,
    with approximate fast centroid-based computation option.
    """
    # Ensure arrays
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
        raise ValueError("labels must be a 1D array of length n_samples.")

    unique_labels, inv = np.unique(labels, return_inverse=True)
    k = unique_labels.size
    if k < 2:
        raise ValueError("Silhouette computation requires at least 2 clusters.")

    # Exact silhouette scores
    if approximation == False:
        silhouette_scores = silhouette_samples(X, labels=labels)
        return silhouette_scores

    # Centroid-based approximate silhouette scores
    n_samples, n_features = X.shape

    if centers is None:
        centers = np.array([X[inv == i].mean(axis=0) for i in range(k)], dtype=float)
    else:
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != n_features:
            raise ValueError(f"centers must have shape (k, d) with d={n_features}.")
        if centers.shape[0] != k:
            raise ValueError(f"centers.shape[0] must equal number of clusters k={k}.")
        if not np.array_equal(unique_labels, np.arange(k)):
            raise ValueError("When passing ndarray centers, labels must be dense 0..k-1.")

    # Squared distances to all centroids
    D_sq = euclidean_distances(X, centers, squared=True)

    # a(i): distance to own centroid
    a = np.sqrt(np.maximum(D_sq[np.arange(n_samples), inv], 0.0))

    # b(i): distance to nearest other centroid
    D_sq[np.arange(n_samples), inv] = np.inf
    b = np.sqrt(np.min(D_sq, axis=1))

    # Silhouette per point
    denom = np.maximum(np.maximum(a, b), 1e-12)
    s_point = (b - a) / denom

    # Singleton clusters -> silhouette = 0
    counts = np.bincount(inv, minlength=k).astype(int)
    s_point[counts[inv] < 2] = 0.0

    silhouette_scores = np.clip(s_point, -1.0, 1.0)

    return silhouette_scores

def micro_sil_score(X, labels, approximation: bool = False, centers=None) -> float:
    """
    Compute the micro-averaged silhouette score.

    This is the mean of all sample-level silhouette scores.
    Larger clusters have more influence because they contain more samples.
    """
    silhouette_scores = sil_samples(
        X,
        labels,
        approximation=approximation,
        centers=centers,
    )
    return float(np.mean(silhouette_scores))


def macro_sil_score(X, labels, approximation: bool = False, centers=None) -> float:
    """
    Compute the macro-averaged silhouette score.

    This first computes the mean silhouette score inside each cluster,
    then averages those cluster means equally.
    Each cluster contributes the same weight regardless of size.
    """
    labels = np.asarray(labels)
    silhouette_scores = sil_samples(
        X,
        labels,
        approximation=approximation,
        centers=centers,
    )

    unique_labels = np.unique(labels)
    cluster_scores = [
        np.mean(silhouette_scores[labels == label])
        for label in unique_labels
    ]

    return float(np.mean(cluster_scores))

def sil_approximation_report(X, labels, centers=None, return_samples=False):
    """
    Compare exact silhouette scores with centroid-based approximate scores.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    labels : array-like of shape (n_samples,)
        Cluster labels for each sample.

    centers : array-like of shape (n_clusters, n_features), optional
        Cluster centers used for the approximate silhouette computation.
        If None, centers are computed from X and labels.

    return_samples : bool, default=False
        If True, include exact scores, approximate scores, and errors
        in the returned dictionary.

    Returns
    -------
    report : dict
        Dictionary with correlation and error metrics.
    """
    exact_scores = sil_samples(X, labels, approximation=False)
    approximate_scores = sil_samples(
        X,
        labels,
        approximation=True,
        centers=centers,
    )

    errors = approximate_scores - exact_scores
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    if np.std(exact_scores) == 0 or np.std(approximate_scores) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(exact_scores, approximate_scores)[0, 1])

    report = {
        "correlation": correlation,
        "mean_absolute_error": float(np.mean(absolute_errors)),
        "mean_squared_error": float(np.mean(squared_errors)),
        "root_mean_squared_error": float(np.sqrt(np.mean(squared_errors))),
        "max_absolute_error": float(np.max(absolute_errors)),
        "mean_error": float(np.mean(errors)),
        "mean_exact_score": float(np.mean(exact_scores)),
        "mean_approximate_score": float(np.mean(approximate_scores)),
        "n_samples": int(len(exact_scores)),
    }

    if return_samples:
        report["exact_scores"] = exact_scores
        report["approximate_scores"] = approximate_scores
        report["errors"] = errors
        report["absolute_errors"] = absolute_errors

    return report
  
def weighted_macro_sil_score(X, labels, cluster_weights, approximation: bool = False, centers=None) -> float:
    """
    Compute a cluster-weighted macro silhouette score.

    First computes the mean silhouette score inside each cluster.
    Then averages those cluster means using user-provided cluster weights.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    labels : array-like of shape (n_samples,)
        Cluster labels.

    cluster_weights : dict or array-like
        Weights for each cluster.

        If dict:
            keys must be cluster labels, values must be weights.
            Example: {0: 0.2, 1: 0.3, 2: 0.5}

        If array-like:
            labels must be dense integers 0, 1, ..., k-1.
            Example: [0.2, 0.3, 0.5]

    approximation : bool, default=False
        If False, use exact silhouette scores.
        If True, use centroid-based approximate silhouette scores.

    centers : array-like of shape (n_clusters, n_features), optional
        Cluster centers for approximate mode.

    Returns
    -------
    score : float
        Cluster-weighted macro silhouette score.
    """
    labels = np.asarray(labels)

    silhouette_scores = sil_samples(
        X,
        labels,
        approximation=approximation,
        centers=centers,
    )

    unique_labels = np.unique(labels)

    cluster_scores = np.array([
        np.mean(silhouette_scores[labels == label])
        for label in unique_labels
    ])

    if isinstance(cluster_weights, dict):
        weights = np.array([
            cluster_weights[label]
            for label in unique_labels
        ], dtype=float)
    else:
        weights = np.asarray(cluster_weights, dtype=float)

        if weights.ndim != 1 or weights.shape[0] != unique_labels.size:
            raise ValueError("cluster_weights must have one weight per cluster.")

        if not np.array_equal(unique_labels, np.arange(unique_labels.size)):
            raise ValueError(
                "When cluster_weights is array-like, labels must be dense integers 0..k-1. "
                "Use a dict for non-dense labels."
            )

    if np.any(weights < 0):
        raise ValueError("cluster_weights must be non-negative.")

    if np.sum(weights) == 0:
        raise ValueError("cluster_weights must sum to a positive value.")

    return float(np.average(cluster_scores, weights=weights))