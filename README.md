# sil-score

<p align="center">
  <a href="https://pypi.org/project/sil-score/"><img src="https://img.shields.io/pypi/v/sil-score.svg?color=blue" alt="PyPI version"></a>&nbsp;&nbsp;
  <a href="https://pypi.org/project/sil-score/"><img src="https://img.shields.io/pypi/pyversions/sil-score.svg" alt="Python versions"></a>&nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>&nbsp;&nbsp;
  <a href="https://pepy.tech/project/sil-score"><img src="https://pepy.tech/badge/sil-score" alt="Downloads"></a>
</p>


`sil-score` is a small Python package for exact and fast approximate silhouette scoring.

It extends the usual silhouette workflow with:

- per-sample silhouette scores
- micro-averaged silhouette score
- macro-averaged silhouette score
- cluster-weighted macro silhouette score
- exact vs approximate comparison report

The exact mode uses scikit-learn's `silhouette_samples`.  
The approximate mode uses Euclidean distances to cluster centroids, making it faster but not identical to the classical silhouette definition.

---

## Installation

Install from [PyPI](https://pypi.org/project/sil-score/):

```bash
pip install sil-score
```

## Quick example

```python
import numpy as np
from sil_score import (
    sil_samples,
    micro_sil_score,
    macro_sil_score,
    weighted_macro_sil_score,
    sil_approximation_report,
)

X = np.array([
    [0.0],
    [2.0],
    [10.0],
    [12.0],
])

labels = np.array([0, 0, 1, 1])

samples = sil_samples(X, labels)
micro = micro_sil_score(X, labels)
macro = macro_sil_score(X, labels)

print(samples)
print(micro)
print(macro)
```

Output:

    [0.81818182 0.77777778 0.77777778 0.81818182]
    0.797979797979798
    0.797979797979798

## Functions

### `sil_samples` 

```python
sil_samples(X, labels, approximation=False, centers=None)
```

Computes the silhouette score for each sample.

By default, it computes the exact silhouette values using scikit-learn.

```python
scores = sil_samples(X, labels)
```

For a faster centroid-based approximation:

```python
scores = sil_samples(X, labels, approximation=True)
```

You can also pass precomputed cluster centers:

```python
scores = sil_samples(
    X,
    labels,
    approximation=True,
    centers=centers,
)
```

---

### `micro_sil_score`

```python
micro_sil_score(X, labels, approximation=False, centers=None)
```

Computes the mean of all sample-level silhouette scores. This is the usual average silhouette score. Larger clusters naturally have more influence because they contain more samples.

```python
# Standard usage
score = micro_sil_score(X, labels)

# Approximate version
score = micro_sil_score(X, labels, approximation=True)
```

---

### `macro_sil_score`

```python
macro_sil_score(X, labels, approximation=False, centers=None)
```

Computes the mean silhouette score inside each cluster, then averages the cluster means equally. This gives every cluster the same importance, regardless of its size.

```python
# Standard usage
score = macro_sil_score(X, labels)

# Approximate version
score = macro_sil_score(X, labels, approximation=True)
```

---

### `weighted_macro_sil_score`

```python
weighted_macro_sil_score(X, labels, cluster_weights, approximation=False, centers=None)
```

Computes a cluster-weighted macro silhouette score. First, it computes the mean silhouette score for each cluster, then combines those cluster means using custom cluster weights.

Using a dictionary:
```python
weights = {
    0: 0.2,
    1: 0.3,
    2: 0.5,
}

score = weighted_macro_sil_score(X, labels, cluster_weights=weights)
```

Using an array:

```python
weights = [0.2, 0.3, 0.5]

score = weighted_macro_sil_score(X, labels, cluster_weights=weights)
```

--- 

### `sil_approximation_report`

```python
sil_approximation_report(X, labels, centers=None, return_samples=False)
```

Compares exact silhouette scores with centroid-based approximate scores. It returns(Pearson) correlation and error metrics:

```python
report = sil_approximation_report(X, labels)
print(report)
```

Example output:

```
{
    "correlation": 0.96,
    "mean_absolute_error": 0.03,
    "mean_squared_error": 0.002,
    "root_mean_squared_error": 0.045,
    "max_absolute_error": 0.12,
    "mean_error": 0.01,
    "mean_exact_score": 0.52,
    "mean_approximate_score": 0.53,
    "n_samples": 300,
}
```

Use `return_samples=True` to also include the exact scores, approximate scores, and per-sample errors.

---

### Exact vs Approximate mode

- **Exact mode**: `sil_samples(X, labels, approximation=False)`. Uses the classical silhouette definition based on distances between samples.
- **Approximate mode**: `sil_samples(X, labels, approximation=True)`. Uses distances from each sample to cluster centroids. This can be significantly faster for larger datasets.

## Requirements 
`sil-score` depends on:
- NumPy
- scikit-learn

## License 
This project is licensed under the MIT License.
