# sil_score

[![PyPI](https://img.shields.io/pypi/v/sil-score.svg)](https://pypi.org/project/sil-score/)
[![Python](https://img.shields.io/pypi/pyversions/sil-score.svg)](https://pypi.org/project/sil-score/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

Install from PyPI:

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
