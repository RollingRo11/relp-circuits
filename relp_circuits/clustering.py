"""ADAG multi-view spectral clustering of circuit features.

For each pair of features (i, j), compute their similarity in two views:

  S^attr_ij(x)    = cosine(Attr(i, x), Attr(j, x))        per prompt x
  S^contrib_ij(x) = cosine(Contrib(i, x), Contrib(j, x))  per prompt x

Aggregate the two views via the **harmonic mean of their ReLU'd values** so
anticorrelated outputs penalise the cluster (paper eq. for S_ij). Then mean
the per-prompt similarities to get a single (n_features × n_features) matrix
and cluster with sklearn's SpectralClustering on the normalised Laplacian.

Output: cluster id per feature, plus a few quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusteringResult:
    cluster_ids: np.ndarray                # (n_features,) int — cluster assignment
    n_clusters: int
    similarity: np.ndarray                 # (n_features, n_features) — aggregated S
    silhouette: float
    coef_of_variation: float               # CV of cluster sizes
    pct_opposing_sign: float               # within-cluster fraction of opposing-sign pairs


def _cosine_per_prompt(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """matrix: (P, n_features, K). Returns (P, n_features, n_features) cosine
    similarities computed on the K-axis for each prompt."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True) + eps     # (P, F, 1)
    unit = matrix / norms
    # cosine(i, j) = unit_i · unit_j over the K axis
    return np.einsum("pik,pjk->pij", unit, unit)


def _harmonic_mean_view(
    sim_attr: np.ndarray, sim_contrib: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """Harmonic mean of ReLU'd cosine sims, per prompt.

    S_ij = 2 · ReLU(s_attr) · ReLU(s_contrib) / (ReLU(s_attr) + ReLU(s_contrib))

    Anticorrelated pairs (negative cosine) zero out via the ReLU. Both views
    must agree (positive in both) for a strong similarity. (Paper Sec. 3.3.)
    """
    a = np.maximum(sim_attr, 0.0)
    b = np.maximum(sim_contrib, 0.0)
    denom = a + b
    return np.where(denom > eps, 2.0 * a * b / (denom + eps), 0.0)


def cluster_features(
    input_attr: np.ndarray,                # (P, F, T_max) — pad with 0 beyond length
    output_contrib: np.ndarray,            # (P, F, K)
    n_clusters: int = 16,
    random_state: int = 0,
) -> ClusteringResult:
    P, F, _T = input_attr.shape
    if P == 0 or F == 0:
        raise ValueError("empty profiles")

    # Per-prompt cosine sims in each view
    sim_attr = _cosine_per_prompt(input_attr)                # (P, F, F)
    sim_contrib = _cosine_per_prompt(output_contrib)         # (P, F, F)

    # Harmonic-mean per prompt, then average across prompts.
    sim_per_pair = _harmonic_mean_view(sim_attr, sim_contrib)
    similarity = sim_per_pair.mean(axis=0)                   # (F, F)
    np.fill_diagonal(similarity, 1.0)

    # sklearn SpectralClustering wants a non-negative affinity matrix; ours is
    # already in [0, 1]. We pass affinity="precomputed".
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(
        n_clusters=min(n_clusters, F),
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
    )
    cluster_ids = sc.fit_predict(similarity).astype(np.int32)

    # Quality metrics.
    from sklearn.metrics import silhouette_score
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    if len(set(cluster_ids)) > 1:
        try:
            sil = float(silhouette_score(distance, cluster_ids, metric="precomputed"))
        except ValueError:
            sil = float("nan")
    else:
        sil = float("nan")

    sizes = np.bincount(cluster_ids)
    cv = float(sizes.std() / max(sizes.mean(), 1e-9))

    # % of within-cluster pairs whose contribution-sign on top-1 logit disagrees.
    # We use sign of the *mean* contribution on top-1 across prompts as the
    # canonical sign per feature.
    contrib_sign = np.sign(output_contrib[..., 0].mean(axis=0))  # (F,)
    pct_opp = 0.0
    pair_count = 0
    for c in range(int(cluster_ids.max()) + 1):
        idx = np.where(cluster_ids == c)[0]
        if idx.size < 2:
            continue
        signs = contrib_sign[idx]
        # number of opposing-sign pairs / total pairs
        pos = (signs > 0).sum()
        neg = (signs < 0).sum()
        n_pairs_c = pos * neg
        n_total_c = idx.size * (idx.size - 1) // 2
        if n_total_c > 0:
            pct_opp += n_pairs_c
            pair_count += n_total_c
    pct_opp = float(pct_opp / pair_count) if pair_count else 0.0

    return ClusteringResult(
        cluster_ids=cluster_ids,
        n_clusters=int(cluster_ids.max()) + 1,
        similarity=similarity.astype(np.float32),
        silhouette=sil,
        coef_of_variation=cv,
        pct_opposing_sign=pct_opp,
    )
