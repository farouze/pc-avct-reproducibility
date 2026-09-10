"""Participant-level paired inference; no approximation from interval widths."""
import numpy as np

def sign_flip_pvalue(values, n_resamples=99999, seed=20260910):
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('Provide at least two finite paired participant contrasts.')
    if n_resamples < 1:
        raise ValueError('n_resamples must be positive.')
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    extreme = 0
    for start in range(0, n_resamples, 500):
        n = min(500, n_resamples-start)
        null = (rng.choice([-1, 1], size=(n, len(d))) * d).mean(axis=1)
        extreme += int((abs(null) >= observed-1e-14).sum())
    return (extreme+1)/(n_resamples+1)

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Provide finite p values in [0, 1].')
    order = np.argsort(p, kind='stable')
    adjusted = np.maximum.accumulate(np.minimum(1, p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p)); result[order] = adjusted
    return result
