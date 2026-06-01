# AIDS HD-IV

Replication code for *High-Dimensional Instrumental Variables Estimation of the Almost Ideal Demand System*.

The notebook reproduces every figure in the paper.

## Files

- `AIDS_HDIV_Appendix.ipynb` — Monte Carlo simulation notebook; figures and tables matching the paper.
- `aids_core.py` — DGP, estimators (OLS, 2SLS, 2SLS-LassoFS, Post-Lasso IV, debiased HD-IV with CLIME, cross-fitting, ridge, and nodewise-Lasso precision variants), CLIME row solvers, diagnostics.
- `aids_experiments.py` — Monte Carlo drivers (`mc_sweep_n`, `mc_sweep_pz`) with per-grid-point caching.
- `aids_plots.py` — Plotting utilities for Monte Carlo outputs.

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

## Reproducing the figures

```
jupyter notebook AIDS_HDIV_Appendix.ipynb
```

Run all cells. Monte Carlo results are cached under `mc_cache/`, so cells that have run before reload their results rather than recomputing. The full run at `n_reps = 100` takes several hours on a multi-core machine; cached outputs in the notebook were produced at this setting.

