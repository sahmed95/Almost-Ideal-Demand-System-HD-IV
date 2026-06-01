"""
aids_experiments.py
===================
Single-run and Monte Carlo drivers for the AIDS HD-IV simulation.

Caching is per-grid-point: each grid value writes its own cache file on
completion, so interrupting the kernel preserves all finished grid points
and restarting the same call resumes at the first uncached point. A
combined cache file is also written at the end, so plotting code that
reads the combined file works unchanged.

Caching tries parquet first and falls back silently to pickle when
pyarrow and fastparquet are not installed.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aids_core import (
    EstimatorConfig, draw_aids_params, extract_targets,
    get_true_params, run_all_estimators, simulate_aids_iv,
)


CACHE_DIR = Path("mc_cache")
CACHE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------

def _has_parquet_engine() -> bool:
    """Check once whether a parquet engine is available."""
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


_PARQUET_OK = _has_parquet_engine()


def _cache_paths(cache_key: str):
    """Return (parquet_path, pickle_path). Search order: parquet, pickle."""
    return (CACHE_DIR / f"{cache_key}.parquet",
            CACHE_DIR / f"{cache_key}.pkl")


def _load_cache(cache_key: str) -> Optional[pd.DataFrame]:
    """Load a cached DataFrame if it exists. Tries parquet first, then pickle."""
    pq, pk = _cache_paths(cache_key)
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass  # fall through to pickle
    if pk.exists():
        try:
            with open(pk, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return None


def _save_cache(df: pd.DataFrame, cache_key: str, verbose: bool = True) -> str:
    """Save DataFrame to cache. Uses parquet if available, else pickle."""
    pq, pk = _cache_paths(cache_key)
    if _PARQUET_OK:
        try:
            df.to_parquet(pq)
            return pq.name
        except Exception:
            pass
    # Fallback: pickle
    with open(pk, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pk.name


# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

def _summarize_mc(name, b_arr, g_arr, bse_arr, gse_arr,
                  beta_true, gii_true, explode_thresh=10.0):
    b = np.asarray(b_arr, dtype=float)
    g = np.asarray(g_arr, dtype=float)
    bse = np.asarray(bse_arr, dtype=float)
    gse = np.asarray(gse_arr, dtype=float)

    def _cov(hat, se, truth):
        with np.errstate(invalid="ignore"):
            lo = hat - 1.96 * se
            hi = hat + 1.96 * se
            mask = np.isfinite(lo) & np.isfinite(hi)
            if not np.any(mask):
                return float("nan")
            hit = (lo[mask] <= truth) & (truth <= hi[mask])
            return float(np.mean(hit))

    return {
        "estimator": name,
        "n_reps": len(b),
        "beta_true": float(beta_true),
        "gii_true": float(gii_true),
        "beta_bias": float(np.nanmean(b - beta_true)),
        "beta_rmse": float(np.sqrt(np.nanmean((b - beta_true) ** 2))),
        "beta_var": float(np.nanvar(b, ddof=1)),
        "beta_q05": float(np.nanquantile(b, 0.05)),
        "beta_q50": float(np.nanquantile(b, 0.50)),
        "beta_q95": float(np.nanquantile(b, 0.95)),
        "beta_se_median": float(np.nanmedian(bse)),
        "beta_nan_rate": float(np.mean(~np.isfinite(b))),
        "beta_explode_rate": float(np.mean(np.abs(b) > explode_thresh)),
        "beta_coverage": _cov(b, bse, beta_true),
        "gii_bias": float(np.nanmean(g - gii_true)),
        "gii_rmse": float(np.sqrt(np.nanmean((g - gii_true) ** 2))),
        "gii_var": float(np.nanvar(g, ddof=1)),
        "gii_q05": float(np.nanquantile(g, 0.05)),
        "gii_q50": float(np.nanquantile(g, 0.50)),
        "gii_q95": float(np.nanquantile(g, 0.95)),
        "gii_se_median": float(np.nanmedian(gse)),
        "gii_nan_rate": float(np.mean(~np.isfinite(g))),
        "gii_explode_rate": float(np.mean(np.abs(g) > explode_thresh)),
        "gii_coverage": _cov(g, gse, gii_true),
    }


# --------------------------------------------------------------------------
# Single-run sweeps
# --------------------------------------------------------------------------

def single_run_sweep_n(
    n_vals: Iterable[int],
    *, T: int = 300, pz: int = 200, i: int = 0,
    seed: int = 20, param_seed: int = 20,
    scenario_kwargs: Optional[dict] = None,
    config: Optional[EstimatorConfig] = None,
    endog_expenditure: bool = False,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """One-simulation-per-grid-point sweep over n_goods."""
    scenario_kwargs = scenario_kwargs or {}
    config = config or EstimatorConfig()
    n_vals = list(n_vals)
    max_n = max(n_vals)
    params_fixed = draw_aids_params(max_n_goods=max_n, s_comp=5,
                                    param_seed=param_seed,
                                    max_pz=pz, k_instr_per_price=5)

    def _one(n_goods):
        rng = np.random.default_rng(seed + 1000 * n_goods)
        summ = simulate_aids_iv(
            T=T, n_goods=n_goods, pz=pz, i=i, rng=rng,
            params_fixed=params_fixed, param_seed=param_seed,
            **scenario_kwargs,
        )
        res = run_all_estimators(summ, i=i, random_state=seed, config=config,
                                 endog_expenditure=endog_expenditure)
        beta_true, gii_true = get_true_params(summ, i=i)
        rows = []
        for (name, bh, bse, gh, gse) in extract_targets(res):
            rows.append({
                "n_goods": n_goods, "estimator": name,
                "beta_true": beta_true, "beta_hat": bh, "beta_se": bse,
                "beta_bias": bh - beta_true,
                "beta_sqerr": (bh - beta_true) ** 2,
                "gii_true": gii_true, "gii_hat": gh, "gii_se": gse,
                "gii_bias": gh - gii_true,
                "gii_sqerr": (gh - gii_true) ** 2,
            })
        return rows

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(n) for n in n_vals
    )
    return pd.DataFrame([r for chunk in nested for r in chunk])


def single_run_sweep_pz(
    pz_vals: Iterable[int],
    *, T: int = 300, n_goods: int = 100, i: int = 0,
    seed: int = 20, param_seed: int = 20,
    scenario_kwargs: Optional[dict] = None,
    config: Optional[EstimatorConfig] = None,
    endog_expenditure: bool = False,
    n_jobs: int = 2,
) -> pd.DataFrame:
    """One-simulation-per-grid-point sweep over number of instruments pz."""
    scenario_kwargs = scenario_kwargs or {}
    config = config or EstimatorConfig()
    pz_vals = list(pz_vals)
    max_pz = max(pz_vals)
    params_fixed = draw_aids_params(max_n_goods=n_goods, s_comp=5,
                                    param_seed=param_seed,
                                    max_pz=max_pz, k_instr_per_price=5)

    def _one(pz):
        rng = np.random.default_rng(seed + 1000 * pz)
        summ = simulate_aids_iv(
            T=T, n_goods=n_goods, pz=pz, i=i, rng=rng,
            params_fixed=params_fixed, param_seed=param_seed,
            **scenario_kwargs,
        )
        res = run_all_estimators(summ, i=i, random_state=seed, config=config,
                                 endog_expenditure=endog_expenditure)
        beta_true, gii_true = get_true_params(summ, i=i)
        rows = []
        for (name, bh, bse, gh, gse) in extract_targets(res):
            rows.append({
                "pz": pz, "estimator": name,
                "beta_true": beta_true, "beta_hat": bh, "beta_se": bse,
                "beta_bias": bh - beta_true,
                "beta_sqerr": (bh - beta_true) ** 2,
                "gii_true": gii_true, "gii_hat": gh, "gii_se": gse,
                "gii_bias": gh - gii_true,
                "gii_sqerr": (gh - gii_true) ** 2,
            })
        return rows

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(pz) for pz in pz_vals
    )
    return pd.DataFrame([r for chunk in nested for r in chunk])


# --------------------------------------------------------------------------
# Monte Carlo helpers
# --------------------------------------------------------------------------

def _mc_one_rep(T, n_goods, pz, i, rep_idx, seed, grid_val,
                params_fixed, param_seed, scenario_kwargs, config,
                random_state, endog_expenditure):
    rng = np.random.default_rng(seed + 10_000 * grid_val + rep_idx)
    summ = simulate_aids_iv(
        T=T, n_goods=n_goods, pz=pz, i=i, rng=rng,
        params_fixed=params_fixed, param_seed=param_seed,
        **scenario_kwargs,
    )
    res = run_all_estimators(summ, i=i, random_state=random_state,
                             config=config, endog_expenditure=endog_expenditure)
    beta_true, gii_true = get_true_params(summ, i=i)
    return beta_true, gii_true, extract_targets(res)


def _collect_store(rep_out):
    """Group rep outputs by estimator name."""
    store = {}
    beta_true = gii_true = None
    for bt, gt, targets in rep_out:
        beta_true, gii_true = bt, gt
        for (name, bh, bse, gh, gse) in targets:
            d = store.setdefault(name, {"b": [], "g": [], "bse": [], "gse": []})
            d["b"].append(bh); d["bse"].append(bse)
            d["g"].append(gh); d["gse"].append(gse)
    return store, beta_true, gii_true


def _fmt_time(seconds: float) -> str:
    """Human-readable duration."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --------------------------------------------------------------------------
# Monte Carlo sweeps with per-grid-point checkpointing
# --------------------------------------------------------------------------

def _mc_sweep_generic(
    grid_vals,
    *, axis_name: str,           # "n_goods" or "pz"
    T, fixed_other, i,           # fixed_other: pz (when axis=n_goods) or n_goods (when axis=pz)
    n_reps, seed,
    scenario_kwargs, config, endog_expenditure,
    random_state, param_seed,
    n_jobs,
    cache_key, force, verbose,
):
    """Shared implementation for mc_sweep_n and mc_sweep_pz with per-point cache."""
    grid_vals = list(grid_vals)

    if cache_key is not None and not force:
        cached = _load_cache(cache_key)
        if cached is not None:
            if verbose:
                print(f"[cache hit] loading combined cache for {cache_key}")
            return cached

    if axis_name == "n_goods":
        max_n_goods = max(grid_vals)
        max_pz_for_params = fixed_other
    else:
        max_n_goods = fixed_other
        max_pz_for_params = max(grid_vals)
    params_fixed = draw_aids_params(
        max_n_goods=max_n_goods, s_comp=5, param_seed=param_seed,
        max_pz=max_pz_for_params, k_instr_per_price=5,
    )

    rows = []
    t_overall = time.time()

    for k_idx, gv in enumerate(grid_vals, 1):
        per_point_key = (f"{cache_key}__{axis_name}{gv}"
                         if cache_key is not None else None)

        # Per-point cache hit?
        if per_point_key is not None and not force:
            cached_part = _load_cache(per_point_key)
            if cached_part is not None:
                rows.extend(cached_part.to_dict("records"))
                if verbose:
                    print(f"[mc {axis_name}] [{k_idx}/{len(grid_vals)}] "
                          f"{axis_name}={gv}  cache hit ({len(cached_part)} rows)")
                continue

        if verbose:
            extra = (f"T={T} pz={fixed_other}" if axis_name == "n_goods"
                     else f"T={T} n={fixed_other}")
            print(f"[mc {axis_name}] [{k_idx}/{len(grid_vals)}] "
                  f"{axis_name}={gv}  {extra} reps={n_reps}  "
                  f"endog_exp={endog_expenditure}")

        t0 = time.time()
        if axis_name == "n_goods":
            n_goods, pz = gv, fixed_other
        else:
            n_goods, pz = fixed_other, gv

        rep_out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_mc_one_rep)(
                T, n_goods, pz, i, r, seed, gv,
                params_fixed, param_seed, scenario_kwargs, config,
                random_state, endog_expenditure,
            ) for r in range(n_reps)
        )
        store, beta_true, gii_true = _collect_store(rep_out)
        local_rows = []
        for name, d in store.items():
            row = _summarize_mc(name, d["b"], d["g"], d["bse"], d["gse"],
                                beta_true, gii_true)
            row[axis_name] = gv
            local_rows.append(row)
            rows.append(row)

        if per_point_key is not None:
            saved = _save_cache(pd.DataFrame(local_rows), per_point_key, verbose=False)
            if verbose:
                print(f"  [checkpoint] {saved}")

        if verbose:
            elapsed = time.time() - t0
            total = time.time() - t_overall
            remaining = len(grid_vals) - k_idx
            eta = total / k_idx * remaining
            print(f"  done in {_fmt_time(elapsed)}  "
                  f"total {_fmt_time(total)}  "
                  f"ETA {_fmt_time(eta)}")

    df = pd.DataFrame(rows)
    if cache_key is not None:
        saved = _save_cache(df, cache_key, verbose=False)
        if verbose:
            print(f"[cache write] combined {saved} ({len(df)} rows)")
    return df


def mc_sweep_n(
    n_vals: Iterable[int],
    *, T: int = 300, pz: int = 200, i: int = 0,
    n_reps: int = 500, seed: int = 0,
    scenario_kwargs: Optional[dict] = None,
    config: Optional[EstimatorConfig] = None,
    endog_expenditure: bool = False,
    random_state: int = 20,
    param_seed: int = 20,
    n_jobs: int = -1,
    cache_key: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Monte Carlo sweep over n_goods. Each grid point spawns n_reps parallel reps.
    """
    return _mc_sweep_generic(
        n_vals, axis_name="n_goods",
        T=T, fixed_other=pz, i=i,
        n_reps=n_reps, seed=seed,
        scenario_kwargs=scenario_kwargs or {},
        config=config or EstimatorConfig(),
        endog_expenditure=endog_expenditure,
        random_state=random_state, param_seed=param_seed,
        n_jobs=n_jobs,
        cache_key=cache_key, force=force, verbose=verbose,
    )


def mc_sweep_pz(
    pz_vals: Iterable[int],
    *, T: int = 300, n_goods: int = 100, i: int = 0,
    n_reps: int = 500, seed: int = 0,
    scenario_kwargs: Optional[dict] = None,
    config: Optional[EstimatorConfig] = None,
    endog_expenditure: bool = False,
    random_state: int = 20,
    param_seed: int = 20,
    n_jobs: int = -1,
    cache_key: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Monte Carlo sweep over number of instruments pz. Caching and checkpointing
    are identical to mc_sweep_n.
    """
    return _mc_sweep_generic(
        pz_vals, axis_name="pz",
        T=T, fixed_other=n_goods, i=i,
        n_reps=n_reps, seed=seed,
        scenario_kwargs=scenario_kwargs or {},
        config=config or EstimatorConfig(),
        endog_expenditure=endog_expenditure,
        random_state=random_state, param_seed=param_seed,
        n_jobs=n_jobs,
        cache_key=cache_key, force=force, verbose=verbose,
    )
