"""
aids_core.py
============
Core building blocks for the AIDS HD-IV simulation.

Contains:
- DGP: parameter draws, simulation of prices, shares, and instruments.
- Estimators: OLS, classical 2SLS, 2SLS with Lasso first stage,
  2SLS with elastic-net first stage, Post-Lasso IV, two-stage Lasso,
  and debiased HD-IV (CLIME precision, optional cross-fitting, ridge
  and nodewise-Lasso precision variants).
- Diagnostics: condition number, minimum eigenvalue, first-stage R^2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy.linalg import eigvals, svd
from scipy.optimize import linprog
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def condition_number(X: np.ndarray) -> float:
    _, s, _ = svd(X, full_matrices=False)
    if np.min(s) == 0:
        return float("inf")
    return float(np.max(s) / np.min(s))


def min_eig_XtX(X: np.ndarray) -> float:
    XtX = X.T @ X
    return float(np.min(np.real(eigvals(XtX))))


# --------------------------------------------------------------------------
# DGP
# --------------------------------------------------------------------------

def draw_aids_params(
    max_n_goods: int,
    s_comp: int = 5,
    param_seed: int = 20,
    max_pz: Optional[int] = None,
    k_instr_per_price: int = 5,
) -> dict:
    """
    Draw structural AIDS parameters (alpha, beta, gamma) and, optionally,
    the full first-stage instrument matrix Pi_full so that
    slicing preserves identities across n-grid and pz-grid sweeps.

    All parameters satisfy adding-up, Slutsky symmetry, and homogeneity.
    """
    rng_params = np.random.default_rng(param_seed)

    alpha = rng_params.uniform(0.02, 0.2, size=max_n_goods)
    alpha = alpha / alpha.sum()

    beta = rng_params.normal(0, 0.1, size=max_n_goods)
    beta -= beta.mean()

    gamma = np.zeros((max_n_goods, max_n_goods))
    own = rng_params.normal(-0.3, 0.05, size=max_n_goods)
    np.fill_diagonal(gamma, own)

    for ii in range(max_n_goods):
        candidates = [j for j in range(max_n_goods) if j != ii]
        kk = min(s_comp, len(candidates))
        comp = rng_params.choice(candidates, size=kk, replace=False)
        gamma[ii, comp] = rng_params.normal(0.08, 0.02, size=kk)

    # Symmetry
    gamma = (gamma + gamma.T) / 2
    # Row-wise homogeneity
    gamma -= gamma.mean(axis=1, keepdims=True)

    out = {"alpha": alpha, "beta": beta, "gamma": gamma}

    if max_pz is not None:
        Pi_full = np.zeros((max_pz, max_n_goods))
        k = min(k_instr_per_price, max_pz)
        for j in range(max_n_goods):
            idx = rng_params.choice(max_pz, size=k, replace=False)
            Pi_full[idx, j] = rng_params.normal(0, 1, size=k)
        out["Pi_full"] = Pi_full
        out["max_pz"] = max_pz
        out["k_instr_per_price"] = k_instr_per_price

    return out


def simulate_aids_iv(
    T: int = 300,
    n_goods: int = 50,
    pz: int = 200,
    i: int = 0,
    *,
    rho_u: float = 0.6,
    rho_x: float = 0.0,
    rho_col: float = 0.95,
    k_instr_per_price: int = 5,
    instr_strength: float = 1.0,
    correlated_prices: bool = True,
    recompute_stone: bool = False,
    noise_sd: float = 0.02,
    rng: Optional[np.random.Generator] = None,
    params_fixed: Optional[dict] = None,
    param_seed: int = 20,
    s_comp: int = 5,
) -> dict:
    """
    Simulate one AIDS dataset (T periods, n_goods goods, pz instruments)
    with a structural share equation and an endogenous-price reduced form.

    log p_{j,t}  = instr_strength * Z_t Pi_j + common_t + v_{j,t} + rho_u u_t
    w_{i,t}      = alpha_i + sum_j gamma_{ij} log p_{j,t}
                 + beta_i log(x_t / P*_t) + u_t + eps_{i,t}

    If params_fixed contains Pi_full and max_pz, the matrix is sliced so
    identities of columns/instruments match across grid points.
    """
    if rng is None:
        rng = np.random.default_rng()

    # instruments and structural shock
    Z = rng.normal(size=(T, pz))
    u = rng.normal(size=T)

    # prices
    if correlated_prices:
        common = rng.normal(0, 1, size=T).cumsum()
        common = common / max(np.std(common), 1e-12)
    else:
        common = np.zeros(T)

    if params_fixed is not None and "Pi_full" in params_fixed:
        Pi = params_fixed["Pi_full"][:pz, :n_goods]
    else:
        k = min(k_instr_per_price, pz)
        Pi = np.zeros((pz, n_goods))
        for j in range(n_goods):
            idx = rng.choice(pz, size=k, replace=False)
            Pi[idx, j] = rng.normal(0, 1, size=k)

    v = rng.normal(scale=(1 - rho_col), size=(T, n_goods))
    log_p = instr_strength * (Z @ Pi) + common[:, None] + v + rho_u * u[:, None]

    # expenditure. If rho_x > 0, log(x) is correlated with the
    # structural demand shock u, making log(x/P*) endogenous. Use
    # endog_expenditure=True in build_yX_from_summary to instrument it.
    common_x = rng.normal(0, 1, size=T).cumsum()
    log_x = 0.5 * common_x + rng.normal(0, 0.3, size=T) + 5.0 + rho_x * u

    # structural parameters
    if params_fixed is None:
        params_fixed = draw_aids_params(
            max_n_goods=n_goods, s_comp=s_comp, param_seed=param_seed,
            max_pz=pz, k_instr_per_price=k_instr_per_price,
        )

    alpha = params_fixed["alpha"][:n_goods].copy()
    alpha = alpha / alpha.sum()
    beta = params_fixed["beta"][:n_goods].copy()
    gamma = params_fixed["gamma"][:n_goods, :n_goods].copy()
    gamma -= gamma.mean(axis=1, keepdims=True)

    # Stone-index deflator (initial weights from alpha)
    log_Pstar0 = log_p @ alpha
    log_x_over_P0 = log_x - log_Pstar0

    # Share equation (all goods at once)
    noise = rng.normal(0, noise_sd, size=(T, n_goods))
    w = (
        alpha[None, :]
        + log_p @ gamma.T
        + np.outer(log_x_over_P0, beta)
        + u[:, None]
        + noise
    )

    if recompute_stone:
        log_Pstar = (log_p * w).sum(axis=1)
        log_x_over_P = log_x - log_Pstar
    else:
        log_Pstar = log_Pstar0
        log_x_over_P = log_x_over_P0

    df = pd.DataFrame({"log_x": log_x, "log_Pstar": log_Pstar})
    df = pd.concat(
        [df, pd.DataFrame(log_p, columns=[f"log_p{j+1}" for j in range(n_goods)])],
        axis=1,
    )
    for j in range(n_goods):
        df[f"w{j+1}"] = w[:, j]

    return {
        "T": T,
        "n_goods": n_goods,
        "df": df,
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "Z": Z,
        "u": u,
        "log_x_over_P": log_x_over_P,
        "log_x_over_P0": log_x_over_P0,
        "params_fixed_seed": param_seed,
    }


def build_yX_from_summary(summary: dict, i: int = 0, *,
                          endog_expenditure: bool = False):
    """Construct (y, X, Z, meta) for share equation i.

    Parameters
    ----------
    endog_expenditure : bool
        If True, column index 1 (log(x/P*)) is treated as endogenous
        along with the price columns, and the IV pipeline will
        instrument it using the same Z matrix.
    """
    df = summary["df"]
    Z = summary["Z"]
    T = df.shape[0]
    n_goods = summary["n_goods"]

    y = df[f"w{i+1}"].values
    log_x_over_P0 = summary["log_x_over_P0"]
    log_p_cols = [f"log_p{j+1}" for j in range(n_goods)]
    logP = df[log_p_cols].values

    X = np.column_stack([np.ones(T), log_x_over_P0, logP])

    endog = list(range(2, 2 + n_goods))
    if endog_expenditure:
        endog = [1] + endog

    meta = {
        "T": T,
        "n_goods": n_goods,
        "idx_beta": 1,
        "idx_gamma_ii": 2 + i,
        "endog_cols_prices": endog,
        "log_p_cols": log_p_cols,
        "endog_expenditure": endog_expenditure,
    }
    return y, X, Z, meta


# --------------------------------------------------------------------------
# Estimators: OLS and classical 2SLS
# --------------------------------------------------------------------------

def fit_ols(y, X):
    res = sm.OLS(y, X).fit()
    return {"b": np.asarray(res.params), "se": np.asarray(res.bse), "res_obj": res}


def fit_2sls_plain(y, X, Z, endog_cols):
    Zc = sm.add_constant(Z, has_constant="add")
    Xhat = X.copy()
    endog_arr = X[:, endog_cols]
    coefs, *_ = np.linalg.lstsq(Zc, endog_arr, rcond=None)
    Xhat[:, endog_cols] = Zc @ coefs
    ss = sm.OLS(y, Xhat).fit()
    return {"b": np.asarray(ss.params), "se": np.asarray(ss.bse),
            "Xhat": Xhat, "res_obj": ss}


# --------------------------------------------------------------------------
# First-stage machinery (Lasso, Elastic Net, optional cross-fitting)
# --------------------------------------------------------------------------

def _make_first_stage(kind: str, cv: int, random_state: int, n_jobs: int):
    """Pipeline for first-stage: Lasso or Elastic Net with CV."""
    if kind == "lasso":
        reg = LassoCV(cv=cv, random_state=random_state, n_jobs=n_jobs,
                      max_iter=5000, tol=1e-3)
    elif kind == "enet":
        reg = ElasticNetCV(cv=cv, random_state=random_state, n_jobs=n_jobs,
                           l1_ratio=[0.3, 0.5, 0.7, 0.9, 0.95],
                           max_iter=5000, tol=1e-3)
    else:
        raise ValueError(f"Unknown first-stage kind: {kind!r}")
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])


def first_stage_predict(
    Z_train, x_train, Z_test=None, *, kind="lasso",
    cv=5, random_state=20, n_jobs=1,
):
    pipe = _make_first_stage(kind, cv, random_state, n_jobs)
    pipe.fit(Z_train, x_train)
    Z_pred = Z_train if Z_test is None else Z_test
    return pipe.predict(Z_pred), pipe


def build_Dhat(
    Z, X, endog_cols, *, kind="lasso", cv=5,
    random_state=20, lasso_n_jobs=1,
):
    """Non-cross-fitted Dhat (in-sample fitted values)."""
    def _one(j):
        xhat_j, model = first_stage_predict(
            Z, X[:, j], kind=kind, cv=cv,
            random_state=random_state, n_jobs=1,
        )
        return j, xhat_j, model

    results = Parallel(n_jobs=lasso_n_jobs, backend="loky")(
        delayed(_one)(j) for j in endog_cols
    )
    Dhat = X.copy()
    fs_models = {}
    for j, xhat_j, model in results:
        Dhat[:, j] = xhat_j
        fs_models[j] = model
    return Dhat, fs_models


def build_Dhat_crossfit(
    Z, X, endog_cols, *, kind="lasso", cv=5, n_splits=5,
    random_state=20, lasso_n_jobs=1,
):
    """
    K-fold cross-fitted Dhat: the prediction for observation i is produced
    by a first-stage fit that did not see observation i.
    """
    n = X.shape[0]
    Dhat = X.copy()
    fs_models = {j: [] for j in endog_cols}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # One task per (fold, endogenous column) pair -- embarrassingly parallel
    def _one(tr_idx, te_idx, j):
        xhat_te, model = first_stage_predict(
            Z[tr_idx], X[tr_idx, j], Z_test=Z[te_idx],
            kind=kind, cv=cv, random_state=random_state, n_jobs=1,
        )
        return j, te_idx, xhat_te, model

    jobs = []
    for tr_idx, te_idx in kf.split(np.arange(n)):
        for j in endog_cols:
            jobs.append((tr_idx, te_idx, j))

    results = Parallel(n_jobs=lasso_n_jobs, backend="loky")(
        delayed(_one)(tr, te, j) for (tr, te, j) in jobs
    )
    for j, te_idx, xhat_te, model in results:
        Dhat[te_idx, j] = xhat_te
        fs_models[j].append(model)

    return Dhat, fs_models


# --------------------------------------------------------------------------
# Second stage: 2SLS-LassoFS, two-stage Lasso, and partialled-out Lasso
# --------------------------------------------------------------------------

def fit_2sls_firststage(y, Xhat):
    """Classical OLS second stage on a given Xhat (works for LassoFS and EnetFS)."""
    ss = sm.OLS(y, Xhat).fit()
    return {"b": np.asarray(ss.params), "se": np.asarray(ss.bse),
            "Xhat": Xhat, "res_obj": ss}




def fit_post_lasso_iv(y, X, Z, endog_cols, *, cv=5, random_state=20,
                     lasso_n_jobs=1, support_threshold=1e-8):
    r"""
    Post-Lasso IV.

    For each endogenous regressor x_j:
      1. Run Lasso of x_j on Z to select instruments. Let S_j be the support.
      2. Refit the first stage by OLS of x_j on Z[:, S_j] -- unshrunk
         estimates on the selected instruments only.
      3. Get the post-selection fitted values \tilde x_j.
    Then stack the \tilde x_j into \tilde D and run classical 2SLS using
    \tilde D as the projected first stage.

    Falls back to including the constant column in S_j (i.e., uses the
    intercept) when Lasso selects nothing, which can happen in finite
    samples for weakly-identified instruments.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    T, p_z = Z.shape

    def _one(j):
        # Lasso selection
        scaler = StandardScaler()
        Z_std = scaler.fit_transform(Z)
        lasso = LassoCV(cv=cv, random_state=random_state, n_jobs=1,
                        max_iter=5000, tol=1e-3)
        lasso.fit(Z_std, X[:, j])
        # Selected instruments: nonzero Lasso coefficients above threshold
        support = np.where(np.abs(lasso.coef_) > support_threshold)[0]

        # refit by OLS on selected support
        if len(support) == 0:
            # Fallback: nothing selected. Use mean as predictor.
            xhat_j = np.full(T, X[:, j].mean())
        else:
            Z_sel = Z[:, support]
            Z_sel_c = sm.add_constant(Z_sel, has_constant="add")
            ols_fs = sm.OLS(X[:, j], Z_sel_c).fit()
            xhat_j = np.asarray(ols_fs.fittedvalues)

        return j, xhat_j

    results = Parallel(n_jobs=lasso_n_jobs, backend="loky")(
        delayed(_one)(j) for j in endog_cols
    )

    # stack into Xhat, then classical 2SLS second stage
    Xhat = X.copy()
    for j, xhat_j in results:
        Xhat[:, j] = xhat_j

    ss = sm.OLS(y, Xhat).fit()
    return {"b": np.asarray(ss.params), "se": np.asarray(ss.bse),
            "Xhat": Xhat, "res_obj": ss}

def fit_two_stage_lasso(y, Xhat, *, kind="lasso", cv=5,
                        random_state=20, lasso_n_jobs=1):
    """Regularized second stage applied directly to Xhat."""
    pipe = _make_first_stage(kind, cv, random_state, lasso_n_jobs)
    pipe.fit(Xhat, y)
    b_hat = pipe.named_steps["reg"].coef_.copy()
    intercept = pipe.named_steps["reg"].intercept_
    b_hat[0] = intercept  # column 0 of X is the constant
    return {
        "b": b_hat,
        "se": np.full_like(b_hat, np.nan, dtype=float),
        "Xhat": Xhat,
        "model": pipe,
    }


def partial_lasso_second_stage(
    y, Dhat, unpen_idx, *, kind="lasso", cv=5,
    random_state=20, lasso_n_jobs=1,
):
    """
    Partial out `unpen_idx` columns, penalize the rest, recover
    unpenalized coefs by OLS on the adjusted outcome. Used as the
    pilot estimator for debiased HD-IV.
    """
    T, p = Dhat.shape
    unpen_idx = sorted(set(unpen_idx))
    pen_idx = [j for j in range(p) if j not in unpen_idx]

    W = Dhat[:, unpen_idx]
    P = Dhat[:, pen_idx]

    YP = np.column_stack([y, P])
    B, *_ = np.linalg.lstsq(W, YP, rcond=None)
    YP_res = YP - W @ B
    y_res = YP_res[:, 0]
    P_res = YP_res[:, 1:]

    pipe = _make_first_stage(kind, cv, random_state, lasso_n_jobs)
    pipe.fit(P_res, y_res)
    theta_pen = pipe.named_steps["reg"].coef_

    y_adj = y - P @ theta_pen
    theta_unpen, *_ = np.linalg.lstsq(W, y_adj, rcond=None)

    theta = np.zeros(p)
    theta[unpen_idx] = theta_unpen
    theta[pen_idx] = theta_pen
    return theta, pipe


# --------------------------------------------------------------------------
# CLIME precision rows + debiased HD-IV
# --------------------------------------------------------------------------


def nodewise_lasso_rows(
    Sigma: np.ndarray, rows: Sequence[int],
    *, cv: int = 5, random_state: int = 20, n_jobs: int = 1,
    Dhat: Optional[np.ndarray] = None,
) -> dict:
    """
    Estimate selected rows of Theta = Sigma^{-1} via nodewise Lasso.

    For each target column j, regress D_j on D_{-j} with Lasso. Recover
    Theta_j from the residuals tau_j and the coefficient gamma_j:
    """
    if Dhat is None:
        # Fall back: treat sqrt(Sigma) as a synthetic design
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, 0.0, None)
        D_eff = eigvecs @ np.diag(np.sqrt(eigvals))  # p x p
        T_eff = D_eff.shape[0]
    else:
        D_eff = Dhat
        T_eff = Dhat.shape[0]

    p = Sigma.shape[0]

    def _solve_row(j):
        idx_other = [k for k in range(p) if k != j]
        Xj = D_eff[:, idx_other]
        yj = D_eff[:, j]
        m = LassoCV(cv=cv, random_state=random_state, n_jobs=1, max_iter=20_000)
        m.fit(Xj, yj)
        gamma = m.coef_ 
        tau_j = yj - Xj @ gamma
        denom = float(tau_j @ yj) / T_eff
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            # Degenerate residual: return a unit vector
            theta_j = np.zeros(p); theta_j[j] = 1.0
            return j, theta_j
        theta_j = np.zeros(p)
        theta_j[j] = 1.0 / denom
        for k_local, k in enumerate(idx_other):
            theta_j[k] = -gamma[k_local] / denom
        return j, theta_j

    out = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_solve_row)(j) for j in rows
    )
    return {j: theta for j, theta in out}



def clime_min_infeasibility(Sigma: np.ndarray, j: int) -> float:
    """
    Compute mu_j* := inf_theta || Sigma @ theta - e_j ||_inf

    Solved as the LP:
        minimize  t
        s.t.      Sigma @ theta - e_j <= t * 1
                 -Sigma @ theta + e_j <= t * 1
                  t >= 0
    Variables: [theta_+ (p), theta_- (p), t]; theta = theta_+ - theta_-.
    """
    p = Sigma.shape[0]
    ej = np.zeros(p)
    ej[j] = 1.0

    # Decision variables: theta_+ (p), theta_- (p), t (1) -> total 2p+1
    # Objective: minimize t
    c = np.zeros(2 * p + 1)
    c[-1] = 1.0

    # Constraints: Sigma @ (theta_+ - theta_-) - e_j <= t * 1
    #              -Sigma @ (theta_+ - theta_-) + e_j <= t * 1
    # Rewritten as Ax <= b form with x = [theta_+, theta_-, t]:
    #     [Sigma, -Sigma, -1] x <= e_j
    #     [-Sigma, Sigma, -1] x <= -e_j
    A1 = np.hstack([Sigma, -Sigma, -np.ones((p, 1))])
    A2 = np.hstack([-Sigma, Sigma, -np.ones((p, 1))])
    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([ej, -ej])

    # theta_+ >= 0, theta_- >= 0, t >= 0
    bounds = [(0, None)] * (2 * p) + [(0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # Return a large but finite value as fallback
        return float("inf")
    return float(res.x[-1])


def clime_row_linprog(Sigma: np.ndarray, j: int, mu: float) -> np.ndarray:
    """
    Solve one row of CLIME via linear programming.

    min ||theta||_1  s.t.  || Sigma @ theta - e_j ||_inf <= mu
    """
    p = Sigma.shape[0]
    ej = np.zeros(p)
    ej[j] = 1.0

    c = np.ones(2 * p)
    A1 = np.hstack([Sigma, -Sigma])
    A2 = np.hstack([-Sigma, Sigma])
    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([mu + ej, mu - ej])
    bounds = [(0, None)] * (2 * p)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CLIME LP failed for row {j}: {res.message}")
    t = res.x
    return t[:p] - t[p:]


def clime_rows(
    Sigma: np.ndarray, rows: Sequence[int], mu, *,
    adaptive: bool = True, kappa: float = 1.2,
    mu_growth: float = 2.0, max_tries: int = 6,
    n_jobs: int = 1,
) -> dict:
    """Per-row CLIME with two tuning modes.

    adaptive=True (default):
        For each row j, compute mu_j* = inf_theta || Sigma @ theta - e_j ||_inf
        and set the row's CLIME tolerance to mu_j = kappa * mu_j*. This is
        guaranteed feasible (provided mu_j* is finite) and adapts to the
        realized sample conditioning of Sigma. The explicit `mu` argument
        is ignored in this mode.

    adaptive=False (rate-based tuning):
        Use the explicit `mu` argument (typically mu = kappa * sqrt(log p / n))
        across all rows. Retry up to `max_tries` with mu *= mu_growth on
        infeasibility.
    """
    def _solve_row_adaptive(j):
        mu_j_star = clime_min_infeasibility(Sigma, j)
        if not np.isfinite(mu_j_star):
            # Fallback: try rate-based with growth budget
            mu_j = kappa * np.sqrt(np.log(max(Sigma.shape[0], 2)) / max(Sigma.shape[0], 1))
            for _ in range(max_tries):
                try:
                    return j, clime_row_linprog(Sigma, j, mu_j)
                except Exception:
                    mu_j *= mu_growth
            # Give up: return a conservative theta (e_j itself, identity-row)
            theta_j = np.zeros(Sigma.shape[0]); theta_j[j] = 1.0
            return j, theta_j
        rate_floor = mu if mu is not None else 0.0
        mu_j = max(kappa * mu_j_star, rate_floor)
        return j, clime_row_linprog(Sigma, j, mu_j)

    def _solve_row_legacy(j):
        mu_j = mu
        last_err = None
        for _ in range(max_tries):
            try:
                return j, clime_row_linprog(Sigma, j, mu_j)
            except Exception as e:
                last_err = e
                mu_j *= mu_growth
        raise last_err

    solver = _solve_row_adaptive if adaptive else _solve_row_legacy
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(solver)(j) for j in rows
    )
    return {j: theta for j, theta in results}


def debias_one_step(
    y, X, Dhat, theta_hat, *,
    target_rows, mu, se_kind: str = "sandwich",
    n_jobs: int = 1,
    ridge_lift: float = 1e-3,
    rank_tol: float = 1e-8,
    on_singular: str = "nan",  # main: skip CLIME at p>=T (NaN); use "ridge" for the regularized variant
    precision_method: str = "clime",       # "clime" or "nodewise"
    precision_tuning: str = "adaptive",
    kappa: float = 1.2,
    cv: int = 5,
    random_state: int = 20,
):
    """
    Gold-Lederer-Tao one-step update applied to theta_hat.

    se_kind:
        "homo"     -- sigma^2 * Theta_jj / n (Gold-Lederer-Tao homoscedastic)
        "sandwich" -- Theta_j' * meat * Theta_j / n, meat = D' diag(u^2) D / n
                      (heteroscedasticity-robust)

    Rank-deficiency handling:
        When p >= T or Sigma_d = D'D/T has near-zero minimum eigenvalue,
        mainline CLIME's per-row LP becomes costly to solve. Detected
        via the smallest eigenvalue of Sigma_d.

        on_singular="ridge"  : regularize Sigma_d <- Sigma_d + lift*I
                               with lift = ridge_lift * trace(Sigma_d)/p.
        on_singular="nan"    : return NaN point estimates and SEs to flag
                               the rank-deficient regime in MC summaries.
        on_singular="raise"  : raise an error.
    """
    T = Dhat.shape[0]
    p = Dhat.shape[1]
    resid = y - X @ theta_hat

    Sigma_d = (Dhat.T @ Dhat) / T
    try:
        min_eig = float(np.linalg.eigvalsh(Sigma_d)[0])
    except np.linalg.LinAlgError:
        min_eig = -np.inf

    rank_deficient = (p >= T) or (min_eig < rank_tol * max(np.trace(Sigma_d) / p, 1.0))

    # Nodewise Lasso never inverts Sigma_d, so the rank-deficiency branch is moot.
    if rank_deficient and precision_method == "clime":
        if on_singular == "nan":
            theta_tilde = theta_hat.copy()
            se = np.full_like(theta_hat, np.nan, dtype=float)
            for j in target_rows:
                theta_tilde[j] = np.nan
            return theta_tilde, se, {}
        if on_singular == "raise":
            raise np.linalg.LinAlgError(
                f"Sigma_d is rank-deficient (p={p}, T={T}, min_eig={min_eig:.2e}); "
                f"CLIME LP has no feasible solution. Set on_singular='ridge' or 'nan'."
            )
        # Default: ridge-regularize.
        lift = ridge_lift * max(np.trace(Sigma_d) / p, 1.0)
        Sigma_d = Sigma_d + lift * np.eye(p)

    if precision_method == "nodewise":
        Theta_rows = nodewise_lasso_rows(
            Sigma_d, target_rows, cv=3, random_state=random_state,
            n_jobs=n_jobs, Dhat=Dhat,
        )
    else:
        # CLIME with either adaptive tuning (default) or fixed rate
        adaptive = (precision_tuning == "adaptive")
        Theta_rows = clime_rows(
            Sigma_d, target_rows, mu=mu,
            adaptive=adaptive, kappa=kappa, n_jobs=n_jobs,
        )

    theta_tilde = theta_hat.copy()
    se = np.full_like(theta_hat, np.nan, dtype=float)

    if se_kind == "homo":
        sigma2 = float(np.mean(resid ** 2))
    else:
        meat = (Dhat.T * (resid ** 2)) @ Dhat / T

    Dresid = Dhat.T @ resid  # (p,)
    for j in target_rows:
        theta_j = Theta_rows[j]
        correction = float(theta_j @ Dresid) / T
        theta_tilde[j] = theta_hat[j] + correction

        if se_kind == "homo":
            var_j = sigma2 * float(theta_j @ theta_j) / T
        else:
            var_j = float(theta_j @ meat @ theta_j) / T
        se[j] = np.sqrt(max(var_j, 0.0))

    return theta_tilde, se, Theta_rows


def fit_debiased_hd_iv(
    y, X, Z, endog_cols, *,
    idx_beta, idx_gamma_ii,
    crossfit: bool = True, n_splits: int = 5,
    first_stage_kind: str = "lasso",
    se_kind: str = "sandwich",
    cv: int = 5, random_state: int = 20,
    mu: Optional[float] = None, kappa: float = 1.2,
    lasso_n_jobs: int = 1,
    Dhat_cached: Optional[np.ndarray] = None,
    fs_models_cached: Optional[dict] = None,
    precision_method: str = "clime",       # "clime" or "nodewise"
    precision_tuning: str = "adaptive",    # "adaptive" or "rate"
    on_singular: str = "nan",              # only matters when precision_method="clime"
):
    """Full debiased HD-IV pipeline with optional cross-fitting."""
    n, p = X.shape

    if Dhat_cached is None:
        if crossfit:
            Dhat, fs_models = build_Dhat_crossfit(
                Z, X, endog_cols, kind=first_stage_kind,
                cv=cv, n_splits=n_splits,
                random_state=random_state, lasso_n_jobs=lasso_n_jobs,
            )
        else:
            Dhat, fs_models = build_Dhat(
                Z, X, endog_cols, kind=first_stage_kind,
                cv=cv, random_state=random_state, lasso_n_jobs=lasso_n_jobs,
            )
    else:
        Dhat = Dhat_cached
        fs_models = fs_models_cached or {}

    unpen_idx = [0, idx_beta, idx_gamma_ii]
    theta_hat, lasso_obj = partial_lasso_second_stage(
        y, Dhat, unpen_idx, kind=first_stage_kind, cv=cv,
        random_state=random_state, lasso_n_jobs=lasso_n_jobs,
    )

    if mu is None:
        mu = kappa * np.sqrt(np.log(max(p, 2)) / n)

    target_rows = [idx_beta, idx_gamma_ii]
    theta_tilde, se, Theta_rows = debias_one_step(
        y, X, Dhat, theta_hat,
        target_rows=target_rows, mu=mu, se_kind=se_kind,
        n_jobs=1,
        precision_method=precision_method,
        precision_tuning=precision_tuning,
        kappa=kappa,
        on_singular=on_singular,
        cv=cv, random_state=random_state,
    )
    return {
        "b_hat": theta_hat,
        "b_tilde": theta_tilde,
        "se_tilde": se,
        "mu": mu,
        "Theta_rows": Theta_rows,
        "Dhat": Dhat,
        "fs_models": fs_models,
        "crossfit": crossfit,
        "se_kind": se_kind,
        "precision_method": precision_method,
        "precision_tuning": precision_tuning,
        "kappa": kappa,
        "on_singular": on_singular,
    }


# --------------------------------------------------------------------------
# One-stop estimator driver
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorConfig:
    """Toggles for which estimators to run and how.

    Estimator tiers (all run by default; tier controls placement in the paper):
      main figures:     OLS, 2SLS, 2SLS-LassoFS, Post-Lasso-IV,
                        Debiased-HDIV, Debiased-HDIV-CF
      appendix figures: 2SLS-EnetFS, Debiased-HDIV-Ridge, Debiased-HDIV-Nodewise
    """
    include_ols: bool = True
    include_2sls_plain: bool = True
    include_2sls_lassoFS: bool = True            # Lasso first stage (Belloni et al. 2012, Sec 3)
    include_post_lasso_iv: bool = True           # NEW: BCCH 2012 post-Lasso IV
    include_2sls_enetFS: bool = True             # appendix robustness (Bae 2018)
    include_two_stage_lasso: bool = False        # default off (Section 5.5 robustness only)
    include_debiased: bool = True
    include_debiased_cf: bool = True
    include_debiased_ridge: bool = False         # appendix robustness; gated by robust_variants_min_n_goods
    include_debiased_nodewise: bool = False      # appendix robustness; gated
    robust_variants_min_n_goods: int = 0
    cv: int = 5
    lasso_n_jobs: int = 1


def run_all_estimators(summary, i=0, random_state=20,
                       config: Optional[EstimatorConfig] = None,
                       endog_expenditure: bool = False):
    """
    Run every estimator in `config` on share equation `i` of `summary`
    and return a dict keyed by estimator name.

    If endog_expenditure=True, log(x/P*) is treated as an additional
    endogenous regressor and instrumented with Z.
    """
    if config is None:
        config = EstimatorConfig()
    y, X, Z, meta = build_yX_from_summary(
        summary, i=i, endog_expenditure=endog_expenditure
    )
    idx_beta = meta["idx_beta"]
    idx_gamma_ii = meta["idx_gamma_ii"]
    endog_cols = meta["endog_cols_prices"]

    out = {"meta": meta}

    if config.include_ols:
        out["ols"] = fit_ols(y, X)
    if config.include_2sls_plain:
        out["2sls_plain"] = fit_2sls_plain(y, X, Z, endog_cols=endog_cols)

    # Shared Lasso-FS Dhat (in-sample) for speed
    need_lasso_fs = (config.include_2sls_lassoFS
                     or config.include_two_stage_lasso
                     or config.include_debiased)
    if need_lasso_fs:
        Dhat_lasso, fs_models_lasso = build_Dhat(
            Z, X, endog_cols, kind="lasso", cv=config.cv,
            random_state=random_state, lasso_n_jobs=config.lasso_n_jobs,
        )
    if config.include_2sls_lassoFS:
        out["2sls_lassoFS"] = fit_2sls_firststage(y, Dhat_lasso)
    if config.include_post_lasso_iv:
        out["post_lasso_iv"] = fit_post_lasso_iv(
            y, X, Z, endog_cols,
            cv=config.cv, random_state=random_state,
            lasso_n_jobs=config.lasso_n_jobs,
        )
    if config.include_two_stage_lasso:
        out["two_stage_lasso"] = fit_two_stage_lasso(
            y, Dhat_lasso, kind="lasso", cv=config.cv,
            random_state=random_state, lasso_n_jobs=config.lasso_n_jobs,
        )
    if config.include_debiased:
        out["debiased_hd_iv"] = fit_debiased_hd_iv(
            y, X, Z, endog_cols,
            idx_beta=idx_beta, idx_gamma_ii=idx_gamma_ii,
            crossfit=False, first_stage_kind="lasso",
            se_kind="sandwich",
            cv=config.cv, random_state=random_state,
            lasso_n_jobs=config.lasso_n_jobs,
            Dhat_cached=Dhat_lasso, fs_models_cached=fs_models_lasso,
        )
    if config.include_debiased_cf:
        # Must re-fit Dhat with cross-fitting; do not reuse in-sample Dhat
        out["debiased_hd_iv_cf"] = fit_debiased_hd_iv(
            y, X, Z, endog_cols,
            idx_beta=idx_beta, idx_gamma_ii=idx_gamma_ii,
            crossfit=True, first_stage_kind="lasso",
            se_kind="sandwich",
            cv=config.cv, random_state=random_state,
            lasso_n_jobs=config.lasso_n_jobs,
        )

    n_goods_here = X.shape[1] - 2
    run_robust_variants = (n_goods_here >= config.robust_variants_min_n_goods)

    # Ridge-CLIME variant: Reuses Dhat_lasso.
    if config.include_debiased_ridge and run_robust_variants:
        out["debiased_hd_iv_ridge"] = fit_debiased_hd_iv(
            y, X, Z, endog_cols,
            idx_beta=idx_beta, idx_gamma_ii=idx_gamma_ii,
            crossfit=False, first_stage_kind="lasso",
            se_kind="sandwich",
            cv=config.cv, random_state=random_state,
            lasso_n_jobs=config.lasso_n_jobs,
            Dhat_cached=Dhat_lasso, fs_models_cached=fs_models_lasso,
            precision_method="clime", on_singular="ridge",
        )
    # Nodewise-Lasso variant: Reuses Dhat_lasso.
    if config.include_debiased_nodewise and run_robust_variants:
        out["debiased_hd_iv_nodewise"] = fit_debiased_hd_iv(
            y, X, Z, endog_cols,
            idx_beta=idx_beta, idx_gamma_ii=idx_gamma_ii,
            crossfit=False, first_stage_kind="lasso",
            se_kind="sandwich",
            cv=config.cv, random_state=random_state,
            lasso_n_jobs=config.lasso_n_jobs,
            Dhat_cached=Dhat_lasso, fs_models_cached=fs_models_lasso,
            precision_method="nodewise",
        )
    if config.include_2sls_enetFS:
        Dhat_enet, _ = build_Dhat(
            Z, X, endog_cols, kind="enet", cv=config.cv,
            random_state=random_state, lasso_n_jobs=config.lasso_n_jobs,
        )
        out["2sls_enetFS"] = fit_2sls_firststage(y, Dhat_enet)

    return out


MAIN_ESTIMATORS = (
    "OLS", "2SLS", "2SLS-LassoFS", "Post-Lasso-IV",
    "Debiased-HDIV", "Debiased-HDIV-CF",
)
APPENDIX_ESTIMATORS = (
    "2SLS-EnetFS", "Debiased-HDIV-Ridge", "Debiased-HDIV-Nodewise",
)


def extract_targets(results, tier="all", include_cf=True, include_enet=True):
    """Flatten estimator results into (name, beta_hat, beta_se, gii_hat, gii_se).

    `tier` controls which estimators appear:
      "main"     -- the six estimators in the main paper figures
      "appendix" -- the three robustness estimators in the appendix
      "all"      -- both (default)
    """
    meta = results["meta"]
    ib, ig = meta["idx_beta"], meta["idx_gamma_ii"]
    rows = []

    def _add(name, b_arr, se_arr):
        if tier == "main"     and name not in MAIN_ESTIMATORS:     return
        if tier == "appendix" and name not in APPENDIX_ESTIMATORS: return
        rows.append((name, float(b_arr[ib]),
                     float(se_arr[ib]) if se_arr is not None else np.nan,
                     float(b_arr[ig]),
                     float(se_arr[ig]) if se_arr is not None else np.nan))

    if "ols" in results:
        _add("OLS", results["ols"]["b"], results["ols"]["se"])
    if "2sls_plain" in results:
        _add("2SLS", results["2sls_plain"]["b"], results["2sls_plain"]["se"])
    if "2sls_lassoFS" in results:
        _add("2SLS-LassoFS", results["2sls_lassoFS"]["b"], results["2sls_lassoFS"]["se"])
    if "post_lasso_iv" in results:
        _add("Post-Lasso-IV", results["post_lasso_iv"]["b"], results["post_lasso_iv"]["se"])
    if include_enet and "2sls_enetFS" in results:
        _add("2SLS-EnetFS", results["2sls_enetFS"]["b"], results["2sls_enetFS"]["se"])
    # Two-stage Lasso is not in the paper's reported slate. When explicitly
    # enabled via include_two_stage_lasso=True, it appears under tier="all" only.
    if tier == "all" and "two_stage_lasso" in results:
        _add("2stage-Lasso", results["two_stage_lasso"]["b"], results["two_stage_lasso"]["se"])
    if "debiased_hd_iv" in results:
        d = results["debiased_hd_iv"]
        _add("Debiased-HDIV", d["b_tilde"], d["se_tilde"])
    if include_cf and "debiased_hd_iv_cf" in results:
        d = results["debiased_hd_iv_cf"]
        _add("Debiased-HDIV-CF", d["b_tilde"], d["se_tilde"])
    if "debiased_hd_iv_ridge" in results:
        d = results["debiased_hd_iv_ridge"]
        _add("Debiased-HDIV-Ridge", d["b_tilde"], d["se_tilde"])
    if "debiased_hd_iv_nodewise" in results:
        d = results["debiased_hd_iv_nodewise"]
        _add("Debiased-HDIV-Nodewise", d["b_tilde"], d["se_tilde"])
    return rows


def get_true_params(summary, i=0):
    p = summary.get("params", {})
    if isinstance(p, dict) and "beta" in p and "gamma" in p:
        return float(p["beta"][i]), float(p["gamma"][i, i])
    return float("nan"), float("nan")


# --------------------------------------------------------------------------
# Diagnostics for the first stage
# --------------------------------------------------------------------------

def lasso_cv_r2(Z, x, *, cv=5, random_state=20, max_iter=5000, tol=1e-3):
    """Cross-validated R^2 of a Lasso regression of x on Z."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=kf, random_state=random_state,
                          max_iter=max_iter, tol=tol, n_jobs=1)),
    ])
    pipe.fit(Z, x)
    best_alpha = pipe.named_steps["lasso"].alpha_

    var_x = float(np.var(x, ddof=1))
    if var_x < 1e-12:
        return float("nan"), float("nan")

    oof = np.zeros_like(x)
    for tr, te in kf.split(Z):
        pipe2 = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(alpha=best_alpha, max_iter=max_iter, tol=tol)),
        ])
        pipe2.fit(Z[tr], x[tr])
        oof[te] = pipe2.predict(Z[te])
    r2_cv = 1.0 - float(np.mean((x - oof) ** 2)) / var_x

    xhat_in = pipe.predict(Z)
    r2_in = 1.0 - float(np.mean((x - xhat_in) ** 2)) / var_x
    return r2_in, r2_cv


def first_stage_r2_all(X, Z, endog_cols, *, cv=5, random_state=20,
                        max_iter=5000, tol=1e-3, n_jobs=1):
    def _one(j):
        _, r2 = lasso_cv_r2(Z, X[:, j], cv=cv, random_state=random_state,
                            max_iter=max_iter, tol=tol)
        return r2
    r2 = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(j) for j in endog_cols
    )
    return np.array(r2)
