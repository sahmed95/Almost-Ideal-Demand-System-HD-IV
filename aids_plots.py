"""
aids_plots.py
=============
Plotting functions for the AIDS HD-IV simulation outputs.

Module-level toggles:
  SHOW_TITLES      bool. False suppresses titles (paper figures).
  SAVE_DIR         Path. When set, every plot is saved to disk with a
                   filename encoding the target (beta/gii), xcol, estimator
                   group, and metric.
  FILENAME_PREFIX  str. Prepended to saved filenames.

Use:
    import aids_plots
    aids_plots.set_show_titles(False)
    aids_plots.set_save_dir("/tmp/aids_figures")
    aids_plots.set_filename_prefix("scenB_n")
    plot_rmse(df, xcol="n_goods", target="gii", only_all=True)
    # writes /tmp/aids_figures/scenB_n_rmse_gii_all.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# Module-level toggles
# ---------------------------------------------------------------------
SHOW_TITLES: bool = True
SAVE_DIR: Optional[Path] = None
FILENAME_PREFIX: str = ""  # e.g. "scenB_n_" to disambiguate when saving
                            

def set_show_titles(flag: bool) -> None:
    """Globally enable or disable plot titles for subsequent calls."""
    global SHOW_TITLES
    SHOW_TITLES = bool(flag)


def set_save_dir(path: Union[str, Path, None]) -> None:
    """Where to save figures. None disables saving."""
    global SAVE_DIR
    SAVE_DIR = Path(path) if path is not None else None
    if SAVE_DIR is not None:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)


def set_filename_prefix(prefix: str) -> None:
    """Prefix added to every saved filename (e.g. to mark the scenario)."""
    global FILENAME_PREFIX
    FILENAME_PREFIX = prefix or ""


# ---------------------------------------------------------------------
# LaTeX label dictionaries
# ---------------------------------------------------------------------
_XCOL_LABEL = {
    "n_goods": r"$n_{\mathrm{goods}}$",
    "pz":      r"$p_z$",
    "T":       r"$T$",
}

_TARGET_LABEL = {
    "beta": r"$\beta$",
    "gii":  r"$\gamma_{ii}$",
}


def _xlab(xcol: str) -> str:
    return _XCOL_LABEL.get(xcol, xcol)


def _tlab(target: str) -> str:
    return _TARGET_LABEL.get(target, target)


def _maybe_title(ax, text: str) -> None:
    if SHOW_TITLES:
        ax.set_title(text)


def _safe_filename(s: str) -> str:
    """Sanitise a string for use as a filename."""
    keep = []
    for c in s:
        if c.isalnum() or c in ("_", "-"):
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def _save_fig(fig, stem: str) -> None:
    """Save the current figure to SAVE_DIR if set."""
    if SAVE_DIR is None:
        return
    name = _safe_filename(f"{FILENAME_PREFIX}_{stem}" if FILENAME_PREFIX else stem)
    out = SAVE_DIR / f"{name}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------
# Estimator groupings
# ---------------------------------------------------------------------
EST_GROUPS = [
    ("Core: OLS / 2SLS / Debiased-HDIV-CF",
     ["OLS", "2SLS", "Debiased-HDIV-CF"]),
    ("Regularized IV: LassoFS / Post-Lasso-IV / Debiased-HDIV-CF",
     ["2SLS-LassoFS", "Post-Lasso-IV", "Debiased-HDIV-CF"]),
    ("Debiased variants: HDIV / HDIV-CF / Ridge / Nodewise",
     ["Debiased-HDIV", "Debiased-HDIV-CF",
      "Debiased-HDIV-Ridge", "Debiased-HDIV-Nodewise"]),
    ("All estimators", None),
]

# Short tag per group for use in saved filenames
_GROUP_TAG = {
    "Core: OLS / 2SLS / Debiased-HDIV-CF": "core",
    "Regularized IV: LassoFS / Post-Lasso-IV / Debiased-HDIV-CF": "regIV",
    "Debiased variants: HDIV / HDIV-CF / Ridge / Nodewise": "debvar",
    "All estimators": "all",
}


def _present(df: pd.DataFrame, ests: Optional[Iterable[str]]) -> list:
    if ests is None:
        return sorted(df["estimator"].unique())
    return [e for e in ests if e in set(df["estimator"].unique())]


def _active_groups(only_all: bool):
    if only_all:
        return [g for g in EST_GROUPS if g[0] == "All estimators"]
    return EST_GROUPS


def _plot_groups(df, xcol, ycol, ylab, title_prefix, *,
                 truth_col: Optional[str] = None,
                 band_cols: Optional[tuple] = None,
                 only_all: bool = False,
                 save_stem: Optional[str] = None):
    """Produce one figure per estimator group."""
    for gname, ests in _active_groups(only_all):
        sel = _present(df, ests)
        if not sel:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for est in sel:
            g_full = df[df["estimator"] == est].sort_values(xcol)
            if band_cols is None:
                g = g_full.dropna(subset=[ycol])
                if g.empty:
                    continue
                ax.plot(g[xcol].to_numpy(), g[ycol].to_numpy(),
                        marker="o", label=est)
            else:
                q05, q50, q95 = band_cols
                g = g_full.dropna(subset=[q50, q05, q95])
                if g.empty:
                    continue
                x = g[xcol].to_numpy()
                ax.plot(x, g[q50].to_numpy(), marker="o", label=est)
                ax.fill_between(x, g[q05].to_numpy(), g[q95].to_numpy(), alpha=0.2)
        if truth_col is not None:
            gt = df.sort_values([xcol, "estimator"]).groupby(xcol).head(1)
            ax.plot(gt[xcol], gt[truth_col], linestyle="--", linewidth=2,
                    color="k", label="truth")
        ax.set_xlabel(_xlab(xcol))
        ax.set_ylabel(ylab)
        _maybe_title(ax, f"{title_prefix}, {gname}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        if save_stem is not None:
            _save_fig(fig, f"{save_stem}_{_GROUP_TAG[gname]}")
        plt.show()


# ---- single-run plots -------------------------------------------------

def plot_single_run(df, xcol, target="beta", metric="bias",
                    only_all: bool = False):
    key_root = "beta" if target == "beta" else "gii"
    key = {"bias": f"{key_root}_bias",
           "sqerr": f"{key_root}_sqerr",
           "hat": f"{key_root}_hat"}[metric]
    ylab = f"{_tlab(target)} {metric}"
    _plot_groups(df, xcol, key, ylab,
                 title_prefix=f"Single run: {_tlab(target)} {metric} vs {_xlab(xcol)}",
                 only_all=only_all,
                 save_stem=f"singlerun_{metric}_{target}_vs_{xcol}")


# ---- MC plots ----------------------------------------------------------

def plot_mc_bands(df_mc, xcol, target="beta", only_all: bool = False):
    if target == "beta":
        q05, q50, q95, truth = "beta_q05", "beta_q50", "beta_q95", "beta_true"
        ylab = r"$\beta$ (MC median, 5–95\%)"
    else:
        q05, q50, q95, truth = "gii_q05", "gii_q50", "gii_q95", "gii_true"
        ylab = r"$\gamma_{ii}$ (MC median, 5–95\%)"
    _plot_groups(df_mc, xcol, q50, ylab,
                 title_prefix=f"MC bands vs {_xlab(xcol)} ({_tlab(target)})",
                 truth_col=truth, band_cols=(q05, q50, q95),
                 only_all=only_all,
                 save_stem=f"bands_{target}_vs_{xcol}")


def plot_rmse(df_mc, xcol, target="beta", only_all: bool = False):
    col = "beta_rmse" if target == "beta" else "gii_rmse"
    ylab = f"RMSE of {_tlab(target)}"
    _plot_groups(df_mc, xcol, col, ylab,
                 title_prefix=f"{_tlab(target)} RMSE vs {_xlab(xcol)}",
                 only_all=only_all,
                 save_stem=f"rmse_{target}_vs_{xcol}")


def plot_coverage(df_mc, xcol, target="beta", only_all: bool = False):
    col = "beta_coverage" if target == "beta" else "gii_coverage"
    ylab = f"Coverage of {_tlab(target)}"
    for gname, ests in _active_groups(only_all):
        sel = _present(df_mc, ests)
        if not sel:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for est in sel:
            g = (df_mc[df_mc["estimator"] == est]
                 .sort_values(xcol)
                 .dropna(subset=[col]))
            if g.empty:
                continue
            ax.plot(g[xcol], g[col], marker="o", label=est)
        ax.axhline(0.95, linestyle="--", linewidth=2, color="k",
                   label="0.95 nominal")
        ax.set_xlabel(_xlab(xcol))
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.05, 1.05)
        _maybe_title(ax, f"{_tlab(target)} coverage vs {_xlab(xcol)}, {gname}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        _save_fig(fig, f"coverage_{target}_vs_{xcol}_{_GROUP_TAG[gname]}")
        plt.show()


def plot_instability(df_mc, xcol, target="gii", only_all: bool = False):
    if target == "beta":
        varcol, explcol, nancol = "beta_var", "beta_explode_rate", "beta_nan_rate"
    else:
        varcol, explcol, nancol = "gii_var", "gii_explode_rate", "gii_nan_rate"
    _plot_groups(df_mc, xcol, varcol, f"MC variance of {_tlab(target)}",
                 title_prefix=f"{_tlab(target)}: variance vs {_xlab(xcol)}",
                 only_all=only_all,
                 save_stem=f"variance_{target}_vs_{xcol}")
    _plot_groups(df_mc, xcol, explcol, f"Explode rate ({_tlab(target)})",
                 title_prefix=f"{_tlab(target)}: explode rate vs {_xlab(xcol)}",
                 only_all=only_all,
                 save_stem=f"explode_{target}_vs_{xcol}")
    _plot_groups(df_mc, xcol, nancol, f"Non-finite rate ({_tlab(target)})",
                 title_prefix=f"{_tlab(target)}: non-finite rate vs {_xlab(xcol)}",
                 only_all=only_all,
                 save_stem=f"nanrate_{target}_vs_{xcol}")
