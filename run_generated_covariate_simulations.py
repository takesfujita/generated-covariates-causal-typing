#!/usr/bin/env python3
"""
Reproducible simulations for:
When AI Generates Covariates: Causal Typing and Estimand Drift in Sequential Experiments

This script is the authoritative implementation of the artificial data-generating mechanisms used to produce the reported simulation tables. It uses artificial data only. It is not a clinical simulation and does not use real patient data.

Outputs:
  results/table4_replicates.csv, results/table4_summary.csv
  results/table5_replicates.csv, results/table5_summary.csv
  results/table6_replicates.csv, results/table6_summary.csv
  results/manifest.json

The manuscript simulation tables should be generated from this script using the seeds and settings recorded in the manifest. The summary CSV files include Monte Carlo standard errors for bias and coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class DGPConfig:
    n_participants: int = 140
    n_sessions: int = 6
    positivity_stress: bool = False
    outcome_compression: bool = False
    nonclassical_representation: bool = False
    observed_state_mar: bool = False


def generate_panel(rng: np.random.Generator, cfg: DGPConfig) -> pd.DataFrame:
    """Generate repeated-session data with known empirical total effect.

    The total effect of setting A from 0 to 1 is 0.45 + 0.45 * U_true because
    A has a direct coefficient 0.20 + 0.45 * U_true and shifts the mediator M by
    0.50, whose outcome coefficient is 0.50.
    """
    rows: List[dict] = []
    n, S = cfg.n_participants, cfg.n_sessions
    C = rng.normal(0, 1, size=n)
    b = rng.normal(0, 0.45, size=n)
    prev_u = np.zeros(n)
    prev_y = np.zeros(n)
    prev_a = np.zeros(n)

    # Coefficients varied in stress tests.
    assign_d_coef = 1.60 if cfg.positivity_stress else 0.80
    outcome_d_coef = 0.00 if cfg.outcome_compression else 0.50

    for s in range(1, S + 1):
        # Design / availability variable is pre-action and partly participant-specific.
        pD = expit(-0.10 + 0.35 * C + 0.12 * (s - 3.5))
        D = rng.binomial(1, pD, size=n)

        # Stable latent state around mean 1.0, with dependence on history and design.
        U_true = (
            0.85
            + 0.35 * C
            + 0.35 * D
            + 0.20 * prev_u
            + 0.08 * prev_a
            + 0.04 * prev_y
            + rng.normal(0, 0.55, size=n)
        )
        U0 = U_true + rng.normal(0, 0.60, size=n)  # coarse pre-action state
        U1 = U_true + rng.normal(0, 0.25, size=n)  # refined generated state marker
        if cfg.nonclassical_representation:
            U1 = U1 + 0.55 * (D - D.mean())

        pA = expit(-0.10 + 0.45 * U0 + assign_d_coef * D + 0.06 * (s - 3.5))
        pA = np.clip(pA, 0.04, 0.96)
        A = rng.binomial(1, pA, size=n)

        M = 0.50 * A + 0.45 * U_true + 0.20 * C + 0.25 * D + rng.normal(0, 0.60, size=n)
        L = M + rng.normal(0, 0.45, size=n)  # post-action leakage covariate
        eps = rng.normal(0, 1.00, size=n)
        Y = (
            b
            + 0.45 * C
            + 0.55 * U_true
            + outcome_d_coef * D
            + 0.20 * prev_y
            + 0.05 * s
            + A * (0.20 + 0.45 * U_true)
            + 0.50 * M
            + eps
        )
        true_total = 0.45 + 0.45 * U_true

        # Observed-state MAR missingness based only on observed pre-action/history variables.
        if cfg.observed_state_mar:
            pR = expit(0.65 + 0.15 * A + 0.25 * U0 + 0.25 * D + 0.10 * C + 0.08 * prev_y - 0.04 * s)
            pR = np.clip(pR, 0.15, 0.98)
            R = rng.binomial(1, pR, size=n)
        else:
            pR = np.ones(n)
            R = np.ones(n, dtype=int)

        for i in range(n):
            rows.append({
                'id': i,
                'session': s,
                'C': C[i],
                'D': D[i],
                'U_true': U_true[i],
                'U0': U0[i],
                'U1': U1[i],
                'A': A[i],
                'pA': pA[i],
                'M': M[i],
                'L': L[i],
                'Y': Y[i],
                'R': R[i],
                'pR': pR[i],
                'prev_y': prev_y[i],
                'prev_a': prev_a[i],
                'true_total': true_total[i],
            })
        prev_u, prev_y, prev_a = U_true, Y, A

    return pd.DataFrame(rows)


def design_matrix(df: pd.DataFrame, method: str) -> Tuple[np.ndarray, List[str], Optional[str]]:
    """Return X matrix, column names, modifier variable used for standardized contrast."""
    n = len(df)
    base = [np.ones(n)]
    names = ['Intercept']

    def add(col: str):
        base.append(df[col].to_numpy(dtype=float))
        names.append(col)

    def add_arr(name: str, arr: np.ndarray):
        base.append(arr.astype(float))
        names.append(name)

    add('A')
    modifier: Optional[str] = None

    if method == 'Untyped naive contrast':
        pass
    elif method == 'Coarse locked state':
        for col in ['U0', 'D', 'C', 'prev_y', 'session']:
            add(col)
        add_arr('A:U0', df['A'].to_numpy() * df['U0'].to_numpy())
        modifier = 'U0'
    elif method == 'Design erasure':
        # Predictive representation replaces design variable D.
        for col in ['U1', 'C', 'prev_y', 'session']:
            add(col)
        add_arr('A:U1', df['A'].to_numpy() * df['U1'].to_numpy())
        modifier = 'U1'
    elif method in ('Admissible refinement', 'IPCW admissible refinement'):
        for col in ['U1', 'U0', 'D', 'C', 'prev_y', 'session']:
            add(col)
        add_arr('A:U1', df['A'].to_numpy() * df['U1'].to_numpy())
        modifier = 'U1'
    elif method == 'Mediator-as-state':
        for col in ['M', 'U0', 'D', 'C', 'prev_y', 'session']:
            add(col)
        # Intentionally no A:M interaction; this targets a path-blocked/direct contrast.
    elif method == 'Post-action leakage':
        for col in ['L', 'U0', 'D', 'C', 'prev_y', 'session']:
            add(col)
        add_arr('A:L', df['A'].to_numpy() * df['L'].to_numpy())
        modifier = 'L'
    elif method == 'Oracle state':
        for col in ['U_true', 'D', 'C', 'prev_y', 'session']:
            add(col)
        add_arr('A:U_true', df['A'].to_numpy() * df['U_true'].to_numpy())
        modifier = 'U_true'
    else:
        raise ValueError(f'Unknown method: {method}')

    return np.column_stack(base), names, modifier


def fit_linear_cluster(df: pd.DataFrame, method: str, use_observed: bool = False, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Fit a linear working model and return standardized A contrast plus cluster-robust SE."""
    if use_observed:
        d = df[df['R'] == 1].copy()
    else:
        d = df.copy()
    X, names, modifier = design_matrix(d, method)
    y = d['Y'].to_numpy(dtype=float)
    if weights is None:
        w = np.ones(len(d))
    else:
        # weights supplied for full df; subset if needed
        w = np.asarray(weights)
        if use_observed and len(w) == len(df):
            w = w[df['R'].to_numpy() == 1]
        w = np.clip(w.astype(float), 0, 50)

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    XtX = Xw.T @ Xw
    ridge = 1e-8 * np.eye(XtX.shape[0])
    beta = np.linalg.solve(XtX + ridge, Xw.T @ yw)
    resid = y - X @ beta

    # Cluster sandwich for WLS estimating equations X_i * w_i * residual_i.
    bread = np.linalg.inv(XtX + ridge)
    ids = d['id'].to_numpy(dtype=int)
    n_clusters = int(ids.max()) + 1
    cluster_scores = np.zeros((n_clusters, X.shape[1]))
    np.add.at(cluster_scores, ids, X * (w * resid)[:, None])
    meat = cluster_scores.T @ cluster_scores
    vcov = bread @ meat @ bread

    c = np.zeros(len(names))
    c[names.index('A')] = 1.0
    if modifier is not None:
        iname = f'A:{modifier}'
        c[names.index(iname)] = df[modifier].mean()  # target mean over all target sessions
    est = float(c @ beta)
    se = float(np.sqrt(max(c @ vcov @ c, 1e-12)))
    return est, se


def one_table4_rep(rng: np.random.Generator, scenario: str) -> List[dict]:
    observed = scenario == 'Observed-state MAR'
    df = generate_panel(rng, DGPConfig(observed_state_mar=observed))
    true = float(df['true_total'].mean())
    methods = [
        'Untyped naive contrast',
        'Coarse locked state',
        'Design erasure',
        'Admissible refinement',
    ]
    if observed:
        methods.append('IPCW admissible refinement')
    methods += ['Mediator-as-state', 'Post-action leakage', 'Oracle state']
    rows = []
    for method in methods:
        if method == 'IPCW admissible refinement':
            weights = 1.0 / np.clip(df['pR'].to_numpy(), 0.05, 1.0)
            est, se = fit_linear_cluster(df, method, use_observed=True, weights=weights)
        else:
            est, se = fit_linear_cluster(df, method, use_observed=observed)
        rows.append({
            'scenario': scenario,
            'method': method,
            'true': true,
            'estimate': est,
            'se': se,
            'covered': int((est - 1.96 * se) <= true <= (est + 1.96 * se)),
        })
    return rows


def one_table5_rep(rng: np.random.Generator, variant: str) -> List[dict]:
    cfg = DGPConfig(
        positivity_stress=variant == 'Positivity stress',
        outcome_compression=variant == 'Outcome compression',
        nonclassical_representation=variant == 'Nonclassical representation',
    )
    df = generate_panel(rng, cfg)
    true = float(df['true_total'].mean())
    rows = []
    for method in ['Design erasure', 'Admissible refinement', 'Oracle state']:
        est, se = fit_linear_cluster(df, method, use_observed=False)
        rows.append({
            'variant': variant,
            'method': method,
            'true': true,
            'estimate': est,
            'se': se,
            'covered': int((est - 1.96 * se) <= true <= (est + 1.96 * se)),
        })
    return rows


def one_table6_rep(rng: np.random.Generator, n: int = 840, k: int = 25) -> List[dict]:
    """Claim-status stress test under no true generated-covariate moderation."""
    A = rng.binomial(1, 0.5, size=n)
    U = rng.normal(size=(n, k))
    # Outcome has treatment main effect and covariate main effects but no treatment interactions.
    beta_u = rng.normal(0, 0.15, size=k)
    y = 0.5 * A + U @ beta_u + rng.normal(0, 1.0, size=n)

    def interaction_t(idx: np.ndarray, j: int) -> Tuple[float, float, float]:
        a = A[idx]
        u = U[idx, j]
        X = np.column_stack([np.ones(len(idx)), a, u, a * u])
        yy = y[idx]
        beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        resid = yy - X @ beta
        sigma2 = float((resid @ resid) / max(len(idx) - X.shape[1], 1))
        vcov = sigma2 * np.linalg.inv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
        est = float(beta[3])
        se = float(np.sqrt(max(vcov[3, 3], 1e-12)))
        return est, se, est / se

    all_idx = np.arange(n)
    stats = [interaction_t(all_idx, j) for j in range(k)]
    j_star = int(np.argmax(np.abs([t[2] for t in stats])))
    est, se, t = stats[j_star]
    same = {
        'procedure': 'Same data selection and inference',
        'estimate': est,
        'se': se,
        'covered': int((est - 1.96 * se) <= 0.0 <= (est + 1.96 * se)),
        'selected_abs_t': abs(t),
    }

    perm = rng.permutation(n)
    sel_idx = perm[: n // 2]
    est_idx = perm[n // 2 :]
    sel_stats = [interaction_t(sel_idx, j) for j in range(k)]
    j_split = int(np.argmax(np.abs([t[2] for t in sel_stats])))
    est2, se2, t2 = interaction_t(est_idx, j_split)
    split = {
        'procedure': 'Split selection and independent inference',
        'estimate': est2,
        'se': se2,
        'covered': int((est2 - 1.96 * se2) <= 0.0 <= (est2 + 1.96 * se2)),
        'selected_abs_t': abs(t2),
    }
    return [same, split]


def summarize_effects(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        err = g['estimate'].to_numpy() - g['true'].to_numpy()
        cov = g['covered'].to_numpy(dtype=float)
        n = len(g)
        p_cov = float(cov.mean())
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            'true': float(g['true'].mean()),
            'estimate': float(g['estimate'].mean()),
            'bias': float(err.mean()),
            'bias_mcse': float(err.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mean_se': float(g['se'].mean()),
            'coverage': p_cov,
            'coverage_mcse': float(math.sqrt(p_cov * (1.0 - p_cov) / n)) if n > 0 else float('nan'),
            'n_rep': int(n),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_claim(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for procedure, g in df.groupby('procedure', dropna=False):
        est = g['estimate'].to_numpy(dtype=float)
        cov = g['covered'].to_numpy(dtype=float)
        n = len(g)
        p_cov = float(cov.mean())
        rows.append({
            'procedure': procedure,
            'mean_estimate': float(est.mean()),
            'mean_estimate_mcse': float(est.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            'rmse': float(np.sqrt(np.mean(est ** 2))),
            'mean_se': float(g['se'].mean()),
            'coverage': p_cov,
            'coverage_mcse': float(math.sqrt(p_cov * (1.0 - p_cov) / n)) if n > 0 else float('nan'),
            'mean_selected_abs_t': float(g['selected_abs_t'].mean()),
            'n_rep': int(n),
        })
    return pd.DataFrame(rows)


def run_all(reps: int, out_dir: Path, master_seed: int) -> None:
    rng_master = np.random.default_rng(master_seed)

    t4_rows: List[dict] = []
    for r in range(reps):
        if r % 100 == 0:
            print(f'table4 rep {r}/{reps}', flush=True)
        seed = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        for scenario in ['Complete', 'Observed-state MAR']:
            for row in one_table4_rep(rng, scenario):
                row.update(rep=r, seed=seed)
                t4_rows.append(row)
    table4 = pd.DataFrame(t4_rows)
    table4.to_csv(out_dir / 'table4_replicates.csv', index=False)
    summarize_effects(table4, ['scenario', 'method']).to_csv(out_dir / 'table4_summary.csv', index=False)

    t5_rows: List[dict] = []
    for r in range(reps):
        if r % 100 == 0:
            print(f'table5 rep {r}/{reps}', flush=True)
        seed = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        for variant in ['Baseline', 'Positivity stress', 'Outcome compression', 'Nonclassical representation']:
            for row in one_table5_rep(rng, variant):
                row.update(rep=r, seed=seed)
                t5_rows.append(row)
    table5 = pd.DataFrame(t5_rows)
    table5.to_csv(out_dir / 'table5_replicates.csv', index=False)
    summarize_effects(table5, ['variant', 'method']).to_csv(out_dir / 'table5_summary.csv', index=False)

    t6_rows: List[dict] = []
    for r in range(reps):
        if r % 100 == 0:
            print(f'table6 rep {r}/{reps}', flush=True)
        seed = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        for row in one_table6_rep(rng):
            row.update(rep=r, seed=seed)
            t6_rows.append(row)
    table6 = pd.DataFrame(t6_rows)
    table6.to_csv(out_dir / 'table6_replicates.csv', index=False)
    summarize_claim(table6).to_csv(out_dir / 'table6_summary.csv', index=False)

    manifest = {
        'master_seed': master_seed,
        'n_replications': reps,
        'created_by_script': 'run_generated_covariate_simulations.py',
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'outputs': {},
        'note': 'Artificial simulations only; no real patient data. Tables are generated by this script from the DGP specified in the script.',
    }
    for p in sorted(out_dir.glob('*')):
        if p.is_file() and p.name != 'manifest.json':
            manifest['outputs'][p.name] = {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--master-seed', type=int, default=20260609)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_all(args.reps, args.out_dir, args.master_seed)


if __name__ == '__main__':
    main()
