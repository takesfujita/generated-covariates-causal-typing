#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
MANIFEST = ROOT / 'manifest.json'

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def summarize_effects(df: pd.DataFrame, by):
    rows = []
    for keys, g in df.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        err = g['estimate'].to_numpy() - g['true'].to_numpy()
        cov = g['covered'].to_numpy(dtype=float)
        n = len(g)
        p_cov = float(cov.mean())
        row = {col: key for col, key in zip(by, keys)}
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

def summarize_claim(df: pd.DataFrame):
    rows = []
    for procedure, g in df.groupby(['procedure'], dropna=False):
        if isinstance(procedure, tuple):
            procedure = procedure[0]
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

def assert_close(a: pd.DataFrame, b: pd.DataFrame, name: str):
    if list(a.columns) != list(b.columns):
        raise SystemExit(f'{name}: column mismatch {list(a.columns)} vs {list(b.columns)}')
    keys = [c for c in a.columns if a[c].dtype == object]
    if keys:
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
    else:
        a = a.reset_index(drop=True); b = b.reset_index(drop=True)
    for col in a.columns:
        if a[col].dtype == object:
            if not (a[col].astype(str).values == b[col].astype(str).values).all():
                raise SystemExit(f'{name}: text mismatch in {col}')
        else:
            if not np.allclose(a[col].values, b[col].values, rtol=1e-12, atol=1e-12):
                raise SystemExit(f'{name}: numeric mismatch in {col}')
    print(f'{name}: summary OK')

def check_manifest_hashes():
    manifest = json.loads(MANIFEST.read_text())
    for item in manifest['outputs']:
        path = ROOT / item['file']
        if not path.exists():
            raise SystemExit(f'manifest: missing file {item["file"]}')
        actual = sha256_file(path)
        if actual != item['sha256']:
            raise SystemExit(f'manifest: hash mismatch for {item["file"]}: {actual} != {item["sha256"]}')
    print('manifest: file hashes OK')

assert_close(summarize_effects(pd.read_csv(RES/'table4_replicates.csv'), ['scenario','method']), pd.read_csv(RES/'table4_summary.csv'), 'table4')
assert_close(summarize_effects(pd.read_csv(RES/'table5_replicates.csv'), ['variant','method']), pd.read_csv(RES/'table5_summary.csv'), 'table5')
assert_close(summarize_claim(pd.read_csv(RES/'table6_replicates.csv')), pd.read_csv(RES/'table6_summary.csv'), 'table6')
check_manifest_hashes()
print('All summaries reproduce from replicate-level CSV files and all manifest hashes match.')
