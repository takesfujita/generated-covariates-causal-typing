#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'run_generated_covariate_simulations.py'
MANIFEST = ROOT / 'manifest.json'
RESULT_FILES = [
    'results/table4_replicates.csv', 'results/table4_summary.csv',
    'results/table5_replicates.csv', 'results/table5_summary.csv',
    'results/table6_replicates.csv', 'results/table6_summary.csv',
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def expected_hashes():
    m = json.loads(MANIFEST.read_text())
    return {item['file']: item['sha256'] for item in m['outputs']}

def main():
    ap = argparse.ArgumentParser(description='Optional re-run verification for the artificial simulations.')
    ap.add_argument('--full', action='store_true', help='Run the full 1000-replication verification and compare hashes.')
    ap.add_argument('--reps', type=int, default=None, help='Override replication count. Defaults to 20 for smoke test or 1000 with --full.')
    ap.add_argument('--master-seed', type=int, default=20260609)
    args = ap.parse_args()
    reps = args.reps if args.reps is not None else (1000 if args.full else 20)
    with tempfile.TemporaryDirectory(prefix='generated_covariates_rerun_') as td:
        out = Path(td) / 'results'
        cmd = [sys.executable, str(SCRIPT), '--reps', str(reps), '--master-seed', str(args.master_seed), '--out-dir', str(out)]
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        if args.full and reps == 1000 and args.master_seed == 20260609:
            h = expected_hashes()
            for rel in RESULT_FILES:
                actual = sha256_file(out / Path(rel).name)
                expected = h[rel]
                if actual != expected:
                    raise SystemExit(f'Hash mismatch after full rerun for {rel}: {actual} != {expected}')
            print('Full rerun hashes match the included replicate and summary CSV files.')
        else:
            for name in ['table4_replicates.csv','table4_summary.csv','table5_replicates.csv','table5_summary.csv','table6_replicates.csv','table6_summary.csv']:
                if not (out / name).exists():
                    raise SystemExit(f'Smoke test did not create {name}')
            print('Smoke test completed and produced all expected output files.')

if __name__ == '__main__':
    main()
