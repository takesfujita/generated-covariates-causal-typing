from pathlib import Path
import sys, json, hashlib, platform, importlib.util
import numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'run_generated_covariate_simulations.py'
RES = ROOT / 'results'
RES.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('sim_module', SCRIPT)
sim = importlib.util.module_from_spec(spec)
sys.modules['sim_module'] = sim
spec.loader.exec_module(sim)

REPS = 1000
MASTER = 20260609
rng_master = np.random.default_rng(MASTER)
seeds4 = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(REPS)]
seeds5 = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(REPS)]
seeds6 = [int(rng_master.integers(0, 2**32 - 1)) for _ in range(REPS)]
variants = ['Baseline', 'Positivity stress', 'Outcome compression', 'Nonclassical representation']

def one5(args):
    r, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for variant in variants:
        for row in sim.one_table5_rep(rng, variant):
            row.update(rep=r, seed=seed)
            rows.append(row)
    return rows

def one6(args):
    r, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for row in sim.one_table6_rep(rng):
        row.update(rep=r, seed=seed)
        rows.append(row)
    return rows

if __name__ == '__main__':
    nproc = min(max(cpu_count() - 1, 1), 8)
    print(f'Running table5 and table6 in parallel with nproc={nproc}', flush=True)
    rows = []
    with Pool(nproc) as pool:
        for i, rr in enumerate(pool.imap(one5, list(enumerate(seeds5)), chunksize=10)):
            if i % 100 == 0:
                print(f'table5 parallel rep {i}/{REPS}', flush=True)
            rows.extend(rr)
    t5 = pd.DataFrame(rows)
    t5.to_csv(RES / 'table5_replicates.csv', index=False)
    sim.summarize_effects(t5, ['variant', 'method']).to_csv(RES / 'table5_summary.csv', index=False)

    rows = []
    with Pool(nproc) as pool:
        for i, rr in enumerate(pool.imap(one6, list(enumerate(seeds6)), chunksize=20)):
            if i % 100 == 0:
                print(f'table6 parallel rep {i}/{REPS}', flush=True)
            rows.extend(rr)
    t6 = pd.DataFrame(rows)
    t6.to_csv(RES / 'table6_replicates.csv', index=False)
    sim.summarize_claim(t6).to_csv(RES / 'table6_summary.csv', index=False)
    print('DONE', flush=True)
