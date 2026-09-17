# When AI Generates Covariates

## Causal Typing and Estimand Drift in Sequential Experiments

**Takes Fujita** — VRI  
**Nobutaka Hattori** — Department of Neurology, Juntendo University School of Medicine

**Paper:** [arXiv:2609.17772](https://arxiv.org/abs/2609.17772)

This repository contains the manuscript source, simulation code, and synthetic results for *When AI Generates Covariates: Causal Typing and Estimand Drift in Sequential Experiments*.

## Overview

A variable extracted from a clinical note, diary, image, or sensor stream may describe a pre-treatment state, a treatment response, a mediator, or an observation process. Using these variables interchangeably can change the causal question being answered.

The paper develops a framework that assigns causal roles and claim status to generated covariates under a fixed estimand. It examines when a generated representation preserves the target and when compression, treatment leakage, or outcome-guided selection changes the interpretation of an estimate.

The framework combines causal typing, an estimand lock, and audit rules. Its compression analysis builds on the established conditional-covariance characterization of bias in the deconfounding-score literature. The paper also gives conditions for cluster-level orthogonal estimation in repeated-session experiments.

The simulations examine design erasure, mediator adjustment, post-action leakage, representation error, and marker selection. All data are synthetic; no patient records are included.

## Repository contents

| Path | Contents |
| --- | --- |
| [manuscript_final.tex](manuscript_final.tex) | Main manuscript, including the bibliography |
| [anc/run_generated_covariate_simulations.py](anc/run_generated_covariate_simulations.py) | Data-generating mechanisms and simulation procedures |
| [anc/results/](anc/results/) | Replicate-level results and summaries for Tables 4–6 |
| [anc/verify_results.py](anc/verify_results.py) | Summary recomputation and package hash checks |
| [anc/verify_full_rerun.py](anc/verify_full_rerun.py) | Smoke run and full simulation rerun |
| [anc/requirements.txt](anc/requirements.txt) | Pinned numerical dependencies |
| [anc/manifest.json](anc/manifest.json) | Simulation settings, recorded environment, and file hashes |

## Check the bundled results

Run the following commands from the repository root, preferably in a virtual environment:

```bash
python3 -m pip install -r anc/requirements.txt
python3 anc/verify_results.py
```

The verification script recomputes the three summary tables from the replicate-level CSV files, compares numeric values with relative and absolute tolerances of `1e-12`, and checks the hashes of the files listed in the manifest. This check uses the bundled data without rerunning the simulations.

## Rerun the simulations

For a 20-replication smoke run:

```bash
python3 anc/verify_full_rerun.py
```

For the full run:

```bash
python3 anc/verify_full_rerun.py --full
```

The full run uses 1,000 replications and master seed `20260609`. It compares the six generated CSV files with the recorded SHA-256 hashes. The recorded simulation environment is Python 3.13.5, NumPy 2.3.5, and pandas 2.2.3. Matching the environment matters for byte-for-byte comparisons. Manuscript tables display selected columns rounded to three decimal places.

## Compile the manuscript

From the repository root, using a LaTeX installation with the required packages:

```bash
pdflatex -interaction=nonstopmode -halt-on-error manuscript_final.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript_final.tex
```

The bibliography is embedded, so BibTeX is not required. The manuscript has no external figures.

## Scope

The code implements controlled examples of the failure modes studied in the paper. Applying the framework to another study requires specifying its target, treatment timing, information available before each decision, and identification assumptions. The synthetic results do not establish validity for a particular clinical dataset.

## Citation

If you use this work or its simulation code, please cite the paper:

```bibtex
@misc{fujita2026generatedcovariates,
  author        = {Fujita, Takes and Hattori, Nobutaka},
  title         = {{When AI Generates Covariates: Causal Typing and Estimand Drift in Sequential Experiments}},
  year          = {2026},
  eprint        = {2609.17772},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.17772}
}
```

Package citation information is also available in [anc/CITATION.cff](anc/CITATION.cff).

## License

The supplementary software and its accompanying software documentation are provided under the [MIT License](anc/LICENSE).

The manuscript was submitted under the [arXiv.org perpetual, non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/). The software license does not apply to the manuscript.
