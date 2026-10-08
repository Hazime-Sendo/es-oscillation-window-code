# Code and verification suite — oscillation-window study (Physica A)

This repository holds the numerical-experiment code, cached result
data, and a verification suite supporting the paper *Onset of a
finite-amplitude oscillation window at the disappearance of a
regime-switching cycle in a seven-variable economic model: an
empirical threshold formula from an energy-budget decomposition*
(submitted to *Physica A: Statistical Mechanics and its
Applications*).

**This repository is the code only.** The manuscript itself (LaTeX
source and compiled PDF) lives in a separate repository — see
[Companion paper repository](#companion-paper-repository) below.

## What this code does

`src/` implements the seven-variable ODE model ("Economic Science" /
ES), the first-harmonic-balance solver, and the empirical formula
$b^*(\lambda)$ (Eq. 9 of the paper). `scripts/` contains the numerical
experiments built on top of `src/` — the bisection searches used to
find the onset threshold $b^*$ of the finite-amplitude oscillation
window described in the paper, and the energy-budget decomposition
used to explain its mechanism. `data/` holds the cached numerical
results those searches produced, and `results/` holds a reproducibility
log. `verification/run_verification.py` is a single script that
re-checks the paper's numeric claims (Sec. 5 and Sec. 7) against `src/`,
`scripts/`, and `data/`, and reports PASS/FAIL.

## Repository structure

```
src/                      Core library: the model, the harmonic-balance
                            solver, and the empirical formula itself.
                            See src/README.md for the file-by-file role.
scripts/                   Experiment drivers, bisection searches, and
                            figure generation, built on src/. Mostly
                            long-running (tens of minutes); not part of
                            the automated verification suite. See
                            scripts/README.md.
data/                      Cached *.json results produced by scripts/
                            and read by verification/run_verification.py.
                            See data/README.md.
results/                   Reproducibility confirmation log
                            (default_analysis_log.txt). See
                            results/README.md.
verification/
  run_verification.py      Single-entry verification suite: re-checks
                            the paper's Sec. 5 / Sec. 7 numeric claims
                            (the T_U, P1, P2 identities; the b*(λ)
                            formula's accuracy; the energy-budget
                            closure; the coupling-removal test) against
                            src/, scripts/, and data/, and prints a
                            clear PASS/FAIL line for each.
  README.md                 What each check does and how to run them
requirements.txt            Dependency list (numpy, scipy; matplotlib
                            only for scripts/make_paper_figures.py)
```

## Quick start — verifying the paper's claims

```
pip install -r requirements.txt
python3 verification/run_verification.py              # all checks (~30s)
python3 verification/run_verification.py --skip-energy # fast checks only (<1s)
```

This reruns the identity checks, the empirical-formula accuracy check,
the energy-budget closure, and the coupling-removal test, and reports
PASS/FAIL against the exact tolerances stated in the paper. See
`verification/README.md` for what each check does, and
`src/README.md` / `scripts/README.md` for the role of every individual
file and how to reproduce every table in the paper from scratch.

## Environment

Python 3.11 (verified at deposit time: 3.11.15). See
[`requirements.txt`](requirements.txt) for the exact package list and
the versions verified at deposit time (NumPy 2.4.4, SciPy 1.17.1,
Matplotlib 3.10.9 — the last of these only needed for
`scripts/make_paper_figures.py`, which is not required to reproduce any
number quoted in the paper). A single CPU is sufficient.

Most scripts in `scripts/` accept `--b`, `--delta`, `--B3`, `--B4`,
`--B5`, `--lam-loss` command-line flags and/or an
`ES_SET="name=value,..."` environment variable to override individual
model coefficients (used for the coupling-removal test). See each
script's own header comment for its specific invocation, and
`scripts/README.md` for a full reproduction command list with
approximate run times.

## Companion paper repository

The manuscript itself (LaTeX source and compiled PDF) is kept in a
separate repository:

**`physica-a-oscillation-window-paper`** *(link to be added once both
repositories are published)*

## License

This code and verification suite are licensed under the **Apache
License, Version 2.0**. See [`LICENSE`](LICENSE) for the full text, or
https://www.apache.org/licenses/LICENSE-2.0 for a summary. Note that
this is a different license from the companion paper repository
(CC BY-SA 4.0), which is appropriate for the manuscript text rather
than for code.

## Citation

A `CITATION.cff` file is included for citing this code; author and DOI
metadata will be finalized once the Zenodo record is minted. For
citing the paper itself, use the `CITATION.cff` in the companion paper
repository.
