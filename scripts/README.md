# `scripts/` — experiment drivers, searches, and figure generation

These scripts produce the manuscript's numbers and (optionally) its
figures. They import the core modules from `../src/` and read/write
the cached result files in `../data/`. Most of them are long-running
bisection searches (tens of minutes each) and are **not** part of the
automated `../verification/run_verification.py` suite — they are
included so the underlying data in `../data/` can be reproduced from
scratch if needed. See `../verification/README.md` for what *is*
checked automatically.

| File | Role (manuscript section) |
|---|---|
| `es_window_bstar.py` | onset threshold $b^*$ versus $\delta$, and versus $\delta$ at fixed $B_4$, by bisection (Sec. 6.1) → `../data/es_window_bstar_results*.json` |
| `es_window_lamc.py`, `es_window_lam_large.py` | $b^*(\lambda)$ by bisection at $\delta=0.05$, extended to large $\lambda$ (Table 1, Sec. 5–6) → `../data/es_window_lamc_results.json`, `es_window_lam_large_results.json` |
| `es_window_hb_bstar.py`, `es_window_hb_lambda.py` | harmonic-balance predictions of $b^*(\delta)$, $b^*(\lambda)$ |
| `es_window_hb_scans.py` | reproducibility scans referenced in Sec. 6.3 |
| `es_window_energy.py` | exact energy-budget / pathway-power decomposition, small-amplitude limit vs. window state (Sec. 6.3); this is the script `run_verification.py`'s check 3b shells out to |
| `es_window_n1_exact.py`, `es_window_fig_data.py` | exact vs. power-law gain $g(\lambda)$; $U$-pathway supply decomposition $S_U, P_2, P_1$ (Sec. 6.4–6.5) → `../data/es_window_paper_data.json` |
| `es_window_verify_structure.py` | checks the exact $T_U$, $P_1$, $P_2$ identities, Eq. (13), against `../data/es_window_paper_data.json` (Sec. 7) |
| `es_window_knockout2.py` | coupling-removal (knockout) test (Sec. 6.2, Sec. 7) → `../data/es_window_knockout2_results.json` |
| `make_paper_figures.py` | figure-generation script, retained for reference; not run for this deposit (no images included — see `../../physica_a_paper/figures_tables_plan.md` in the companion paper repository) |

## Reproduction (approximate run times, one CPU; run from inside `scripts/`)

```
python3 es_window_verify_structure.py             # Sec. 7: T_U ~1e-8, P1/P2 ~1e-15 to 1e-17       (< 1 s)
python3 es_window_hb_scans.py                     # Sec. 6.3 scans                                  (~4 min)
python3 es_window_energy.py 0.15 0.05 0.69 2.25   # Sec. 6.3 energy-budget numbers                  (~30 s)
python3 ../src/es_window_eval.py 0.134 0.05       # window present? (b=0.133: no, b=0.134: yes)     (~40 s per point)
python3 es_window_lamc.py                         # b*(λ) bisection search, δ=0.05                  (~40 min)
python3 es_window_bstar.py                        # b*(δ) bisection search                          (~25 min)
```

The bisection searches call `../src/es_window_eval.py` in subprocesses,
one per $(b,\delta,\lambda)$ trial point, which is why they are slow
relative to the other scripts.

## Notes for readers

Only the files listed here and in `../src/README.md` were used to
produce the numbers in the manuscript. Any other script that may
appear in a working copy of this project (earlier exploratory
variants, one-off checks) was not used and is not included here.
