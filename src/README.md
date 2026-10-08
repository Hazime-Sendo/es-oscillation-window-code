# `src/` — core library modules

These are the reusable building blocks that every script in `../scripts/`
and `../verification/run_verification.py` imports from, or invokes as a
subprocess. None of them are run directly to produce a result on their
own (except `es_window_eval.py`, which is a single-point CLI oracle) —
they define the model, the solvers, and the empirical formula itself.

| File | Role (manuscript section) |
|---|---|
| `es_core_integrated.py` | the model, Eqs. (1)–(8) and default parameters (Sec. 2) |
| `es_shock_analysis_integrated.py` | reference cycle, Floquet multipliers, pulse response, sustained input $s^*_\pm$ (Sec. 2, Sec. 3.1). Options: `--b --delta --B3 --B4 --B5 --lam-loss`; env `ES_SET="name=value,..."` overrides individual coefficients |
| `es_window_eval.py` | window criterion at one $(b,\delta,\lambda)$ point (Sec. 4); the core oracle every bisection search in `../scripts/` calls via subprocess |
| `es_window_hb.py` | first-harmonic-balance solver, effective growth rate $\rho(A)$ (Sec. 6.3) |
| `es_window_bstar_formula.py` | **the empirical formula, Eq. (9) of the manuscript**, with the fit/validation bracket data and a self-test reproducing Table 1 |

## Reproduction

```
python3 es_window_bstar_formula.py          # prints Table 1 and the λ→∞ limit 0.1214       (< 1 s)
python3 es_shock_analysis_integrated.py     # Sec. 2 / 3.1 numbers, writes a log like
                                             # ../results/default_analysis_log.txt            (~3.5 min)
```

## Notes for readers

All coefficients in the empirical formula, Eq. (9) of the manuscript,
are least-squares fits to harmonic-balance quantities evaluated at
directly simulated thresholds (Sec. 3.2, Sec. 6.5); they are not
analytically derived, and this is stated explicitly in both the
manuscript and in the header comment of `es_window_bstar_formula.py`.
The bracket data hard-coded in its self-test are the (window absent,
window present] values from the direct bisection search described in
Sec. 4 (produced by `../scripts/es_window_bstar.py`,
`es_window_lamc.py`, and `es_window_lam_large.py`); they match Table 1
of the manuscript exactly.
