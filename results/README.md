# `results/` — reproducibility confirmation logs

| File | Contents |
|---|---|
| `default_analysis_log.txt` | Raw console log of a full default-parameter run of `../src/es_shock_analysis_integrated.py`. The numbers quoted in the manuscript's Sec. 2 and Sec. 3.1 (reference-cycle period $T_0$, switching-cycle multiples, $s^*_\pm$) are read from this log; `../scripts/make_paper_figures.py` also parses it (via the `ES_DEFAULT_LOG` environment variable, defaulting to this file) to label Fig. 1. |

This folder is for human-readable confirmation that a full run actually
happened and produced the numbers quoted in the paper — it is not
itself consumed by `../verification/run_verification.py`, which checks
the manuscript's claims directly against `../data/` and `../src/`.
