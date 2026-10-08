# `data/` — cached numerical results

These JSON files are the cached output of the long-running searches in
`../scripts/` (tens of minutes to run from scratch; see
`../scripts/README.md`). They are what the manuscript's tables and
figure descriptions draw on, and what `../verification/run_verification.py`
reads directly for its fast checks (1, 2, 2b, 3a, 4).

| File | Produced by |
|---|---|
| `es_window_bstar_results.json`, `es_window_bstar_results_B4_-0.5.json`, `es_window_bstar_results_B4_0.5.json` | `../scripts/es_window_bstar.py` |
| `es_window_bstar_validation.json` | the 4 independent validation points (Table 1) |
| `es_window_lamc_results.json` | `../scripts/es_window_lamc.py` |
| `es_window_lam_large_results.json` | `../scripts/es_window_lam_large.py` |
| `es_window_hb_bstar_results.json` | `../scripts/es_window_hb_bstar.py` |
| `es_window_hb_lambda_results.json` | `../scripts/es_window_hb_lambda.py` |
| `es_window_n1_exact_results.json` | `../scripts/es_window_n1_exact.py` |
| `es_window_paper_data.json` | `../scripts/es_window_fig_data.py`; read by `../scripts/es_window_verify_structure.py` and by checks 1 and 3a of `../verification/run_verification.py` |
| `es_window_knockout2_results.json` | `../scripts/es_window_knockout2.py`; read by check 4 of `../verification/run_verification.py` |

None of these files are hand-edited; each is regenerated in place by
re-running its producing script.
