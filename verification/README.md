# Verification suite

`run_verification.py` is a single, self-contained script that re-checks
the quantitative claims made in Sec. 5 ("Error evaluation") and Sec. 7
("Verification") of the manuscript (`main.tex` in the companion paper
repository), against the cached data in `../data/` and the modules in
`../src/` and `../scripts/`. It introduces no new computation — it
re-evaluates the same closed-form identities and the same empirical
formula, and prints a clear PASS/FAIL line for each claim. Tolerances
are set tight, around the actually-measured values (not generous
margins), so a PASS is informative about the paper's stated digits
rather than something any plausible result would satisfy.

This is meant to be the one thing a reviewer, or anyone browsing the
GitHub/Zenodo deposit, can run to confirm the manuscript's numbers are
reproducible, without first having to read all the files in `../src/`
and `../scripts/`.

## What it checks

1. **Exact $T_U$, $P_1$, $P_2$ identities** (Sec. 6.3 Eq. 13, Sec. 7) —
   recomputes both sides of the closed-form expressions from
   `es_window_paper_data.json` and checks the relative/absolute
   differences against the measured ranges: $T_U$ relative difference
   in $[9.6\times10^{-9},\,7.3\times10^{-8}]$ across the seven fitted
   $\lambda$ values; $P_1,P_2$ absolute difference at most
   $8.3\times10^{-17}$ (two of the seven values agree exactly at
   double precision). Both the lower and upper end of the $T_U$ range
   are checked, not just the worst case.
2. **Empirical formula $b^*(\lambda)$, Eq. (9)** (Sec. 3.2, Sec. 5,
   Table 1) — imports `es_window_bstar_formula.py` directly and checks
   the formula against the bracket midpoints for both the 7 fit points
   and the 4 independent validation points, against the paper's stated
   $2.0\times10^{-4}$ / $1.8\times10^{-4}$ maxima. Also checks the
   $\lambda\to\infty$ limit (0.1214).
3. **Energy-budget closure** (Sec. 6.3, Sec. 7) — this is *two separate
   checks against two different claims* in the paper, which are easy to
   conflate but use different methods and have very different
   precision:
   - **3a. Across the seven harmonic-balance thresholds** (Sec. 7's
     "Energy-budget closure" paragraph) — reads the cached `total`
     field directly from `es_window_paper_data.json` for each of the
     seven fitted $\lambda$ values and checks the closure falls in
     $[6.8\times10^{-5},\,1.34\times10^{-3}]$, the paper's stated
     range. The residual grows monotonically with $\lambda$ and tracks
     the growth of the harmonic-balance waveform's second-harmonic
     content (negligible at $\lambda=1$, $\sim27\%$ of the fundamental
     at $\lambda=20$), so it is most likely harmonic-balance truncation
     error rather than the root-solve tolerance of the harmonic-balance
     equations — this attribution has not been independently confirmed
     (e.g. by tightening the root-solver and checking the residual is
     unchanged), so it is offered as the more likely explanation, not
     a settled one. Instant — reads cached data only.
     Note: `total` ($=b\cdot P_1+P_2+\text{rest}$) is the real,
     non-tautological closure quantity. A different combination,
     $b\cdot P_1+P_2-S_U$, is exactly zero *by construction* at
     machine precision for every $\lambda$ (since $b$ is defined as
     $(S_U-P_2)/P_1$) and must not be confused with `total` — the two
     are easy to conflate if "closure" isn't pinned to a specific
     formula.
   - **3b. At a representative window state** (Sec. 6.3's narrative
     point) — runs `es_window_energy.py` (directly re-integrates the
     full ODE system at $b=0.15,\delta=0.05,s=0.69,\lambda=2.25$,
     $\sim30$s) and checks the sum of pathway powers closes to within
     $10^{-9}$ of zero. The paper reports $\sim10^{-11}$; this check
     reads the script's `total` variable directly (via
     `importlib`) at full double precision, rather than regex-parsing
     its printed output (which is rounded to 5 decimal places and so
     could only confirm "rounds to zero," not the true magnitude —
     see "Known limitations," below, for why this used to be a
     problem). If `es_window_energy.py` calls `sys.exit(0)` at import
     time (it does this when no oscillation window exists at the given
     parameters), the suite catches that `SystemExit` explicitly and
     records check 3b as a FAIL, rather than letting it propagate out
     of `main()` and silently end the whole script with exit code 0 --
     an earlier version of this suite only caught `Exception`, which
     does *not* catch `SystemExit` (`SystemExit` is a `BaseException`),
     so a run that hit this case looked like a clean pass to a CI
     system or a reviewer checking only the exit code: check 4 and the
     final summary line were simply never printed, but the process
     still exited 0. Skip check 3b (and its `scipy`/`matplotlib`
     dependency, see "Dependencies" below) with `--skip-energy`.
4. **Coupling-removal (knockout) test** (Sec. 6.2, Sec. 7) — checks the
   cached results in `es_window_knockout2_results.json` against the
   necessary/not-necessary classification reported in the paper. $\chi$
   and the $\tau$-side regime term are checked at **all three** of
   $b=0.15,\,0.20,\,0.25$ (window absent at each), matching the paper's
   explicit "does not appear (up to $b=0.25$)" claim; $\gamma_{G\to r}$
   and all $\varepsilon$-related couplings are checked at the single
   $b$ value the paper discusses for each (window still present); and
   $\lambda=1$ is checked at $b=0.15$ (window still present). The
   lookup is an **exact** match on the `es_set` field (not a
   substring match), and the script raises an error rather than
   silently overwriting if two rows in the data file ever share an
   `es_set` value.

## Usage

```
python3 run_verification.py              # all checks (under 10s, dominated by check 3b)
python3 run_verification.py --skip-energy # checks 1, 2, 3a, 4 only (<1s)
```

Exit code is 0 if every check passes, 1 otherwise — suitable for use in
a CI workflow if one is added to this repository later.

## Known limitations

- **Check 2 only confirms the formula fits its own cached brackets, not
  that the brackets were correctly found.** The bracket values in
  `es_window_bstar_formula.py`'s `FIT_POINTS`/`VALIDATION_POINTS` (the
  bisection-search results $b^*(\lambda)\in(\text{lo},\text{hi}]$) are
  themselves the output of `es_window_lamc.py`, `es_window_bstar.py`,
  and `es_window_lam_large.py` in `../scripts/`. This suite does
  not re-run those scripts (each takes tens of minutes) and so does not
  independently verify the bracket search itself — only that Eq. (9)
  reproduces the cached bracket midpoints. Re-running those scripts
  from scratch is the way to verify the brackets themselves.
- **Checks 1 and 2 verify internal self-consistency of the cached data,
  not that the data-generating code itself is correct.** Check 1
  recomputes $T_U$ from the same cached $g$ values using the same
  formula that produced them; a bug shared between the two computations
  would not be caught. Check 2 is checking against cached bracket
  midpoints with the same caveat as above.
- **The three empirical-formula components ($S_U$, $P_2$, $P_1$)
  individually deviate from the cached harmonic-balance values by more
  than the combined formula's claimed accuracy** — up to
  $9.7\times10^{-4}$ ($S_U$), $6.7\times10^{-4}$ ($P_2$), and
  $4.2\times10^{-3}$ ($P_1$, the loosest of the three, though still
  under 2% relative) — while $b^*(\lambda)$ itself is accurate to
  $2.0\times10^{-4}$. This is possible because the three fitted
  functions enter Eq. (9) together and their errors partly cancel; this
  suite does not independently check each component's fit (only the
  combined formula, in check 2), and the paper does not claim
  component-level accuracy independent of the combined result.
- **The coupling removed by fixing $\gamma_w=0$ is not checked here**
  because the paper makes no claim about it (its classification in the
  cached data is "undetermined" / `present: null`, not a clean
  present/absent result). If a future revision of the paper adds a
  claim about this variant, a check should be added for it.
- **`check_identities`'s hardcoded `0.5`** is $\gamma_w$, the model's
  default damping-related parameter (`self.gamma_w = 0.5` in
  `../src/es_core_integrated.py`), used there as $\beta_w=\gamma_w/\omega$;
  this is noted in a code comment at the point of use.

## What this suite does not check

This suite confirms that the manuscript's *numbers are reproducible from
the cached data and the closed-form/empirical formulas*, at the stated
precision. It is not a from-scratch regeneration of the study, and does
not re-run (this is by design -- the omitted steps take from minutes to
tens of minutes each, some of them per data point):

- **Re-searching the bisection brackets.** Only the 11 brackets already
  cached in `es_window_bstar_formula.py` (`FIT_POINTS`/`VALIDATION_POINTS`)
  are checked (see "Known limitations" above); the other b*(lambda)
  values appearing only in Table 1 of the manuscript, and the bracket
  search itself, are not re-run here. (Spot-checking a few of these --
  the lambda=1 and lambda=2.25 endpoints, and the delta=0.01/0.2 points
  -- by re-running `es_window_lamc.py` is a reasonable manual check.)
- **The rest of Sec. 3's reference-cycle properties**: $s^*_\pm$,
  the hysteresis interval, $T_0$, the Floquet multipliers, and the
  320-run pulse-response grid (Sec. 3, Sec. 7) are not re-verified by
  this suite. They would need `src/es_shock_analysis_integrated.py`
  run directly (its own `__main__` block reproduces them).
- **Regenerating `es_window_paper_data.json` itself** (the harmonic-balance
  solutions underlying checks 1, 2, and 3a) from the harmonic-balance
  solver in `../src/es_window_hb.py` and the driver scripts in
  `../scripts/` is not done by this suite; it only reads the cached
  file.

If a reviewer needs those confirmed independently, the relevant script
under `../src/` or `../scripts/` should be run directly; each has its
own module docstring describing what it computes and roughly how long
it takes.

## Dependencies

NumPy and SciPy (already required by `../src/` and `../scripts/`), plus
**matplotlib**, which check 3b pulls in indirectly: it imports
`../scripts/es_window_energy.py`, which imports
`../src/es_shock_analysis_integrated.py`, which imports `matplotlib` at
module level (for that module's own, unrelated figure-generation code)
and also computes the model's reference limit cycle as a side effect of
being imported, which costs a few seconds (measured: under 10 seconds
total for the whole suite, including check 3b, on a typical machine).
`run_verification.py --skip-energy` skips check 3b and avoids both the
`matplotlib` dependency and that cost. No files are written; the script
only reads from `../src/`, `../scripts/`, and `../data/`, and prints to
stdout.
