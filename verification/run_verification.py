#!/usr/bin/env python3
"""
run_verification.py — single-entry verification suite for the claims made in
Sec. 5 ("Error evaluation") and Sec. 7 ("Verification") of the manuscript,
which lives in the companion paper repository (main.tex there).

This script does not introduce any new computation: it re-evaluates the same
closed-form identities and the same empirical formula that the scripts in
../src/ and ../scripts/ already compute, using the same cached data files in
../data/, and checks the results against the numeric tolerances actually
stated in the paper. It is meant to be the one thing a reviewer or a
GitHub/Zenodo visitor runs to confirm the manuscript's numbers are
reproducible, without having to read every script in ../src/ and ../scripts/
first.

Checks performed (each prints PASS/FAIL and the measured value):
  1. Exact T_U, P1, P2 identities (Sec. 6.3, Eq. 13; Sec. 7)
       -- claimed: 9.6e-9 to 7.3e-8 relative, for T_U; at most 8.3e-17
          absolute, for P1 and P2 (two of the seven points agree
          exactly). Tolerances are tight (within ~2x of the claimed
          worst case), not generous margins, so PASS here is actually
          informative about the paper's claimed digits.
  2. Empirical formula for b*(lambda), Eq. (9) (Sec. 3.2, Sec. 5, Table 1)
       -- claimed: max 2.0e-4 on the 7 fit points, max 1.8e-4 on the 4
          independent validation points
  3a. Energy-budget closure across the seven harmonic-balance thresholds
      (Sec. 7's "Energy-budget closure" paragraph)
       -- claimed: closes to within 1.34e-3 (ranging 6.8e-5 to 1.34e-3
          across the seven fitted lambda values), most likely reflecting
          truncation of the harmonic-balance solution at the first
          harmonic (the residual tracks the growing second-harmonic
          content, not the root-solve tolerance -- see the paper's
          corrected Sec. 7 text). A coarser check than 3b below. Reads
          the cached "total" field directly from es_window_paper_data.json;
          instant. Checks both the upper bound (max) and the lower bound
          (min) of the claimed range.
  3b. Energy-budget closure at a representative window state (Sec. 6.3)
       -- claimed: closes to ~1e-11, i.e. essentially floating-point
          precision (b=0.15, delta=0.05, s=0.69, lambda=2.25), obtained
          by directly integrating the full ODE system -- a different,
          much tighter check than 3a, using a different method. Imports
          ../scripts/es_window_energy.py directly and reads its "total"
          variable at full double precision (earlier versions of this
          suite parsed the script's printed output, which is rounded to
          5 decimal places and could only confirm "rounds to zero", not
          the claimed ~1e-11 figure). Takes under 10 seconds on a typical
          machine (dominated by computing the model's reference limit
          cycle at import time); skip it with --skip-energy for a fast
          run of checks 1, 2, 3a, and 4 only.
  4. Coupling-removal (knockout) test (Sec. 6.2, Sec. 7)
       -- checks that the cached results in
          ../data/es_window_knockout2_results.json agree with the
          necessary/not-necessary classification reported in the paper,
          including that the window stays absent through b=0.25 (not
          just b=0.15) for the two couplings reported as necessary.

Usage:
  python3 run_verification.py                # runs all checks
  python3 run_verification.py --skip-energy   # skips the ODE check (3b), <1s

Exit code is 0 if every check passes, 1 otherwise.

Known limitations of this suite (see also verification/README.md):
  - Check 2 confirms Eq. (9) reproduces the bracket midpoints cached in
    es_window_bstar_formula.py's FIT_POINTS/VALIDATION_POINTS. It does
    NOT re-run the bisection searches (es_window_lamc.py,
    es_window_bstar.py, es_window_lam_large.py) that originally
    produced those brackets -- those scripts are included in
    ../scripts/ but take tens of minutes each and are not part
    of this automated suite. Re-running them is the way to verify the
    brackets themselves, not just the formula's fit to them.
  - Check 4 omits one coupling-removal variant present in the cached
    data ("gamma_w=0", fixing w): its result was inconclusive (window
    presence could not be classified either way) and the paper makes no
    claim about it, so there is nothing to check it against.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "scripts"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))

PASS, FAIL = "PASS", "FAIL"
results = []  # (name, status, detail)


def record(name, ok, detail):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}: {detail}")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Check 1: exact T_U, P1, P2 identities (Sec. 6.3 Eq. 13 / Sec. 7)
# ----------------------------------------------------------------------
def check_identities():
    data_path = os.path.join(DATA, "es_window_paper_data.json")
    D = json.load(open(data_path))
    chi, xi, kG, gG, gam = 0.8, 0.05, 0.4, -0.2, 1.1  # gamma = alpha_tau - B3 = 0.8 - (-0.3)
    # beta_w = gamma_w / omega (paper's Eq. for T_U = g/(1-i*beta_w*g)); gamma_w = 0.5
    # is the model's default (src/es_core_integrated.py), not an arbitrary constant.
    beta_w_num = 0.5
    diffs_TU, diffs_P1, diffs_P2 = [], [], []
    for q in D:
        g = q["g_hb"] * np.exp(1j * q["g_hb_arg"])
        T = q["T_U_re"] + 1j * q["T_U_im"]
        om = q["omega"]
        Tm = g / (1 - 1j * (beta_w_num / om) * g)
        Ht = 1 / (gam + 1j * om)
        Hg = 1 / (xi + 1j * om)
        l2 = q["ell2"]
        P1f = 2 * chi * np.real(T * Ht) / (1 + l2)
        P2f = 2 * gG * np.real(-kG * chi * T * Ht * Hg) / (1 + l2)
        diffs_TU.append(abs(T - Tm) / abs(T))
        diffs_P1.append(abs(P1f - q["P1"]))
        diffs_P2.append(abs(P2f - q["P2"]))
    worst_TU, best_TU = max(diffs_TU), min(diffs_TU)
    worst_P1, worst_P2 = max(diffs_P1), max(diffs_P2)
    # tight tolerances (within ~2x of the claimed worst case), not generous margins:
    # a PASS here actually confirms the paper's claimed digits, not just "same order".
    ok = (best_TU >= 5e-9 and worst_TU <= 1.5e-7 and worst_P1 <= 2e-16 and worst_P2 <= 2e-16)
    record(
        "1. T_U / P1 / P2 exact identities",
        ok,
        f"rel. diff T_U range {best_TU:.1e} to {worst_TU:.1e} (claimed 9.6e-9 to 7.3e-8), "
        f"max abs diff P1={worst_P1:.1e}, P2={worst_P2:.1e} (claimed: at most 8.3e-17, two "
        f"of the seven points exact)",
    )


# ----------------------------------------------------------------------
# Check 2: empirical formula for b*(lambda), Eq. (9) / Table 1
# ----------------------------------------------------------------------
def check_formula():
    mod = load_module("es_window_bstar_formula", os.path.join(SRC, "es_window_bstar_formula.py"))
    worst_fit = 0.0
    for lam, (lo, hi) in mod.FIT_POINTS.items():
        mid = 0.5 * (lo + hi)
        worst_fit = max(worst_fit, abs(mod.b_star(lam) - mid))
    worst_val = 0.0
    for lam, (lo, hi) in mod.VALIDATION_POINTS.items():
        mid = 0.5 * (lo + hi)
        worst_val = max(worst_val, abs(mod.b_star(lam) - mid))
    # paper states max 2.0e-4 (fit) / 1.8e-4 (validation); allow a small margin
    ok = worst_fit <= 2.1e-4 and worst_val <= 1.9e-4
    record(
        "2. Empirical formula b*(lambda), Eq. (9)",
        ok,
        f"max |formula - bracket midpoint|: fit points={worst_fit:.5f} (paper: 0.00020), "
        f"validation points={worst_val:.5f} (paper: 0.00018)",
    )
    limit = (mod.S0 - mod.P2_0) / mod.P1_0
    ok_limit = abs(limit - 0.1214) < 1e-3
    record(
        "2b. lambda -> infinity limit",
        ok_limit,
        f"(S0-P2_0)/P1_0 = {limit:.4f} (paper: 0.1214, numerical extrapolation range 0.1211-0.1218)",
    )


# ----------------------------------------------------------------------
# Check 3a: energy-budget closure across the 7 harmonic-balance thresholds
# (Sec. 7's "Energy-budget closure" paragraph) -- fast, reads cached JSON
# ----------------------------------------------------------------------
def check_energy_budget_hb():
    data_path = os.path.join(DATA, "es_window_paper_data.json")
    D = json.load(open(data_path))
    vals = [abs(q["total"]) for q in D]
    worst, best = max(vals), min(vals)
    # paper states the range is 6.8e-5 to 1.34e-3 across the seven lambda values;
    # check both ends, with a tight tolerance rather than a generous margin.
    ok = (5e-5 <= best <= 8e-5) and (1.2e-3 <= worst <= 1.4e-3)
    record(
        "3a. Energy-budget closure across 7 HB thresholds",
        ok,
        f"range of |total| = {best:.2e} to {worst:.2e} (paper: 6.8e-5 to 1.34e-3; most "
        f"likely harmonic-balance truncation error -- the residual tracks the growing "
        f"second-harmonic content with lambda, not the root-solve tolerance)",
    )


# ----------------------------------------------------------------------
# Check 3b: energy-budget closure at a representative window state
# (Sec. 6.3) -- slow, directly integrates the ODE system
# ----------------------------------------------------------------------
def check_energy_budget_window_state():
    script = os.path.join(SCRIPTS, "es_window_energy.py")
    # Import the script as a module (instead of shelling out and parsing its printed
    # output) so we can read its "total" variable at full double precision. The
    # script's own print statements round to 5 decimal places, which could only
    # confirm "rounds to zero", not the claimed ~1e-11 figure -- this is the fix for
    # that limitation, not just a documentation note about it.
    saved_argv = sys.argv
    try:
        sys.argv = ["es_window_energy.py", "0.15", "0.05", "0.69", "2.25"]
        mod = load_module("es_window_energy", script)
        closure = float(mod.total)
    except SystemExit as e:
        # es_window_energy.py calls sys.exit(0) at import time if no oscillation
        # window exists for the given (b, delta, s, lambda) -- e.g. if the script or
        # these parameters are ever changed. SystemExit is a BaseException, not an
        # Exception, so a plain "except Exception" does NOT catch it: it would
        # propagate straight out of main() and exit the whole process with code 0
        # (its own "success" code), silently skipping check 4 and the final
        # PASS/FAIL summary while still looking like a clean run to a CI system or a
        # reviewer. Catch it explicitly and record it as a FAIL instead.
        record(
            "3b. Energy-budget closure at window state",
            False,
            f"es_window_energy.py exited (sys.exit({e.code})) while being imported, instead of "
            f"computing a closure -- most likely no oscillation window exists at these parameters "
            f"(b=0.15, delta=0.05, s=0.69, lambda=2.25); treated as FAIL rather than allowed to "
            f"silently end the whole script",
        )
        return
    except Exception as e:
        record("3b. Energy-budget closure at window state", False, f"could not import es_window_energy.py: {e}")
        return
    finally:
        sys.argv = saved_argv
    ok = abs(closure) <= 1e-9  # paper claims ~1e-11; tight but allows platform/solver variation
    record(
        "3b. Energy-budget closure at window state",
        ok,
        f"sum of pathway powers = {closure:+.3e} at full precision (paper: closes to ~1e-11; "
        f"b=0.15, delta=0.05, s=0.69, lambda=2.25)",
    )


# ----------------------------------------------------------------------
# Check 4: coupling-removal (knockout) test, qualitative (Sec. 6.2 / 7)
# ----------------------------------------------------------------------
def check_knockout():
    path = os.path.join(DATA, "es_window_knockout2_results.json")
    D = json.load(open(path))

    # by_esset keys on the "es_set" field by EXACT match (not substring -- an
    # earlier version of this comment said "substring", which was never what the
    # code did: by_esset.get(es_set) is a plain dict lookup). Guard against two
    # rows silently sharing an es_set, which a dict comprehension would resolve by
    # keeping only the last one without any warning.
    by_esset = {}
    for label, v in D.items():
        es = v["es_set"]
        if es in by_esset:
            raise RuntimeError(
                f"duplicate es_set {es!r} in {path} (rows {by_esset[es][0]!r} and {label!r}); "
                f"cannot verify check 4 safely until the data file is de-duplicated"
            )
        by_esset[es] = (label, v)

    # (es_set, [(b value, expected window presence), ...], claim label)
    # chi=0 and kappa_tau=0 are checked at b=0.15, 0.20, AND 0.25, matching the
    # paper's explicit "does not appear (up to b=0.25)" claim (Sec. 6.2). The other
    # "necessary"/"not necessary" rows are checked only at b=0.15 (or b=0.20 for
    # g_to_r=0, which the paper says only *raises* the threshold rather than
    # eliminating the window), since the paper makes no "up to" claim for them.
    expected = [
        ("", [("0.15", True)], "baseline (no coupling removed): window present"),
        ("lam_loss=1", [("0.15", True)], "lambda=1 (symmetric U, no loss aversion): window still present"),
        ("chi=0", [("0.15", False), ("0.2", False), ("0.25", False)],
         "chi (U -> tau) removed: window absent through b=0.25 -- necessary coupling"),
        ("kappa_tau=0", [("0.15", False), ("0.2", False), ("0.25", False)],
         "tau-side regime term removed: window absent through b=0.25 -- necessary coupling"),
        ("g_to_r=0", [("0.2", True)], "G -> r removed: window still present at b=0.20 (not necessary, threshold rises)"),
        ("eps_to_r=0", [("0.15", True)], "epsilon -> r removed: window still present"),
        ("B5=-1", [("0.15", True)], "epsilon -> tau removed: window still present"),
        ("gamma_eps_Lambda=0", [("0.15", True)], "epsilon -> Lambda removed: window still present"),
        ("gamma_eps_tau=0", [("0.15", True)], "tau -> epsilon removed: window still present"),
    ]
    all_ok = True
    details = []
    n_checks = 0
    for es_set, b_checks, label in expected:
        found = by_esset.get(es_set)
        if found is None:
            all_ok = False
            details.append(f"  [MISSING] es_set={es_set!r} not found in results file")
            continue
        _, row = found
        for b_key, want_present in b_checks:
            n_checks += 1
            got_present = row["results"].get(b_key, {}).get("present")
            ok = got_present == want_present
            all_ok = all_ok and ok
            details.append(f"  [{PASS if ok else FAIL}] {label} (window present at b={b_key}: {got_present})")
    record("4. Coupling-removal (knockout) test", all_ok, f"{n_checks} checks across {len(expected)} rows, see detail below")
    print("\n".join(details))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-energy", action="store_true",
                     help="skip the energy-budget ODE re-integration (check 3b, under 10s)")
    args = ap.parse_args()

    print("Verification suite for:")
    print("  Onset of a finite-amplitude oscillation window at the disappearance of a")
    print("  regime-switching cycle in a seven-variable economic model")
    print("  (checks Sec. 5 and Sec. 7 claims against ../src/, ../scripts/, ../data/)\n")

    check_identities()
    check_formula()
    check_energy_budget_hb()
    if not args.skip_energy:
        check_energy_budget_window_state()
    else:
        print("[SKIP] 3b. Energy-budget closure at window state (--skip-energy)")
    check_knockout()

    print()
    n_fail = sum(1 for _, status, _ in results if status == FAIL)
    if n_fail == 0:
        print(f"ALL {len(results)} CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"{n_fail} OF {len(results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
