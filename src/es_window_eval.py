"""
es_window_eval.py — 統合版 ESCore の「切替なしの持続振動の窓」を 1 点 (b, δ) で評価する

窓 = 正側の持続入力 s で、レジーム切替サイクルが消えた直後に現れる、収縮レジーム内の切替のない持続振動。
手順（es_shock_analysis_integrated.py と同じ物差し）
  1. 正側の臨界 s*+（切替サイクルが消える s）を find_critical で求める（許容 tol）。
  2. s*+ + Δ（Δ の格子）で 100 周期積分し、最後の 20 周期窓の r の幅・切替数を見る。
       切替数 0 かつ 幅 > 0.02 かつ 最後の 2 窓の比 > 0.9   → 「持続振動」（窓の内側）
       それ以外で幅 < 0.02                                   → 「定常点」（減衰）
  3. 窓の下端は s*+（分解能 Δ の最小値）、上端は「持続する最大の Δ」と「次の減衰する Δ」の区間。

使い方: python es_window_eval.py <b> <delta> [tol_crit] [Δ1,Δ2,...] [--B4 値] [--lam 値]   出力: JSON を 1 行
        --lam は損失回避係数 λ（既定 2.25）。
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
_argv = list(sys.argv)
B4 = "0"
if "--B4" in _argv:                                   # --B4 値 を位置引数から取り除く
    i = _argv.index("--B4"); B4 = _argv[i + 1]; del _argv[i:i + 2]
LAM = "2.25"
if "--lam" in _argv:                                  # --lam 値 を位置引数から取り除く
    i = _argv.index("--lam"); LAM = _argv[i + 1]; del _argv[i:i + 2]
sys.argv = _argv
b = sys.argv[1]; delta = sys.argv[2]
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0005
DELTAS = ([float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else
          [0.002, 0.004, 0.008, 0.015, 0.03, 0.05, 0.08, 0.12, 0.17, 0.24, 0.35, 0.5])
sys.argv = ["x", "--b", b, "--delta", delta, "--B4", B4, "--lam-loss", LAM]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import es_shock_analysis_integrated as E
from scipy.integrate import solve_ivp

NP = 100


def classify(s_):
    T = NP * E.T0
    sol = solve_ivp(E.make_rhs(E.p, s_), (0, T), E.GAM[int(0.3 * E.N)], method="LSODA", rtol=1e-9, atol=1e-11,
                    max_step=0.5, t_eval=np.arange(0, T, 0.5))
    if not np.isfinite(sol.y).all():
        return "逃走", float("nan")
    w = []
    for k in range(NP // 20):
        m = (sol.t >= k * 20 * E.T0) & (sol.t < (k + 1) * 20 * E.T0)
        r = sol.y[1][m]; w.append(float(r.max() - r.min()))
    last = sol.t >= (NP - 20) * E.T0
    nsw = int((np.diff(E.regime_series(sol.y[1][last], sol.y[5][last])) != 0).sum())
    if nsw > 0:
        return "切替サイクル", w[-1]
    if w[-1] > 0.02 and w[-1] / max(w[-2], 1e-300) > 0.9:
        return "持続振動", w[-1]
    return ("減衰中" if w[-1] > 0.02 else "定常点"), w[-1]


sc, kind = E.find_critical(+1, TOL)
out = dict(b=float(b), delta=float(delta), B4=float(B4), lam=float(LAM), T0=float(E.T0), s_crit=float(sc), kind_after=kind)
if sc == sc:
    rows = [(d,) + classify(sc + d) for d in DELTAS]
    pers = [(d, a) for d, k, a in rows if k == "持続振動"]
    out["present"] = bool(pers)
    out["scan"] = [(d, k, a) for d, k, a in rows]
    if pers:
        last = max(d for d, a in pers)
        nxt = [d for d, k, a in rows if d > last and k != "持続振動"]
        out.update(amp_first=pers[0][1], amp_at_edge=pers[-1][1], width_lo=last, width_hi=(min(nxt) if nxt else None))
else:
    out["present"] = None
print(json.dumps(out, ensure_ascii=False))
