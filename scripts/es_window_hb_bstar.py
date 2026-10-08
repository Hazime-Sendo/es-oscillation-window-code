"""
es_window_hb_bstar.py — 調和バランスの実効成長率 ρ(A) から、窓が現れる b*(δ) を予測する

窓（正側の持続入力での切替なしの持続振動）は、ρ(A)（es_window_hb.py）が A>0 で正の山を持つとき存在する。
折り返し（b*）は max_A ρ(A) = 0。各 δ で b を振って max ρ を求め、ゼロを横切る b を線形補間して b_pred* とする。
s は各 (b,δ) の s*+（切替サイクルが消える s。es_window_bstar_results.json の実測点から補間）+ 0.003 に置く。
実測 b*(δ) は 0.1178 / 0.1335 / 0.1616（δ=0.01 / 0.05 / 0.2）。

使い方: python es_window_hb_bstar.py     （約 10 分）
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
from es_window_hb import rho_curve

BASE = json.load(open(os.path.join(DATA, "es_window_bstar_results.json")))
GRID = {0.01: [0.105, 0.110, 0.114, 0.1178, 0.122, 0.128],
        0.05: [0.120, 0.126, 0.130, 0.1335, 0.138, 0.145],
        0.2:  [0.145, 0.152, 0.158, 0.1616, 0.166, 0.172]}
A_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
MEASURED = {0.01: 0.1178, 0.05: 0.1335, 0.2: 0.1616}


def s_crit_interp(delta):
    pts = {}
    for e in BASE[f"{delta:g}"]["log"]:
        if e.get("s_crit") == e.get("s_crit") and e.get("s_crit") is not None:
            pts[round(e["b"], 5)] = e["s_crit"]
    bs = np.array(sorted(pts)); ss = np.array([pts[b] for b in bs])
    A = np.polyfit(bs, ss, 1) if len(bs) >= 2 else None
    return lambda b: float(np.interp(b, bs, ss)) if bs.min() <= b <= bs.max() else float(np.polyval(A, b))


def run():
    res = json.load(open(os.path.join(DATA, "es_window_hb_bstar_results.json"))) if os.path.exists(os.path.join(DATA, "es_window_hb_bstar_results.json")) else {}
    for delta, blist in GRID.items():
        sc = s_crit_interp(delta); key = f"{delta:g}"
        rows = []
        print(f"\n===== δ={delta:g}（実測 b*={MEASURED[delta]}）=====", flush=True)
        for b in blist:
            t0 = time.time(); s = sc(b) + 0.003
            out = rho_curve(b, delta, s, A_list=A_LIST)
            cv = [d for d in out["curve"] if d["ok"] and d["A"] >= 0.2]
            if not cv:
                print(f"   b={b}: 収束せず", flush=True); continue
            imax = int(np.argmax([d["rho"] for d in cv])); rmax = cv[imax]["rho"]; Amax = cv[imax]["A"]
            lam_re = out["lam0"].real
            rows.append(dict(b=b, s=s, rho_max=rmax, A_at_max=Amax, re_lambda=lam_re))
            print(f"   b={b:.4f} s={s:.4f}: max ρ = {rmax:+.5f} (A={Amax:.2f}),  Re λ(ロック状態)={lam_re:+.4f}   [{time.time() - t0:.0f}s]", flush=True)
        # ゼロ交差
        bpred = None
        for i in range(len(rows) - 1):
            r0, r1 = rows[i]["rho_max"], rows[i + 1]["rho_max"]
            if r0 < 0 <= r1:
                bpred = rows[i]["b"] + (rows[i + 1]["b"] - rows[i]["b"]) * (-r0) / (r1 - r0); break
        res[key] = dict(rows=rows, b_pred=bpred, b_meas=MEASURED[delta])
        print(f"  → 予測 b* = {bpred}   実測 b* = {MEASURED[delta]}", flush=True)
        json.dump(res, open(os.path.join(DATA, "es_window_hb_bstar_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    run()
