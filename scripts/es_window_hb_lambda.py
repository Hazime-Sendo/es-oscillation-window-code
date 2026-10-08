"""
es_window_hb_lambda.py — δ=0.05 で、b*(λ) を調和バランスから予測して実測と比べる（1/λ 則を理論側から取りに行く）

各 λ で b を振り、max_A ρ(A)=0 となる b を予測 b*(λ) とする（es_window_hb_bstar.py と同じ手続き。s は各 (b,λ) の s*+ + 0.003）。
s*+ は、実測（es_window_lamc.py / es_window_bstar.py の評価ログ）から補間する。
実測 b*(λ) (δ=0.05): λ=1: 0.14703, 1.5: 0.13891, 2.25: 0.13338, 3: 0.13066, 4: 0.12828

使い方: python es_window_hb_lambda.py     （約 8〜10 分）
"""
import os, sys, re, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
from es_window_hb import rho_curve

# Optional raw logs from es_window_lamc.py runs, if present alongside this
# script (used only to interpolate s*+ more finely; falls back to the
# JSON results below and the hardcoded MEAS table if absent).
LOGS = [os.path.join(HERE, "lamc.log"), os.path.join(HERE, "lamc2.log")]
MEAS = {1.0: 0.14703, 1.5: 0.13891, 2.25: 0.13338, 3.0: 0.13066, 4.0: 0.12828}
A_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]


def scrit_table():
    tab = {}
    pat = re.compile(r"b=([\d.]+) δ=([\d.]+) λ=([\d.]+): s\*\+=([+-][\d.]+)")
    for f in LOGS:
        if os.path.exists(f):
            for m in pat.finditer(open(f, encoding="utf-8", errors="replace").read()):
                b, d, l, s = float(m[1]), float(m[2]), round(float(m[3]), 2), float(m[4])
                if abs(d - 0.05) < 1e-9: tab.setdefault(l, {})[round(b, 5)] = s
    base = json.load(open(os.path.join(DATA, "es_window_bstar_results.json")))["0.05"]["log"]
    for e in base:
        tab.setdefault(2.25, {})[round(e["b"], 5)] = e["s_crit"]
    return tab


def main():
    tab = scrit_table(); out = {}
    for lam in (1.0, 1.5, 2.25, 3.0, 4.0):
        pts = tab[round(lam, 2)] if round(lam, 2) in tab else tab[lam]
        bs = np.array(sorted(pts)); ss = np.array([pts[b] for b in bs]); coef = np.polyfit(bs, ss, 1)
        sc = lambda b: float(np.polyval(coef, b))
        bm = MEAS[lam]; grid = [round(bm + dx, 5) for dx in (-0.006, -0.003, 0.0, 0.003, 0.006)]
        rows = []
        print(f"\n===== λ={lam:g}（実測 b*={bm}; s*+ の点 {len(bs)} 個から線形補間）=====", flush=True)
        for b in grid:
            t0 = time.time(); s = sc(b) + 0.003
            r = rho_curve(b, 0.05, s, lam=lam, A_list=A_LIST); cv = [d for d in r["curve"] if d["ok"] and d["A"] >= 0.2]
            im = int(np.argmax([d["rho"] for d in cv])); rows.append(dict(b=b, s=s, rho_max=cv[im]["rho"], A_at_max=cv[im]["A"]))
            print(f"   b={b:.5f} s={s:.4f}: max ρ = {cv[im]['rho']:+.5f} (A={cv[im]['A']:.2f})  [{time.time() - t0:.0f}s]", flush=True)
        bpred = None
        for i in range(len(rows) - 1):
            r0, r1 = rows[i]["rho_max"], rows[i + 1]["rho_max"]
            if r0 < 0 <= r1:
                bpred = rows[i]["b"] + (rows[i + 1]["b"] - rows[i]["b"]) * (-r0) / (r1 - r0); break
        out[f"{lam:g}"] = dict(rows=rows, b_pred=bpred, b_meas=bm)
        print(f"  → 予測 b* = {bpred}   実測 b* = {bm}", flush=True)
        json.dump(out, open(os.path.join(DATA, "es_window_hb_lambda_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
