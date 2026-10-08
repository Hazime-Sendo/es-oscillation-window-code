"""
es_window_lam_large.py — 大きな λ（10, 20）で b*(λ) を実測し、調和バランスでも予測して、2 つの候補の形を判別する

  経験則      b* = b_∞ + c/λ            （λ=1〜4 の実測 5 点: b_∞=0.12226, c=0.02485）
  理論側の形  b* = b_∞' + c'/N1(λ)      （U の描像関数 N1(λ): ⟨U⟩=0 を課した基本波ゲイン; 5 点: b_∞'=0.10189, c'=0.04606）
δ=0.05。λ=10 と 20 で 2 つの形の予測が 0.001〜0.002 分かれる。
手順: (1) es_window_lamc.b_star で b*(λ) を直接測る（評価ログから s*+ も得る） → (2) その s*+ で調和バランス max ρ=0 の b を予測する。
使い方: python es_window_lam_large.py   （約 20 分）
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
import es_window_lamc as LC
from es_window_hb import rho_curve

PRED = {10.0: dict(inv_lambda=0.12226 + 0.02485 / 10.0, inv_N1=0.10189 + 0.04606 / 2.1561, start=0.124),
        20.0: dict(inv_lambda=0.12226 + 0.02485 / 20.0, inv_N1=0.10189 + 0.04606 / 2.3819, start=0.122)}
A_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]


def main():
    out = {}
    for lam in (10.0, 20.0):
        t0 = time.time(); pr = PRED[lam]
        print(f"\n===== λ={lam:g}: 予測 b*  1/λ 則 = {pr['inv_lambda']:.5f},  1/N1 形 = {pr['inv_N1']:.5f} =====", flush=True)
        lo, hi, log = LC.b_star(0.05, lam, pr["start"], step=0.004)
        pts = {}
        for e in log:
            if e.get("s_crit") == e.get("s_crit") and e.get("s_crit") is not None:
                pts[round(e["b"], 5)] = e["s_crit"]
        bm = 0.5 * (lo + hi) if lo is not None else None
        print(f"  実測 b*(λ={lam:g}) ∈ ({lo}, {hi}]   （{time.time() - t0:.0f}s）", flush=True)
        bs = np.array(sorted(pts)); ss = np.array([pts[b] for b in bs]); coef = np.polyfit(bs, ss, 1)
        rows = []
        for dx in (-0.006, -0.003, 0.0, 0.003, 0.006):
            b = round(bm + dx, 5); s = float(np.polyval(coef, b)) + 0.003
            r = rho_curve(b, 0.05, s, lam=lam, A_list=A_LIST); cv = [d for d in r["curve"] if d["ok"] and d["A"] >= 0.2]
            im = int(np.argmax([d["rho"] for d in cv])); rows.append(dict(b=b, s=s, rho_max=cv[im]["rho"], A_at_max=cv[im]["A"]))
            print(f"   HB: b={b:.5f} s={s:.4f}: max ρ = {cv[im]['rho']:+.5f} (A={cv[im]['A']:.2f})", flush=True)
        bpred = None
        for i in range(len(rows) - 1):
            r0, r1 = rows[i]["rho_max"], rows[i + 1]["rho_max"]
            if r0 < 0 <= r1:
                bpred = rows[i]["b"] + (rows[i + 1]["b"] - rows[i]["b"]) * (-r0) / (r1 - r0); break
        out[f"{lam:g}"] = dict(b_absent=lo, b_present=hi, b_hb=bpred, pred_inv_lambda=pr["inv_lambda"], pred_inv_N1=pr["inv_N1"], rows=rows)
        print(f"  → 実測 {bm:.5f} / HB 予測 {bpred} / 1/λ 則 {pr['inv_lambda']:.5f} / 1/N1 形 {pr['inv_N1']:.5f}", flush=True)
        json.dump(out, open(os.path.join(DATA, "es_window_lam_large_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
