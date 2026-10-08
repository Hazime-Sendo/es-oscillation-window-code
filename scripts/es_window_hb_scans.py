"""
es_window_hb_scans.py — 論文 Sec. V.C で引用する 2 つの調和バランス走査（再現用）
  (1) δ を振る（b=0.135, s=0.635 固定）: ρ の最大値とその振幅 A_max（A_max ≈ 3.0√δ, max ρ は √δ にほぼ線形に減少）
  (2) s を振る（b=0.14, δ=0.05）: max ρ の s 依存（s=0.60 で +0.011 → s=0.70 で −0.002）
使い方: python es_window_hb_scans.py         （約 4 分）
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
from es_window_hb import rho_curve

A_LIST = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.7, 2.0]


def peak(b, delta, s, amax=2.0):
    """max over 0.15 <= A <= amax. b* の解析で使った範囲は A <= 1.4（それより大きい振幅は別の枝で、走査の端に最大が来ることがある）"""
    o = rho_curve(b, delta, s, A_list=[a for a in A_LIST if a <= amax]); cv = [d for d in o["curve"] if d["ok"] and d["A"] >= 0.15]
    i = int(np.argmax([d["rho"] for d in cv])); return cv[i]["rho"], cv[i]["A"]


print("(1) delta scan at b=0.135, s=0.635")
rows = []
for dl in (0.003, 0.01, 0.03, 0.05):
    r, A = peak(0.135, dl, 0.635); rows.append((dl, r, A)); print(f"  delta={dl:<6g} max rho={r:+.5f}  A_max={A:.2f}  A_max/sqrt(delta)={A / np.sqrt(dl):.2f}")
sl = np.diff([r for _, r, _ in rows]) / np.diff([np.sqrt(d) for d, _, _ in rows]); print("  d(max rho)/d sqrt(delta) between points:", np.round(sl, 3))
print("(2) s scan at b=0.14, delta=0.05; max over A <= 1.4 (range used for the threshold analysis)")
for s in (0.60, 0.64, 0.65, 0.66, 0.70):
    r, A = peak(0.14, 0.05, s, 1.4); print(f"  s={s:.2f}  max rho={r:+.5f}  A_max={A:.2f}")
r, A = peak(0.14, 0.05, 0.60, 2.0); print(f"  (for reference: s=0.60 with A up to 2.0: max rho={r:+.5f} at A={A:.2f}, the edge of the scan)")
