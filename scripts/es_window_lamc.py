"""
es_window_lamc.py — 窓の境界を (b, λ, δ) で調べる: 「λ_c(b,δ)=2.25 の等高線が b*(δ) と一致するか」

定義（すべて es_window_eval.py の同じ判定 = 「窓」の有無）
  λ_c(b,δ)  : b, δ を固定して損失回避 λ を振ったとき、窓が現れる λ の下限（λ>λ_c で窓あり）。
  b*(λ,δ)   : λ, δ を固定して b を振ったとき、窓が現れる b の下限（b>b* で窓あり）。
  s は毎回、その設定での s*+（切替サイクルが消える s）からの相対位置で測る
  （λ を変えると s*+ が動くため、s を固定して比べると窓の移動と消滅を区別できない）。

内容
  A. 各 δ について、b*(δ)（λ=2.25）の上端 b_hi と下端 b_lo で λ_c を測る。等高線が一致するなら
     λ_c(b_hi) ≲ 2.25 ≲ λ_c(b_lo)（差は b* の幅 ×（dλ_c/db）程度）。
  B. δ=0.05 で b*(λ) を λ=1, 1.5, 3, 4 で直接測る（λ_c(b) の曲線を b 方向から描く）。
結果は es_window_lamc_results.json に逐次保存する。  使い方: python es_window_lamc.py   （約 30〜40 分）
"""
import sys, os, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
RES = os.path.join(DATA, "es_window_lamc_results.json")
BASE = json.load(open(os.path.join(DATA, "es_window_bstar_results.json")))     # B4=0, λ=2.25 の b*(δ)
DELTAS_SCAN = "0.002,0.004,0.008,0.015,0.03,0.06"


def evaluate(b, delta, lam, tol=0.0005):
    out = subprocess.run([sys.executable, "-u", os.path.join(SRC, "es_window_eval.py"), f"{b:.5f}", f"{delta:g}",
                          f"{tol:g}", DELTAS_SCAN, "--lam", f"{lam:g}"], capture_output=True, text=True, cwd=SRC).stdout.strip().splitlines()
    d = json.loads(out[-1])
    print(f"   b={b:.5f} δ={delta:g} λ={lam:.3f}: s*+={d['s_crit']:+.4f}, 窓={'あり' if d['present'] else 'なし'}"
          + (f", 振幅 {d['amp_first']:.2f}" if d.get("present") else ""), flush=True)
    return d


def save(res):
    json.dump(res, open(RES, "w"), ensure_ascii=False, indent=1)


def lam_c(b, delta, tol=0.01, lam_min=1.0, lam_max=6.0):
    """λ を振って窓の下限を求める。返り値 (λ_absent, λ_present)。窓が λ_min でも存在すれば λ_present=λ_min, λ_absent=None"""
    d = evaluate(b, delta, 2.25); log = [d]
    if d["present"]:
        hi = 2.25
        for lam in (2.0, 1.5, lam_min):
            dd = evaluate(b, delta, lam); log.append(dd)
            if not dd["present"]:
                lo = lam; break
            hi = lam
        else:
            return None, hi, log
    else:
        lo = 2.25; hi = None
        for lam in (2.75, 3.5, 4.5, lam_max):
            dd = evaluate(b, delta, lam); log.append(dd)
            if dd["present"]:
                hi = lam; break
            lo = lam
        if hi is None:
            return lo, None, log
    while hi - lo > tol:
        mid = round(0.5 * (lo + hi), 4); dm = evaluate(b, delta, mid); log.append(dm)
        if dm["present"]: hi = mid
        else: lo = mid
    return lo, hi, log


def b_star(delta, lam, b_start, step=0.005, tol=0.0005, b_min=0.05, b_max=0.30):
    d = evaluate(b_start, delta, lam); log = [d]
    direction = -1 if d["present"] else +1
    b = b_start; prev = b
    while bool(d["present"]) == (direction == -1):
        prev = b; b = round(b + direction * step, 5)
        if b < b_min or b > b_max:
            return None, None, log
        d = evaluate(b, delta, lam); log.append(d)
    lo, hi = (prev, b) if direction == +1 else (b, prev)
    while hi - lo > tol:
        mid = round(0.5 * (lo + hi), 5); dm = evaluate(mid, delta, lam); log.append(dm)
        if dm["present"]: hi = mid
        else: lo = mid
    return lo, hi, log


def run():
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    # A. b*(δ) の上端・下端での λ_c
    jobs = [("A", 0.05, "b_present"), ("A", 0.01, "b_present"), ("A", 0.2, "b_present"), ("A", 0.05, "b_absent")]
    for kind, delta, edge in jobs:
        key = f"A|{delta:g}|{edge}"
        if key in res and res[key].get("done"): continue
        b = BASE[f"{delta:g}"][edge]; t0 = time.time()
        print(f"\n===== A: λ_c(b={b}, δ={delta:g})  [b*(δ) の{'上端(窓あり側)' if edge == 'b_present' else '下端(窓なし側)'}] =====", flush=True)
        lo, hi, log = lam_c(b, delta)
        res[key] = dict(b=b, delta=delta, edge=edge, lam_absent=lo, lam_present=hi, done=True, seconds=time.time() - t0); save(res)
        print(f"  → λ_c ∈ ({lo}, {hi}]   ({time.time() - t0:.0f}s)", flush=True)
    # B. δ=0.05 で b*(λ)
    guess = {1.0: 0.145, 1.5: 0.14, 3.0: 0.128, 4.0: 0.125}
    for lam in (1.0, 4.0, 1.5, 3.0):
        key = f"B|0.05|{lam:g}"
        if key in res and res[key].get("done"): continue
        t0 = time.time(); print(f"\n===== B: b*(λ={lam:g}, δ=0.05) =====", flush=True)
        lo, hi, log = b_star(0.05, lam, guess[lam])
        res[key] = dict(delta=0.05, lam=lam, b_absent=lo, b_present=hi, done=True, seconds=time.time() - t0); save(res)
        print(f"  → b* ∈ ({lo}, {hi}]   ({time.time() - t0:.0f}s)", flush=True)


def run_delta_scan():
    """δ=0.01, 0.2 で b*(λ=1), b*(λ=4) を測る（b* = b_∞(δ) + c(δ)/λ の係数を決めるため）"""
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for delta, lam, start in ((0.01, 1.0, 0.1315), (0.01, 4.0, 0.1127), (0.2, 1.0, 0.1753), (0.2, 4.0, 0.1565)):
        key = f"B|{delta:g}|{lam:g}"
        if key in res and res[key].get("done"): continue
        t0 = time.time(); print(f"\n===== B: b*(λ={lam:g}, δ={delta:g}) =====", flush=True)
        lo, hi, log = b_star(delta, lam, start)
        res[key] = dict(delta=delta, lam=lam, b_absent=lo, b_present=hi, done=True, seconds=time.time() - t0); save(res)
        print(f"  → b* ∈ ({lo}, {hi}]   ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if "--delta-scan" in sys.argv:
        run_delta_scan()
    else:
        run()
