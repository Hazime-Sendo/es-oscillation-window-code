"""
es_window_n1_exact.py — U のべき乗近似を外した、基本波ゲイン N1(λ) の厳密な再構成（δ=0.05）

U(x) = V(x)（x≥0）, −λ·V(−x)（x<0）,  V(y) = (y²+δ²)^0.44 − δ^0.88   （実際の U。べき乗 |x|^0.88 の近似は使わない）

各 λ の窓の閾値（実測 b*）で、調和バランスの解から x(t)=r−w を取り出し、次を比べる:
  g_HB    : 実波形の U と x のフーリエ基本波の比  U₁/x₁                     （厳密。波形の高調波も含む）
  g_model : 振幅 a=|x₁| の余弦入力 x̄'+a·cosθ（x̄' は ⟨U⟩=0 を満たす）に対する厳密な描像関数 N1_exact/a
  g_pl    : べき乗近似（V=|x|^0.88, a≫δ）の描像関数 N1_pl(λ)·a^(p−1)         （前回の近似）
そのうえで b*(λ) を 1/g（と 1/λ）で当てはめて残差を比べる。

使い方: python es_window_n1_exact.py       （約 3 分）
"""
import os, sys, json
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
from es_window_hb import rho_curve

DELTA = 0.05; P_EXP = 0.88
TH = np.linspace(0, 2 * np.pi, 200001)


def U_exact(x, lam, dl=DELTA):
    v = (x * x + dl * dl) ** (P_EXP / 2) - dl ** P_EXP
    return np.where(x >= 0, v, -lam * v)


def U_pl(x, lam):
    return np.where(x >= 0, np.abs(x) ** P_EXP, -lam * np.abs(x) ** P_EXP)


def describing(Ufun, lam, a):
    """振幅 a の余弦入力 x̄+a·cosθ で ⟨U⟩=0 となる x̄ を求め、基本波ゲイン N1（cos 係数）を返す"""
    F = lambda xb: np.trapezoid(Ufun(xb + a * np.cos(TH), lam), TH) / (2 * np.pi)
    xb = brentq(F, -0.999999 * a, 0.999999 * a, xtol=1e-14)
    N1 = np.trapezoid(Ufun(xb + a * np.cos(TH), lam) * np.cos(TH), TH) / np.pi
    return xb, N1


def N1_pl_unit(lam):
    xb, n1 = describing(U_pl, lam, 1.0); return xb, n1                  # a=1 のとき（べき乗は a^p にスケール）


def load_rows():
    r1 = json.load(open(os.path.join(DATA, "es_window_hb_lambda_results.json")))
    r2 = json.load(open(os.path.join(DATA, "es_window_lam_large_results.json")))
    out = {}
    for k, v in r1.items(): out[float(k)] = (v["b_meas"], v["rows"][2])
    for k, v in r2.items():
        bm = 0.5 * (v["b_absent"] + v["b_present"]); out[float(k)] = (bm, v["rows"][2])
    return out


def main():
    rows = load_rows(); res = []
    print(f"δ={DELTA}: 各 λ の窓の閾値（実測 b*, その s, HB で max ρ となる振幅 A）での x=r−w の波形と U の基本波ゲイン")
    print("   λ     b*       s      A     a=|x₁|  x̄/a(実)  x̄/a(厳密) x̄/a(冪)   g_HB     g_厳密   g_冪     g_HB/g_厳密  高調波 |x₂/x₁|")
    for lam in sorted(rows):
        bm, row = rows[lam]; s, Aat = row["s"], row["A_at_max"]
        A_list = sorted(set([0.05, 0.1, 0.2, 0.4, Aat]))
        out = rho_curve(bm, DELTA, s, lam=lam, A_list=A_list)
        d = [c for c in out["curve"] if abs(c["A"] - Aat) < 1e-9][0]
        if not d["ok"]:
            print(f"  λ={lam}: 収束せず"); continue
        Z = d["Z"]; ns = Z.shape[1]; th = 2 * np.pi * np.arange(ns) / ns
        r = d["rbar"] + d["A"] * np.cos(th); w = Z[2]; x = r - w
        Ut = U_exact(x, lam)
        four = lambda y, n=1: 2 * np.mean(y * np.exp(-1j * n * th))
        x1, U1, x2 = four(x), four(Ut), four(x, 2)
        a = abs(x1); xbar = float(x.mean())
        g_hb = U1 / x1
        xb_m, N1m = describing(U_exact, lam, a); g_model = N1m / a
        xb_p, N1p = N1_pl_unit(lam); g_pl = N1p * a ** (P_EXP - 1)
        res.append(dict(lam=lam, b=bm, s=s, A=d["A"], a=a, xbar_a=xbar / a, xbar_a_model=xb_m / a, xbar_a_pl=xb_p, g_hb=g_hb, g_model=g_model, g_pl=g_pl,
                        U_mean=float(Ut.mean()), harm2=abs(x2) / a))
        print(f"  {lam:5.2f}  {bm:.5f}  {s:.4f}  {d['A']:.2f}  {a:.4f}  {xbar / a:+.3f}   {xb_m / a:+.3f}    {xb_p:+.3f}   {abs(g_hb):.4f}  {g_model:.4f}  {g_pl:.4f}   "
              f"{abs(g_hb) / g_model:.3f}     {abs(x2) / a:.3f}   (arg g_HB={np.degrees(np.angle(g_hb)):+.1f}°, ⟨U⟩={Ut.mean():+.1e})")
    L = np.array([q["lam"] for q in res]); b = np.array([q["b"] for q in res])
    print("\nb*(λ) の当てはめ  b* = b0 + c/X  （X を変える。7 点）")
    for label, X in (("λ", L), ("g_HB", np.array([abs(q["g_hb"]) for q in res])), ("g_厳密", np.array([q["g_model"] for q in res])),
                     ("g_冪", np.array([q["g_pl"] for q in res]))):
        A = np.column_stack([np.ones_like(X), 1 / X]); c = np.linalg.lstsq(A, b, rcond=None)[0]; e = np.abs(A @ c - b)
        print(f"   X={label:6s}: b0={c[0]:.5f}, c={c[1]:+.5f},  残差の最大 {e.max():.5f}  （λ≤4 の 5 点 {e[:5].max():.5f} / λ=10,20 {e[5:].max():.5f}）")
    json.dump([{k: (complex(v).real if k.startswith("g_hb") is False and False else v) for k, v in q.items() if k != "g_hb"} | dict(g_hb_abs=abs(q["g_hb"]), g_hb_arg=float(np.angle(q["g_hb"]))) for q in res],
              open(os.path.join(DATA, "es_window_n1_exact_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__" and "--power" not in sys.argv:
    main()


# ---------------------------------------------------------------------------------------------
# 追加: U 経路の供給を、HB 解の実波形から厳密に分解し、閾値での供給（U 経路）と残り（赤字）を λ ごとに比べる
#   P_U = b⟨r̃ τ̃_U⟩ + g_G⟨r̃ G̃_U⟩,   τ_U = フィルタ(χ·U; γ),   G_U = フィルタ(−κ_G·τ_U; ξ)
#   S_U/E = P_U / ⟨E⟩,   ⟨E⟩ = ½(⟨r̃²⟩+⟨L̃²⟩);  合計 = ⟨r̃·F⟩/⟨E⟩（閾値ではほぼ 0）、残り = 合計 − S_U
# ---------------------------------------------------------------------------------------------
def power_split():
    from es_window_hb import coeffs
    rows = load_rows(); print("\nU 経路の供給と残り（閾値の実測 b*, HB の実波形）:")
    print("   λ     b*       S_U/E    残り(赤字)/E   合計/E     P1=⟨r̃τ_U⟩/E   P2=g_G⟨r̃G_U⟩/E   b*·P1+P2=S_U/E")
    out = []
    for lam in sorted(rows):
        bm, row = rows[lam]; s, Aat = row["s"], row["A_at_max"]
        A_list = sorted(set([0.05, 0.1, 0.2, 0.4, Aat]))
        o = rho_curve(bm, DELTA, s, lam=lam, A_list=A_list); d = [c for c in o["curve"] if abs(c["A"] - Aat) < 1e-9][0]
        C = coeffs(bm, DELTA, lam); Z = d["Z"]; ns = Z.shape[1]; th = 2 * np.pi * np.arange(ns) / ns
        L, tau, w, G, Lam, eps = Z; r = d["rbar"] + d["A"] * np.cos(th); om = d["om"]; T = 2 * np.pi / om
        wn = 2 * np.pi * np.fft.fftfreq(ns, d=T / ns)
        tl = lambda y: y - y.mean()
        filt = lambda src, rate: np.real(np.fft.ifft(np.fft.fft(tl(src)) / (rate + 1j * wn)))
        Ut = U_exact(r - w, lam); gam = C["aT"] - C["B3"]
        tauU = filt(C["CHI"] * Ut, gam); GU = filt(-C["kG"] * tauU, C["XI"])
        rt, Lt = tl(r), tl(L); E = 0.5 * (np.mean(rt ** 2) + np.mean(Lt ** 2))
        P1 = float(np.mean(rt * tauU)) / E; P2 = C["gGr"] * float(np.mean(rt * GU)) / E
        F = (C["A_"] * r + C["B"] * tau - C["K"] * L - C["D"] * np.exp(r) + C["KAP"] * np.tanh(C["BR"] * (r - Lam)) + C["gGr"] * G + C["e_r"] * eps)
        total = float(np.mean(rt * F)) / E; SU = C["B"] * P1 + P2
        out.append((lam, bm, SU, total - SU, total, P1, P2))
        print(f"  {lam:5.2f}  {bm:.5f}  {SU:+.4f}   {total - SU:+.4f}     {total:+.4f}     {P1:+.4f}        {P2:+.4f}          {SU:+.4f}")
    a = np.array(out)
    print(f"\n  S_U/E の範囲 {a[:, 2].min():+.4f} 〜 {a[:, 2].max():+.4f}（平均 {a[:, 2].mean():+.4f}, 相対ばらつき {100 * a[:, 2].std() / abs(a[:, 2].mean()):.1f}%）")
    print(f"  残り/E の範囲 {a[:, 3].min():+.4f} 〜 {a[:, 3].max():+.4f}")
    # 単一の供給定数 D0 で b*(λ) を再構成: b* = (D0 − P2)/P1
    D0 = float(np.mean(a[:, 2]))
    bpred = (D0 - a[:, 6]) / a[:, 5]
    print(f"  S_U/E = D0={D0:+.4f}（一定）とした再構成 b* = (D0−P2)/P1 の実測との差の最大: {np.abs(bpred - a[:, 1]).max():.5f}")
    print("   λ    実測 b*   再構成 b*")
    for q, bp in zip(a, bpred): print(f"  {q[0]:5.2f}  {q[1]:.5f}   {bp:.5f}")


if __name__ == "__main__" and "--power" in sys.argv:
    power_split()
