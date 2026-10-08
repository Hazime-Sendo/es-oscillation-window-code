"""
es_window_fig_data.py — 論文用データの保存（δ=0.05, 各 λ の窓の閾値での HB 波形から）

保存する量（es_window_paper_data.json）:
  g の 3 系列（実波形 g_HB, 厳密な描像関数 g_exact, べき乗近似 g_pl）, 振幅 a=|x₁|, 高調波 |x₂/x₁|, 平均のずれ x̄/a（実・厳密・冪）,
  U 経路の供給の分解（S_U, P1, P2, 残り, 合計）,
  基本波だけの伝達関数による P1, P2 の再現:  T_U = U₁/r₁,  H_τ = 1/(γ+iω),  H_G = 1/(ξ+iω),  ℓ² = |L₁/r₁|²
     P1 = 2χ·Re[T_U·H_τ]/(1+ℓ²),   P2 = 2 g_G·Re[−κ_G χ T_U H_τ H_G]/(1+ℓ²)      （r は純粋な余弦なので基本波のみが仕事をする）
  w が U の積分（w' = γ_w U）なので、x=r−w の基本波について  T_U = g/(1 − iβg),  β = γ_w/ω
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
import es_window_n1_exact as N
from es_window_hb import rho_curve, coeffs


def main():
    rows = N.load_rows(); out = []
    for lam in sorted(rows):
        bm, row = rows[lam]; s, Aat = row["s"], row["A_at_max"]
        o = rho_curve(bm, N.DELTA, s, lam=lam, A_list=sorted(set([0.05, 0.1, 0.2, 0.4, Aat])))
        d = [c for c in o["curve"] if abs(c["A"] - Aat) < 1e-9][0]
        C = coeffs(bm, N.DELTA, lam); L, tau, w, G, Lam, eps = d["Z"]; ns = len(L); th = 2 * np.pi * np.arange(ns) / ns
        r = d["rbar"] + d["A"] * np.cos(th); x = r - w; Ut = N.U_exact(x, lam)
        four = lambda y, n=1: 2 * np.mean(y * np.exp(-1j * n * th))
        r1, L1, x1, U1, x2 = four(r), four(L), four(x), four(Ut), four(x, 2)
        a = abs(x1); xbar = float(x.mean()); om = d["om"]
        g_hb = U1 / x1; T_U = U1 / r1
        xb_m, N1m = N.describing(N.U_exact, lam, a); xb_p, N1p = N.N1_pl_unit(lam)
        # 時間領域の分解（es_window_n1_exact.power_split と同じ）
        T = 2 * np.pi / om; wn = 2 * np.pi * np.fft.fftfreq(ns, d=T / ns); tl = lambda y: y - y.mean()
        filt = lambda src, rate: np.real(np.fft.ifft(np.fft.fft(tl(src)) / (rate + 1j * wn)))
        gam = C["aT"] - C["B3"]; tauU = filt(C["CHI"] * Ut, gam); GU = filt(-C["kG"] * tauU, C["XI"])
        rt, Lt = tl(r), tl(L); E = 0.5 * (np.mean(rt ** 2) + np.mean(Lt ** 2))
        P1 = float(np.mean(rt * tauU)) / E; P2 = C["gGr"] * float(np.mean(rt * GU)) / E
        F = (C["A_"] * r + C["B"] * tau - C["K"] * L - C["D"] * np.exp(r) + C["KAP"] * np.tanh(C["BR"] * (r - Lam)) + C["gGr"] * G + C["e_r"] * eps)
        total = float(np.mean(rt * F)) / E; SU = C["B"] * P1 + P2
        Ht = 1 / (gam + 1j * om); Hg = 1 / (C["XI"] + 1j * om); l2 = abs(L1 / r1) ** 2
        P1f = float(2 * C["CHI"] * np.real(T_U * Ht) / (1 + l2))
        P2f = float(2 * C["gGr"] * np.real(-C["kG"] * C["CHI"] * T_U * Ht * Hg) / (1 + l2))
        beta = C["gW"] / om; T_model = g_hb / (1 - 1j * beta * g_hb)
        out.append(dict(lam=lam, b=bm, s=s, A=d["A"], omega=om, a=a, a_over_delta=a / N.DELTA, xbar_a=xbar / a, xbar_a_exact=xb_m / a, xbar_a_pl=xb_p,
                        g_hb=abs(g_hb), g_hb_arg=float(np.angle(g_hb)), g_exact=N1m / a, g_pl=N1p * a ** (0.88 - 1), harm2=abs(x2) / a,
                        T_U_re=float(T_U.real), T_U_im=float(T_U.imag), T_model_err=float(abs(T_U - T_model) / abs(T_U)), beta=beta,
                        SU=SU, P1=P1, P2=P2, rest=total - SU, total=total, P1_fund=P1f, P2_fund=P2f, ell2=l2, gamma=gam))
        print(f"λ={lam:5.2f}: g_HB={abs(g_hb):.4f} g_exact={N1m / a:.4f} g_pl={N1p * a ** (-0.12):.4f}  P1={P1:.4f} (基本波 {P1f:.4f})  P2={P2:.4f} (基本波 {P2f:.4f})  |T_U−g/(1−iβg)|/|T_U|={abs(T_U - T_model) / abs(T_U):.3f}", flush=True)
    json.dump(out, open(os.path.join(DATA, "es_window_paper_data.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
