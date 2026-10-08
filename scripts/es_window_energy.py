"""
es_window_energy.py — 窓の持続振動のエネルギー収支（どの項が振動にエネルギーを供給しているか）

r–L の振動エネルギー  E = ½(r̃² + L̃²)   （~ は 1 周期平均からのずれ）
  dE/dt = r̃·ṙ + L̃·L̇   → 1 周期で平均するとゼロ（定常な持続振動）。
  ṙ = a r + b τ − k L − d e^r + κ·tanh(β(r−Λ)) + g·G + e_r·ε,   L̇ = p r − q L
  k=p のとき (p−k)⟨r̃L̃⟩=0 なので、各項の平均仕事 P_i = ⟨r̃ · (項 i の変動)⟩ の総和 = ⟨dE/dt⟩ = 0（検算に使う）。
τ を通る経路は、τ̇ = −γτ + α r + χU + κ·tanh(β(r−Λ−B4)) + e_τ ε  （γ = α − B3）が線形なので、
  周期定常解を送り元 j ごとに分解する（τ_j = 各送り元への線形フィルタ応答、Σ_j τ_j = τ）:
     j = r 自身（α r）／ U（χ U(r−w)）／ τ 側のレジーム項 ／ ε
  P_τ,j = b ⟨r̃ τ̃_j⟩ が、その経路が r に与える仕事。

小振幅の極限との比較: 窓の内側のロック状態（平衡点）を線形化し、振動数の近い複素固有モードで同じ収支を計算する。
  固有ベクトルの振幅 v（複素）から、各項の平均仕事 ½Re(conj(v_r)·f_i) を求める（τ の送り元は 1/(λ+γ) のフィルタ）。
  U 経路は U′(0)=0（平滑化 δ>0）のため、小振幅では寄与がゼロになる。

使い方: python es_window_energy.py <b> <delta> <s> [lam_loss]
"""
import sys, os, math, warnings
import numpy as np
warnings.filterwarnings("ignore")
b, dl, s_val = sys.argv[1], sys.argv[2], float(sys.argv[3])
LAM = sys.argv[4] if len(sys.argv) > 4 else "2.25"
sys.argv = ["x", "--b", b, "--delta", dl, "--lam-loss", LAM]
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
import es_shock_analysis_integrated as E
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

p = E.p
T0 = E.T0
rhs = E.make_rhs(p, s_val)
NP = 100
sol = solve_ivp(rhs, (0, NP * T0), E.GAM[int(0.3 * E.N)], method="LSODA", rtol=1e-11, atol=1e-13, max_step=0.2, dense_output=True)
tt = np.arange((NP - 25) * T0, NP * T0, 0.05); Y = sol.sol(tt)
r = Y[1]
span = r[-int(20 * T0 / 0.05):].max() - r[-int(20 * T0 / 0.05):].min()
print(f"=== b={b}, δ={dl}, λ={LAM}, s={s_val}: T0={T0:.1f};  最後の 20 周期の r の幅 = {span:.3f}" + ("" if span > 0.02 else "  → 持続振動なし（この s は窓の外）"))
if span <= 0.02:
    sys.exit(0)
mean_r = r.mean()
up = np.where((r[:-1] < mean_r) & (r[1:] >= mean_r))[0]
cross = [brentq(lambda t: sol.sol(t)[1] - mean_r, tt[i], tt[i + 1], xtol=1e-12) for i in up[-14:]]
T = float(np.mean(np.diff(cross[-8:])))
t_start = cross[-4]
Nn = 4096
ts = t_start + T * np.arange(Nn) / Nn
S = sol.sol(ts)                                             # 1 周期分（[L,r,τ,w,G,Λ,ε]）
closure = np.abs(sol.sol(t_start + T) - sol.sol(t_start)).max()
print(f"振動周期 T = {T:.3f}, 1 周期の閉じ誤差 {closure:.1e}")
L, r, tau, w, G, Lam, eps = S
C = dict(a=p.a, b=p.b, k=p.k, d=p.d, kap=p.kappa, BR=p.beta_regime, gGr=p.g_to_r, e_r=getattr(p, "eps_to_r", 0.1),
         P=p.p, Q=p.q, aT=p.alpha_tau, B3=p.B3, B4=p.B4, chi=p.chi, e_tau=0.2 * (1 + p.B5))


def Ufun(x):
    v = (x * x + p.delta ** 2) ** 0.44 - p.delta ** 0.88
    return np.where(x >= 0, v, -p.lam_loss * v)


U = Ufun(r - w)
th_r = np.tanh(C["BR"] * (r - Lam)); th_t = np.tanh(C["BR"] * (r - Lam - C["B4"])); er = np.exp(r)
tilde = lambda x: x - x.mean()
rt, Lt = tilde(r), tilde(L)
avg = lambda x, y: float(np.mean(tilde(x) * tilde(y)))

# τ の送り元ごとの周期定常応答（周波数領域の線形フィルタ）
gam = C["aT"] - C["B3"]
om = 2 * np.pi * np.fft.fftfreq(Nn, d=T / Nn)
def filt(src):
    F = np.fft.fft(src - src.mean()); H = F / (gam + 1j * om)
    return np.real(np.fft.ifft(H))
sources = {"r 自身 (α·r)": C["aT"] * r, "U(r−w) (χ·U)": C["chi"] * U, "τ 側レジーム項 (κ·tanh)": C["kap"] * th_t, "ε (e_τ·ε)": C["e_tau"] * eps}
tau_j = {k: filt(v) for k, v in sources.items()}
tau_sum = sum(tau_j.values())
err_tau = np.abs(tau_sum - tilde(tau)).max() / np.abs(tilde(tau)).max()
print(f"τ の分解の検算: Σ_j τ_j と τ の変動の最大相対誤差 = {err_tau:.1e}")

E_mean = 0.5 * (np.mean(rt ** 2) + np.mean(Lt ** 2))
rows = [("a·r（r の自己増幅）", C["a"] * np.mean(rt * rt)),
        ("−q·L（L の散逸）", -C["Q"] * np.mean(Lt * Lt)),
        ("(p−k)·rL（結合の非対称、k=p ならゼロ）", (C["P"] - C["k"]) * np.mean(rt * Lt)),
        ("−d·e^r（指数の散逸）", -C["d"] * avg(r, er)),
        ("κ·tanh(β(r−Λ))（r 側のレジーム項）", C["kap"] * avg(r, th_r)),
        ("g·G（制度 G のブレーキ）", C["gGr"] * avg(r, G)),
        ("e_r·ε（ε→r）", C["e_r"] * avg(r, eps))]
for k, tj in tau_j.items():
    rows.append((f"b·τ ← {k}", C["b"] * float(np.mean(rt * tj))))
total = sum(v for _, v in rows)
print(f"\n平均振動エネルギー ⟨E⟩ = {E_mean:.4f}    r の幅 {r.max() - r.min():.3f}, L の幅 {L.max() - L.min():.3f}")
print("  項                                          平均仕事 P_i      ⟨E⟩ あたりの率 P_i/⟨E⟩")
for name, v in rows:
    print(f"  {name:44s} {v:+10.5f}       {v / E_mean:+9.4f}")
pump = sum(v for _, v in rows if v > 0); damp = sum(v for _, v in rows if v < 0)
print(f"  {'合計（検算: ≈0）':44s} {total:+10.5f}       {total / E_mean:+9.4f}")
print(f"  供給の合計 {pump:+.5f}（率 {pump / E_mean:+.4f}）,  散逸の合計 {damp:+.5f}（率 {damp / E_mean:+.4f}）")


# ---------- G の仕事を、送り元ごとに分解（G̈ = κ_G(r−τ) − ξ·G は線形）----------
xiG, kG = p.xi, p.kappa_G
def filtG(src):
    F = np.fft.fft(src - src.mean()); return np.real(np.fft.ifft(F / (xiG + 1j * om)))
G_from = {"r 自身": filtG(kG * r)}
for k_, tj in tau_j.items():
    G_from[f"−τ ← {k_}"] = filtG(-kG * tj)
err_G = np.abs(sum(G_from.values()) - tilde(G)).max() / np.abs(tilde(G)).max()
print(f"\nG の分解の検算: Σ 送り元 と G の変動の最大相対誤差 = {err_G:.1e}")
print("  G→r（ブレーキ g·G）の仕事を、G の送り元ごとに:                   平均仕事      ⟨E⟩ あたりの率")
PG = {}
for k_, gj in G_from.items():
    PG[k_] = C["gGr"] * float(np.mean(rt * gj)); print(f"    G ← {k_:34s}                {PG[k_]:+10.5f}      {PG[k_] / E_mean:+9.4f}")
P_U_direct = C["b"] * float(np.mean(rt * tau_j["U(r−w) (χ·U)"])); P_U_viaG = PG["−τ ← U(r−w) (χ·U)"]
print(f"  → U(r−w) 経路の合計 = 直接 (b·τ←U) {P_U_direct:+.5f} + G 経由 (g·G←−τ←U) {P_U_viaG:+.5f} = {P_U_direct + P_U_viaG:+.5f}"
      f"   （⟨E⟩ あたり {(P_U_direct + P_U_viaG) / E_mean:+.4f}）")

# ---------- 小振幅の極限（ロック状態の固有モード）との比較 ----------
from scipy.optimize import root
guess = Y[:, -int(20 * T0 / 0.05):].mean(axis=1)
yeq = root(lambda y: np.array(rhs(0, y)), guess, method="hybr", tol=1e-13).x
Jm = np.zeros((7, 7))
for j in range(7):
    e = np.zeros(7); e[j] = 1e-6 * max(1.0, abs(yeq[j]))
    Jm[:, j] = (np.array(rhs(0, yeq + e)) - np.array(rhs(0, yeq - e))) / (2 * e[j])
lam_, V_ = np.linalg.eig(Jm)
w_osc = 2 * np.pi / T
cx = [i for i in range(7) if lam_[i].imag > 1e-6]
kk = min(cx, key=lambda i: abs(lam_[i].imag - w_osc))
lam_k, v = lam_[kk], V_[:, kk]
vL, vr, vt, vw, vG, vLam, ve = v
rs, Lrs, Lams = yeq[1], yeq[0], yeq[5]
sech2 = lambda x: 1 - np.tanh(x) ** 2
conj = np.conj
half = lambda z: 0.5 * float(np.real(z))
lin_rows = [("a·r（r の自己増幅）", half(C["a"] * conj(vr) * vr)),
            ("−q·L（L の散逸）", half(-C["Q"] * conj(vL) * vL)),
            ("(p−k)·rL（結合の非対称、k=p ならゼロ）", half((C["P"] - C["k"]) * conj(vr) * vL)),
            ("−d·e^r（指数の散逸）", half(-C["d"] * math.exp(rs) * conj(vr) * vr)),
            ("κ·tanh(β(r−Λ))（r 側のレジーム項）", half(C["kap"] * C["BR"] * sech2(C["BR"] * (rs - Lams)) * conj(vr) * (vr - vLam))),
            ("g·G（制度 G のブレーキ）", half(C["gGr"] * conj(vr) * vG)),
            ("e_r·ε（ε→r）", half(C["e_r"] * conj(vr) * ve))]
Hf = 1 / (lam_k + gam)
lin_src = {"r 自身 (α·r)": C["aT"] * vr, "U(r−w) (χ·U)": C["chi"] * 0.0 * (vr - vw),      # U′(0)=0
           "τ 側レジーム項 (κ·tanh)": C["kap"] * C["BR"] * sech2(C["BR"] * (rs - Lams - C["B4"])) * (vr - vLam), "ε (e_τ·ε)": C["e_tau"] * ve}
for k_, sv in lin_src.items():
    lin_rows.append((f"b·τ ← {k_}", half(C["b"] * conj(vr) * Hf * sv)))
E_lin = 0.25 * (abs(vr) ** 2 + abs(vL) ** 2)
tot_lin = sum(x for _, x in lin_rows)
print(f"\n---- 小振幅の極限（ロック状態の固有モード λ={lam_k.real:+.4f}{lam_k.imag:+.4f}i, 測定 ω={w_osc:.4f}）との比較 ----")
print("  項                                          小振幅の率 P/E    窓の率 P/E     差（窓−小振幅）")
for (name, vl), (_, vn) in zip(lin_rows, rows):
    print(f"  {name:44s} {vl / E_lin:+10.4f}      {vn / E_mean:+9.4f}     {vn / E_mean - vl / E_lin:+9.4f}")
print(f"  {'合計':44s} {tot_lin / E_lin:+10.4f}      {total / E_mean:+9.4f}     （小振幅の合計の目安 2Reλ = {2 * lam_k.real:+.4f}）")
