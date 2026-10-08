"""
es_window_hb.py — 窓の持続振動の 1 次の調和バランス（振幅 A の関数としての実効成長率 ρ(A)）

仮定: r(t) = r̄ + A·cos(ωt)。r 以外の変数 z=(L, τ, w, G, Λ, ε) は、この r(t) に駆動された周期定常解を
時間領域で厳密に求める（U(r−w) や tanh、e^r の非線形性は近似しない）。
r の式 ṙ = F(r, z) について、F の 1 次のフーリエ成分で次の 3 条件を課す:
   ⟨F⟩ = 0                          （平均のつり合い → r̄）
   2⟨F·sin ωt⟩ + A·ω = 0             （r′ の sin 成分 = −Aω → ω）
   ρ(A) = 2⟨F·cos ωt⟩ / A            （r′ の cos 成分 = σA → 実効成長率 ρ。持続振動は ρ=0）
A→0 で ρ → Re λ（ロック状態の固有モードの成長率）。ρ の最大値がゼロになる点が折り返し（窓の出現 b*）。

使い方: python es_window_hb.py <b> <delta> <s> [lam_loss]     （検証: 既知の窓と比べる）
"""
import sys, os, math, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from es_core_integrated import ESCore
from scipy.integrate import solve_ivp
from scipy.optimize import root


def coeffs(b, delta, lam=2.25, B4=0.0):
    p = ESCore(b=b, delta=delta, lam_loss=lam, B4=B4)
    return dict(P=p.p, Q=p.q, A_=p.a, K=p.k, D=p.d, KAP=p.kappa, BR=p.beta_regime, B=p.b, CHI=p.chi, XI=p.xi,
                aT=p.alpha_tau, gW=p.gamma_w, kG=p.kappa_G, gGr=p.g_to_r, lamr=p.lam_rate, gEL=p.gamma_eps_Lambda,
                gEt=p.gamma_eps_tau, B3=p.B3, B4=p.B4, dl=p.delta, LAM=p.lam_loss, e_tau=0.2 * (1 + p.B5), e_r=0.1)


def make_U(C):
    dl, LAM = C["dl"], C["LAM"]
    def U(x):
        v = (x * x + dl * dl) ** 0.44 - dl ** 0.88
        return v if x >= 0 else -LAM * v
    return U


def full_rhs(C, s):
    tanh, exp = math.tanh, math.exp
    U = make_U(C)
    def rhs(t, y):
        L, r, tau, w, G, Lam, eps = y
        u = U(r - w)
        dr = (C["A_"] * r + C["B"] * tau - C["K"] * L - C["D"] * exp(min(r, 700.0)) + C["KAP"] * tanh(C["BR"] * (r - Lam))
              + C["gGr"] * G + C["e_r"] * eps)
        dtau = C["aT"] * (r - tau) + C["B3"] * tau + C["CHI"] * u + C["KAP"] * tanh(C["BR"] * (r - Lam - C["B4"])) + C["e_tau"] * eps
        return [C["P"] * r - C["Q"] * L, dr, dtau, C["gW"] * u, C["kG"] * (r - tau) - C["XI"] * G,
                C["lamr"] * (r - Lam) + C["gEL"] * eps, -0.3 * eps + C["gEt"] * tau + s]
    return rhs


def locked_equilibrium(C, s):
    """ロック状態の平衡点（s の下で十分長く積分した末尾）と、複素固有値（最も不安定なもの）"""
    rhs = full_rhs(C, s)
    sol = solve_ivp(rhs, (0, 6000), [0, 0, 0, 0, 0, -0.5, 0], method="LSODA", rtol=1e-10, atol=1e-12, max_step=5.0)
    y0 = sol.y[:, -1]
    eq = root(lambda y: np.array(rhs(0, y)), y0, method="hybr", tol=1e-13).x
    J = np.zeros((7, 7))
    for j in range(7):
        e = np.zeros(7); e[j] = 1e-6 * max(1.0, abs(eq[j]))
        J[:, j] = (np.array(rhs(0, eq + e)) - np.array(rhs(0, eq - e))) / (2 * e[j])
    lam = np.linalg.eigvals(J); cx = [z for z in lam if z.imag > 1e-6]
    return eq, max(cx, key=lambda z: z.real)


class HB:
    def __init__(self, C, s, npre=24, ns=1024):
        self.C, self.s, self.npre, self.ns = C, s, npre, ns
        self.U = make_U(C)
        self.z0 = None                                   # 連続解法の初期値 [L,τ,w,G,Λ,ε]

    def slaves(self, rbar, A, om):
        C, s, U = self.C, self.s, self.U
        tanh = math.tanh; cos = math.cos
        def f(t, z):
            L, tau, w, G, Lam, eps = z
            r = rbar + A * cos(om * t)
            u = U(r - w)
            return [C["P"] * r - C["Q"] * L,
                    C["aT"] * (r - tau) + C["B3"] * tau + C["CHI"] * u + C["KAP"] * tanh(C["BR"] * (r - Lam - C["B4"])) + C["e_tau"] * eps,
                    C["gW"] * u, C["kG"] * (r - tau) - C["XI"] * G, C["lamr"] * (r - Lam) + C["gEL"] * eps, -0.3 * eps + C["gEt"] * tau + s]
        T = 2 * np.pi / om
        te = self.npre * T + T * np.arange(self.ns) / self.ns
        sol = solve_ivp(f, (0, (self.npre + 1) * T), self.z0, method="LSODA", rtol=1e-9, atol=1e-11, t_eval=te)
        return te, sol.y

    def residuals(self, rbar, A, om, want_rho=False):
        C = self.C
        te, Z = self.slaves(rbar, A, om)
        L, tau, w, G, Lam, eps = Z
        th = om * te
        r = rbar + A * np.cos(th)
        F = (C["A_"] * r + C["B"] * tau - C["K"] * L - C["D"] * np.exp(r) + C["KAP"] * np.tanh(C["BR"] * (r - Lam))
             + C["gGr"] * G + C["e_r"] * eps)
        mean = float(F.mean()); Sn = 2 * float((F * np.sin(th)).mean()); Cn = 2 * float((F * np.cos(th)).mean())
        self.last = Z[:, 0].copy()
        if want_rho:
            return mean, Sn + A * om, Cn / A, Z
        return mean, Sn + A * om

    def solve(self, A, rbar0, om0):
        """(r̄, ω) を解いて ρ(A) を返す。z0（スレーブ初期値）は直前の解から引き継ぐ。"""
        def fun(x):
            m, sq = self.residuals(x[0], A, x[1])
            self.z0 = self.last.copy()
            return [m, sq]
        sol = root(fun, [rbar0, om0], method="hybr", tol=1e-9)
        m, sq, rho, Z = self.residuals(sol.x[0], A, sol.x[1], want_rho=True)
        self.z0 = Z[:, 0].copy()
        return dict(A=A, rbar=sol.x[0], om=sol.x[1], rho=rho, ok=bool(sol.success) and abs(m) < 1e-5 and abs(sq) < 1e-5, Z=Z)


def rho_curve(b, delta, s, lam=2.25, B4=0.0, A_list=None, verbose=False):
    C = coeffs(b, delta, lam, B4)
    eq, lam0 = locked_equilibrium(C, s)
    hb = HB(C, s)
    hb.z0 = np.array([eq[0], eq[2], eq[3], eq[4], eq[5], eq[6]])
    rb, om = eq[1], lam0.imag
    if A_list is None:
        A_list = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0]
    out = []
    for A in A_list:
        d = hb.solve(A, rb, om)
        if d["ok"]:
            rb, om = d["rbar"], d["om"]
        out.append(d)
        if verbose:
            print(f"   A={A:5.2f}  r̄={d['rbar']:+.4f}  ω={d['om']:.4f}  ρ={d['rho']:+.5f}  {'' if d['ok'] else '(収束せず)'}", flush=True)
    return dict(eq=eq, lam0=lam0, curve=out)


if __name__ == "__main__":
    b, dl, s = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 2.25
    res = rho_curve(b, dl, s, lam, verbose=True)
    print(f"ロック状態の固有値: {res['lam0'].real:+.5f}{res['lam0'].imag:+.5f}i;  ρ(A→0) の目安 = Re λ")
    cv = [d for d in res["curve"] if d["ok"]]
    for i in range(len(cv) - 1):
        if cv[i]["rho"] * cv[i + 1]["rho"] < 0:
            a0, a1, r0, r1 = cv[i]["A"], cv[i + 1]["A"], cv[i]["rho"], cv[i + 1]["rho"]
            print(f"ρ=0 の交差: A ≈ {a0 + (a1 - a0) * (-r0) / (r1 - r0):.3f}  ({'外側=安定な窓の振幅' if r0 > 0 else '内側=不安定'})")
    print("最大 ρ =", max(d["rho"] for d in cv))
