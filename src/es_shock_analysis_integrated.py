"""
統合版 ESCore (es_core_integrated.py) の ε ショック応答解析
（es_shock_analysis.py = AB_fixed 版の解析を、同じ物差しのまま統合版へ移植したもの）

問い: ショックを受けたあと「元のサイクルへの復帰」か「新しいサイクルへの移行」か。

判別の物差し（AB_fixed 版と同一）
  1. 基準サイクル Γ（周期 T0, Floquet 乗数）を Newton 法で厳密に求める。
  2. 応答軌道の Γ への尺度化距離 d(t) = 最近傍距離 / 各変数の範囲（RMS）。
       最終 2 周期で d < tol  →  「復帰」（同一サイクル。位相ずれは別途報告）
       d ≥ tol かつ周期的     →  「新サイクル」（周期・振幅を報告）
       d ≥ tol かつ定常       →  「定常点（レジーム固定）」
  3. 位相シフト Δφ: 応答の最終位相 − 無摂動の到達位相。
  4. レジーム遷移: Schmitt トリガ (|r−Λ|>0.3) で活動/収縮を判定し、
       n_pert − n_exp（位相シフトから期待される切替数との差）を「切替の過不足」とする。
       偶数(±2)は同一サイクルのままの「サイクル・スリップ」（位相が半周期超ずれた）。

ショックの 2 種類
  - パルス:   ε(t_s) += A  （瞬間ジャンプ。ε は自然減衰 0.3 で戻る）
  - 持続入力: dε/dt に一定入力 s を加え続ける（パラメータの持続的なずれに相当）

AB_fixed 版との差（移植上の変更）
  - 状態の並びは AB_fixed 版と同じ [L,r,τ,w,G,Λ,ε] に揃えた（ESCore 本体は [r,L,w,τ,G,Λ,ε]）。
  - 周期が長い（T0≈200）ため、持続入力の積分は 25 周期（AB_fixed 版は 40）、ヒステリシス探索は 20/30 周期（30/60）。
  - 既定は確定仕様（b=0.1, δ=0.05, B3=−0.3, B5=0）。ε は基準サイクル上でも 0 でない（τ に駆動される）。

実行:  python es_shock_analysis_integrated.py                       (既定 b=0.1)
       python es_shock_analysis_integrated.py --b 0.05              (b を変える。図は _b0.05 つき)
       python es_shock_analysis_integrated.py --B3 0                (歪みなし。既定は確定仕様の B3=−0.3, B5=0。図は _B3_0 つき)
       python es_shock_analysis_integrated.py --B5 0.5              (B5 を変える。図は _B5_0.5 つき)
       python es_shock_analysis_integrated.py --quick               (粗い格子)
"""
import sys, os, math, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from es_core_integrated import ESCore

QUICK = "--quick" in sys.argv


def _arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


_DEF = ESCore()
B_VAL, DELTA_U = _arg("--b", _DEF.b), _arg("--delta", _DEF.delta)
B3_VAL, B4_VAL, B5_VAL = _arg("--B3", _DEF.B3), _arg("--B4", _DEF.B4), _arg("--B5", _DEF.B5)
LAM_VAL = _arg("--lam-loss", _DEF.lam_loss)                       # 損失回避係数 λ（既定 2.25）
SUFFIX = ((f"_b{B_VAL:g}" if B_VAL != _DEF.b else "") + (f"_delta{DELTA_U:g}" if DELTA_U != _DEF.delta else "")
          + (f"_B3_{B3_VAL:g}" if B3_VAL != _DEF.B3 else "") + (f"_B4_{B4_VAL:g}" if B4_VAL != _DEF.B4 else "")
          + (f"_B5_{B5_VAL:g}" if B5_VAL != _DEF.B5 else "") + (f"_lam{LAM_VAL:g}" if LAM_VAL != _DEF.lam_loss else ""))
NAMES = ["L", "r", "τ", "w", "G", "Λ", "ε"]
p = ESCore(b=B_VAL, delta=DELTA_U, B3=B3_VAL, B4=B4_VAL, B5=B5_VAL, lam_loss=LAM_VAL)
# 環境変数 ES_SET="name=value,name=value" で係数を上書きできる（結合の切除実験用。既定では何もしない）。
#   ESCore の属性に加え、eps_to_r（r 式の ε の係数, 既定 0.1）と kappa_tau（τ 側レジーム項の強さ, 既定 κ）も指定できる。
_SET = os.environ.get("ES_SET", "")
for _kv in filter(None, _SET.split(",")):
    _k, _v = _kv.split("="); setattr(p, _k.strip(), float(_v))


# =========================
# 1. 高速 rhs（ESCore.F と同一式。並びは [L,r,τ,w,G,Λ,ε]。u_eps は ε への外部入力）
# =========================
def make_rhs(p, u_eps=0.0):
    tanh, exp = math.tanh, math.exp
    P, Q, A_, K, D, KAP, BR = p.p, p.q, p.a, p.k, p.d, p.kappa, p.beta_regime
    B, CHI, XI = p.b, p.chi, p.xi
    aT, gW, kG, gGr = p.alpha_tau, p.gamma_w, p.kappa_G, p.g_to_r
    lamr, gEL, gEt = p.lam_rate, p.gamma_eps_Lambda, p.gamma_eps_tau
    B3, B4, B5 = p.B3, p.B4, p.B5
    dl, LAM = p.delta, p.lam_loss
    E_R, KAPT = getattr(p, "eps_to_r", 0.1), getattr(p, "kappa_tau", KAP)

    def U(x):
        v = (x * x + dl * dl) ** 0.44 - dl ** 0.88
        return v if x >= 0 else -LAM * v

    def rhs(t, y):
        L, r, tau, w, G, Lam, eps = y
        u = U(r - w)
        dr = (A_ * r + B * tau - K * L - D * exp(min(r, 700.0)) + KAP * tanh(BR * (r - Lam))
              + gGr * G + E_R * eps)
        dtau = aT * (r - tau) + B3 * tau + CHI * u + KAPT * tanh(BR * (r - Lam - B4)) + 0.2 * (1 + B5) * eps
        return [P * r - Q * L, dr, dtau, gW * u, kG * (r - tau) - XI * G,
                lamr * (r - Lam) + gEL * eps, -0.3 * eps + gEt * tau + u_eps]
    return rhs


RHS0 = make_rhs(p)
_rng = np.random.default_rng(0)


def _to_core(y):                 # [L,r,τ,w,G,Λ,ε] → ESCore の [r,L,w,τ,G,Λ,ε]
    L, r, tau, w, G, Lam, eps = y
    return np.array([r, L, w, tau, G, Lam, eps])


def _from_core(f):               # ESCore.F の出力 → [L,r,τ,w,G,Λ,ε] の並び
    dr, dL, dw, dtau, dG, dLam, deps = f
    return np.array([dL, dr, dtau, dw, dG, dLam, deps])


if not _SET:                     # 係数を上書きした場合は ESCore.F と一致しないので確認しない
    assert max(np.abs(np.array(RHS0(0, y)) - _from_core(p.F(_to_core(y)))).max()
               for y in _rng.uniform(-3, 3, (100, 7))) < 1e-12, "rhs が ESCore.F と一致しない"


# =========================
# 2. 基準サイクル（Newton 法）と Floquet 乗数
# =========================
def up_crossings(sol, lam_mid, t_lo, t_hi, dt=0.1):
    """Λ が lam_mid を上向きに横切る時刻（密出力+brentq。scipy のイベント検出は急峻な軌道で破綻するため使わない）"""
    from scipy.optimize import brentq
    tt = np.arange(t_lo, t_hi, dt); v = sol.sol(tt)[5] - lam_mid
    idx = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    return [brentq(lambda t: sol.sol(t)[5] - lam_mid, tt[i], tt[i + 1], xtol=1e-12) for i in idx]


def find_reference_cycle():
    Y0 = np.array([0.2, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0])          # [L,r,τ,w,G,Λ,ε]
    s = solve_ivp(RHS0, (0, 9000), Y0, method="LSODA", rtol=1e-10, atol=1e-12, max_step=0.5)
    tail = s.y[:, s.t > 5000]
    lam_mid = 0.5 * (tail[5].max() + tail[5].min())        # 断面: Λ=lam_mid（上向き通過）
    s2 = solve_ivp(RHS0, (0, 3000), s.y[:, -1], method="LSODA", rtol=1e-10, atol=1e-12,
                   max_step=0.5, dense_output=True)
    tc = up_crossings(s2, lam_mid, 0, 3000)
    Tg = float(np.median(np.diff(tc)))                      # 周期の推定値

    def P_map(x6):
        y0 = np.array([x6[0], x6[1], x6[2], x6[3], x6[4], lam_mid, x6[5]])
        sol = solve_ivp(RHS0, (0, 2.2 * Tg), y0, method="LSODA", rtol=1e-11, atol=1e-13,
                        max_step=0.5, dense_output=True)
        te = up_crossings(sol, lam_mid, 0.5 * Tg, 2.2 * Tg)[0]
        yE = sol.sol(te)
        return np.array([yE[0], yE[1], yE[2], yE[3], yE[4], yE[6]]), te

    def dP(x, h=1e-6):
        J = np.zeros((6, 6))
        for j in range(6):
            e = np.zeros(6); e[j] = h * max(1, abs(x[j]))
            J[:, j] = (P_map(x + e)[0] - P_map(x - e)[0]) / (2 * e[j])
        return J

    yc = s2.sol(tc[-2]); x = np.array([yc[0], yc[1], yc[2], yc[3], yc[4], yc[6]])
    for _ in range(8):
        xn, _T = P_map(x); res = xn - x
        if np.abs(res).max() < 1e-10:
            break
        x = x - np.linalg.solve(dP(x) - np.eye(6), res)
    T0 = P_map(x)[1]; mu = np.linalg.eigvals(dP(x))
    y_sec = np.array([x[0], x[1], x[2], x[3], x[4], lam_mid, x[5]])
    return T0, mu, y_sec, lam_mid


t_start = time.time()
T0, MU, Y_SEC, LAM_MID = find_reference_cycle()
N = 8000
_s = solve_ivp(RHS0, (0, T0), Y_SEC, method="LSODA", rtol=1e-11, atol=1e-13, max_step=0.5,
               dense_output=True)
PH = np.linspace(0, 1, N, endpoint=False)
GAM = _s.sol(PH * T0).T                                  # 基準軌道 (N,7)
SCALE = np.maximum(GAM.max(axis=0) - GAM.min(axis=0), 1e-3)     # 下限: 基準サイクル上で恒等的に 0 の変数（τ→ε を切った場合の ε）があっても割り算が破綻しない
TREE = cKDTree(GAM / SCALE)
DELTA = 0.3


def regime_series(r, Lam):
    d = r - Lam; st = 1 if d[0] > 0 else -1; out = np.empty(len(d), int)
    for i, x in enumerate(d):
        if st == 1 and x < -DELTA: st = -1
        elif st == -1 and x > DELTA: st = 1
        out[i] = st
    return out


_reg2 = regime_series(np.tile(GAM[:, 1], 2), np.tile(GAM[:, 5], 2))[N:]
SW_CUM = np.concatenate([[0], np.cumsum(np.diff(_reg2) != 0)])[:N]
NSW = int((np.diff(np.concatenate([_reg2, [_reg2[0]]])) != 0).sum())
SW_PHASES = PH[np.where(np.diff(np.concatenate([_reg2, [_reg2[0]]])) != 0)[0]]


def S_total(x):
    k = int(np.floor(x)); f = x - k
    return k * NSW + SW_CUM[min(int(f * N), N - 1)]


wrap = lambda x: (x + 0.5) % 1.0 - 0.5


# =========================
# 3. 応答の計算と判別
# =========================
def describe_tail(t, Y, tail_periods=6):
    sel = t > t[-1] - tail_periods * T0; tt = t[sel]; Z = Y[:, sel]
    var = (Z.max(axis=1) - Z.min(axis=1)) / SCALE
    if var.max() < 1e-3:
        return dict(kind="定常点", state=Z[:, -1].copy())
    lam = Z[5]; mid = 0.5 * (lam.max() + lam.min())
    up = np.where((lam[:-1] < mid) & (lam[1:] >= mid))[0]
    per = np.diff(tt[up]) if len(up) >= 3 else np.array([])
    if len(per) >= 2 and per.std() / per.mean() < 0.02:
        n_sw = int((np.diff(regime_series(Z[1], Z[5])) != 0).sum())      # 観測窓内のレジーム切替数
        return dict(kind="周期" if n_sw > 0 else "固定振動", period=per.mean(), n_sw=n_sw,
                    r_lo=Z[1].min(), r_hi=Z[1].max(), lam_lo=Z[5].min(), lam_hi=Z[5].max(),
                    eps_mean=Z[6].mean())
    return dict(kind="不規則/未収束")


N_OBS = max(14, int(np.ceil(6 / -np.log(np.abs(MU).max()))) + 4)      # 回復に十分な観測周期数


def respond(phi, A, T_end=None, u=None, tol=0.01, dt=0.5):
    """phi: 基準サイクル上の位相, A: ε へのジャンプ, u=(s,t_on,t_off): 持続入力"""
    T_end = N_OBS * T0 if T_end is None else T_end
    y = GAM[int(phi * N) % N].copy(); y[6] += A
    esc = lambda t, y: 1e4 - np.max(np.abs(y)); esc.terminal = True
    if u is None: segs = [(0, T_end, 0.0)]
    else:
        s_, ton, toff = u; toff = min(toff, T_end)
        segs = [g for g in [(0, ton, 0.0), (ton, toff, s_), (toff, T_end, 0.0)] if g[1] > g[0]]
    ts, ys, escaped = [], [], False
    for a, b, uu in segs:
        sol = solve_ivp(make_rhs(p, uu), (a, b), y, method="LSODA", rtol=1e-9, atol=1e-11,
                        max_step=0.5, t_eval=np.arange(a, b, dt), events=[esc])
        ts.append(sol.t); ys.append(sol.y)
        if sol.status == 1: escaped = True; break
        y = solve_ivp(make_rhs(p, uu), (a, b), y, method="LSODA", rtol=1e-9, atol=1e-11,
                      max_step=0.5).y[:, -1]
    t = np.concatenate(ts); Y = np.concatenate(ys, axis=1)
    escaped = escaped or not np.isfinite(Y).all()
    out = dict(t=t, Y=Y, phi0=phi, A=A, escaped=escaped)
    if escaped: out["outcome"] = "逃走"; return out
    d, j = TREE.query(Y.T / SCALE); d = d / np.sqrt(7)
    bad = np.where(d >= tol)[0]
    out["t_rec"] = 0.0 if len(bad) == 0 else (t[bad[-1] + 1] if bad[-1] + 1 < len(t) else np.inf)
    out["d_tail"] = d[-int(2 * T0 / dt):].max()
    out["dphi"] = wrap(PH[j[-1]] - (phi + T_end / T0) % 1.0)
    reg = regime_series(Y[1], Y[5]); n_pert = int((np.diff(reg) != 0).sum())
    n_exp = S_total(phi + t[-1] / T0 + out["dphi"]) - S_total(phi)
    out["d_regime"] = n_pert - int(n_exp)
    if out["d_tail"] < tol: out["outcome"] = "復帰"
    else:
        dt_ = describe_tail(t, Y); out["outcome"] = {"定常点": "定常点(レジーム固定)",
                                                    "固定振動": "新サイクル(レジーム固定の振動)",
                                                    "周期": "新サイクル"}.get(dt_["kind"], "要精査")
    return out


# =========================
# 4. 解析 A: パルスショック（強度 × 位相）
# =========================
def impulse_map(amps, phis):
    R = {}
    for A in amps:
        for ph in phis:
            o = respond(ph, A)
            R[(A, ph)] = dict(outcome=o["outcome"], dphi=o.get("dphi", np.nan),
                              t_rec=o.get("t_rec", np.nan), d_regime=o.get("d_regime", 0))
    return R


# =========================
# 5. 解析 B: 持続ショック
# =========================
def attractor_under(s, T_periods=25, phi=0.3):
    o = respond(phi, 0.0, T_end=T_periods * T0, u=(s, 0.0, 1e9))
    if o["escaped"]: return dict(kind="逃走")
    return describe_tail(o["t"], o["Y"], tail_periods=8)


def bisect_critical(lo, hi, tol):
    while abs(hi - lo) > tol:                     # lo: サイクル側, hi: 定常点側
        mid = 0.5 * (lo + hi)
        if attractor_under(mid)["kind"] == "周期": lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def find_critical(sign, tol, s0=0.02, grow=1.5, s_max=20.0):
    """s=0 から sign 方向へ幾何級数で s を増やし、サイクルでなくなる最初の点を括ってから二分法。
    返り値: (臨界 s*, 越えた先の到達先の種類)"""
    lo, s = 0.0, s0
    while s < s_max:
        k = attractor_under(sign * s)["kind"]
        if k != "周期":
            return bisect_critical(sign * lo, sign * s, tol), k
        lo, s = s, s * grow
    return np.nan, "臨界なし"


def relax(y0, s, T):
    sol = solve_ivp(make_rhs(p, s), (0, T), y0, method="LSODA", rtol=1e-9, atol=1e-11,
                    max_step=0.5, t_eval=np.arange(0, T, 0.5))
    return sol.t, sol.y


def hysteresis_lower(s_hi, s_lock, tol):
    """ロック状態(s=s_lock で到達)から s を下げて、ロックが保たれる下限 s_low を探す"""
    _, Y = relax(GAM[int(0.3 * N)], s_lock, 20 * T0); ylock = Y[:, -1]
    lo, hi = 0.0, s_hi
    if describe_tail(*relax(ylock, s_hi * 0.99, 30 * T0), 8)["kind"] == "周期":
        return np.nan                             # 臨界直下でロックが保たれない = ヒステリシスなし
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); t2, Y2 = relax(ylock, mid, 30 * T0)
        if describe_tail(t2, Y2, 8)["kind"] != "周期": hi = mid
        else: lo = mid
    return hi


# =========================
# 6. 図
# =========================
def setup_font():
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        try: fm.fontManager.addfont(f)
        except Exception: pass
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]


def make_figure(path, imp, amps, phis, s_scan, s_crit_neg, s_crit_pos, s_low, examples):
    setup_font()
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    # (a) 基準サイクル
    ax = fig.add_subplot(gs[0, 0]); tt = PH * T0
    reg = regime_series(np.tile(GAM[:, 1], 2), np.tile(GAM[:, 5], 2))[N:]
    y_lo, y_hi = GAM[:, 1].min() - 0.06 * SCALE[1], GAM[:, 1].max() + 0.12 * SCALE[1]   # 基準サイクルの r の範囲
    ax.fill_between(tt, y_lo, y_hi, where=reg > 0, color="#fdd0a2", alpha=0.5, lw=0, label="活動レジーム")
    ax.fill_between(tt, y_lo, y_hi, where=reg < 0, color="#c6dbef", alpha=0.5, lw=0, label="収縮レジーム")
    ax.plot(tt, GAM[:, 1], color="#08519c", label="r"); ax.plot(tt, GAM[:, 5], color="#cb181d", label="Λ")
    ax.set_ylim(y_lo, y_hi); ax.set_xlabel("時間"); ax.set_title(f"(a) 基準サイクル  T0={T0:.1f}")
    ax.text(0.02, 0.04, "Floquet |μ|=" + ", ".join(f"{m:.2g}" for m in np.sort(np.abs(MU))[::-1][:3]) + "…",
            transform=ax.transAxes, fontsize=8); ax.legend(fontsize=7, loc="upper right", ncol=2)

    # (b) 位相シフトのヒートマップ
    ax = fig.add_subplot(gs[0, 1])
    M = np.array([[imp[(A, ph)]["dphi"] for ph in phis] for A in amps])
    slip = np.array([[imp[(A, ph)]["d_regime"] != 0 for ph in phis] for A in amps])
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   extent=[0, 1, -0.5, len(amps) - 0.5])
    for i in range(len(amps)):
        for j in range(len(phis)):
            if slip[i, j]: ax.plot((j + 0.5) / len(phis), i, "k.", ms=4)
    ax.set_yticks(range(len(amps))); ax.set_yticklabels([f"{a:+g}" for a in amps], fontsize=7)
    for x in SW_PHASES: ax.axvline(x, color="k", ls="--", lw=0.8)
    ax.set_xlabel("ショックの位相 φ（破線=基準のレジーム切替）"); ax.set_ylabel("ε へのジャンプ量 A")
    ax.set_title("(b) 漸近位相シフト Δφ（●=サイクル・スリップ）"); plt.colorbar(im, ax=ax, fraction=0.046)

    # (c) 位相応答曲線
    ax = fig.add_subplot(gs[0, 2])
    cand = [(1, "#b2182b"), (-1, "#2166ac"), (4, "#ef8a62"), (-4, "#67a9cf")]
    if not any(A in amps for A, _ in cand):                       # 粗い格子(--quick)向けの代替
        cand = [(2, "#b2182b"), (-2, "#2166ac"), (8, "#ef8a62"), (-8, "#67a9cf")]
    for A, c in cand:
        if A in amps: ax.plot(phis, [imp[(A, ph)]["dphi"] for ph in phis], "o-", ms=3, color=c, label=f"A={A:+d}")
    for x in SW_PHASES: ax.axvline(x, color="k", ls="--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("ショックの位相 φ"); ax.set_ylabel("Δφ")
    ax.set_title("(c) 位相応答曲線（負のショックで不連続）"); ax.legend(fontsize=7)

    # (d) 回復時間
    ax = fig.add_subplot(gs[1, 0])
    for sign, c, lab in [(1, "#b2182b", "A>0"), (-1, "#2166ac", "A<0")]:
        aa = [a for a in amps if a * sign > 0]; aa = sorted(aa, key=abs)
        med = [np.median([imp[(a, ph)]["t_rec"] for ph in phis]) / T0 for a in aa]
        mx = [np.max([imp[(a, ph)]["t_rec"] for ph in phis]) / T0 for a in aa]
        ax.plot([abs(a) for a in aa], med, "o-", color=c, label=lab + " 中央値")
        ax.plot([abs(a) for a in aa], mx, "s--", color=c, alpha=0.5, label=lab + " 最大")
    ax.set_xscale("log"); ax.set_xlabel("|A|"); ax.set_ylabel("Γ 復帰までの時間（周期）")
    ax.set_title("(d) 回復時間（全て『復帰』）"); ax.legend(fontsize=7)

    # (e) 持続ショックの分岐図
    ax = fig.add_subplot(gs[1, 1])
    sw = [(s, d) for s, d in s_scan if d["kind"] == "周期"]; lk = [(s, d) for s, d in s_scan if d["kind"] == "固定振動"]
    ss = [s for s, d in sw]; pr = [d["period"] / T0 for s, d in sw]
    ax.plot(ss, pr, "o-", color="#08519c", label="切替を伴うサイクル（周期比）")
    if lk: ax.plot([s for s, d in lk], [d["period"] / T0 for s, d in lk], "^", color="#6a3d9a", label="レジーム固定の小振動（周期比）")
    ax.axvspan(-1e3, s_crit_neg, color="#fdd0a2", alpha=0.6, label="活動レジーム固定")
    ax.axvspan(s_crit_pos, 1e3, color="#c6dbef", alpha=0.6, label="収縮レジーム固定")
    if np.isfinite(s_low):
        ax.axvspan(s_low, s_crit_pos, facecolor="none", hatch="///", edgecolor="gray", label="共存（ヒステリシス）")
    ax.axvline(s_crit_neg, color="k", lw=0.8); ax.axvline(s_crit_pos, color="k", lw=0.8)
    sm = 1.6 * max(abs(s_crit_neg), abs(s_crit_pos)); ax.set_xlim(-sm, sm)
    allp = pr + [d["period"] / T0 for s, d in lk]
    ax.set_ylim(0, 1.15 * max([1.0] + allp)); ax.set_xlabel("持続入力 s"); ax.set_ylabel("周期 / T0")
    ax.set_title(f"(e) 新サイクルへの移行  s*={s_crit_neg:+.3f}, {s_crit_pos:+.3f}"); ax.legend(fontsize=6.5, loc="upper center")

    # (f) 例
    sub = gs[1, 2].subgridspec(3, 1, hspace=0.55)
    titles = ["(f-1) パルス A=+4: " + examples[0][0]["outcome"] + "（位相 Δφ=%+.2f）" % examples[0][0].get("dphi", np.nan),
              "(f-2) 持続 s=%+.3f: サイクルが変形" % examples[1][2],
              "(f-3) 持続 s=%+.3f を 10 周期→解除" % examples[2][2]]
    for k, (o, hold, _s) in enumerate(examples):
        a = fig.add_subplot(sub[k]); tmax = (20 if k == 2 else 10) * T0; m = o["t"] <= tmax
        base = solve_ivp(RHS0, (0, tmax), GAM[int(o["phi0"] * N)], method="LSODA", rtol=1e-9, atol=1e-11,
                         max_step=0.5, t_eval=o["t"][m])
        a.plot(base.t, base.y[1], color="gray", lw=1, label="無摂動")
        a.plot(o["t"][m], o["Y"][1][m], color="#08519c", lw=1, label="応答 r")
        if hold: a.axvspan(0, hold * T0, color="#fdd0a2", alpha=0.4, lw=0)
        a.set_title(titles[k], fontsize=8); a.set_ylim(GAM[:, 1].min() - 0.06 * SCALE[1], GAM[:, 1].max() + 0.12 * SCALE[1])
        a.tick_params(labelsize=7)
        if k == 0: a.legend(fontsize=6, loc="lower right", ncol=2)
    _dist = "".join(f", {n}={v:g}" for n, v in (("B3", B3_VAL), ("B4", B4_VAL), ("B5", B5_VAL)) if v)
    fig.suptitle(f"統合版 ESCore (b={B_VAL:g}, δ={DELTA_U:g}{_dist}): ε ショックへの応答とレジーム遷移", fontsize=13)
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


# =========================
# 実行
# =========================
if __name__ == "__main__":
    print(f"基準サイクル: T0={T0:.4f}, Floquet |μ|={np.round(np.sort(np.abs(MU))[::-1][:3], 4)} (…残りは≈0)")
    print(f"  範囲: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, SCALE)) + f"; 1周期の切替数={NSW}, 切替位相={np.round(SW_PHASES, 3)}")

    # A. パルス
    amps = [-8, -2, -0.5, 0.5, 2, 8] if QUICK else [-32, -16, -8, -4, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, 4, 8, 16, 32]
    phis = np.linspace(0, 1, 8 if QUICK else 20, endpoint=False)
    t0 = time.time(); imp = impulse_map(amps, phis)
    print(f"\n[A] パルスショック {len(amps)}強度×{len(phis)}位相 = {len(amps) * len(phis)} 通り ({time.time() - t0:.0f}s)")
    from collections import Counter
    print("  結果:", dict(Counter(v["outcome"] for v in imp.values())))
    print("  A       位相シフト範囲      回復(周期) 中央値/最大   サイクル・スリップ")
    for A in amps:
        rr = [imp[(A, ph)] for ph in phis]; dp = np.array([r["dphi"] for r in rr]); tr = np.array([r["t_rec"] for r in rr]) / T0
        print(f"  {A:+6.1f}  [{dp.min():+.3f},{dp.max():+.3f}]      {np.median(tr):4.1f}/{tr.max():4.1f}            {sum(1 for r in rr if r['d_regime'] != 0)}/{len(rr)}")

    # B. 持続
    tol = 0.01 if QUICK else 0.002
    s_neg, kind_neg = find_critical(-1, tol); s_pos, kind_pos = find_critical(+1, tol)
    S_CAP = 20.0
    s_neg_p = s_neg if np.isfinite(s_neg) else -S_CAP; s_pos_p = s_pos if np.isfinite(s_pos) else S_CAP
    s_lock = 1.3 * s_pos_p
    s_low = hysteresis_lower(0.99 * s_pos_p, s_lock, tol * 2) if np.isfinite(s_pos) else np.nan
    print(f"\n[B] 持続ショック: レジーム切替サイクルが消える臨界 s*− ≈ {s_neg:+.3f} (越えると: {kind_neg}), "
          f"s*+ ≈ {s_pos:+.3f} (越えると: {kind_pos}); ロックの保持下限 s_low ≈ {s_low:.3f}")
    n_s = 6 if QUICK else 16
    s_list = list(np.linspace(0.97 * s_neg_p, 0.97 * s_pos_p, n_s))
    outside = [1.5 * s_neg_p, 4 * s_neg_p, 1.5 * s_pos_p, 4 * s_pos_p]        # 臨界の外側
    s_scan = [(s, attractor_under(s)) for s in sorted(s_list + outside)]
    print("  s        到達先")
    for s, d in s_scan:
        if d["kind"] in ("周期", "固定振動"):
            tag = "周期(切替あり)" if d["kind"] == "周期" else "固定振動(切替なし)"
            print(f"  {s:+7.3f}  {tag} T={d['period']:.1f} ({d['period'] / T0:.2f}倍), r∈[{d['r_lo']:+.2f},{d['r_hi']:+.2f}]")
        else: print(f"  {s:+7.3f}  {d['kind']}")
    if np.isfinite(s_low):
        print(f"  ヒステリシス: s∈[{s_low:.3f},{s_pos:.3f}] で切替サイクルとロック状態が共存")
    else:
        print("  ヒステリシス: 正側の臨界の直下でロック状態は保持されない（共存なし）")

    # 解除後の復帰
    print("\n[C] 持続ショック 10 周期 → 解除後")
    print("  s      結果  位相シフト  解除後の回復(周期)")
    for s in ([0.83 * s_neg_p, 1.2 * s_pos_p] if QUICK else [2.5 * s_neg_p, 0.83 * s_neg_p, 0.4 * s_pos_p, 1.2 * s_pos_p]):
        o = respond(0.3, 0.0, T_end=24 * T0, u=(s, 0.0, 10 * T0))
        print(f"  {s:+.3f}  {o['outcome']}   {o['dphi']:+.3f}     {(o['t_rec'] - 10 * T0) / T0:.1f}")

    s_ex1, s_ex2 = 0.83 * s_neg_p, 1.2 * s_pos_p
    examples = [(respond(0.3, 4.0, T_end=12 * T0), 0, 0.0),
                (respond(0.3, 0.0, T_end=12 * T0, u=(s_ex1, 0.0, 1e9)), 0, s_ex1),
                (respond(0.3, 0.0, T_end=24 * T0, u=(s_ex2, 0.0, 10 * T0)), 10, s_ex2)]
    make_figure(f"es_shock_response_integrated{SUFFIX}.png", imp, amps, phis, s_scan, s_neg_p, s_pos_p, s_low, examples)
    print(f"\n図: es_shock_response_integrated{SUFFIX}.png   (総時間 {time.time() - t_start:.0f}s)")
