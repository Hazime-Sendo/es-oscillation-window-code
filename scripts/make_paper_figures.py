"""
make_paper_figures.py — 論文用の全図（英語表記）を、保存済みの数値実験データと短いシミュレーションから生成する。

Fig.1  regime-switching cycle and response to sustained input          （既定条件の全面解析ログ + シミュレーション）
Fig.2  the oscillation window: waveform and mode shape                  （調和バランス解を初期値にした直接シミュレーション）
Fig.3  onset threshold b*: dependence on delta, B4, lambda               （es_window_bstar_results*.json, es_window_lamc_results.json）
Fig.4  harmonic balance: rho(A), predicted vs measured b*                （es_window_hb*.py の再計算 + 保存結果）
Fig.5  energy budget: small-amplitude limit vs window                    （es_window_energy.py の出力）
Fig.6  fundamental gain of U: waveform / exact describing function / power-law （es_window_paper_data.json）
Fig.7  U-path supply decomposition vs lambda                             （同上）
Fig.8  empirical formula vs numerical b*(lambda), with validation         （es_window_bstar_formula.py）

使い方: python make_paper_figures.py [出力ディレクトリ]      既定 ./paper_figures
"""
import os, sys, re, json, subprocess, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SRC = os.path.normpath(os.path.join(HERE, "..", "src")); sys.path.insert(0, SRC)
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "paper_figures")
os.makedirs(OUT, exist_ok=True)
ANALYSIS_LOG = os.environ.get("ES_DEFAULT_LOG", os.path.join(RESULTS, "default_analysis_log.txt"))

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7.5, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GRAY, C_PURPLE = "#1f5fa8", "#c0392b", "#2e8b57", "#e67e22", "#7f8c8d", "#7d3c98"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", name)


def jload(name):
    return json.load(open(os.path.join(DATA, name)))


def regime_series(r, lam, delta=0.3):
    d = r - lam; st = 1 if d[0] > 0 else -1; out = np.empty(len(d), int)
    for i, x in enumerate(d):
        if st == 1 and x < -delta: st = -1
        elif st == -1 and x > delta: st = 1
        out[i] = st
    return out


# ----------------------------------------------------------------------------------------------------------
def fig1():
    from scipy.integrate import solve_ivp
    from es_window_hb import coeffs, full_rhs
    C = coeffs(0.1, 0.05); rhs = full_rhs(C, 0.0)
    s = solve_ivp(rhs, (0, 6000), [0.2, 0.5, 0, 0.5, 0, 0, 0], method="LSODA", rtol=1e-10, atol=1e-12, max_step=0.5, dense_output=True)
    tt = np.arange(4000, 4400, 0.1); Y = s.sol(tt); r, tau, Lam = Y[1], Y[2], Y[5]
    lam_mid = 0.5 * (Lam.max() + Lam.min()); up = np.where((Lam[:-1] < lam_mid) & (Lam[1:] >= lam_mid))[0]
    t0i, t1i = up[-3], up[-2]; sl = slice(t0i, t1i + 1); t = tt[sl] - tt[t0i]; T0 = tt[t1i] - tt[t0i]
    reg = regime_series(r[sl], Lam[sl])
    txt0 = open(ANALYSIS_LOG, encoding="utf-8", errors="replace").read()
    T0_log = float(re.search(r"T0=([\d.]+)", txt0)[1])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    a = ax[0]
    a.fill_between(t, -6, 4.2, where=reg > 0, color="#fde3c8", lw=0, label="active regime")
    a.fill_between(t, -6, 4.2, where=reg < 0, color="#d6e6f5", lw=0, label="contraction regime")
    a.plot(t, r[sl], color=C_BLUE, label=r"$r$"); a.plot(t, Lam[sl], color=C_RED, label=r"$\Lambda$"); a.plot(t, tau[sl], color=C_GREEN, lw=0.9, ls="--", label=r"$\tau$")
    a.set_xlim(0, T0); a.set_ylim(-6, 4.2); a.set_xlabel("time"); a.set_ylabel("state"); a.set_title(f"(a) reference cycle, $T_0={T0_log:.1f}$"); a.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=0.8, handlelength=1.4)
    b = ax[1]
    txt = open(ANALYSIS_LOG, encoding="utf-8", errors="replace").read().replace("\x00", "")
    pts = [(float(m[1]), float(m[3])) for m in re.finditer(r"^\s+([-+]\d+\.\d+)\s+周期\(切替あり\) T=([\d.]+) \(([\d.]+)倍\)", txt, flags=re.M)]
    crit = re.search(r"s\*− ≈ ([-+]\d+\.\d+).*s\*\+ ≈ ([-+]\d+\.\d+).*s_low ≈ ([\d.]+)", txt)
    sm, sp, slow = float(crit[1]), float(crit[2]), float(crit[3])
    pts.sort(); b.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C_BLUE, ms=3, label="switching cycle: $T/T_0$")
    b.axvspan(-0.45, sm, color="#fde3c8", alpha=0.7, lw=0); b.axvspan(sp, 0.7, color="#d6e6f5", alpha=0.7, lw=0)
    b.axvspan(slow, sp, facecolor="none", hatch="////", edgecolor=C_GRAY, lw=0)
    b.text(-0.27, 2.3, "active-regime\nlock", fontsize=7, ha="center"); b.text(0.6, 2.3, "contraction-\nregime lock", fontsize=7, ha="center")
    b.text(0.5 * (slow + sp), 0.25, "bistable", fontsize=7, ha="center", rotation=90)
    b.set_xlim(-0.45, 0.7); b.set_ylim(0, 2.9); b.set_xlabel("sustained input $s$"); b.set_ylabel(r"$T/T_0$")
    b.set_title(rf"(b) sustained input ($s^*_-={sm:+.3f}$, $s^*_+={sp:+.3f}$)")
    b.legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig1_cycle_and_sustained_input")


# ----------------------------------------------------------------------------------------------------------
def fig2():
    from scipy.integrate import solve_ivp
    from es_window_hb import coeffs, full_rhs, rho_curve
    b0, dl, s0 = 0.15, 0.05, 0.69
    out = rho_curve(b0, dl, s0, A_list=[0.1, 0.3, 0.6, 1.0, 1.457]); d = out["curve"][-1]
    Z = d["Z"]; y0 = np.array([Z[0, 0], d["rbar"] + d["A"], Z[1, 0], Z[2, 0], Z[3, 0], Z[4, 0], Z[5, 0]])
    C = coeffs(b0, dl); rhs = full_rhs(C, s0)
    sol = solve_ivp(rhs, (0, 4500), y0, method="LSODA", rtol=1e-10, atol=1e-12, max_step=0.2, dense_output=True)
    t = np.arange(0, 4500, 0.2); Y = sol.sol(t)
    names = ["L", "r", r"$\tau$", "w", "G", r"$\Lambda$", r"$\varepsilon$"]; keys = ["L", "r", "tau", "w", "G", "Lam", "eps"]
    # 減衰する比較例: b=0.10, s=0.545（窓の外）
    C2 = coeffs(0.10, dl); sol2 = solve_ivp(full_rhs(C2, 0.545), (0, 4500), y0 * 1.0, method="LSODA", rtol=1e-10, atol=1e-12, max_step=0.2, dense_output=True)
    Y2 = sol2.sol(t)
    fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.32)
    a = fig.add_subplot(gs[0, :])
    m = (t > 3800) & (t < 4100)
    a.plot(t[m] - 3800, Y[1][m], color=C_BLUE, label=r"$r$ ($b=0.15$, $s=0.69$)"); a.plot(t[m] - 3800, Y[5][m] * 1.0, color=C_RED, label=r"$\Lambda$")
    a.plot(t[m] - 3800, Y[2][m], color=C_GREEN, lw=0.9, ls="--", label=r"$\tau$")
    a.set_ylim(-3.4, 2.2); a.set_xlabel("time since $t=3800$"); a.set_ylabel("state"); a.set_title(r"(a) persistent oscillation without regime switching"); a.legend(ncol=3, loc="lower center")
    a = fig.add_subplot(gs[1, 0])
    env = lambda y: np.array([np.ptp(y[(t >= k * 500) & (t < (k + 1) * 500)]) for k in range(9)])
    a.plot(np.arange(9) * 500 + 250, env(Y[1]), "o-", color=C_BLUE, label=r"window ($b=0.15$)"); a.plot(np.arange(9) * 500 + 250, env(Y2[1]), "s-", color=C_GRAY, label=r"no window ($b=0.10$)")
    a.set_yscale("log"); a.set_ylim(1e-8, 10); a.set_xlabel("time"); a.set_ylabel(r"peak-to-peak of $r$"); a.set_title("(b) persistence vs decay"); a.legend()
    a = fig.add_subplot(gs[1, 1])
    mm = (t >= 3000); tt_, Yt = t[mm], Y[:, mm]
    r_ = Yt[1]; up = np.where((r_[:-1] < r_.mean()) & (r_[1:] >= r_.mean()))[0]; T = float(np.mean(np.diff(tt_[up][-8:]))); om = 2 * np.pi / T
    sel = (tt_ >= tt_[up[-12]]) & (tt_ < tt_[up[-12]] + 10 * T)
    c1 = []
    for i in range(7):
        y = Yt[i][sel]; tt2 = tt_[sel]; c1.append(2 * np.mean(y * np.exp(-1j * om * tt2)))
    c1 = np.array(c1); rel = c1 / c1[1]; order = [1, 2, 3, 0, 4, 5, 6]
    x = np.arange(7)
    a.bar(x, [abs(rel[i]) for i in order], color=C_BLUE, width=0.6)
    for xi, i in zip(x, order): a.text(xi, abs(rel[i]) + 0.04, (f"{np.degrees(np.angle(rel[i])):+.0f}°" if abs(np.degrees(np.angle(rel[i]))) > 0.5 else "0°"), ha="center", fontsize=7)
    a.set_xticks(x); a.set_xticklabels([names[i] for i in order]); a.set_ylabel("amplitude ratio to $r$"); a.set_ylim(0, 1.25)
    a.set_title(f"(c) first harmonic ($T={T:.1f}$); labels: phase vs $r$")
    fig.tight_layout(); save(fig, "fig2_window_oscillation")


# ----------------------------------------------------------------------------------------------------------
def fig3():
    base = jload("es_window_bstar_results.json"); lam_r = jload("es_window_lamc_results.json")
    r5 = jload("es_window_bstar_results_B4_-0.5.json"); r6 = jload("es_window_bstar_results_B4_0.5.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a = ax[0]; ds = [0.01, 0.05, 0.2]
    for B4, R, col in ((-0.5, r5, C_ORANGE), (0.0, base, C_BLUE), (0.5, R6 := r6, C_GREEN)):
        bs = [R[f"{d:g}"]["b_present"] for d in ds]; a.plot(ds, bs, "o-", color=col, label=("$B_4=0$" if B4 == 0 else f"$B_4={B4:+g}$"))
    dd = np.linspace(0.008, 0.25, 100); a.plot(dd, 0.105 + 0.126 * np.sqrt(dd), ":", color=C_BLUE, lw=1, label=r"$0.105+0.126\sqrt{\delta}$ ($B_4=0$)")
    a.axhline(0.15, color=C_GRAY, ls="--", lw=0.7); a.text(0.0105, 0.1515, "$b=0.15$", fontsize=7, color=C_GRAY)
    a.set_xscale("log"); a.set_xlabel(r"smoothing width $\delta$"); a.set_ylabel(r"onset threshold $b^*$"); a.set_title(r"(a) $b^*(\delta)$, $\lambda=2.25$"); a.legend(loc="upper left")
    a = ax[1]
    pts = {}
    for k, v in lam_r.items():
        if k.startswith("B|"):
            _, d, l = k.split("|"); pts.setdefault(float(d), []).append((float(l), 0.5 * (v["b_absent"] + v["b_present"])))
    for d in (0.01, 0.05, 0.2):
        pts.setdefault(d, []).append((2.25, 0.5 * (base[f"{d:g}"]["b_absent"] + base[f"{d:g}"]["b_present"])))
    for d, col in zip((0.01, 0.05, 0.2), (C_ORANGE, C_BLUE, C_GREEN)):
        p = sorted(pts[d]); x = np.array([1 / q[0] for q in p]); y = np.array([q[1] for q in p]); a.plot(x, y, "o", color=col, label=rf"$\delta={d:g}$")
        A = np.column_stack([np.ones_like(x), x]); c = np.linalg.lstsq(A, y, rcond=None)[0]; xx = np.linspace(0.2, 1.05, 20); a.plot(xx, c[0] + c[1] * xx, "-", color=col, lw=0.9)
    a.set_xlabel(r"$1/\lambda$"); a.set_ylabel(r"$b^*$"); a.set_title(r"(b) $b^*$ vs $1/\lambda$"); a.legend()
    fig.tight_layout(); save(fig, "fig3_onset_threshold")


# ----------------------------------------------------------------------------------------------------------
def fig4():
    from es_window_hb import rho_curve
    hb = jload("es_window_hb_bstar_results.json"); hl = jload("es_window_hb_lambda_results.json"); big = jload("es_window_lam_large_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    a = ax[0]; rows = {round(r["b"], 4): r["s"] for r in hb["0.05"]["rows"]}
    A_LIST = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
    for b, col in ((0.120, C_BLUE), (0.1335, C_GREEN), (0.145, C_RED)):
        o = rho_curve(b, 0.05, rows[round(b, 4)], A_list=A_LIST); cv = [d for d in o["curve"] if d["ok"]]
        a.plot([d["A"] for d in cv], [d["rho"] for d in cv], "o-", ms=3, color=col, label=f"$b={b}$")
    a.axhline(0, color="k", lw=0.7); a.set_xlabel("amplitude $A$"); a.set_ylabel(r"effective growth rate $\rho(A)$"); a.set_title(r"(a) $\rho(A)$, $\delta=0.05$"); a.legend(loc="lower right")
    a = ax[1]; ds = [0.01, 0.05, 0.2]
    a.plot(ds, [hb[f"{d:g}"]["b_meas"] for d in ds], "ko-", label="direct simulation"); a.plot(ds, [hb[f"{d:g}"]["b_pred"] for d in ds], "s--", color=C_ORANGE, label="harmonic balance")
    a.set_xscale("log"); a.set_xlabel(r"$\delta$"); a.set_ylabel(r"$b^*$"); a.set_title(r"(b) $b^*(\delta)$: prediction"); a.legend(loc="upper left")
    a = ax[2]
    L, m, h = [], [], []
    for k, v in hl.items(): L.append(float(k)); m.append(v["b_meas"]); h.append(v["b_pred"])
    for k, v in big.items(): L.append(float(k)); m.append(0.5 * (v["b_absent"] + v["b_present"])); h.append(v["b_hb"])
    o = np.argsort(L); L = np.array(L)[o]; m = np.array(m)[o]; h = np.array(h)[o]
    a.plot(L, m, "ko-", label="direct simulation"); a.plot(L, h, "s--", color=C_ORANGE, label="harmonic balance"); a.set_xscale("log"); a.set_xlabel(r"$\lambda$"); a.set_ylabel(r"$b^*$")
    a.set_title(r"(c) $b^*(\lambda)$: prediction, $\delta=0.05$"); a.legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig4_harmonic_balance")


# ----------------------------------------------------------------------------------------------------------
def fig5():
    res = subprocess.run([sys.executable, os.path.join(HERE, "es_window_energy.py"), "0.15", "0.05", "0.69", "2.25"], capture_output=True, text=True, cwd=HERE).stdout
    blk = res.split("小振幅の極限")[-1]
    rows = re.findall(r"^\s+(.+?)\s+([+-]\d\.\d{4})\s+([+-]\d\.\d{4})\s+([+-]\d\.\d{4})\s*$", blk, flags=re.M)
    rows = [r for r in rows if not r[0].startswith("合計")][:11]
    names = [r"$a\,r$", r"$-qL$", r"$(p-k)rL$", r"$-d\,e^{r}$", r"$\kappa\tanh(\cdot)$ in $\dot r$", r"$g_G G$", r"$e_r\varepsilon$",
             r"$b\tau\leftarrow r$", r"$b\tau\leftarrow U$", r"$b\tau\leftarrow$ regime", r"$b\tau\leftarrow\varepsilon$"]
    lin = np.array([float(r[1]) for r in rows]); win = np.array([float(r[2]) for r in rows])
    keep = [i for i in range(len(rows)) if abs(lin[i]) > 5e-4 or abs(win[i]) > 5e-4]
    fig, ax = plt.subplots(figsize=(6.6, 3.0)); x = np.arange(len(keep)); w = 0.38
    ax.bar(x - w / 2, lin[keep], w, color=C_GRAY, label=f"small-amplitude limit (sum $=${lin.sum():+.3f}$=2\\,$Re$\\lambda$)")
    ax.bar(x + w / 2, win[keep], w, color=C_BLUE, label=f"oscillation window (sum $=${win.sum():+.3f})")
    ax.axhline(0, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels([names[i] for i in keep], rotation=25, ha="right")
    ax.set_ylabel(r"mean power per unit energy, $P_i/\langle E\rangle$"); ax.set_title(r"Energy budget of the $r$–$L$ oscillation ($b=0.15$, $\delta=0.05$, $s=0.69$)"); ax.legend(loc="lower right")
    fig.tight_layout(); save(fig, "fig5_energy_budget")
    json.dump(dict(rows=[[names[i], lin[i], win[i]] for i in range(len(rows))]), open(os.path.join(OUT, "fig5_data.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------
def fig6():
    D = jload("es_window_paper_data.json"); L = np.array([q["lam"] for q in D])
    g_hb = np.array([abs(q["g_hb"]) for q in D]); g_ex = np.array([q["g_exact"] for q in D]); g_pl = np.array([q["g_pl"] for q in D]); h2 = np.array([q["harm2"] for q in D])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a = ax[0]; a.plot(L, g_pl, "^-", color=C_RED, label=r"power-law approximation ($|x|^{0.88}$)"); a.plot(L, g_ex, "s-", color=C_GREEN, label="exact describing function (cosine input)")
    a.plot(L, g_hb, "o-", color=C_BLUE, label="waveform (harmonic balance)"); a.set_xscale("log"); a.set_xlabel(r"loss aversion $\lambda$"); a.set_ylabel(r"fundamental gain $g=U_1/x_1$")
    a.set_ylim(0.8, 3.4); a.set_title("(a) gain of the value function"); a.legend(loc="upper left")
    for l, p, h in zip(L, g_pl, g_hb):
        if l in (1.0, 20.0): a.annotate(f"+{100 * (p / h - 1):.0f}%", (l, p), textcoords="offset points", xytext=((14 if l == 1.0 else -10), 5), ha="center", fontsize=7, color=C_RED)
    a = ax[1]; a.plot(L, g_ex / g_hb, "s-", color=C_GREEN, label=r"$g_{\rm exact}/g_{\rm waveform}$"); a.plot(L, g_pl / g_hb, "^-", color=C_RED, label=r"$g_{\rm power}/g_{\rm waveform}$")
    a.set_xscale("log"); a.set_xlabel(r"$\lambda$"); a.set_ylabel("ratio"); a.axhline(1, color="k", lw=0.6)
    a2 = a.twinx(); a2.plot(L, h2, "d:", color=C_PURPLE, label=r"2nd harmonic $|x_2/x_1|$"); a2.set_ylabel(r"$|x_2/x_1|$", color=C_PURPLE); a2.set_ylim(0, 0.5); a2.spines["right"].set_visible(True)
    a.set_title("(b) deviation grows with harmonic content"); h1, l1 = a.get_legend_handles_labels(); h2_, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2_, l1 + l2, loc="upper left")
    fig.tight_layout(); save(fig, "fig6_gain_of_U")


# ----------------------------------------------------------------------------------------------------------
def fig7():
    D = jload("es_window_paper_data.json"); L = np.array([q["lam"] for q in D]); SU = np.array([q["SU"] for q in D]); P2 = np.array([q["P2"] for q in D])
    Dir = np.array([q["b"] * q["P1"] for q in D])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    xx = np.linspace(0.04, 1.05, 50)
    a = ax[0]
    for y, col, lab, mk in ((SU, C_BLUE, r"$S_U$ (required supply)", "o"), (P2, C_GREEN, r"$P_2$ (via $G$)", "s"), (Dir, C_ORANGE, r"$b^*P_1$ (direct)", "^")):
        x = 1 / L; A = np.column_stack([np.ones_like(x), x]); c = np.linalg.lstsq(A, y, rcond=None)[0]
        a.plot(x, y, mk, color=col, label=lab); a.plot(xx, c[0] + c[1] * xx, "-", color=col, lw=0.9)
    a.axhspan(0.123, 0.133, color="#eeeeee", zorder=0); a.text(0.55, 0.1385, "onset: U-path supply fills the deficit $\\approx 0.13$", fontsize=7, color=C_GRAY, ha="center")
    a.set_xlabel(r"$1/\lambda$"); a.set_ylabel(r"supply rate per unit energy"); a.set_ylim(0, 0.15); a.set_title("(a) all components are affine in $1/\\lambda$"); a.legend(loc="center", bbox_to_anchor=(0.55, 0.42))
    a = ax[1]
    a.semilogx(L, SU, "o-", color=C_BLUE, label=r"$S_U$"); a.semilogx(L, P2, "s-", color=C_GREEN, label=r"$P_2$"); a.semilogx(L, Dir, "^-", color=C_ORANGE, label=r"$b^*P_1$"); a.semilogx(L, SU - P2, "k:", label=r"$S_U-P_2$")
    a.set_xlabel(r"$\lambda$"); a.set_ylabel("supply rate"); a.set_title(r"(b) $b^*P_1=S_U-P_2$ (identity at threshold)"); a.legend(loc="center right")
    fig.tight_layout(); save(fig, "fig7_supply_decomposition")


# ----------------------------------------------------------------------------------------------------------
def fig8():
    import es_window_bstar_formula as F
    fig = plt.figure(figsize=(6.4, 4.3)); gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
    a = fig.add_subplot(gs[0]); r = fig.add_subplot(gs[1], sharex=a)
    ll = np.logspace(0, np.log10(20.5), 200); a.plot(ll, [F.b_star(l) for l in ll], "-", color=C_BLUE, lw=1.2, label="empirical formula")
    for pts, col, mk, lab in ((F.FIT_POINTS, "k", "o", "fit points (7)"), (F.VALIDATION_POINTS, C_RED, "D", "independent validation (4)")):
        lam = np.array(list(pts)); lo = np.array([v[0] for v in pts.values()]); hi = np.array([v[1] for v in pts.values()]); mid = 0.5 * (lo + hi)
        a.errorbar(lam, mid, yerr=0.5 * (hi - lo), fmt=mk, color=col, ms=4, capsize=2, label=lab + ": bracket midpoint")
        r.errorbar(lam, [F.b_star(l) - m for l, m in zip(lam, mid)], yerr=0.5 * (hi - lo), fmt=mk, color=col, ms=4, capsize=2)
    a.set_ylabel(r"$b^*(\lambda)$"); a.legend(loc="upper right"); plt.setp(a.get_xticklabels(), visible=False)
    a.set_title(r"Empirical formula vs numerical onset threshold ($\delta=0.05$)")
    r.axhline(0, color="k", lw=0.6); r.set_xscale("log"); r.set_xlabel(r"$\lambda$"); r.set_ylabel("formula $-$ midpoint"); r.set_ylim(-0.0005, 0.0005)
    save(fig, "fig8_empirical_formula")


if __name__ == "__main__":
    which = sys.argv[2:] or ["1", "2", "3", "4", "5", "6", "7", "8"]
    for w in which:
        globals()["fig" + w]()
