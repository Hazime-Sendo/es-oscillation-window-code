"""
es_window_bstar.py — 「窓が立ち上がる臨界 b*(δ)」の探索と作図

各 δ について、正側の持続入力で現れる「切替なしの持続振動の窓」(es_window_eval.py) が初めて現れる b* を求める。
  - δ ごとの探索方向: 窓の有無が b で切り替わる場所を、粗い刻みで括ってから二分法（許容 0.0005）で詰める
  - b* の少し上（b*+0.005, +0.010, +0.015）で、窓の幅（上端の区間）と振幅を測る
結果は es_window_bstar_results.json に逐次保存し、es_window_bstar_curve.png に臨界曲線を描く。

使い方: python es_window_bstar.py                    （B4=0 の探索＋作図。約 20〜30 分）
        python es_window_bstar.py --B4 -0.5          （B4=−0.5 の探索。結果は es_window_bstar_results_B4_-0.5.json）
        python es_window_bstar.py --plot             （保存済みの JSON から B4=0 の図）
        python es_window_bstar.py --plot-B4          （B4=−0.5, 0, +0.5 の比較図 es_window_bstar_curve_B4.png）
"""
import sys, os, json, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
DELTAS_TO_RUN = [0.01, 0.05, 0.2]
B_TOL = 0.0005
GUESS = {0.01: 0.118, 0.05: 0.1335, 0.2: 0.1616}          # B4=0 で得た b*（探索の初期値）
B4 = float(sys.argv[sys.argv.index("--B4") + 1]) if "--B4" in sys.argv else 0.0


def res_path(b4):
    return os.path.join(DATA, "es_window_bstar_results.json" if b4 == 0 else f"es_window_bstar_results_B4_{b4:g}.json")


RES = res_path(B4)


def evaluate(b, delta):
    out = subprocess.run([sys.executable, "-u", os.path.join(SRC, "es_window_eval.py"), f"{b:.5f}", f"{delta:g}", "--B4", f"{B4:g}"],
                         capture_output=True, text=True, cwd=SRC).stdout.strip().splitlines()
    d = json.loads(out[-1])
    print(f"   b={b:.4f} δ={delta:g}: s*+={d['s_crit']:+.4f}, 窓={'あり' if d['present'] else 'なし' if d['present'] is not None else '臨界なし'}"
          + (f", 振幅 {d['amp_first']:.2f}, 上端 Δ∈[{d['width_lo']}, {d['width_hi']}]" if d.get("present") else ""), flush=True)
    return d


def save(res):
    json.dump(res, open(RES, "w"), ensure_ascii=False, indent=1)


def search(delta, start_b, step=0.01, b_min=0.03, b_max=0.30):
    """窓の有無が切り替わる b を括って二分法で詰める。窓ありなら下へ、窓なしなら上へ start_b から step 刻みで進む。
    返り値: (b_absent, b_present, log)"""
    b = start_b; d = evaluate(b, delta); log = [d]
    going_down = bool(d["present"])
    prev = b
    while True:
        b = round(b + (-step if going_down else step), 5)
        if b < b_min or b > b_max:
            return None, None, log
        prev_d = d; d = evaluate(b, delta); log.append(d)
        if bool(d["present"]) != going_down:              # 窓の有無が切り替わった
            lo_absent, hi_present = (b, prev_d["b"]) if going_down else (prev_d["b"], b)
            break
    while hi_present - lo_absent > B_TOL:
        mid = round(0.5 * (lo_absent + hi_present), 5); dm = evaluate(mid, delta); log.append(dm)
        if dm["present"]: hi_present = mid
        else: lo_absent = mid
    return lo_absent, hi_present, log


def run():
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    for delta in DELTAS_TO_RUN:
        key = f"{delta:g}"
        if key in res and res[key].get("done"):
            continue
        t0 = time.time(); print(f"\n===== δ={delta:g}: b* の探索 =====", flush=True)
        lo, hi, log = search(delta, GUESS[delta])
        entry = dict(b_absent=lo, b_present=hi, log=log)
        if hi is not None:
            entry["above"] = []
            for off in ((0.005, 0.010, 0.015) if B4 == 0 else (0.010,)):
                entry["above"].append(dict(offset=off, **evaluate(round(hi + off, 5), delta)))
        entry["done"] = True; entry["seconds"] = time.time() - t0
        res[key] = entry; save(res)
        print(f"  → δ={delta:g}: b* ∈ ({lo}, {hi}]  ({time.time() - t0:.0f}s)", flush=True)


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        try: fm.fontManager.addfont(f)
        except Exception: pass
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    res = json.load(open(RES))
    ds = sorted(float(k) for k in res if res[k].get("b_present") is not None)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    # (a) 臨界曲線 b*(δ)
    ax = axs[0]
    bs = [res[f"{d:g}"]["b_present"] for d in ds]; lo = [res[f"{d:g}"]["b_absent"] for d in ds]
    ax.errorbar(ds, bs, yerr=[np.array(bs) - np.array(lo), np.zeros(len(ds))], fmt="o-", color="#08519c", capsize=4)
    for d, b in zip(ds, bs): ax.annotate(f"{b:.4f}", (d, b), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(0.1, color="gray", ls=":", lw=0.8); ax.axhline(0.15, color="gray", ls=":", lw=0.8)
    ax.text(ds[0], 0.1015, "運用 b=0.1", fontsize=8, color="gray"); ax.text(ds[0], 0.1515, "運用上限 b=0.15", fontsize=8, color="gray")
    ax.fill_between(ds, bs, 0.3, color="#fdd0a2", alpha=0.35, label="窓あり（b>b*）")
    ax.fill_between(ds, 0.0, bs, color="#c6dbef", alpha=0.35, label="窓なし（b<b*）")
    ax.set_xscale("log"); ax.set_xlabel("δ（U の平滑化幅）"); ax.set_ylabel("b")
    ax.set_ylim(0.03, 0.31); ax.set_title("(a) 窓が立ち上がる臨界曲線 b*(δ)"); ax.legend(fontsize=8, loc="upper left")
    # (b) 窓の幅
    ax = axs[1]
    for d, c in zip(ds, ["#e6550d", "#08519c", "#31a354"]):
        e = res[f"{d:g}"]
        xs = [a["offset"] for a in e.get("above", []) if a.get("present")]
        ys = [0.5 * ((a["width_lo"] or 0) + (a["width_hi"] if a["width_hi"] is not None else 0.5)) for a in e.get("above", []) if a.get("present")]
        lo_ = [(a["width_lo"] or 0) for a in e.get("above", []) if a.get("present")]
        hi_ = [(a["width_hi"] if a["width_hi"] is not None else 0.5) for a in e.get("above", []) if a.get("present")]
        if xs:
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo_), np.array(hi_) - np.array(ys)], fmt="o-", color=c, capsize=3, label=f"δ={d:g}")
    ax.set_xlabel("b − b*"); ax.set_ylabel("窓の幅（s*+ からの上端 Δ、区間の中点と範囲）"); ax.set_title("(b) 窓の幅の立ち上がり"); ax.legend(fontsize=8)
    # (c) 窓の振幅
    ax = axs[2]
    for d, c in zip(ds, ["#e6550d", "#08519c", "#31a354"]):
        e = res[f"{d:g}"]
        xs = [a["offset"] for a in e.get("above", []) if a.get("present")]; ys = [a["amp_first"] for a in e.get("above", []) if a.get("present")]
        if xs: ax.plot(xs, ys, "o-", color=c, label=f"δ={d:g}")
    ax.set_xlabel("b − b*"); ax.set_ylabel("窓の振幅（s*+ の直上、r の幅）"); ax.set_title("(c) 窓の振幅（有限振幅で生まれる）"); ax.legend(fontsize=8)
    fig.suptitle("統合版 ESCore（B3=−0.3, B5=0, B4=0）: 切替なし持続振動の窓の臨界曲線", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(HERE, "es_window_bstar_curve.png"), dpi=140); plt.close(fig)
    print("作図: es_window_bstar_curve.png")


def plot_B4():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
        try: fm.fontManager.addfont(f)
        except Exception: pass
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = {-0.5: "#e6550d", 0.0: "#08519c", 0.5: "#31a354"}
    data = {}
    for b4 in (-0.5, 0.0, 0.5):
        p = res_path(b4)
        if os.path.exists(p): data[b4] = json.load(open(p))
    ax = axs[0]
    for b4, res in data.items():
        ds = sorted(float(k) for k in res if res[k].get("b_present") is not None)
        bs = [res[f"{d:g}"]["b_present"] for d in ds]; lo = [res[f"{d:g}"]["b_absent"] for d in ds]
        ax.errorbar(ds, bs, yerr=[np.array(bs) - np.array(lo), np.zeros(len(ds))], fmt="o-", color=colors[b4], capsize=3, label=f"B4={b4:+g}")
        for d, b in zip(ds, bs): ax.annotate(f"{b:.4f}", (d, b), textcoords="offset points", xytext=(5, 5), fontsize=8, color=colors[b4])
    ax.axhline(0.1, color="gray", ls=":", lw=0.8); ax.axhline(0.15, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("δ"); ax.set_ylabel("b*（窓が初めて現れる b）"); ax.set_title("(a) b*(δ) の B4 依存（点線: 運用 0.1 / 上限 0.15）"); ax.legend(fontsize=8)
    ax = axs[1]                                          # δ ごとの b*(B4)
    for d, mk in zip([0.01, 0.05, 0.2], ["o", "s", "^"]):
        xs = [b4 for b4 in data if data[b4].get(f"{d:g}", {}).get("b_present") is not None]
        ys = [data[b4][f"{d:g}"]["b_present"] for b4 in xs]; o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], mk + "-", label=f"δ={d:g}")
    ax.axhline(0.15, color="gray", ls=":", lw=0.8); ax.set_xlabel("B4"); ax.set_ylabel("b*"); ax.set_title("(b) b*(B4)（δ ごと）"); ax.legend(fontsize=8)
    ax = axs[2]                                          # 窓の直上の振幅
    for b4, res in data.items():
        ds = sorted(float(k) for k in res if res[k].get("b_present") is not None); amps = []
        for d in ds:
            last = [x for x in res[f"{d:g}"]["log"] if x.get("present")]
            amps.append(min(last, key=lambda x: x["b"])["amp_first"] if last else np.nan)
        ax.plot(ds, amps, "o-", color=colors[b4], label=f"B4={b4:+g}")
    ax.set_xscale("log"); ax.set_xlabel("δ"); ax.set_ylabel("b* の直上の窓の振幅（r の幅）"); ax.set_title("(c) 窓が生まれる振幅"); ax.legend(fontsize=8)
    fig.suptitle("統合版 ESCore（B3=−0.3, B5=0）: 窓の臨界曲線 b*(δ) の B4 依存", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(HERE, "es_window_bstar_curve_B4.png"), dpi=140); plt.close(fig)
    print("作図: es_window_bstar_curve_B4.png")


if __name__ == "__main__":
    if "--plot-B4" in sys.argv:
        plot_B4()
    else:
        if "--plot" not in sys.argv:
            run()
        if B4 == 0:
            plot()
