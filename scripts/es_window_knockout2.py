"""
es_window_knockout2.py — 窓を支える結合の特定（結合を切る実験・改訂版）

改訂の理由: es_window_knockout.py は「窓の振動状態から、s を固定したまま結合を切る」判定だった。結合を切ると
s*+（切替サイクルが消える s）が動き、窓の位置も動くので、「窓が消えた」と「窓が s の範囲から外れた」を区別できない。
本スクリプトは、切った各モデルについて es_window_eval.py と同じ判定（s を、そのモデルの s*+ からの相対位置で測る）で、
窓の有無を b=0.15 と b=0.20 で調べる。
  b=0.15 で窓なし・b=0.20 で窓あり → その結合は窓の存在に必須ではない（b* を上げるだけ）
  どちらも窓なし                    → 必須の候補（さらに b を上げて確認）
  切替サイクルが存在しない/暴走     → 判定不能

使い方: python es_window_knockout2.py [delta]    （約 15 分）
"""
import sys, os, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
DELTA = sys.argv[1] if len(sys.argv) > 1 else "0.05"
GRID = "0.002,0.004,0.008,0.015,0.03,0.06"
VARIANTS = [
    ("基準（切除なし）",                    ""),
    ("U を対称にする (損失回避 λ=1)",        "lam_loss=1"),
    ("U→τ を切る (χ=0)",                    "chi=0"),
    ("w を固定 (γ_w=0)",                    "gamma_w=0"),
    ("G→r を切る",                          "g_to_r=0"),
    ("τ 側のレジーム項を切る (κ_τ=0)",       "kappa_tau=0"),
    ("ε→r を切る (0.1→0)",                  "eps_to_r=0"),
    ("ε→τ を切る (0.2→0)",                  "B5=-1"),
    ("ε→Λ を切る",                          "gamma_eps_Lambda=0"),
    ("τ→ε を切る",                          "gamma_eps_tau=0"),
]


def evaluate(b, es_set):
    env = dict(os.environ, ES_SET=es_set)
    proc = subprocess.run([sys.executable, "-u", os.path.join(SRC, "es_window_eval.py"), f"{b:.4f}", DELTA, "0.001", GRID],
                          capture_output=True, text=True, cwd=SRC, env=env)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        return dict(present=None, note="実行失敗（切替サイクルが求まらない等）")
    d = json.loads(lines[-1])
    return d


def fmt(d):
    if d.get("present") is None:
        return "判定不能（臨界なし）"
    return f"窓あり(振幅 {d['amp_first']:.2f})" if d["present"] else "窓なし"


def run():
    out = {}
    print(f"δ={DELTA}: 窓の有無（s は各モデルの s*+ からの相対位置）")
    print("  切除した項                          b=0.15            b=0.20            b=0.25        判定")
    only = os.environ.get("KO_ONLY")
    for label, es_set in VARIANTS:
        if only and only not in label:
            continue
        t0 = time.time()
        row = {}
        for b in (0.15, 0.20):
            row[b] = evaluate(b, es_set)
        if row[0.15].get("present") is False and row[0.20].get("present") is False:
            row[0.25] = evaluate(0.25, es_set)
        p15, p20 = row[0.15].get("present"), row[0.20].get("present")
        p25 = row.get(0.25, {}).get("present")
        if p15 is None and p20 is None:
            verdict = "判定不能"
        elif p15:
            verdict = "窓は残る（b=0.15）"
        elif p20:
            verdict = "b* が 0.15〜0.20 に上がる（必須ではない）"
        elif p25:
            verdict = "b* が 0.20〜0.25 に上がる（必須ではない）"
        else:
            verdict = "b≤0.25 で窓なし（必須の候補）"
        print(f"  {label:34s}  {fmt(row[0.15]):16s}  {fmt(row[0.20]):16s}  " + (f"{fmt(row[0.25]):16s}" if 0.25 in row else " " * 16) + f"  {verdict}  [{time.time() - t0:.0f}s]", flush=True)
        out[label] = dict(es_set=es_set, results={str(b): dict(present=r.get("present"), amp=r.get("amp_first"), s_crit=r.get("s_crit")) for b, r in row.items()}, verdict=verdict)
    json.dump(out, open(os.path.join(DATA, "es_window_knockout2_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    run()
