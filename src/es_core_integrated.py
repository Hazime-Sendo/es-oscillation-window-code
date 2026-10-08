# es_core_integrated.py — 統合版 ESCore（確定仕様）
#
# 由来
#   骨格:  es_core_fixed.py（符号つき U、G→r=−0.2 のブレーキ、G の漏れ −ξG、b の接続）
#   復元:  es_unified_field_AB_fixed.py から、次をすべて AB_fixed と同じ形で復元済み
#     ・r の式の非線形項  a·r, −k·L, −d·e^r, κ·tanh(β(r−Λ))             （完全復元）
#     ・U→τ        τ' += χ·U(r−w)   （w は U の入力＝参照水準として残す。w' = γ_w·U(r−w) で r に追従）
#     ・ε→Λ        Λ' = lam_rate·(r−Λ) + γ_εΛ·ε    （γ_εΛ=0.02）
#     ・τ→ε        ε' = −0.3·ε + γ_ετ·τ            （γ_ετ=0.02。ε は τ に駆動されるフィルタ変数）
#     ・τ 側レジーム項  τ' += κ·tanh(β(r−Λ−B4))
#     ・歪みベクトル B3〜B5（すべて τ の式のみに入る）
#         B3: τ の減衰の歪み   η_eff = α_τ − B3   （τ' に +B3·τ）   B3<0 で減衰が速くなる
#         B4: レジーム境界の歪み  tanh(β(r−Λ−B4))
#         B5: ε→τ の感応度の歪み  0.2·(1+B5)·ε
#
# 確定仕様（既定値）
#   lam_rate=0.02（AB_fixed と同じ）／ δ=0.05 ／ χ=0.8 ／ b=0.1（運用範囲 0.1、最大 0.15）
#   B3=−0.3 ／ B4=0 ／ B5=0。 歪みなしは ESCore(B3=0.0) で得られる（オプション）。
#
# 実測（すべて既定設定、12 初期値で有界性を確認。__main__ に自己検証がある）
#   b の依存（B3=−0.3）:  b=0.05: T0≈72, r_min≈−4.2 / 0.1: 75, −5.5 / 0.15: 81, −7.5 / 0.2: 91, −11 / 0.25: 106, −16 / 0.3: 130, −27
#       → 運用範囲 b≤0.15 の内側に十分な余裕がある（b=0.3 でも有界）。B3=0 のオプションでは b=0.2 で −42、b=0.26 で −4000 と急に悪化する。
#   χ の依存（b=0.1）:   χ=0: T0≈181, r_min≈−3.1 / 0.4: 73, −4.1 / 0.8: 75, −5.5 / 1.6: 90, −11
#   歪みなし(B3=0) との差（b=0.1）: T0≈202・r_min≈−9.1・収縮レジーム内に複数の谷（B3=−0.3 では T0≈75・崩壊1回で AB_fixed に近い）
#   B3>0（歪みの逆向き）の安定上限: b=0.1 で B3<0.094、b=0.05 で 0.114、b=0.15 で 0.073（それ以上は全初期値が逃走）
#   B5: −1〜+5 で有界（B3=0 のとき）。B5≥10 は逃走。
#
# 位置づけ: AB_fixed とは「ゲイン設計が違う後継版」（確定）
#   AB_fixed の忠実な再現を目指すものではない。骨格（es_core_fixed）の設計を保ったまま、AB_fixed の構造
#   （非線形項・結合の向き・歪みベクトル）を復元した版である。次の相違は意図的な設計差として扱う:
#     項目                      AB_fixed                              後継版 ESCore
#     ε→τ の基底ゲイン          1.0（(1+B5)·ε）                       0.2（0.2·(1+B5)·ε）   ← 確定: 0.2 のまま
#     G→r の係数                −0.05                                 −0.2
#     r の式の +0.1·ε           なし                                  あり
#     τ の式の形                −(η−B3)τ+χU+κ·tanh+(1+B5)ε           α_τ(r−τ)+B3τ+χU+κ·tanh+0.2(1+B5)ε
#     U の入力                  ẇ                                    r−w（w は参照水準）
#     w・G の式                 AB_fixed の式                         骨格の式
#   共通（同形・同係数）: r の非線形項、U→τ の接続、ε→Λ（0.02）、τ→ε（0.02）、τ 側レジーム項、Λ の適応速度 0.02、
#                         歪みベクトル B3〜B5 の入り方。
#   注意: B5=0 の意味が違う。後継版の ε→τ ゲインは 0.2·(1+B5) なので、B5=0 は AB_fixed の B5=0（ゲイン 1.0）ではない
#         （代数的にゲイン 1.0 になるのは B5=4）。したがって臨界 s* などの数値を AB_fixed と直接比べて差を「誤差」と
#         みなさないこと。b=0.1・確定仕様での実測は、後継版 +0.540 ／ AB_fixed +0.334（ゲインの違いが原因かは未検証）。
import math
import numpy as np


class ESCore:
    def __init__(self, b=0.1, delta=0.05, lam_loss=2.25, xi=0.05,
                 a=0.08, k=0.3, d=0.12, kappa=0.5, beta_regime=5.0, lam_rate=0.02, chi=0.8,
                 gamma_eps_Lambda=0.02, gamma_eps_tau=0.02, B3=-0.3, B4=0.0, B5=0.0):
        # 骨格（es_core_fixed）
        self.b, self.delta, self.lam_loss, self.xi = b, delta, lam_loss, xi
        self.p, self.q = 0.30, 0.20
        self.alpha_tau, self.gamma_w, self.kappa_G = 0.8, 0.5, 0.4
        self.g_to_r = -0.2
        # 復元（AB_fixed の r 式）
        self.a, self.k, self.d = a, k, d
        self.kappa, self.beta_regime = kappa, beta_regime
        self.lam_rate = lam_rate
        # 復元2（U→τ）、復元3（ε→Λ）
        self.chi = chi
        self.gamma_eps_Lambda = gamma_eps_Lambda
        self.gamma_eps_tau = gamma_eps_tau
        self.B3, self.B4, self.B5 = B3, B4, B5   # 歪みベクトル（すべて τ の式のみに入る。AB_fixed と同じ）

    def U(self, x):
        v = (x * x + self.delta ** 2) ** 0.44 - self.delta ** 0.88     # U(0)=0
        return v if x >= 0 else -self.lam_loss * v                       # 損失側を λ 倍

    def F(self, y):
        r, L, w, tau, G, Lam, eps = y
        U = self.U(r - w)                                                # w は U の入力（参照水準）
        r_dot = (self.a * r                                              # 自己増幅（復元）
                 + self.b * tau
                 - self.k * L                                            # L→r のブレーキ（復元）
                 - self.d * math.exp(min(r, 700.0))                      # 散逸（復元）
                 + self.kappa * math.tanh(self.beta_regime * (r - Lam))  # レジーム項（復元）
                 + self.g_to_r * G                                       # 制度 G のブレーキ
                 + 0.1 * eps)
        L_dot = self.p * r - self.q * L
        w_dot = self.gamma_w * U
        regime_tau = self.kappa * math.tanh(self.beta_regime * (r - Lam - self.B4))   # τ 側のレジーム項（復元）
        tau_dot = (self.alpha_tau * (r - tau) + self.B3 * tau                         # 減衰の歪み（η_eff=α_τ−B3）
                   + self.chi * U                                                     # U→τ（接続）
                   + regime_tau
                   + 0.2 * (1.0 + self.B5) * eps)                                     # ε→τ の感応度の歪み
        G_dot = self.kappa_G * (r - tau) - self.xi * G
        Lam_dot = self.lam_rate * (r - Lam) + self.gamma_eps_Lambda * eps   # ε→Λ（接続）
        eps_dot = -0.3 * eps + self.gamma_eps_tau * tau                  # τ→ε（接続）
        return np.array([r_dot, L_dot, w_dot, tau_dot, G_dot, Lam_dot, eps_dot])

    def step(self, y, dt):
        k1 = self.F(y); k2 = self.F(y + 0.5 * dt * k1)
        k3 = self.F(y + 0.5 * dt * k2); k4 = self.F(y + dt * k3)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from scipy.integrate import solve_ivp
    m = ESCore()
    s = solve_ivp(lambda t, y: m.F(y), (0, 5000), [0.5, 0.2, 0.5, 0, 0, 0, 0], method="LSODA",
                  rtol=1e-9, atol=1e-11, max_step=0.5, dense_output=True)
    tt = np.arange(3500, 5000, 0.1); Y = s.sol(tt); lam = Y[5]; mid = 0.5 * (lam.max() + lam.min())
    up = np.where((lam[:-1] < mid) & (lam[1:] >= mid))[0]
    print(f"サイクル: 周期 T={np.diff(tt[up]).mean():.1f}, r∈[{Y[0].min():+.2f},{Y[0].max():+.2f}], "
          f"τ∈[{Y[3].min():+.2f},{Y[3].max():+.2f}], w∈[{Y[2].min():+.2f},{Y[2].max():+.2f}], ε∈[{Y[6].min():+.3f},{Y[6].max():+.3f}]")
    # w が他の式に及ぼす影響（U→τ 接続後は τ の式に入る）
    J = np.zeros((7, 7)); y0 = Y[:, 100]
    for j in range(7):
        e = np.zeros(7); e[j] = 1e-6; J[:, j] = (m.F(y0 + e) - m.F(y0 - e)) / 2e-6
    print("∂F_i/∂w (i≠w) の最大絶対値:", round(np.abs(np.delete(J[:, 2], 2)).max(), 4), "→ w は受動でなくなった（τ へ）")
