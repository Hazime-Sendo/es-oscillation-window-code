"""
es_window_verify_structure.py — Sec. V.E の関係式の検算（es_window_paper_data.json から）
  T_U = g/(1−iβg), β=γ_w/ω           （w が U の積分であることからの厳密な関係）
  P1 = 2χ Re[T_U H_τ]/(1+ℓ²),  P2 = 2 g_G Re[−κ_G χ T_U H_τ H_G]/(1+ℓ²)     H_τ=1/(γ+iω), H_G=1/(ξ+iω)
使い方: python es_window_verify_structure.py
"""
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
D = json.load(open(os.path.join(DATA, "es_window_paper_data.json")))
chi, xi, kG, gG, gam = 0.8, 0.05, 0.4, -0.2, 1.1
print("  lam   |T_U| 相対差(T_U vs g/(1-i b g))   P1 差       P2 差")
for q in D:
    g = q["g_hb"] * np.exp(1j * q["g_hb_arg"]); T = q["T_U_re"] + 1j * q["T_U_im"]; om = q["omega"]
    Tm = g / (1 - 1j * (0.5 / om) * g); Ht = 1 / (gam + 1j * om); Hg = 1 / (xi + 1j * om); l2 = q["ell2"]
    P1f = 2 * chi * np.real(T * Ht) / (1 + l2); P2f = 2 * gG * np.real(-kG * chi * T * Ht * Hg) / (1 + l2)
    print(f"  {q['lam']:5.2f}  {abs(T - Tm) / abs(T):.1e}                        {abs(P1f - q['P1']):.1e}   {abs(P2f - q['P2']):.1e}")
