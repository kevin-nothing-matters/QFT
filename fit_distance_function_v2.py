"""
Gap 1 v2: Distance function fit with weighted regression on above-floor bins only.

Key fix: weight each bin by 1/std to downweight noisy high-dG bins.
Also fits d = a * log(1/MI) + c to allow non-zero intercept.
"""

import numpy as np
import json
from scipy.optimize import curve_fit
from itertools import combinations

RESULTS_DIR = "/home/3x-agent/qft/results"

def load_summary(depth):
    with open(f"{RESULTS_DIR}/depth{depth}_ensemble_summary.json") as f:
        d = json.load(f)
    summary = {int(k): v for k, v in d["summary"].items()}
    valid = {dG: v for dG, v in summary.items() if v["median"] > 1e-11}
    dG_vals = np.array(sorted(valid.keys()), dtype=float)
    means   = np.array([valid[dG]["mean"] for dG in sorted(valid.keys())])
    stds    = np.array([valid[dG]["std"]  for dG in sorted(valid.keys())])
    return dG_vals, means, stds

def check_triangle_inequality(distances_by_dG):
    bins = sorted(distances_by_dG.keys())
    ordinary_pass = ultra_pass = total = 0
    for i, j, k in combinations(range(len(bins)), 3):
        d_ij = distances_by_dG[bins[i]]
        d_jk = distances_by_dG[bins[j]]
        d_ik = distances_by_dG[bins[k]]
        if d_ik <= d_ij + d_jk + 1e-10: ordinary_pass += 1
        if d_ik <= max(d_ij, d_jk) + 1e-10: ultra_pass += 1
        total += 1
    return ordinary_pass/total, ultra_pass/total

results = {}

for depth in [8, 10]:
    print(f"\n{'='*60}")
    print(f"DEPTH {depth}")
    print(f"{'='*60}")

    dG_vals, means, stds = load_summary(depth)
    weights = 1.0 / np.clip(stds, 1e-15, None)
    weights /= weights.sum()

    print(f"Bins: dG = {list(dG_vals.astype(int))}")
    print(f"Means: {[f'{m:.4e}' for m in means]}")

    depth_results = {}

    # ── 1. Simple log: d = a * log(1/MI) ──────────────────────────────────
    print(f"\n--- d = a * log(1/MI) ---")
    log_mi = np.log(1.0 / np.clip(means, 1e-15, None))
    # Weighted linear regression: dG = a * log(1/MI)
    # Force through origin: a = sum(w * dG * log_mi) / sum(w * log_mi^2)
    a_fit = np.sum(weights * dG_vals * log_mi) / np.sum(weights * log_mi**2)
    d_pred = a_fit * log_mi
    ss_res = np.sum(weights * (dG_vals - d_pred)**2)
    ss_tot = np.sum(weights * (dG_vals - np.average(dG_vals, weights=weights))**2)
    r2 = 1 - ss_res/ss_tot
    print(f"  a = {a_fit:.6f}")
    print(f"  R² (weighted) = {r2:.6f}")
    print(f"  Predicted dG: {[f'{d:.3f}' for d in d_pred]}")
    print(f"  Actual dG:    {list(dG_vals.astype(int))}")
    print(f"  Residuals:    {[f'{r:.3f}' for r in dG_vals - d_pred]}")

    d_by_dG = {int(dG_vals[i]): float(d_pred[i]) for i in range(len(dG_vals))}
    ord_pct, ult_pct = check_triangle_inequality(d_by_dG)
    print(f"  Ordinary triangle: {100*ord_pct:.1f}%  |  Ultrametric: {100*ult_pct:.1f}%")

    depth_results["log_simple"] = {
        "a": float(a_fit), "r2": float(r2),
        "ordinary_pct": ord_pct, "ultra_pct": ult_pct,
        "d_predicted": list(d_pred), "dG_actual": list(dG_vals),
        "residuals": list(dG_vals - d_pred),
    }

    # ── 2. Log with intercept: d = a * log(1/MI) + c ──────────────────────
    print(f"\n--- d = a * log(1/MI) + c ---")
    X = np.column_stack([log_mi, np.ones_like(log_mi)])
    W = np.diag(weights)
    try:
        coeffs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ dG_vals, rcond=None)[0]
        a2, c2 = coeffs
        d_pred2 = a2 * log_mi + c2
        ss_res2 = np.sum(weights * (dG_vals - d_pred2)**2)
        r2_2 = 1 - ss_res2/ss_tot
        print(f"  a = {a2:.6f},  c = {c2:.6f}")
        print(f"  R² (weighted) = {r2_2:.6f}")
        print(f"  Predicted dG: {[f'{d:.3f}' for d in d_pred2]}")
        print(f"  Residuals:    {[f'{r:.3f}' for r in dG_vals - d_pred2]}")

        d_by_dG2 = {int(dG_vals[i]): float(d_pred2[i]) for i in range(len(dG_vals))}
        ord_pct2, ult_pct2 = check_triangle_inequality(d_by_dG2)
        print(f"  Ordinary triangle: {100*ord_pct2:.1f}%  |  Ultrametric: {100*ult_pct2:.1f}%")

        depth_results["log_intercept"] = {
            "a": float(a2), "c": float(c2), "r2": float(r2_2),
            "ordinary_pct": ord_pct2, "ultra_pct": ult_pct2,
        }
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 3. Summary ─────────────────────────────────────────────────────────
    print(f"\n--- Summary depth {depth} ---")
    print(f"  Simple log R²:    {depth_results['log_simple']['r2']:.6f}")
    if "log_intercept" in depth_results:
        print(f"  Log+intercept R²: {depth_results['log_intercept']['r2']:.6f}")
    print(f"  Theoretical prediction: a = 1/2 (from alpha=2 and MI ~ exp(-2*dG))")
    print(f"  Fitted a = {depth_results['log_simple']['a']:.6f}  (target: 0.5)")

    results[str(depth)] = depth_results

# ── Cross-depth consistency ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CROSS-DEPTH CONSISTENCY")
print(f"{'='*60}")
for depth in [8, 10]:
    a = results[str(depth)]["log_simple"]["a"]
    r2 = results[str(depth)]["log_simple"]["r2"]
    print(f"  depth={depth}: a = {a:.6f}, R² = {r2:.6f}")
print(f"  Theoretical a = 0.5 (from MI ~ exp(-alpha*dG), alpha=2, dG=a*log(1/MI) => a=1/alpha)")

out_path = f"{RESULTS_DIR}/distance_function_fit_v2.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
