"""
Gap 1: Derive the distance function f such that d(i,j) = f(1/MI(i,j)) is a proper metric.

Tests two candidates motivated by theory:
  1. Logarithmic: d = a * log(1/MI) = -a * log(MI)   [motivated by Ryu-Takayanagi]
  2. Power law:   d = MI^(-b)                          [alternative]

For each candidate:
  - Fits parameters using ensemble mean MI per dG bin
  - Computes R² goodness of fit against graph distance dG
  - Verifies metric axioms:
      (i)  d(i,j) >= 0, equality iff i=j          [positivity]
      (ii) d(i,j) = d(j,i)                         [symmetry — trivially satisfied]
      (iii) d(i,k) <= d(i,j) + d(j,k)              [triangle inequality]
      (iv)  d(i,k) <= max(d(i,j), d(j,k))          [ultrametric — strong triangle]

Results saved to results/distance_function_fit.json
"""

import numpy as np
import json
from scipy.optimize import curve_fit
from itertools import combinations

RESULTS_DIR = "/home/3x-agent/qft/results"

# ── Load ensemble summaries ────────────────────────────────────────────────

def load_summary(depth):
    with open(f"{RESULTS_DIR}/depth{depth}_ensemble_summary.json") as f:
        d = json.load(f)
    summary = {int(k): v for k, v in d["summary"].items()}
    # Only use bins above the epsilon floor
    valid = {dG: v for dG, v in summary.items() if v["median"] > 1e-11}
    dG_vals = np.array(sorted(valid.keys()), dtype=float)
    means   = np.array([valid[dG]["mean"] for dG in sorted(valid.keys())])
    return dG_vals, means

# ── Candidate distance functions ───────────────────────────────────────────

def log_distance(mi, a):
    """d = a * log(1/MI) = -a * log(MI)"""
    return a * np.log(1.0 / np.clip(mi, 1e-15, None))

def power_distance(mi, b):
    """d = MI^(-b)"""
    return np.power(np.clip(mi, 1e-15, None), -b)

# ── Metric axiom checks ────────────────────────────────────────────────────

def check_triangle_inequality(distances_by_dG, label):
    """
    Sample triples of dG bins and check both ordinary and strong triangle inequalities.
    distances_by_dG: dict {dG: mean_distance}
    """
    bins = sorted(distances_by_dG.keys())
    n = len(bins)
    ordinary_pass = 0
    ultra_pass = 0
    total = 0

    for i, j, k in combinations(range(n), 3):
        d_ij = distances_by_dG[bins[i]]  # smallest dG -> smallest d
        d_jk = distances_by_dG[bins[j]]
        d_ik = distances_by_dG[bins[k]]  # largest dG -> largest d

        # Ordinary triangle: d(i,k) <= d(i,j) + d(j,k)
        if d_ik <= d_ij + d_jk + 1e-10:
            ordinary_pass += 1
        # Strong (ultrametric): d(i,k) <= max(d(i,j), d(j,k))
        if d_ik <= max(d_ij, d_jk) + 1e-10:
            ultra_pass += 1
        total += 1

    print(f"  {label}:")
    print(f"    Ordinary triangle inequality: {ordinary_pass}/{total} ({100*ordinary_pass/total:.1f}%)")
    print(f"    Ultrametric (strong triangle): {ultra_pass}/{total} ({100*ultra_pass/total:.1f}%)")
    return ordinary_pass/total, ultra_pass/total

# ── Main ───────────────────────────────────────────────────────────────────

results = {}

for depth in [8, 10]:
    print(f"\n{'='*60}")
    print(f"DEPTH {depth}")
    print(f"{'='*60}")

    dG_vals, means = load_summary(depth)
    print(f"Valid bins: dG = {list(dG_vals.astype(int))}")
    print(f"Mean MI:    {[f'{m:.4e}' for m in means]}")

    depth_results = {}

    # ── 1. Logarithmic fit ─────────────────────────────────────────────────
    print(f"\n--- Logarithmic: d = a * log(1/MI) ---")
    try:
        popt, _ = curve_fit(log_distance, means, dG_vals, p0=[1.0])
        a_fit = popt[0]
        d_pred = log_distance(means, a_fit)
        ss_res = np.sum((dG_vals - d_pred)**2)
        ss_tot = np.sum((dG_vals - dG_vals.mean())**2)
        r2 = 1 - ss_res/ss_tot
        print(f"  a = {a_fit:.4f}")
        print(f"  R² = {r2:.6f}")
        print(f"  Predicted distances: {[f'{d:.2f}' for d in d_pred]}")
        print(f"  Actual dG values:    {list(dG_vals.astype(int))}")

        # Check metric axioms at the level of bin means
        d_by_dG = {int(dG_vals[i]): float(d_pred[i]) for i in range(len(dG_vals))}
        ord_pct, ult_pct = check_triangle_inequality(d_by_dG, "Log distance")

        depth_results["log"] = {
            "a": float(a_fit), "r2": float(r2),
            "ordinary_triangle_pct": ord_pct,
            "ultrametric_pct": ult_pct,
            "d_predicted": list(d_pred),
            "dG_actual": list(dG_vals),
        }
    except Exception as e:
        print(f"  Fit failed: {e}")

    # ── 2. Power law fit ───────────────────────────────────────────────────
    print(f"\n--- Power law: d = MI^(-b) ---")
    try:
        popt2, _ = curve_fit(power_distance, means, dG_vals, p0=[0.5], maxfev=5000)
        b_fit = popt2[0]
        d_pred2 = power_distance(means, b_fit)
        ss_res2 = np.sum((dG_vals - d_pred2)**2)
        r2_2 = 1 - ss_res2/ss_tot
        print(f"  b = {b_fit:.4f}")
        print(f"  R² = {r2_2:.6f}")
        print(f"  Predicted distances: {[f'{d:.2f}' for d in d_pred2]}")

        d_by_dG2 = {int(dG_vals[i]): float(d_pred2[i]) for i in range(len(dG_vals))}
        ord_pct2, ult_pct2 = check_triangle_inequality(d_by_dG2, "Power distance")

        depth_results["power"] = {
            "b": float(b_fit), "r2": float(r2_2),
            "ordinary_triangle_pct": ord_pct2,
            "ultrametric_pct": ult_pct2,
            "d_predicted": list(d_pred2),
            "dG_actual": list(dG_vals),
        }
    except Exception as e:
        print(f"  Fit failed: {e}")

    # ── 3. Winner ──────────────────────────────────────────────────────────
    print(f"\n--- Summary ---")
    if "log" in depth_results and "power" in depth_results:
        log_r2 = depth_results["log"]["r2"]
        pow_r2 = depth_results["power"]["r2"]
        winner = "logarithmic" if log_r2 > pow_r2 else "power law"
        print(f"  R² log = {log_r2:.6f}")
        print(f"  R² pow = {pow_r2:.6f}")
        print(f"  Winner: {winner}")
        depth_results["winner"] = winner

    results[str(depth)] = depth_results

# ── Save ───────────────────────────────────────────────────────────────────
out_path = f"{RESULTS_DIR}/distance_function_fit.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nSaved: {out_path}")
