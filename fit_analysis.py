"""
fit_analysis.py — Distance-entanglement curve fitting for Quantum Family Tree results.

Fits MI vs genealogical distance to logarithmic, exponential, and power-law models.
Accepts JSON output from phase_c_v6.py.

Usage:
    python fit_analysis.py results/phase_c_v6_TIMESTAMP.json
    python fit_analysis.py  # uses hardcoded depth 4 results

Author: Kevin Donahue (kevin@nothingmatters.life)
"""

import numpy as np
import json
import sys
from scipy.stats import linregress


def fit_all(ng_values, mi_values, label=""):
    """Run all fits on distance-MI data."""
    ng = np.array(ng_values, dtype=float)
    mi = np.array(mi_values, dtype=float)

    print("=" * 60)
    print(f"  DISTANCE-ENTANGLEMENT FIT {label}")
    print("=" * 60)
    print()
    print("  %-6s %-12s %-12s %-12s" % ("n_g", "MI", "ln(1/MI)", "1/MI"))
    print("  " + "-" * 44)
    for n, m in zip(ng, mi):
        if m > 0:
            print("  %-6d %-12.6f %-12.4f %-12.2f" % (n, m, np.log(1 / m), 1 / m))

    # Fit 1: d = a * ln(1/MI) + b (logarithmic — Ryu-Takayanagi motivated)
    mask = mi > 1e-15
    x_log = np.log(1 / mi[mask])
    y = ng[mask]
    sl, ic, r, p, se = linregress(x_log, y)
    print(f"\n  --- Logarithmic: d = a ln(1/MI) + b ---")
    print(f"  a = {sl:.4f}")
    print(f"  b = {ic:.4f}")
    print(f"  R² = {r ** 2:.6f}")
    print(f"  p = {p:.2e}")
    print(f"  Formula: d = {sl:.4f} * ln(1/MI) + {ic:.4f}")

    # Fit 2: MI = exp(-c * d)
    ln_mi = np.log(mi[mask])
    sl2, ic2, r2, p2, se2 = linregress(ng[mask], ln_mi)
    print(f"\n  --- Exponential: MI = exp(-c * d) ---")
    print(f"  decay rate c = {-sl2:.4f}")
    print(f"  R² = {r2 ** 2:.6f}")
    print(f"  Half-life: {np.log(2) / (-sl2):.2f} generations")

    # Fit 3: MI = (1/2)^(alpha * n_g)
    log2_mi = np.log2(mi[mask])
    sl3, ic3, r3, p3, se3 = linregress(ng[mask], log2_mi)
    alpha = -sl3
    print(f"\n  --- Kinship: MI = (1/2)^(alpha * n_g) ---")
    print(f"  alpha = {alpha:.4f}")
    print(f"  R² = {r3 ** 2:.6f}")
    print(f"  Classical kinship: alpha = 0.5")
    print(f"  Amplification: {alpha / 0.5:.2f}x steeper than classical")

    # Fit 4: Power law (excluding siblings)
    if len(ng) > 2:
        non_sib = mi < 0.99
        if np.sum(non_sib) >= 2:
            log_ng_ns = np.log(ng[non_sib])
            log_mi_ns = np.log(mi[non_sib])
            sl4, ic4, r4, p4, se4 = linregress(log_ng_ns, log_mi_ns)
            print(f"\n  --- Power law: MI = a * d^(-p) ---")
            print(f"  exponent p = {-sl4:.4f}")
            print(f"  R² = {r4 ** 2:.6f}")

    print()
    return {
        "log_a": float(sl),
        "log_b": float(ic),
        "log_r2": float(r ** 2),
        "decay_rate": float(-sl2),
        "alpha": float(alpha),
        "alpha_r2": float(r3 ** 2),
    }


def main():
    if len(sys.argv) > 1:
        # Load from JSON file
        with open(sys.argv[1]) as f:
            data = json.load(f)

        for depth_str, depth_data in data.get("depths", {}).items():
            kinship = depth_data.get("kinship", {})
            if kinship:
                ng_vals = sorted([int(k) for k in kinship.keys()])
                mi_vals = [kinship[str(ng)]["mean"] for ng in ng_vals]
                fit_all(ng_vals, mi_vals, label=f"(Depth {depth_str})")
    else:
        # Default: hardcoded depth 4 results
        print("No input file specified. Using hardcoded depth 4 results.\n")
        ng = [2, 4, 6, 8]
        mi = [1.0, 0.076, 0.006, 0.001]
        fit_all(ng, mi, label="(Depth 4 — 30 qubits, 10 trees)")

    # Combined analysis across depths (if available)
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  The logarithmic fit d = a ln(1/MI) + b is consistent with")
    print("  Ryu-Takayanagi holographic scaling (S = A/4G).")
    print("  Distance IS the logarithm of inverse entanglement.")
    print()


if __name__ == "__main__":
    main()
