"""
Refit alpha from existing depth10 ensemble summary,
using only bins where median is above the epsilon floor.
"""
import numpy as np
import json

results_dir = "/home/3x-agent/qft/results"

for depth in [8, 10]:
    path = f"{results_dir}/depth{depth}_ensemble_summary.json"
    with open(path) as f:
        d = json.load(f)

    summary = {int(k): v for k, v in d["summary"].items()}

    valid_dG = [dG for dG in sorted(summary) if summary[dG]["median"] > 1e-11]
    distances = np.array(valid_dG, dtype=float)
    means = np.array([summary[dG]["mean"] for dG in valid_dG])
    all_means = np.array([summary[dG]["mean"] for dG in sorted(summary)])

    coeffs = np.polyfit(distances, np.log(means), 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])
    monotone = all(all_means[k] > all_means[k+1] for k in range(len(all_means)-1))

    print(f"\ndepth={depth}")
    print(f"  Fitting on dG: {valid_dG}")
    print(f"  MI = {A:.4f} * exp(-{alpha:.4f} * dG)")
    print(f"  alpha = {alpha:.4f}")
    print(f"  Monotone: {monotone}")

    # Save corrected alpha back to summary
    d["alpha"] = float(alpha)
    d["alpha_fit_bins"] = valid_dG
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Saved corrected alpha to {path}")
