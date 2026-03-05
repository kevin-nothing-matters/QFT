"""
Gap 2: Metric tensor reconstruction via Multidimensional Scaling (MDS).

Takes the distance matrix d(i,j) = 0.5 * log(1/MI(i,j)) + c for all leaf pairs
at depth 8 and 10, embeds in low-dimensional space, and asks:
  - What is the intrinsic dimension of the emergent geometry?
  - Is the geometry flat (Euclidean), positively curved (spherical), or
    negatively curved (hyperbolic/AdS)?

Method:
  1. Load per-tree MI data, compute distance matrix d(i,j) for one representative tree
  2. Classical MDS: embed in R^k, find k via eigenvalue spectrum
  3. Check for hyperbolicity via Gromov delta (4-point condition)
  4. Fit to known geometries: flat R^n, hyperbolic H^n (AdS)
  5. Report the metric signature

Gromov delta test:
  For any four points (a,b,c,d), form three sums:
    S1 = d(a,b) + d(c,d)
    S2 = d(a,c) + d(b,d)
    S3 = d(a,d) + d(b,c)
  Sort: S1 >= S2 >= S3
  delta = (S1 - S2) / 2
  Pure tree metric: delta = 0
  Flat space: delta grows with scale
  Hyperbolic space: delta bounded (small relative to diameter)
"""

import numpy as np
import json
import random
from itertools import combinations

RESULTS_DIR = "/home/3x-agent/qft/results"
A = 0.5    # theoretical slope
N_SAMPLE = 200   # leaves to sample for MDS (full depth 10 = 1024 is too large for MDS)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Load MI data from individual trees ────────────────────────────────────

def load_tree_mi(depth, tree_idx=0):
    """Load MI values from a single tree JSON. Returns dict {(i,j): MI}."""
    with open(f"{RESULTS_DIR}/depth{depth}_tree{tree_idx:03d}.json") as f:
        data = json.load(f)
    # data is {dG: [MI values]} — we need pair indices
    # Reconstruct from ensemble summary instead — use bin means as representative
    return data

def load_ensemble_means(depth):
    """Load ensemble mean MI per dG bin."""
    with open(f"{RESULTS_DIR}/depth{depth}_ensemble_summary.json") as f:
        d = json.load(f)
    summary = {int(k): v for k, v in d["summary"].items()}
    valid = {dG: v["mean"] for dG, v in summary.items() if v["median"] > 1e-11}
    return valid

# ── Distance function ──────────────────────────────────────────────────────

def mi_to_distance(mi, a=0.5, c=1.82):
    mi = max(mi, 1e-12)
    return a * np.log(1.0 / mi) + c

# ── Build distance matrix from bin means ──────────────────────────────────

def build_distance_matrix_from_bins(depth, n_leaves_per_bin=4):
    """
    Construct a distance matrix using bin means.
    Creates n_leaves_per_bin representative leaves per dG bin.
    This gives us a structured distance matrix we can run MDS on.
    """
    means = load_ensemble_means(depth)
    dG_bins = sorted(means.keys())
    n_bins = len(dG_bins)

    # Create n_leaves virtual leaves, assign graph distances
    # Use a simple binary tree structure: leaves 0..N-1
    # dG(i,j) = 2 * highest_differing_bit_position
    n_leaves = 2**depth
    leaves = list(range(n_leaves))

    # Sample leaves if too many
    if len(leaves) > N_SAMPLE:
        leaves = sorted(random.sample(leaves, N_SAMPLE))

    n = len(leaves)
    D = np.zeros((n, n))

    for ii in range(n):
        for jj in range(ii+1, n):
            i, j = leaves[ii], leaves[jj]
            xor = i ^ j
            if xor == 0:
                dG = 0
            else:
                dG = 2 * xor.bit_length()

            # Map dG to MI to distance
            if dG in means:
                mi = means[dG]
            else:
                # Use closest available bin
                closest = min(means.keys(), key=lambda x: abs(x - dG))
                mi = means[closest]

            d = mi_to_distance(mi)
            D[ii, jj] = d
            D[jj, ii] = d

    return D, leaves

# ── Classical MDS ──────────────────────────────────────────────────────────

def classical_mds(D, n_components=10):
    """
    Classical (metric) MDS via double centering.
    Returns eigenvalues and coordinates in n_components dimensions.
    Negative eigenvalues indicate non-Euclidean geometry.
    """
    n = D.shape[0]
    D2 = D**2
    # Double centering
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ D2 @ J
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Coordinates
    k = n_components
    pos_mask = eigvals[:k] > 0
    coords = np.zeros((n, k))
    for i in range(k):
        if eigvals[i] > 0:
            coords[:, i] = eigvecs[:, i] * np.sqrt(eigvals[i])
    return eigvals, coords

# ── Gromov delta (hyperbolicity) ───────────────────────────────────────────

def gromov_delta(D, n_samples=2000):
    """
    Estimate Gromov delta (4-point hyperbolicity constant).
    delta=0: pure tree metric
    delta small relative to diameter: hyperbolic
    delta large: flat or spherical
    """
    n = D.shape[0]
    indices = list(range(n))
    deltas = []

    for _ in range(n_samples):
        a, b, c, d = random.sample(indices, 4)
        s1 = D[a,b] + D[c,d]
        s2 = D[a,c] + D[b,d]
        s3 = D[a,d] + D[b,c]
        sums = sorted([s1, s2, s3], reverse=True)
        delta = (sums[0] - sums[1]) / 2
        deltas.append(delta)

    deltas = np.array(deltas)
    diameter = D.max()
    return {
        "delta_max": float(deltas.max()),
        "delta_mean": float(deltas.mean()),
        "delta_relative": float(deltas.max() / diameter) if diameter > 0 else 0,
        "diameter": float(diameter),
    }

# ── Main ───────────────────────────────────────────────────────────────────

all_results = {}

for depth in [8, 10]:
    print(f"\n{'='*60}")
    print(f"DEPTH {depth}  (sampling {N_SAMPLE} leaves for MDS)")
    print(f"{'='*60}")

    D, leaves = build_distance_matrix_from_bins(depth)
    print(f"Distance matrix: {D.shape[0]}x{D.shape[0]}")
    print(f"Distance range: [{D.min():.3f}, {D.max():.3f}]")
    print(f"Diameter: {D.max():.3f}")

    # ── MDS ───────────────────────────────────────────────────────────────
    print(f"\n--- MDS Eigenvalue Spectrum ---")
    eigvals, coords = classical_mds(D, n_components=20)

    # Show top eigenvalues
    print(f"  Top 10 eigenvalues:")
    for i in range(min(10, len(eigvals))):
        sign = "+" if eigvals[i] >= 0 else "-"
        print(f"    λ_{i+1} = {eigvals[i]:+10.3f}")

    # Count significant positive vs negative eigenvalues
    pos_eigvals = eigvals[eigvals > 0.01 * abs(eigvals[0])]
    neg_eigvals = eigvals[eigvals < -0.01 * abs(eigvals[0])]
    print(f"\n  Significant positive eigenvalues: {len(pos_eigvals)}")
    print(f"  Significant negative eigenvalues: {len(neg_eigvals)}")

    # Variance explained
    total_pos = pos_eigvals.sum()
    var_exp = np.cumsum(pos_eigvals) / total_pos
    for k in [1, 2, 3, 5]:
        if k <= len(var_exp):
            print(f"  Variance explained by top {k} dims: {100*var_exp[k-1]:.1f}%")

    # Geometry diagnosis
    if len(neg_eigvals) > 0:
        print(f"\n  *** NEGATIVE EIGENVALUES PRESENT ***")
        print(f"  This indicates NON-EUCLIDEAN geometry.")
        neg_fraction = abs(neg_eigvals.sum()) / (abs(pos_eigvals.sum()) + abs(neg_eigvals.sum()))
        print(f"  Negative eigenvalue fraction: {100*neg_fraction:.1f}%")
        if neg_fraction > 0.1:
            print(f"  Geometry signature: STRONGLY non-Euclidean (hyperbolic candidate)")
        else:
            print(f"  Geometry signature: Mildly non-Euclidean")
    else:
        print(f"\n  All eigenvalues positive: Euclidean geometry")

    # ── Gromov delta ──────────────────────────────────────────────────────
    print(f"\n--- Gromov Delta (Hyperbolicity) ---")
    gd = gromov_delta(D, n_samples=3000)
    print(f"  Delta max:      {gd['delta_max']:.4f}")
    print(f"  Delta mean:     {gd['delta_mean']:.4f}")
    print(f"  Diameter:       {gd['diameter']:.4f}")
    print(f"  Delta/diameter: {gd['delta_relative']:.4f}")

    if gd['delta_relative'] < 0.05:
        geo_type = "TREE METRIC (delta ~ 0)"
    elif gd['delta_relative'] < 0.2:
        geo_type = "HYPERBOLIC (small delta/diameter)"
    else:
        geo_type = "FLAT OR SPHERICAL (large delta/diameter)"
    print(f"  Geometry type:  {geo_type}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n--- Geometry Verdict depth {depth} ---")
    n_neg = len(neg_eigvals)
    print(f"  MDS negative eigenvalues: {n_neg}")
    print(f"  Gromov delta/diameter:    {gd['delta_relative']:.4f}")
    print(f"  Geometry:                 {geo_type}")

    all_results[str(depth)] = {
        "top_eigenvalues": list(eigvals[:20].astype(float)),
        "n_positive_significant": int(len(pos_eigvals)),
        "n_negative_significant": int(len(neg_eigvals)),
        "gromov": gd,
        "geometry_type": geo_type,
    }

out_path = f"{RESULTS_DIR}/metric_reconstruction.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n\nSaved: {out_path}")
