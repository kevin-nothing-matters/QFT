"""
MDS Geometry Figure
===================
Produces a 2D MDS embedding of the information distance matrix at depth 8.
Colors leaves by genealogical cluster (generation of common ancestor).

Run: python make_mds_figure.py
Requires: results/depth8_ensemble_summary.json (or any depth8_tree*.json)
Output: mds_geometry.png (publication-ready figure)
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
import os

RESULTS_DIR = os.path.expanduser("~/qft/results")
OUTPUT_PATH = os.path.expanduser("~/qft/results/mds_geometry.png")

# ── Load MI data ───────────────────────────────────────────────────────────

def graph_dist(i, j, depth=8):
    """Genealogical distance = 2 * (depth - lca_depth)."""
    if i == j: return 0
    for k in range(depth):
        if (i >> (depth-1-k)) != (j >> (depth-1-k)):
            return 2 * (depth - k)
    return 0

def load_mi_matrix(depth=8, tree_idx=0):
    """Load per-tree MI values into a full n_leaves x n_leaves matrix."""
    n = 2**depth
    fname = f"{RESULTS_DIR}/depth{depth}_tree{tree_idx:03d}.json"
    
    if not os.path.exists(fname):
        # Try ensemble summary
        fname2 = f"{RESULTS_DIR}/depth{depth}_ensemble_summary.json"
        if os.path.exists(fname2):
            print(f"Per-tree file not found, using ensemble means from {fname2}")
            with open(fname2) as f:
                data = json.load(f)
            summary = data.get("summary", data)
            # Build MI matrix from mean values by dG
            mi_by_dG = {}
            for dG_str, stats in summary.items():
                dG = int(dG_str)
                if isinstance(stats, dict):
                    mi_by_dG[dG] = stats.get("mean", stats.get("median", 1e-12))
                else:
                    mi_by_dG[dG] = float(stats)
            M = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        M[i,j] = 1.0
                    else:
                        dG = graph_dist(i, j, depth)
                        M[i,j] = mi_by_dG.get(dG, 1e-12)
            return M
        raise FileNotFoundError(f"No data found for depth {depth} tree {tree_idx}")
    
    with open(fname) as f:
        data = json.load(f)
    
    # Reconstruct MI matrix from per-dG lists
    pairs_by_dG = {}
    all_pairs = list(combinations(range(n), 2))
    for i, j in all_pairs:
        dG = graph_dist(i, j, depth)
        pairs_by_dG.setdefault(dG, []).append((i,j))
    
    M = np.zeros((n, n))
    for i in range(n):
        M[i,i] = 1.0  # self-MI = max
    
    for dG_str, mi_vals in data.items():
        dG = int(dG_str)
        if dG not in pairs_by_dG: continue
        for idx, (i,j) in enumerate(pairs_by_dG[dG]):
            if idx < len(mi_vals):
                val = mi_vals[idx]
                M[i,j] = M[j,i] = max(float(val), 1e-15)
    
    return M

# ── Classical MDS ─────────────────────────────────────────────────────────

def classical_mds(D, n_components=2):
    """
    Classical (metric) MDS from distance matrix D.
    Returns coordinates in n_components dimensions.
    """
    n = D.shape[0]
    # Double-center the squared distance matrix
    D2 = D**2
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J @ D2 @ J
    # Eigendecomposition
    vals, vecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:,idx]
    # Take top n_components
    vals_pos = np.maximum(vals[:n_components], 0)
    coords = vecs[:,:n_components] * np.sqrt(vals_pos)
    # Variance explained
    total_var = np.sum(np.maximum(vals, 0))
    var_explained = np.sum(vals_pos) / total_var if total_var > 0 else 0
    return coords, var_explained, vals

# ── Cluster coloring ──────────────────────────────────────────────────────

def get_cluster_label(leaf, depth=8, n_colors=4):
    """
    Color by top-level subtree (first n_color_bits bits of leaf index).
    n_colors=4 means 4 quadrants, 64 leaves each.
    n_colors=8 means 8 octants, 32 leaves each.
    """
    n_color_bits = int(np.log2(n_colors))
    return (leaf >> (depth - n_color_bits)) & (n_colors - 1)

def get_generation_cluster(leaf_i, leaf_j, depth=8):
    """Genealogical distance between leaves."""
    return graph_dist(leaf_i, leaf_j, depth)

# ── Main ──────────────────────────────────────────────────────────────────

DEPTH = 8
N = 2**DEPTH
EPS = 1e-12

print(f"Loading depth {DEPTH} MI data ({N} leaves)...")
M = load_mi_matrix(DEPTH, tree_idx=0)
print(f"MI matrix loaded. Min non-diagonal: {np.min(M[M > 0] if np.any(M > 0) else [0]):.2e}")
print(f"Siblings (dG=2) mean MI: {np.mean([M[i,i+1] for i in range(0,N-1,2)]):.4f}")

# Build distance matrix
print("Building distance matrix...")
D = -np.log(np.maximum(M, EPS))
np.fill_diagonal(D, 0)
print(f"Distance range: {D.min():.2f} to {D[D < 1e10].max():.2f}")

# Apply MDS
print("Running classical MDS...")
coords, var_exp, eigenvalues = classical_mds(D, n_components=2)
print(f"Variance explained by 2D embedding: {var_exp*100:.1f}%")
print(f"Top 5 eigenvalues: {eigenvalues[:5]}")

# ── Figure ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# Color scheme: 8 top-level clusters (octants of tree)
N_CLUSTERS = 8
cluster_labels = np.array([get_cluster_label(i, DEPTH, N_CLUSTERS) for i in range(N)])

# Use a colormap with good cluster separation
cmap = plt.cm.get_cmap('tab10', N_CLUSTERS)
colors = [cmap(cluster_labels[i]) for i in range(N)]

# ── Left panel: full embedding colored by cluster ─────────────────────────
ax1 = axes[0]
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=[cmap(c)], s=8, alpha=0.8, linewidths=0,
                label=f'Cluster {c+1}')

ax1.set_title(f'MDS Embedding — Depth {DEPTH} ({N} leaves)',
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('MDS Dimension 1', fontsize=10)
ax1.set_ylabel('MDS Dimension 2', fontsize=10)
ax1.legend(loc='upper right', fontsize=7, ncol=2,
           markerscale=2, framealpha=0.9)
ax1.text(0.02, 0.02,
         f'Variance explained: {var_exp*100:.1f}%\nU = 0.983 ± 0.006',
         transform=ax1.transAxes, fontsize=9,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)

# ── Right panel: color by genealogical distance from leaf 0 ───────────────
ax2 = axes[1]
# Distance from leaf 0 = genealogical distance from leaf 0
dist_from_0 = D[0, :]
sc = ax2.scatter(coords[:, 0], coords[:, 1],
                 c=dist_from_0, cmap='plasma',
                 s=8, alpha=0.9, linewidths=0)
plt.colorbar(sc, ax=ax2, label='Information distance from leaf 0\n'
             r'$d_{0j} = -\log(\mathrm{MI}_{0j})$',
             shrink=0.8)

# Mark leaf 0
ax2.scatter(coords[0, 0], coords[0, 1], c='red', s=60,
            zorder=5, marker='*', label='Leaf 0 (reference)')
ax2.legend(fontsize=9)
ax2.set_title('Information Distance from Reference Leaf',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('MDS Dimension 1', fontsize=10)
ax2.set_ylabel('MDS Dimension 2', fontsize=10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2)

plt.suptitle(
    'Emergent Geometry from Quantum Entanglement Correlations\n'
    r'$d_{ij} = -\log\,\mathrm{MI}_{ij}$, Classical MDS Embedding, Depth 8',
    fontsize=11, y=1.02
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\nFigure saved: {OUTPUT_PATH}")

# ── Eigenvalue spectrum (diagnostic) ─────────────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 4))
n_show = min(30, len(eigenvalues))
x = np.arange(1, n_show+1)
colors_ev = ['steelblue' if v >= 0 else 'red' for v in eigenvalues[:n_show]]
ax.bar(x, eigenvalues[:n_show], color=colors_ev, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Eigenvalue index', fontsize=11)
ax.set_ylabel('Eigenvalue', fontsize=11)
ax.set_title('MDS Eigenvalue Spectrum\n'
             '(Negative eigenvalues indicate non-Euclidean geometry)',
             fontsize=11)
ax.text(0.98, 0.95,
        'Blue = positive\nRed = negative (non-Euclidean)',
        transform=ax.transAxes, fontsize=9,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
eigenval_path = OUTPUT_PATH.replace('mds_geometry.png', 'mds_eigenvalues.png')
plt.savefig(eigenval_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Eigenvalue spectrum saved: {eigenval_path}")

# Print summary stats
print(f"\nGeometry Summary:")
print(f"  Positive eigenvalues: {np.sum(eigenvalues > 0)}")
print(f"  Negative eigenvalues: {np.sum(eigenvalues < 0)}")
print(f"  Largest eigenvalue:   {eigenvalues[0]:.2f}")
print(f"  2nd largest:          {eigenvalues[1]:.2f}")
print(f"  First negative:       {eigenvalues[eigenvalues < 0][0]:.4f}" 
      if np.any(eigenvalues < 0) else "  No negative eigenvalues")
print(f"\n  Interpretation:")
if np.sum(eigenvalues[:2]) / np.sum(eigenvalues[eigenvalues>0]) > 0.8:
    print("  > 80% variance in 2D — strong low-dimensional structure")
if np.any(eigenvalues < 0):
    print("  Negative eigenvalues present — geometry is non-Euclidean")
    print("  This is the expected signature of a hyperbolic/tree metric")
