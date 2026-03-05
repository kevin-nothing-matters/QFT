"""
MDS Geometry Figure v2
======================
Produces a 2D MDS embedding of the information distance matrix at depth 8.
Uses actual per-tree MI values (tree 0) for realistic within-cluster scatter.
Colors leaves by top-level subtree cluster.

Run: python make_mds_figure.py
Requires: ~/qft/results/depth8_tree000.json
Output: ~/qft/results/mds_geometry.png
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.expanduser("~/qft/results")
OUTPUT_PATH = os.path.expanduser("~/qft/results/mds_geometry.png")
DEPTH = 8
N = 2**DEPTH
EPS = 1e-12

# ── Genealogical distance ─────────────────────────────────────────────────

def graph_dist(i, j, depth=8):
    if i == j: return 0
    for k in range(depth):
        if (i >> (depth-1-k)) != (j >> (depth-1-k)):
            return 2 * (depth - k)
    return 0

# ── Load per-tree MI into full N×N matrix ─────────────────────────────────

print(f"Loading depth {DEPTH} tree 0 MI data ({N} leaves)...")
fname = f"{RESULTS_DIR}/depth{DEPTH}_tree000.json"
with open(fname) as f:
    raw = json.load(f)

# raw keys are dG strings, values are lists of MI for all pairs at that dG
# Build ordered pair list matching the simulation's output order
from itertools import combinations
pairs_by_dG = {}
for i, j in combinations(range(N), 2):
    dG = graph_dist(i, j, DEPTH)
    pairs_by_dG.setdefault(dG, []).append((i, j))

M = np.full((N, N), EPS)
np.fill_diagonal(M, 1.0)

for dG_str, mi_list in raw.items():
    dG = int(dG_str)
    pairs = pairs_by_dG.get(dG, [])
    for idx, mi_val in enumerate(mi_list):
        if idx < len(pairs):
            i, j = pairs[idx]
            val = max(float(mi_val), EPS)
            M[i, j] = M[j, i] = val

print(f"MI matrix loaded.")
print(f"Siblings (dG=2) sample MI: {[f'{M[i,i+1]:.3f}' for i in range(0,8,2)]}")
print(f"Cousins  (dG=4) sample MI: {[f'{M[i,i+2]:.4f}' for i in range(0,8,4)]}")

# ── Distance matrix ───────────────────────────────────────────────────────

D = -np.log(np.maximum(M, EPS))
np.fill_diagonal(D, 0)
print(f"Distance range: {D[D>0].min():.2f} — {D[D<1e10].max():.2f}")

# ── Classical MDS ─────────────────────────────────────────────────────────

print("Running classical MDS...")
D2 = D**2
J = np.eye(N) - np.ones((N, N)) / N
B = -0.5 * J @ D2 @ J
vals, vecs = np.linalg.eigh(B)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]
vals2 = np.maximum(vals[:2], 0)
coords = vecs[:, :2] * np.sqrt(vals2)
var_exp = np.sum(vals2) / np.sum(np.maximum(vals, 0))
print(f"Variance explained (2D): {var_exp*100:.1f}%")
print(f"Negative eigenvalues: {np.sum(vals < 0)} (non-Euclidean signature)")

# ── Cluster labels (top 3 bits = 8 clusters of 32 leaves each) ───────────

N_CLUSTERS = 8
n_bits = 3  # log2(8)
cluster_labels = np.array([(i >> (DEPTH - n_bits)) for i in range(N)])

# ── Figure ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

cmap = matplotlib.colormaps.get_cmap('tab10')

# Left: colored by cluster
ax1 = axes[0]
for c in range(N_CLUSTERS):
    mask = cluster_labels == c
    ax1.scatter(coords[mask, 0], coords[mask, 1],
                c=[cmap(c)], s=12, alpha=0.85, linewidths=0,
                label=f'Cluster {c+1}')

ax1.set_title(f'MDS Embedding — Depth {DEPTH} ({N} leaves, single tree)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('MDS Dimension 1', fontsize=10)
ax1.set_ylabel('MDS Dimension 2', fontsize=10)
ax1.legend(loc='upper right', fontsize=8, ncol=2, markerscale=2, framealpha=0.9)
ax1.text(0.02, 0.02,
         f'Variance explained: {var_exp*100:.1f}%\nU = 0.983 ± 0.006',
         transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)

# Right: colored by distance from leaf 0
ax2 = axes[1]
dist_from_0 = D[0, :]
sc = ax2.scatter(coords[:, 0], coords[:, 1],
                 c=dist_from_0, cmap='plasma',
                 s=12, alpha=0.9, linewidths=0)
plt.colorbar(sc, ax=ax2,
             label=r'Information distance from leaf 0, $d_{0j} = -\log\,\mathrm{MI}_{0j}$',
             shrink=0.8)
ax2.scatter(coords[0, 0], coords[0, 1], c='red', s=80,
            zorder=5, marker='*', label='Leaf 0 (reference)')
ax2.legend(fontsize=9)
ax2.set_title('Information Distance from Reference Leaf', fontsize=12, fontweight='bold')
ax2.set_xlabel('MDS Dimension 1', fontsize=10)
ax2.set_ylabel('MDS Dimension 2', fontsize=10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2)

plt.suptitle(
    'Emergent Geometry from Quantum Entanglement Correlations\n'
    r'$d_{ij} = -\log\,\mathrm{MI}_{ij}$, Classical MDS Embedding, Depth 8',
    fontsize=11, y=1.01
)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"\nFigure saved: {OUTPUT_PATH}")
print("\nGeometry summary:")
print(f"  Positive eigenvalues: {np.sum(vals > 0)}")
print(f"  Negative eigenvalues: {np.sum(vals < 0)}")
print(f"  First negative: {vals[vals < 0][0]:.6f}" if np.any(vals < 0) else "  No negatives")
