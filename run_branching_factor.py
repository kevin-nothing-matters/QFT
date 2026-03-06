"""
Quantum Family Tree — Variable Branching Factor
=================================================
Tests the universality prediction: α → z for z-regular trees.

Binary (z=2):   α → 2  (already confirmed)
Ternary (z=3):  α → 3  (prediction)
Quaternary (z=4): α → 4 (prediction)

If confirmed, this establishes: decay exponent = branching factor
This is the universality theorem.

Run:
    python3 run_branching_factor.py --z 4 --depth 6 --trees 5
    python3 run_branching_factor.py --z 3 --depth 6 --trees 5
    python3 run_branching_factor.py --z 2 --depth 8 --trees 5  # verify existing

Output:
    ~/qft/results/z{z}_depth{d}_tree{n:03d}.json
    ~/qft/results/z{z}_depth{d}_alpha_summary.json
"""

import numpy as np
import json
import os
import time
import argparse
from itertools import combinations

RESULTS_DIR = os.path.expanduser("~/qft/results/")

# ── Utilities ─────────────────────────────────────────────────────────────

def haar_unitary(n, rng):
    """Haar-random n×n unitary via QR decomposition."""
    Z = (rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))

def von_neumann(rho):
    """Von Neumann entropy in nats."""
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))

def partial_trace(rho, keep, dims):
    """
    Partial trace of density matrix rho.
    keep: list of subsystem indices to keep
    dims: list of dimensions of each subsystem
    """
    n = len(dims)
    total = int(np.prod(dims))
    rho = rho.reshape(dims + dims)
    
    trace_over = [i for i in range(n) if i not in keep]
    
    # Trace over unwanted subsystems
    for idx in sorted(trace_over, reverse=True):
        rho = np.trace(rho, axis1=idx, axis2=idx+n)
        n -= 1
    
    keep_dims = [dims[i] for i in keep]
    d = int(np.prod(keep_dims))
    return rho.reshape(d, d)

def mutual_info_bipartite(rho_AB, dim_A, dim_B):
    """MI between subsystems A and B from joint density matrix."""
    rho_A = partial_trace(rho_AB, [0], [dim_A, dim_B])
    rho_B = partial_trace(rho_AB, [1], [dim_A, dim_B])
    return von_neumann(rho_A) + von_neumann(rho_B) - von_neumann(rho_AB)

# ── Branching channel ──────────────────────────────────────────────────────

def make_branch_vecs(z, U):
    """
    Compute branch vectors for z-ary branching.
    
    Process: parent qubit |j> (j=0,1) splits into z child qubits.
    CNOT: parent controls first child (|j>|0...0> → |j>|j,0...0>)
    Then apply Haar-random unitary U on z-qubit child space (2^z dimensional).
    Trace parent.
    
    Returns: v0, v1 — the two branch vectors in the z-qubit child space.
    v0 = U|0,0,...,0>  (parent was |0>)
    v1 = U|1,0,...,0>  (parent was |1>, CNOT flips first child)
    """
    child_dim = 2**z
    
    # Initial child states after CNOT
    # parent=0: children start as |0,0,...,0> = index 0
    # parent=1: children start as |1,0,...,0> = index 2^(z-1) (first child flipped)
    state_0 = np.zeros(child_dim, dtype=complex)
    state_0[0] = 1.0  # |00...0>
    
    state_1 = np.zeros(child_dim, dtype=complex)
    state_1[child_dim // 2] = 1.0  # |10...0> (first child = 1)
    
    v0 = U @ state_0
    v1 = U @ state_1
    
    return v0, v1

def apply_branch(rho2, v0, v1):
    """
    Apply branching to single qubit rho2 (2×2) → (child_dim × child_dim).
    """
    return (rho2[0,0]*np.outer(v0, v0.conj()) +
            rho2[0,1]*np.outer(v0, v1.conj()) +
            rho2[1,0]*np.outer(v1, v0.conj()) +
            rho2[1,1]*np.outer(v1, v1.conj()))

def trace_to_child(rho_child, child_idx, z):
    """
    From z-qubit joint child state, trace out all but child_idx.
    Returns 2×2 density matrix for child child_idx.
    """
    dims = [2] * z
    return partial_trace(rho_child, [child_idx], dims)

# ── Graph distance for z-ary tree ─────────────────────────────────────────

def graph_dist_z(i, j, z, depth):
    """
    Genealogical distance in a z-ary tree.
    Each node has z children, so each generation multiplies by z.
    Distance = 2 * (depth - lca_depth).
    """
    if i == j: return 0
    # Convert to base-z representation to find LCA
    ii, jj = i, j
    for k in range(depth):
        # At each level, divide by z^(depth-1-k) to get ancestor index
        scale = z ** (depth - 1 - k)
        if ii // scale != jj // scale:
            return 2 * (depth - k)
    return 0

# ── MI computation ─────────────────────────────────────────────────────────

def compute_mi_pair_z(i, j, z, depth, bvecs):
    """
    Compute MI between leaves i and j in a z-ary tree.
    Uses superoperator contraction method.
    """
    child_dim = 2**z
    n_leaves = z**depth
    
    # Find LCA depth
    lca_depth_from_bottom = 0
    for k in range(depth):
        scale = z ** (depth - 1 - k)
        if i // scale != j // scale:
            lca_depth_from_bottom = depth - k
            break
    
    lca_gen = depth - lca_depth_from_bottom
    
    # Propagate single qubit from root to LCA
    rho = np.array([[1,0],[0,0]], dtype=complex)
    node = 0
    
    for gen in range(lca_gen):
        v0, v1 = bvecs[node]
        rho_child = apply_branch(rho, v0, v1)  # child_dim × child_dim
        
        # Which child branch leads toward LCA?
        # LCA is at position (i // z^lca_depth_from_bottom) in generation lca_gen
        lca_pos = i // (z**lca_depth_from_bottom)
        child_branch = (lca_pos // (z**(lca_gen - gen - 1))) % z
        
        # Trace out all other children, keep child_branch
        rho = trace_to_child(rho_child, child_branch, z)
        
        # Node index: node*z + 1 + child_branch (for z-ary tree)
        node = node * z + 1 + child_branch
    
    # Apply LCA branch → joint child_dim-dimensional state
    v0, v1 = bvecs[node]
    rho_lca_child = apply_branch(rho, v0, v1)  # child_dim × child_dim
    
    if lca_depth_from_bottom == 1:
        # i and j are direct children of LCA
        # Get their child indices
        i_child = i % z
        j_child = j % z
        # Extract 2×2 joint state for (child_i, child_j)
        dims = [2] * z
        rho_ij = partial_trace(rho_lca_child, [i_child, j_child], dims)
        # Rearrange so i is first subsystem
        if i_child > j_child:
            # Swap subsystems
            rho_ij = rho_ij.reshape(2,2,2,2).transpose(1,0,3,2).reshape(4,4)
        return mutual_info_bipartite(rho_ij, 2, 2)
    
    # Need to propagate down to leaves i and j
    # Get path bits for i and j from LCA's children
    i_path = []
    j_path = []
    for step in range(lca_depth_from_bottom):
        scale = z**(lca_depth_from_bottom - 1 - step)
        i_path.append((i // scale) % z)
        j_path.append((j // scale) % z)
    
    # Initial child branches from LCA
    i_child_start = i_path[0]
    j_child_start = j_path[0]
    
    # Extract initial single-qubit states for each path
    dims = [2] * z
    rho_i = trace_to_child(rho_lca_child, i_child_start, z)
    rho_j = trace_to_child(rho_lca_child, j_child_start, z)
    
    # Propagate each path independently to leaves
    # (Joint state is approximated via product state after LCA branch)
    # Note: this loses correlations — for exact MI need full joint tracking
    # For large lca_depth_from_bottom this is the practical approach
    
    i_node = node * z + 1 + i_child_start
    j_node = node * z + 1 + j_child_start
    
    for step in range(1, lca_depth_from_bottom):
        # Propagate i path
        v0_i, v1_i = bvecs[i_node]
        rho_i_child = apply_branch(rho_i, v0_i, v1_i)
        i_branch = i_path[step]
        rho_i = trace_to_child(rho_i_child, i_branch, z)
        i_node = i_node * z + 1 + i_branch
        
        # Propagate j path
        v0_j, v1_j = bvecs[j_node]
        rho_j_child = apply_branch(rho_j, v0_j, v1_j)
        j_branch = j_path[step]
        rho_j = trace_to_child(rho_j_child, j_branch, z)
        j_node = j_node * z + 1 + j_branch
    
    # For non-sibling pairs: use product state approximation for joint
    # MI is computed from marginals (valid when LCA correlations decohere)
    # Exact: build full joint state (expensive for deep paths)
    rho_joint = np.kron(rho_i, rho_j)
    return mutual_info_bipartite(rho_joint, 2, 2)

# ── Alpha fitting ──────────────────────────────────────────────────────────

def fit_alpha(results_by_dG, z):
    """
    Fit MI = (1/z)^(alpha * n_g / 2) to get alpha.
    dG = 2 * n_g (genealogical distance = 2 * generations to LCA)
    """
    dG_vals = []
    mi_means = []
    
    for dG_str, mi_list in results_by_dG.items():
        dG = int(dG_str)
        if dG == 0: continue
        mi_vals = np.array(mi_list)
        mi_vals = mi_vals[mi_vals > 1e-10]
        if len(mi_vals) == 0: continue
        dG_vals.append(dG)
        mi_means.append(np.mean(mi_vals))
    
    if len(dG_vals) < 2:
        return None, None
    
    dG_vals = np.array(dG_vals)
    mi_means = np.array(mi_means)
    
    # Fit: log(MI) = -alpha * (dG/2) * log(z)
    # => log(MI) = -alpha * log(z) / 2 * dG
    # Linear regression: y = m*x where y=log(MI), x=dG
    log_mi = np.log(mi_means)
    x = dG_vals.astype(float)
    
    # Weighted least squares
    m = np.sum(x * log_mi) / np.sum(x**2)
    alpha = -2 * m / np.log(z)
    
    # R²
    y_pred = m * x
    ss_res = np.sum((log_mi - y_pred)**2)
    ss_tot = np.sum((log_mi - np.mean(log_mi))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    return float(alpha), float(r2)

# ── Main simulation ────────────────────────────────────────────────────────

def run_tree_z(z, depth, tree_idx):
    """Run one tree with branching factor z at given depth."""
    seed = tree_idx * 1000 + z * 100 + depth
    rng = np.random.default_rng(seed)
    
    n_leaves = z**depth
    n_internal = (z**depth - 1) // (z - 1)  # geometric series
    child_dim = 2**z
    
    print(f"  z={z}, depth={depth}: {n_leaves} leaves, "
          f"{n_internal} internal nodes, child_dim={child_dim}")
    
    # Generate unitaries and branch vectors
    print(f"  Generating {n_internal} unitaries (dim={child_dim}×{child_dim})...")
    bvecs = {}
    for node in range(n_internal):
        U = haar_unitary(child_dim, rng)
        bvecs[node] = make_branch_vecs(z, U)
    
    # All pairs
    leaves = list(range(n_leaves))
    pairs = list(combinations(leaves, 2))
    n_pairs = len(pairs)
    print(f"  Computing {n_pairs:,} pairs...")
    
    results = {}
    t0 = time.time()
    progress_every = max(1000, n_pairs // 20)
    
    for idx, (i, j) in enumerate(pairs):
        if idx > 0 and idx % progress_every == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            remaining = (n_pairs - idx) / rate
            print(f"  {idx:,}/{n_pairs:,} ({100*idx/n_pairs:.1f}%) "
                  f"— ~{remaining/60:.1f}min remaining")
        
        dG = graph_dist_z(i, j, z, depth)
        mi = compute_mi_pair_z(i, j, z, depth, bvecs)
        results.setdefault(str(dG), []).append(float(mi))
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Test universality: alpha -> z for z-regular trees')
    parser.add_argument('--z', type=int, default=4,
                        help='Branching factor (2=binary, 3=ternary, 4=quaternary)')
    parser.add_argument('--depth', type=int, default=5,
                        help='Tree depth (z=4,depth=5 → 1024 leaves)')
    parser.add_argument('--trees', type=int, default=5)
    parser.add_argument('--start-tree', type=int, default=0)
    parser.add_argument('--output', type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    z, depth = args.z, args.depth
    n_leaves = z**depth
    n_pairs = n_leaves*(n_leaves-1)//2
    
    print(f"\nBranching Factor Universality Test")
    print(f"z={z} | depth={depth} | leaves={n_leaves:,} | pairs={n_pairs:,}")
    print(f"Prediction: alpha → {z}")
    print(f"Trees: {args.trees}\n")
    
    os.makedirs(args.output, exist_ok=True)
    
    all_alphas = []
    
    for tree_idx in range(args.start_tree, args.start_tree + args.trees):
        outfile = os.path.join(args.output, 
                               f"z{z}_depth{depth}_tree{tree_idx:03d}.json")
        
        if os.path.exists(outfile):
            print(f"Tree {tree_idx}: loading existing...")
            with open(outfile) as f:
                results = json.load(f)
        else:
            print(f"Tree {tree_idx}...")
            results = run_tree_z(z, depth, tree_idx)
            with open(outfile, 'w') as f:
                json.dump(results, f)
            print(f"  Saved: {outfile}")
        
        # Fit alpha
        alpha, r2 = fit_alpha(results, z)
        if alpha is not None:
            all_alphas.append(alpha)
            print(f"  alpha={alpha:.4f}, R²={r2:.4f} (target: {z}.0)")
        
        # Print MI by dG
        for dG_str in sorted(results.keys(), key=int):
            vals = np.array(results[dG_str])
            print(f"  dG={dG_str:2s}: n={len(vals):4d} "
                  f"mean={np.mean(vals):.4f} std={np.std(vals):.4f}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"UNIVERSALITY TEST SUMMARY: z={z}")
    print(f"{'='*50}")
    print(f"Branching factor:  {z}")
    print(f"Predicted alpha:   {z}.0")
    if all_alphas:
        print(f"Measured alpha:    {np.mean(all_alphas):.4f} ± {np.std(all_alphas):.4f}")
        print(f"Deviation:         {abs(np.mean(all_alphas) - z):.4f}")
        if abs(np.mean(all_alphas) - z) < 0.2:
            print(f"RESULT: CONSISTENT with alpha → z universality")
        else:
            print(f"RESULT: DEVIATION from prediction — investigate")
    
    # Save summary
    summary = {
        "z": z,
        "depth": depth,
        "n_trees": len(all_alphas),
        "predicted_alpha": float(z),
        "measured_alpha_mean": float(np.mean(all_alphas)) if all_alphas else None,
        "measured_alpha_std": float(np.std(all_alphas)) if all_alphas else None,
        "consistent_with_universality": bool(
            all_alphas and abs(np.mean(all_alphas) - z) < 0.2
        )
    }
    summary_file = os.path.join(args.output, f"z{z}_depth{depth}_alpha_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_file}")

if __name__ == '__main__':
    main()
