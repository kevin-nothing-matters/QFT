import math
import itertools
import time
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    xp = np
    GPU = False

# ── Core math ──────────────────────────────────────────────────────────────
def kron(a, b): return xp.kron(a, b)
def eye(n): return xp.eye(n, dtype=xp.complex128)

def von_neumann_entropy_bits(rho, eps=1e-15):
    w = xp.clip(xp.real(xp.linalg.eigvalsh(rho)), eps, 1.0)
    return float(xp.sum(-w * xp.log2(w)))

def partial_trace(rho4, keep):
    r = rho4.reshape(2, 2, 2, 2)
    if keep == 0: return xp.trace(r, axis1=1, axis2=3)
    return xp.trace(r, axis1=0, axis2=2)

def mutual_information(rho4):
    return (von_neumann_entropy_bits(partial_trace(rho4, 0))
          + von_neumann_entropy_bits(partial_trace(rho4, 1))
          - von_neumann_entropy_bits(rho4))

# ── Branching channel ──────────────────────────────────────────────────────
def cnot_c0t1():
    U = xp.zeros((4, 4), dtype=xp.complex128)
    U[0,0]=1; U[1,1]=1; U[3,2]=1; U[2,3]=1
    return U

def embed_2q_into_3q(U2, which_pair):
    a, b = which_pair
    U3 = xp.zeros((8, 8), dtype=xp.complex128)
    for s in range(8):
        bits = [(s >> (2-q)) & 1 for q in (0,1,2)]
        ip = (bits[a] << 1) | bits[b]
        for op in range(4):
            bo = bits.copy(); bo[a]=(op>>1)&1; bo[b]=op&1
            U3[(bo[0]<<2)|(bo[1]<<1)|bo[2], s] += U2[op, ip]
    return U3

def make_haar_unitary_4(rng):
    X = (rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(X)
    return Q * (np.diag(R) / np.abs(np.diag(R)))

def make_kraus(H):
    """Build Kraus operators for: CNOT(p->c1), CNOT(p->c2), Haar(c1,c2), trace p.
    A has shape (4,2): maps 1 parent qubit -> 2 child qubits."""
    Ut = embed_2q_into_3q(H, (1,2)) @ embed_2q_into_3q(cnot_c0t1(),(0,2)) @ embed_2q_into_3q(cnot_c0t1(),(0,1))
    K = []
    for s in (0, 1):
        A = xp.zeros((4, 2), dtype=xp.complex128)
        for p in (0, 1):
            col = Ut[:, (p<<2)|0]
            for c1 in (0,1):
                for c2 in (0,1): A[(c1<<1)|c2, p] = col[(s<<2)|(c1<<1)|c2]
        K.append(A)
    return K

def apply_1to2(K, rho1):
    """Apply branching channel: 1-qubit rho1 -> 2-qubit joint rho."""
    return sum(A @ rho1 @ A.conj().T for A in K)

def apply_1to1(K, rho1, keep_child):
    """Apply branching channel and trace out the non-target child."""
    return partial_trace(apply_1to2(K, rho1), keep_child)

# ── Tree construction ──────────────────────────────────────────────────────
def build_gates(depth, seed):
    """Build gates_dict[(gen, node_idx)] = Kraus ops.
    Node indices: root=0, left child of node n = 2n+1, right = 2n+2."""
    rng = np.random.default_rng(seed)
    gates = {}
    for gen in range(depth):
        for node in range(2**gen - 1, 2**(gen+1) - 1):
            H = make_haar_unitary_4(rng)
            gates[(gen, node)] = make_kraus(H)
    return gates

def get_path(leaf, depth):
    """Path from root to leaf as list of child choices (0=left,1=right)."""
    return [(leaf >> (depth-1-g)) & 1 for g in range(depth)]

def graph_dist(i, j, depth):
    if i == j: return 0
    return 2 * (i ^ j).bit_length()

# ── Correct MI computation via joint state propagation ────────────────────
def compute_mi_pair(i, j, depth, gates):
    """
    Exact MI(leaf_i, leaf_j) via joint density matrix propagation.

    Algorithm:
      1. Walk root -> LCA maintaining single-qubit state (pure root |0><0|)
      2. At LCA apply branching -> joint 2-qubit state of LCA's two children
      3. Propagate joint 4x4 state down both paths simultaneously
         (valid because paths below LCA are causally independent)
      4. Return final 4x4 joint density matrix of (leaf_i, leaf_j)
    """
    pi, pj = get_path(i, depth), get_path(j, depth)

    # Find LCA
    lca_gen = depth
    for k in range(depth):
        if pi[k] != pj[k]:
            lca_gen = k
            break

    # Walk root -> LCA, propagating single-qubit state
    rho = xp.array([[1., 0.], [0., 0.]], dtype=xp.complex128)  # pure |0><0|
    node = 0
    for gen in range(lca_gen):
        choice = pi[gen]
        rho = apply_1to1(gates[(gen, node)], rho, keep_child=choice)
        node = 2*node + 1 + choice

    # At LCA: get joint 2-qubit state of both children
    rho_joint = apply_1to2(gates[(lca_gen, node)], rho)  # 4x4

    # Track which child leads to i vs j
    ni = 2*node + 1 + pi[lca_gen]
    nj = 2*node + 1 + pj[lca_gen]

    # Propagate joint state down both paths independently
    for gen in range(lca_gen + 1, depth):
        ci, cj = pi[gen], pj[gen]
        K_i, K_j = gates[(gen, ni)], gates[(gen, nj)]

        # Apply K_i to qubit-0 of rho_joint, keep child ci
        I2 = eye(2)
        rho_8 = sum(kron(A, I2) @ rho_joint @ kron(A, I2).conj().T for A in K_i)
        r8 = rho_8.reshape(2, 2, 2, 2, 2, 2)
        rho_joint = (xp.trace(r8, axis1=1, axis2=4) if ci == 0
                     else xp.trace(r8, axis1=0, axis2=3)).reshape(4, 4)

        # Apply K_j to qubit-1 of rho_joint, keep child cj
        rho_8b = sum(kron(I2, B) @ rho_joint @ kron(I2, B).conj().T for B in K_j)
        r8b = rho_8b.reshape(2, 2, 2, 2, 2, 2)
        rho_joint = (xp.trace(r8b, axis1=2, axis2=5) if cj == 0
                     else xp.trace(r8b, axis1=1, axis2=4)).reshape(4, 4)

        ni = 2*ni + 1 + ci
        nj = 2*nj + 1 + cj

    return rho_joint

# ── Main run ───────────────────────────────────────────────────────────────
def run_exhaustive(depth=6, n_trees=20, start_tree=0, results_dir="/home/3x-agent/qft/results"):
    os.makedirs(results_dir, exist_ok=True)
    n_leaves = 2**depth
    all_pairs = list(itertools.combinations(range(n_leaves), 2))
    n_pairs = len(all_pairs)

    print(f"Backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}")
    print(f"depth={depth}  leaves={n_leaves}  pairs={n_pairs}  trees={n_trees}")
    print(f"Results → {results_dir}\n")

    all_bins = {}
    grand_start = time.time()

    for tree_idx in range(start_tree, n_trees):
        seed = 1000 + tree_idx
        gates = build_gates(depth, seed)
        tree_bins = {}
        t0 = time.time()

        for k, (i, j) in enumerate(all_pairs):
            dG = graph_dist(i, j, depth)
            rho_ij = compute_mi_pair(i, j, depth, gates)
            Iij = max(mutual_information(rho_ij), 1e-12)
            tree_bins.setdefault(dG, []).append(Iij)
            all_bins.setdefault(dG, []).append(Iij)

            if (k+1) % 500 == 0 or k == n_pairs-1:
                elapsed = time.time() - t0
                rate = (k+1) / elapsed
                eta = (n_pairs - k - 1) / rate
                print(f"  Tree {tree_idx+1:2d}/{n_trees} | pair {k+1:5d}/{n_pairs} | "
                      f"{rate:.0f} pairs/s | ETA {eta:.0f}s", flush=True)

        print(f"  Tree {tree_idx+1:2d} done in {time.time()-t0:.1f}s:")
        for dG in sorted(tree_bins):
            vals = np.array(tree_bins[dG])
            print(f"    dG={dG:2d}  n={len(vals):4d}  mean={vals.mean():.6f}  std={vals.std():.6f}")

        tree_out = {str(k): [float(x) for x in v] for k, v in tree_bins.items()}
        with open(f"{results_dir}/depth{depth}_tree{tree_idx:03d}.json", "w") as f:
            json.dump(tree_out, f)
        print()

    # Ensemble summary
    print(f"\n{'='*60}")
    print(f"ENSEMBLE SUMMARY: depth={depth}, {n_trees} trees, {n_pairs} pairs/tree")
    print(f"Total elapsed: {time.time()-grand_start:.1f}s")
    print(f"{'='*60}")

    summary = {}
    for dG in sorted(all_bins):
        vals = np.array(all_bins[dG])
        summary[dG] = {"mean": float(vals.mean()), "std": float(vals.std()),
                       "n": len(vals), "median": float(np.median(vals))}
        print(f"dG={dG:2d}  n={len(vals):6d}  mean={vals.mean():.6f}  "
              f"std={vals.std():.6f}  median={np.median(vals):.6f}")

    # Fit alpha
    distances = np.array(sorted(summary.keys()), dtype=float)
    means = np.array([summary[int(d)]["mean"] for d in distances])
    log_means = np.log(np.clip(means, 1e-12, None))
    coeffs = np.polyfit(distances, log_means, 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])
    monotone = all(means[k] > means[k+1] for k in range(len(means)-1))
    print(f"\nExponential fit: MI = {A:.4f} * exp(-{alpha:.4f} * dG)")
    print(f"alpha (depth {depth}) = {alpha:.4f}")
    print(f"Monotone decay: {monotone}")

    with open(f"{results_dir}/depth{depth}_ensemble_summary.json", "w") as f:
        json.dump({"depth": depth, "n_trees": n_trees, "n_pairs": n_pairs,
                   "alpha": alpha, "A": A, "monotone": monotone,
                   "summary": {str(k): v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved: {results_dir}/depth{depth}_ensemble_summary.json")
    return all_bins, alpha

if __name__ == "__main__":
    run_exhaustive(depth=10, n_trees=20, start_tree=10)
