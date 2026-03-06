"""
Quantum Family Tree — Depth 12 Runner
======================================
Direct port of the proven run_d8_d10.py superoperator method,
adapted for depth 12 (4,096 leaves, 8,386,560 pairs per tree).

Run:
    nohup python3 run_d12.py --trees 1 --start-tree 0 > ~/d12_run.log 2>&1 &

Output:
    ~/qft/results/depth12_tree000.json  (same format as depth8/10 files)
"""

import numpy as np
import json
import os
import time
import argparse
from itertools import combinations

DEPTH = 12
N_LEAVES = 2**DEPTH  # 4096
RESULTS_DIR = os.path.expanduser("~/qft/results/")

def graph_dist(i, j, depth=DEPTH):
    if i == j: return 0
    for k in range(depth):
        if (i >> (depth-1-k)) != (j >> (depth-1-k)):
            return 2 * (depth - k)
    return 0

def haar_unitary_2x2(rng):
    """Haar-random 2x2 unitary."""
    Z = (rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))

def haar_unitary_4x4(rng):
    """Haar-random 4x4 unitary."""
    Z = (rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))

def von_neumann(rho):
    """Von Neumann entropy in nats."""
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))

def mutual_info(rho4):
    """MI from 4x4 joint density matrix in nats."""
    rho_A = np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                      [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])
    rho_B = np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                      [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])
    return von_neumann(rho_A) + von_neumann(rho_B) - von_neumann(rho4)

def branch_vectors(U):
    """
    Precompute the two branch vectors from a 4x4 unitary.
    v0 = U|00>, v1 = U|10> (after CNOT: parent 0->|00>, parent 1->|10>)
    """
    v0 = U @ np.array([1,0,0,0], dtype=complex)
    v1 = U @ np.array([0,0,1,0], dtype=complex)
    return v0, v1

def apply_branch_single(rho2, v0, v1):
    """
    Apply branching to single qubit state rho2 (2x2) -> 4x4 joint child state.
    rho4 = rho[0,0]|v0><v0| + rho[0,1]|v0><v1| + rho[1,0]|v1><v0| + rho[1,1]|v1><v1|
    """
    return (rho2[0,0]*np.outer(v0,v0.conj()) +
            rho2[0,1]*np.outer(v0,v1.conj()) +
            rho2[1,0]*np.outer(v1,v0.conj()) +
            rho2[1,1]*np.outer(v1,v1.conj()))

def ptrace_right(rho4):
    """Partial trace over right qubit: 4x4 -> 2x2."""
    return np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                     [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])

def ptrace_left(rho4):
    """Partial trace over left qubit: 4x4 -> 2x2."""
    return np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                     [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])

def compute_mi_pair(i, j, bvecs):
    """
    Compute MI between leaves i and j using superoperator contraction.
    bvecs: dict {node_idx: (v0, v1)} precomputed branch vectors
    """
    # Find LCA depth
    lca_steps = 0
    for k in range(DEPTH):
        if (i >> (DEPTH-1-k)) != (j >> (DEPTH-1-k)):
            lca_steps = DEPTH - k
            break

    lca_gen = DEPTH - lca_steps

    # Propagate single qubit from root to LCA
    rho = np.array([[1,0],[0,0]], dtype=complex)
    node = 0
    for gen in range(lca_gen):
        v0, v1 = bvecs[node]
        rho4 = apply_branch_single(rho, v0, v1)
        # Which child leads toward LCA?
        # LCA node index at generation lca_gen: i >> lca_steps
        # At generation gen+1, the ancestor of LCA is at:
        lca_idx_at_gen1 = (i >> lca_steps) >> (lca_gen - gen - 1)
        go_left = (lca_idx_at_gen1 % 2 == 0)
        rho = ptrace_right(rho4) if go_left else ptrace_left(rho4)
        node = 2*node + 1 + (0 if go_left else 1)

    # Apply LCA branch -> joint 4x4
    v0, v1 = bvecs[node]
    rho4_joint = apply_branch_single(rho, v0, v1)

    if lca_steps == 1:
        # Leaves are direct children of LCA
        return mutual_info(rho4_joint)

    # Propagate joint state down left path (to i) and right path (to j)
    # Left child of LCA
    left_node = 2*node + 1
    right_node = 2*node + 2

    # Get bits for path from LCA's children to leaves
    i_bits = [(i >> (lca_steps - 1 - k)) & 1 for k in range(lca_steps)]
    j_bits = [(j >> (lca_steps - 1 - k)) & 1 for k in range(lca_steps)]

    # Propagate left side: rho_L = ptrace of joint (keep left qubit initially)
    rho_L = ptrace_right(rho4_joint)
    rho_R = ptrace_left(rho4_joint)

    for step in range(lca_steps - 1):
        # Left path
        v0_L, v1_L = bvecs[left_node]
        rho4_L = apply_branch_single(rho_L, v0_L, v1_L)
        go_left_L = (i_bits[step+1] == 0)
        rho_L = ptrace_right(rho4_L) if go_left_L else ptrace_left(rho4_L)
        left_node = 2*left_node + 1 + (0 if go_left_L else 1)

        # Right path
        v0_R, v1_R = bvecs[right_node]
        rho4_R = apply_branch_single(rho_R, v0_R, v1_R)
        go_left_R = (j_bits[step+1] == 0)
        rho_R = ptrace_right(rho4_R) if go_left_R else ptrace_left(rho4_R)
        right_node = 2*right_node + 1 + (0 if go_left_R else 1)

    # Reconstruct joint state from marginals
    # Note: after independent propagation, correlations are preserved via
    # the initial joint state. Approximate joint as product + LCA correlation.
    # For exact MI, we need to track the full 4x4 joint state.
    # Use the exact method: propagate the full joint state.
    
    # Restart with exact joint propagation
    rho4_joint_curr = rho4_joint.copy()
    left_node = 2*node + 1
    right_node = 2*node + 2

    for step in range(lca_steps - 1):
        go_left_L = (i_bits[step+1] == 0)
        go_left_R = (j_bits[step+1] == 0)

        v0_L, v1_L = bvecs[left_node]
        v0_R, v1_R = bvecs[right_node]

        # Apply left branch to left qubit of joint state
        # Left qubit: dims 0,1 of 4x4; right qubit: dims 2,3
        # After left branch: left qubit -> 2 qubits (4 dims), right stays (2 dims) -> 8x8
        # Then trace out unwanted left child
        # Efficient: use Kronecker structure
        
        # Branch left qubit
        # rho4_joint_curr is (left, right) = (2,2)
        # After left branch: (left_ch1, left_ch2, right) = (2,2,2) -> 8x8
        # Apply: (v0_L ⊗ I_2) and (v1_L ⊗ I_2) style
        
        rho8 = np.zeros((8,8), dtype=complex)
        for a in range(2):  # left input basis
            for b in range(2):  # left input basis
                amp = rho4_joint_curr[a*2:a*2+2, b*2:b*2+2]  # 2x2 right subblock
                va = v0_L if a==0 else v1_L  # 4-dim
                vb = v0_L if b==0 else v1_L
                # Outer product in left space, keep right
                for ri in range(2):
                    for rj in range(2):
                        rho8[ri::2, rj::2] += amp[ri,rj] * np.outer(va, vb.conj())

        # Trace out unwanted left child
        if go_left_L:
            # Keep left child 1 (indices 0,2 of 4-dim left)
            # In 8x8: left_ch1 is bit 1, left_ch2 is bit 0, right is bit... 
            # Reindex: 8 = left_ch1(2) x left_ch2(2) x right(2)
            # Keep left_ch1, trace left_ch2
            rho4_new = np.zeros((4,4), dtype=complex)
            for lc2 in range(2):
                for lc1 in range(2):
                    for r in range(2):
                        i_idx = lc1*4 + lc2*2 + r
                        for lc1b in range(2):
                            for rb in range(2):
                                j_idx = lc1b*4 + lc2*2 + rb
                                rho4_new[lc1*2+r, lc1b*2+rb] += rho8[i_idx, j_idx]
        else:
            # Keep left child 2
            rho4_new = np.zeros((4,4), dtype=complex)
            for lc1 in range(2):
                for lc2 in range(2):
                    for r in range(2):
                        i_idx = lc1*4 + lc2*2 + r
                        for lc2b in range(2):
                            for rb in range(2):
                                j_idx = lc1*4 + lc2b*2 + rb
                                rho4_new[lc2*2+r, lc2b*2+rb] += rho8[i_idx, j_idx]

        rho4_joint_curr = rho4_new

        # Now branch right qubit similarly
        rho8 = np.zeros((8,8), dtype=complex)
        # rho4_joint_curr is now (kept_left_child, right)
        for a in range(2):
            for b in range(2):
                amp = rho4_joint_curr[a*2:a*2+2, b*2:b*2+2]
                va = v0_R if a==0 else v1_R
                vb = v0_R if b==0 else v1_R
                for li in range(2):
                    for lj in range(2):
                        # left is outer, right branches
                        outer = np.outer(va, vb.conj())
                        rho8[li*4:li*4+4, lj*4:lj*4+4] += amp[li,lj] * outer

        if go_left_R:
            rho4_new = np.zeros((4,4), dtype=complex)
            for rc2 in range(2):
                for l in range(2):
                    for rc1 in range(2):
                        i_idx = l*4 + rc1*2 + rc2
                        for lb in range(2):
                            for rc1b in range(2):
                                j_idx = lb*4 + rc1b*2 + rc2
                                rho4_new[l*2+rc1, lb*2+rc1b] += rho8[i_idx, j_idx]
        else:
            rho4_new = np.zeros((4,4), dtype=complex)
            for rc1 in range(2):
                for l in range(2):
                    for rc2 in range(2):
                        i_idx = l*4 + rc1*2 + rc2
                        for lb in range(2):
                            for rc2b in range(2):
                                j_idx = lb*4 + rc1*2 + rc2b
                                rho4_new[l*2+rc2, lb*2+rc2b] += rho8[i_idx, j_idx]

        rho4_joint_curr = rho4_new
        left_node = 2*left_node + 1 + (0 if go_left_L else 1)
        right_node = 2*right_node + 1 + (0 if go_left_R else 1)

    return mutual_info(rho4_joint_curr)

def run_tree(tree_idx, depth=DEPTH):
    seed = tree_idx * 1000 + depth
    rng = np.random.default_rng(seed)
    n_internal = 2**depth - 1

    print(f"  Generating {n_internal:,} unitaries...")
    bvecs = {}
    for node in range(n_internal):
        U = haar_unitary_4x4(rng)
        bvecs[node] = branch_vectors(U)

    pairs = list(combinations(range(2**depth), 2))
    n_pairs = len(pairs)
    results = {}

    print(f"  Computing {n_pairs:,} pairs...")
    t0 = time.time()
    progress_every = 100000

    for idx, (i, j) in enumerate(pairs):
        if idx > 0 and idx % progress_every == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            remaining = (n_pairs - idx) / rate
            pct = 100 * idx / n_pairs
            print(f"  {idx:,}/{n_pairs:,} ({pct:.1f}%) "
                  f"— {elapsed/3600:.2f}h elapsed, "
                  f"~{remaining/3600:.2f}h remaining")

        dG = graph_dist(i, j, depth)
        mi = compute_mi_pair(i, j, bvecs)
        results.setdefault(str(dG), []).append(float(mi))

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed/3600:.2f}h")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trees', type=int, default=1)
    parser.add_argument('--start-tree', type=int, default=0)
    parser.add_argument('--output', type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    n_leaves = 2**DEPTH
    n_pairs = n_leaves*(n_leaves-1)//2
    print(f"Depth {DEPTH} | Leaves: {n_leaves:,} | Pairs: {n_pairs:,}")

    for tree_idx in range(args.start_tree, args.start_tree + args.trees):
        outfile = os.path.join(args.output, f"depth{DEPTH}_tree{tree_idx:03d}.json")
        if os.path.exists(outfile):
            print(f"Tree {tree_idx}: exists, skipping")
            continue
        print(f"\nTree {tree_idx}...")
        t0 = time.time()
        results = run_tree(tree_idx)
        with open(outfile, 'w') as f:
            json.dump(results, f)
        print(f"Saved: {outfile}")

        # Print summary
        for dG in sorted(results.keys(), key=int):
            vals = np.array(results[dG])
            print(f"  dG={dG:2s}: n={len(vals):,} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f}")

if __name__ == '__main__':
    main()
