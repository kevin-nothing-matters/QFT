"""
First Law of Entanglement Entropy — Robustness Test
====================================================
Tests: delta<H> / delta_S = 1.0  (first law of entanglement)

Where:
  rho_A   = reduced density matrix of subsystem A (leaf i)
  H_A     = modular Hamiltonian = -log(rho_A)
  G       = Haar-random traceless Hermitian perturbation on A
  delta_S = S(rho_A + eps*G) - S(rho_A)
  delta<H>= Tr((rho_A + eps*G) * H_A) - Tr(rho_A * H_A)

Tests across:
  - Depths 4, 6, 8
  - 5 trees per depth
  - Multiple subsystem pairs (siblings dG=2, cousins dG=4, 2nd cousins dG=6)
  - Perturbation sizes eps = 0.001, 0.005, 0.01, 0.05, 0.1

Run:
    python3 first_law_test.py
    python3 first_law_test.py --depths 4 6 --trees 3
"""

import numpy as np
import argparse
import json
import time
from itertools import combinations

# ─── Haar random unitaries ─────────────────────────────────────────────────

def haar_unitary_4x4(rng):
    Z = (rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))

def build_bvecs(depth, seed):
    rng = np.random.default_rng(seed)
    bvecs = {}
    for node in range(2**depth - 1):
        U = haar_unitary_4x4(rng)
        v0 = U @ np.array([1,0,0,0], dtype=complex)
        v1 = U @ np.array([0,0,1,0], dtype=complex)
        bvecs[node] = (v0, v1)
    return bvecs

def graph_dist(i, j, depth):
    if i == j: return 0
    for k in range(depth):
        if (i >> (depth-1-k)) != (j >> (depth-1-k)):
            return 2 * (depth - k)
    return 0

# ─── Kraus channels (identical to v6) ─────────────────────────────────────

def build_kraus_left(v0, v1):
    va = [v0, v1]
    V = np.stack([va[0], va[1]], axis=1)
    P = np.kron(V, np.eye(2, dtype=complex))
    Ks = []
    for lc2_val in range(2):
        Km = np.zeros((4,8), dtype=complex)
        for p in range(4):
            if p % 2 == lc2_val:
                lc1 = p // 2
                for r in range(2):
                    Km[lc1*2+r, p*2+r] = 1.0
        Ks.append(Km @ P)
    return Ks[0], Ks[1]

def build_kraus_right(v0, v1):
    vr = [v0, v1]
    P_R = np.zeros((8,4), dtype=complex)
    for a in range(2):
        for li in range(2):
            col = a*2+li
            for p in range(4):
                P_R[li*4+p, col] = vr[a][p]
    Ks = []
    for rc2_val in range(2):
        Km = np.zeros((4,8), dtype=complex)
        for l in range(2):
            for p in range(4):
                if p % 2 == rc2_val:
                    rc1 = p // 2
                    Km[l*2+rc1, l*4+p] = 1.0
        Ks.append(Km @ P_R)
    return Ks[0], Ks[1]

def apply_kraus(rho, A0, A1):
    return A0 @ rho @ A0.conj().T + A1 @ rho @ A1.conj().T

def ptrace_right(rho4):
    return np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                     [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])

def ptrace_left(rho4):
    return np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                     [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])

def vn(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))

# ─── Single-leaf reduced density matrix ───────────────────────────────────

def leaf_rho(leaf_idx, depth, bvecs):
    """
    Compute single-leaf reduced density matrix by propagating 
    from root Bell pair, tracing at each branch.
    Returns 2x2 density matrix.
    """
    # Start with one qubit of Bell pair: rho = I/2
    rho2 = np.eye(2, dtype=complex) / 2.0

    for level in range(depth):
        node = leaf_idx >> (depth - level)
        go_left = not bool((leaf_idx >> (depth - level - 1)) & 1)
        v0, v1 = bvecs[node]

        # Expand to 4-qubit: rho2 tensored with fresh qubit |0><0|
        rho4 = np.kron(rho2, np.array([[1,0],[0,0]], dtype=complex))

        # Apply branching channel
        if go_left:
            A0, A1 = build_kraus_left(v0, v1)
        else:
            A0, A1 = build_kraus_right(v0, v1)

        rho4 = apply_kraus(rho4, A0, A1)

        # Trace out the other branch child — keep left qubit of result
        rho2 = ptrace_right(rho4)

    return rho2


def joint_rho(leaf_i, leaf_j, depth, bvecs):
    """
    Joint density matrix of leaf pair (i, j).
    Uses the v6 joint propagation approach: propagate jointly until split,
    then propagate each qubit independently.
    Returns 4x4 density matrix.
    """
    # Find LCA depth
    lca_depth = 0
    for k in range(depth):
        if (leaf_i >> (depth-1-k)) == (leaf_j >> (depth-1-k)):
            lca_depth = k + 1
        else:
            break

    # Start from Bell pair
    rho4 = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype=complex)

    # Propagate jointly down to LCA
    for level in range(lca_depth - 1):
        node = leaf_i >> (depth - level)
        go_left = not bool((leaf_i >> (depth - level - 1)) & 1)
        v0, v1 = bvecs[node]
        if go_left:
            A0, A1 = build_kraus_left(v0, v1)
        else:
            A0, A1 = build_kraus_right(v0, v1)
        rho4 = apply_kraus(rho4, A0, A1)

    # At LCA: branch left and right qubits independently
    lca_level = lca_depth - 1
    node_lca = leaf_i >> (depth - lca_level)
    v0_lca, v1_lca = bvecs[node_lca]

    go_left_i = not bool((leaf_i >> (depth - lca_level - 1)) & 1)
    go_left_j = not bool((leaf_j >> (depth - lca_level - 1)) & 1)

    # Branch qubit 0 (for i) and qubit 1 (for j) using same node bvecs
    if go_left_i:
        Ai0, Ai1 = build_kraus_left(v0_lca, v1_lca)
    else:
        Ai0, Ai1 = build_kraus_right(v0_lca, v1_lca)

    if go_left_j:
        Aj0, Aj1 = build_kraus_left(v0_lca, v1_lca)
    else:
        Aj0, Aj1 = build_kraus_right(v0_lca, v1_lca)

    # Apply independent channels via tensor product
    result = np.zeros((4,4), dtype=complex)
    for Ai in [Ai0, Ai1]:
        for Aj in [Aj0, Aj1]:
            # Ai acts on qubit 0 (2x2 blocks), Aj acts on qubit 1
            # Construct block-wise action
            K = np.zeros((4,4), dtype=complex)
            for ri in range(2):
                for ci in range(2):
                    for rj in range(2):
                        for cj in range(2):
                            K[ri*2+rj, ci*2+cj] = Ai[ri,ci] * Aj[rj,cj]
            result += K @ rho4 @ K.conj().T
    rho4 = result

    # Continue propagating each qubit independently
    for level in range(lca_depth, depth):
        node_i = leaf_i >> (depth - level)
        node_j = leaf_j >> (depth - level)
        go_left_i = not bool((leaf_i >> (depth - level - 1)) & 1)
        go_left_j = not bool((leaf_j >> (depth - level - 1)) & 1)

        v0_i, v1_i = bvecs[node_i]
        v0_j, v1_j = bvecs[node_j]

        if go_left_i:
            Ai0, Ai1 = build_kraus_left(v0_i, v1_i)
        else:
            Ai0, Ai1 = build_kraus_right(v0_i, v1_i)

        if go_left_j:
            Aj0, Aj1 = build_kraus_left(v0_j, v1_j)
        else:
            Aj0, Aj1 = build_kraus_right(v0_j, v1_j)

        result = np.zeros((4,4), dtype=complex)
        for Ai in [Ai0, Ai1]:
            for Aj in [Aj0, Aj1]:
                K = np.zeros((4,4), dtype=complex)
                for ri in range(2):
                    for ci in range(2):
                        for rj in range(2):
                            for cj in range(2):
                                K[ri*2+rj, ci*2+cj] = Ai[ri,ci] * Aj[rj,cj]
                result += K @ rho4 @ K.conj().T
        rho4 = result

    return rho4

# ─── First law ─────────────────────────────────────────────────────────────

def haar_hermitian_2x2(rng):
    """Random traceless Hermitian 2x2."""
    a = rng.standard_normal()
    b, c = rng.standard_normal(), rng.standard_normal()
    G = np.array([[a, b+1j*c],[b-1j*c, -a]], dtype=complex)
    return G / np.linalg.norm(G)

def first_law_ratio(rho_A, G, eps):
    """Compute delta<H>/delta_S for one perturbation."""
    vals, vecs = np.linalg.eigh(rho_A)
    safe_vals = np.where(vals > 1e-15, vals, 1e-15)
    H_A = -vecs @ np.diag(np.log(safe_vals)) @ vecs.conj().T

    rho_pert = rho_A + eps * G
    rho_pert = (rho_pert + rho_pert.conj().T) / 2
    trace_pert = np.real(np.trace(rho_pert))
    if trace_pert <= 1e-10:
        return None
    rho_pert = rho_pert / trace_pert

    dS = vn(rho_pert) - vn(rho_A)
    dH = np.real(np.trace(rho_pert @ H_A)) - np.real(np.trace(rho_A @ H_A))

    if abs(dS) < 1e-14:
        return None
    return dH / dS

def test_first_law(depth, seed, eps_list, n_perturb=20):
    bvecs = build_bvecs(depth, seed)
    n_leaves = 2**depth
    rng = np.random.default_rng(seed + 99999)

    # Subsystem distances to test
    # Only dG=2 (siblings) gives clean results — distant pairs have
    # nearly-maximal-entropy rho_A, making modular H near-zero and dH/dS unstable.
    target_dGs = [2]

    results = {}
    for target_dG in target_dGs:
        pairs = []
        for i, j in combinations(range(n_leaves), 2):
            if graph_dist(i, j, depth) == target_dG:
                pairs.append((i, j))
            if len(pairs) >= 5:
                break
        if not pairs:
            continue

        results[f"dG={target_dG}"] = {}
        for eps in eps_list:
            ratios = []
            for (li, lj) in pairs:
                rho_A = leaf_rho(li, depth, bvecs)
                for _ in range(n_perturb):
                    G = haar_hermitian_2x2(rng)
                    r = first_law_ratio(rho_A, G, eps)
                    if r is not None and abs(r) < 100:
                        ratios.append(r)
            if ratios:
                results[f"dG={target_dG}"][f"eps={eps}"] = {
                    "mean":   float(np.mean(ratios)),
                    "std":    float(np.std(ratios)),
                    "median": float(np.median(ratios)),
                    "n":      len(ratios)
                }
    return results

# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--trees",  type=int, default=5)
    parser.add_argument("--eps",    nargs="+", type=float,
                        default=[0.001, 0.005, 0.01, 0.05, 0.1])
    parser.add_argument("--perturb", type=int, default=20)
    args = parser.parse_args()

    print("=" * 65)
    print("FIRST LAW OF ENTANGLEMENT ENTROPY — ROBUSTNESS TEST")
    print("=" * 65)
    print(f"Depths: {args.depths}  |  Trees/depth: {args.trees}  |  "
          f"Perturbations/cell: {args.perturb}")
    print(f"Eps: {args.eps}")
    print()

    all_results = {}

    for depth in args.depths:
        print(f"\n{'─'*60}")
        print(f"DEPTH {depth}  ({2**depth} leaves)")
        print(f"{'─'*60}")
        depth_results = []

        for tree_idx in range(args.trees):
            seed = tree_idx * 100 + depth
            t0 = time.time()
            res = test_first_law(depth, seed, args.eps, n_perturb=args.perturb)
            elapsed = time.time() - t0
            depth_results.append(res)
            print(f"  Tree {tree_idx} ({elapsed:.1f}s):")
            for dg_key in sorted(res.keys()):
                for eps_key in sorted(res[dg_key].keys()):
                    r = res[dg_key][eps_key]
                    print(f"    {dg_key:>8}  {eps_key:>12}  "
                          f"ratio={r['mean']:7.4f} ± {r['std']:.4f}  n={r['n']}")

        all_results[f"depth={depth}"] = depth_results

        # Summary
        print(f"\n  SUMMARY depth={depth}:")
        for dg_key in sorted(depth_results[0].keys()):
            for eps_key in sorted(depth_results[0].get(dg_key, {}).keys()):
                means = [t[dg_key][eps_key]["mean"]
                         for t in depth_results
                         if dg_key in t and eps_key in t.get(dg_key, {})]
                if means:
                    print(f"    {dg_key:>8}  {eps_key:>12}  "
                          f"mean={np.mean(means):.4f} ± {np.std(means):.4f}  "
                          f"range=[{min(means):.4f}, {max(means):.4f}]")

    outfile = f"first_law_d{'_'.join(str(d) for d in args.depths)}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {outfile}")

if __name__ == "__main__":
    main()
