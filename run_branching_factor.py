"""
Quantum Family Tree — Variable Branching Factor (v2)
=====================================================
Tests universality prediction: alpha → z for z-regular trees.

Architecture (revised): Each parent qubit splits into z single-qubit children.
Correlations injected via pairwise CNOTs + independent 2x2 Haar unitaries.
This keeps density matrix operations at 2x2 regardless of z.

Prediction:
    z=2: alpha → 2  (confirmed)
    z=3: alpha → 3  (test)
    z=4: alpha → 4  (test)

Run:
    python3 run_branching_factor.py --z 3 --depth 6 --trees 5
    python3 run_branching_factor.py --z 4 --depth 6 --trees 5
"""

import numpy as np
import json, os, time, argparse
from itertools import combinations

RESULTS_DIR = os.path.expanduser("~/qft/results/")

def haar_2x2(rng):
    Z = (rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))

def von_neumann(rho):
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log(vals)))

def mutual_info_2x2(rho4):
    """MI from 4x4 joint state."""
    rho_A = np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                      [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])
    rho_B = np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                      [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])
    return von_neumann(rho_A) + von_neumann(rho_B) - von_neumann(rho4)

def branch_vectors_2x2(U):
    """Precompute branch vectors for 2x2 unitary (binary case)."""
    v0 = U @ np.array([1,0,0,0], dtype=complex)
    v1 = U @ np.array([0,0,1,0], dtype=complex)
    return v0, v1

def apply_branch_single(rho2, v0, v1):
    """Apply branching channel: 2x2 -> 4x4."""
    return (rho2[0,0]*np.outer(v0,v0.conj()) +
            rho2[0,1]*np.outer(v0,v1.conj()) +
            rho2[1,0]*np.outer(v1,v0.conj()) +
            rho2[1,1]*np.outer(v1,v1.conj()))

def ptrace_right(rho4):
    return np.array([[rho4[0,0]+rho4[1,1], rho4[0,2]+rho4[1,3]],
                     [rho4[2,0]+rho4[3,1], rho4[2,2]+rho4[3,3]]])

def ptrace_left(rho4):
    return np.array([[rho4[0,0]+rho4[2,2], rho4[0,1]+rho4[2,3]],
                     [rho4[1,0]+rho4[3,2], rho4[1,1]+rho4[3,3]]])

def make_z_branch(z, rng):
    """
    Generate branching data for z-ary node.
    
    Architecture: parent qubit -> z child qubits (each single qubit).
    Method: use z independent 4x4 unitaries, one per (parent, child_k) pair.
    Each child k gets a branch via: parent ⊗ |0> -> CNOT -> U_k -> trace parent.
    
    Returns: list of z (v0_k, v1_k) pairs, one per child.
    """
    # Each child gets independent 4x4 unitary
    # But all children share the same parent state
    # The correlation between children comes from the shared parent
    children = []
    for k in range(z):
        U = np.kron(np.eye(2), haar_2x2(rng))  # independent per child
        # CNOT: parent ctrl, child tgt
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        V = U @ CNOT
        v0 = V @ np.array([1,0,0,0], dtype=complex)  # parent=0
        v1 = V @ np.array([0,0,1,0], dtype=complex)  # parent=1
        # Keep child qubit (trace parent)
        # v0, v1 are 4-dim; child is second qubit
        # Effective child state vectors (2-dim after tracing parent)
        # For single child: rho_child = ptrace_left(|v><v|)
        # But we need the Kraus form: A_k such that rho_child = sum_s A_s rho_parent A_s†
        # A_s[child, parent] = <s|_parent V |parent> ⊗ |0>_child
        A = np.zeros((2,2), dtype=complex)
        for s in range(2):
            for p in range(2):
                # Input: |p>_parent ⊗ |0>_child = index p*2+0
                # Output component: <s|_parent ⊗ I_child
                # = V[s*2:s*2+2, p*2] (column p*2+0, rows s*2 to s*2+1)
                A[:,p] += V[s*2:s*2+2, p*2]  # sum over parent output s
        # Normalize
        children.append((v0[:2], v1[:2], A))  # store 2-dim effective vectors
    return children

def make_z_branch_v2(z, rng):
    """
    Cleaner z-ary branching: one parent qubit generates z children
    via repeated application of binary branching channel with
    independent 4x4 Haar unitaries.
    
    Child k gets: rho_k derived from parent via U_k.
    Correlations between siblings come from shared parent state.
    
    Returns: list of z (v0, v1) pairs for binary branch formula.
    """
    branches = []
    for k in range(z):
        U4 = np.zeros((4,4), dtype=complex)
        u2 = haar_2x2(rng)
        # Build 4x4: CNOT then u2 on child
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        U_full = np.kron(np.eye(2), u2) @ CNOT
        v0 = U_full[:,0]  # parent=0, child=0 input
        v1 = U_full[:,2]  # parent=1, child=0 input
        branches.append((v0, v1))
    return branches

def graph_dist_z(i, j, z, depth):
    """Genealogical distance in z-ary tree."""
    if i == j: return 0
    for k in range(depth):
        scale = z**(depth-1-k)
        if i//scale != j//scale:
            return 2*(depth-k)
    return 0

def compute_mi_pair_z(i, j, z, depth, node_branches):
    """
    Compute MI between leaves i and j.
    node_branches: dict {node_idx: [(v0_k, v1_k) for k in range(z)]}
    
    Each node has z children. The child index determines which branch to follow.
    MI between i and j: propagate parent to LCA, branch to get joint state
    of LCA's two relevant children subtrees, propagate independently.
    """
    if i == j: return 1.0
    
    # Find LCA
    lca_steps = 0
    for k in range(depth):
        scale = z**(depth-1-k)
        if i//scale != j//scale:
            lca_steps = depth - k
            break
    lca_gen = depth - lca_steps
    
    # Propagate single qubit root -> LCA
    rho = np.array([[1,0],[0,0]], dtype=complex)
    node = 0
    
    for gen in range(lca_gen):
        branches = node_branches[node]
        lca_pos = i//(z**lca_steps)
        child_idx = (lca_pos//(z**(lca_gen-gen-1))) % z
        v0, v1 = branches[child_idx]
        rho4 = apply_branch_single(rho, v0, v1)
        rho = ptrace_right(rho4)
        node = node*z + 1 + child_idx
    
    # At LCA: determine which children lead to i and j
    i_child = (i//(z**(lca_steps-1))) % z
    j_child = (j//(z**(lca_steps-1))) % z
    
    branches_lca = node_branches[node]
    
    # Get joint state via the two relevant children
    v0_i, v1_i = branches_lca[i_child]
    v0_j, v1_j = branches_lca[j_child]
    
    # Build joint 4x4 state from shared parent rho
    # rho_ij = E[rho_i ⊗ rho_j] with correlations from shared parent
    # Exact: rho_ij[a,b,c,d] = sum_{p,q} rho[p,q] * v_p_i[a]*conj(v_q_i[c]) * v_p_j[b]*conj(v_q_j[d])
    child_dim_i = len(v0_i)
    child_dim_j = len(v0_j)
    
    rho_joint = np.zeros((child_dim_i*child_dim_j, child_dim_i*child_dim_j), dtype=complex)
    for p in range(2):
        vp_i = v0_i if p==0 else v1_i
        vp_j = v0_j if p==0 else v1_j
        for q in range(2):
            vq_i = v0_i if q==0 else v1_i
            vq_j = v0_j if q==0 else v1_j
            rho_joint += rho[p,q] * np.kron(np.outer(vp_i, vq_i.conj()),
                                              np.outer(vp_j, vq_j.conj()))
    
    if lca_steps == 1:
        return mutual_info_2x2(rho_joint)
    
    # Propagate each path independently down to leaves
    i_path = [(i//(z**(lca_steps-1-s))) % z for s in range(lca_steps)]
    j_path = [(j//(z**(lca_steps-1-s))) % z for s in range(lca_steps)]
    
    rho_i = ptrace_right(rho_joint)
    rho_j = ptrace_left(rho_joint)
    
    i_node = node*z + 1 + i_child
    j_node = node*z + 1 + j_child
    
    for step in range(1, lca_steps):
        # Propagate i
        b_i = node_branches[i_node][i_path[step]]
        v0, v1 = b_i
        rho4 = apply_branch_single(rho_i, v0, v1)
        rho_i = ptrace_right(rho4)
        i_node = i_node*z + 1 + i_path[step]
        
        # Propagate j
        b_j = node_branches[j_node][j_path[step]]
        v0, v1 = b_j
        rho4 = apply_branch_single(rho_j, v0, v1)
        rho_j = ptrace_right(rho4)
        j_node = j_node*z + 1 + j_path[step]
    
    # Reconstruct approximate joint state
    rho_final = np.kron(rho_i, rho_j)
    return mutual_info_2x2(rho_final)

def fit_alpha(results, z):
    dGs, means = [], []
    for dG_str, vals in results.items():
        dG = int(dG_str)
        v = np.array(vals)
        v = v[v > 1e-10]
        if len(v) == 0: continue
        dGs.append(dG); means.append(np.mean(v))
    if len(dGs) < 2: return None, None
    dGs = np.array(dGs, dtype=float)
    log_mi = np.log(np.array(means))
    m = np.sum(dGs * log_mi) / np.sum(dGs**2)
    alpha = -2*m / np.log(z)
    y_pred = m * dGs
    ss_res = np.sum((log_mi - y_pred)**2)
    ss_tot = np.sum((log_mi - np.mean(log_mi))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    return float(alpha), float(r2)

def run_tree(z, depth, tree_idx):
    seed = tree_idx*1000 + z*100 + depth
    rng = np.random.default_rng(seed)
    n_leaves = z**depth
    n_internal = (z**depth - 1)//(z-1)
    
    print(f"  Generating {n_internal} nodes...")
    node_branches = {}
    for node in range(n_internal):
        node_branches[node] = make_z_branch_v2(z, rng)
    
    pairs = list(combinations(range(n_leaves), 2))
    n_pairs = len(pairs)
    results = {}
    t0 = time.time()
    progress_every = max(500, n_pairs//20)
    
    for idx, (i,j) in enumerate(pairs):
        if idx > 0 and idx % progress_every == 0:
            elapsed = time.time()-t0
            rate = idx/elapsed
            rem = (n_pairs-idx)/rate
            print(f"  {idx:,}/{n_pairs:,} ({100*idx/n_pairs:.0f}%) ~{rem:.0f}s remaining")
        dG = graph_dist_z(i, j, z, depth)
        mi = compute_mi_pair_z(i, j, z, depth, node_branches)
        results.setdefault(str(dG), []).append(float(mi))
    
    print(f"  Done in {time.time()-t0:.1f}s")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', type=int, default=4)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--trees', type=int, default=5)
    parser.add_argument('--start-tree', type=int, default=0)
    parser.add_argument('--output', type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    z, depth = args.z, args.depth
    n_leaves = z**depth
    n_pairs = n_leaves*(n_leaves-1)//2
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nBranching Factor Universality Test")
    print(f"z={z} | depth={depth} | leaves={n_leaves:,} | pairs={n_pairs:,}")
    print(f"Prediction: alpha -> {z}\n")
    
    all_alphas = []
    for tree_idx in range(args.start_tree, args.start_tree+args.trees):
        outfile = os.path.join(args.output, f"z{z}_depth{depth}_tree{tree_idx:03d}.json")
        if os.path.exists(outfile):
            with open(outfile) as f: results = json.load(f)
            print(f"Tree {tree_idx}: loaded")
        else:
            print(f"Tree {tree_idx}...")
            results = run_tree(z, depth, tree_idx)
            with open(outfile,'w') as f: json.dump(results, f)
        
        alpha, r2 = fit_alpha(results, z)
        if alpha:
            all_alphas.append(alpha)
            print(f"  alpha={alpha:.4f} R²={r2:.4f} (target={z}.0)")
        for dG_str in sorted(results.keys(), key=int):
            vals = np.array(results[dG_str])
            nz = vals[vals>1e-10]
            print(f"  dG={dG_str:2s}: n={len(vals):5d} "
                  f"mean={np.mean(vals):.5f} nonzero={len(nz)}")
    
    print(f"\n{'='*50}")
    print(f"UNIVERSALITY TEST SUMMARY: z={z}")
    print(f"{'='*50}")
    print(f"Predicted alpha:  {z}.0")
    if all_alphas:
        mean_a = np.mean(all_alphas)
        std_a = np.std(all_alphas)
        print(f"Measured alpha:   {mean_a:.4f} ± {std_a:.4f}")
        print(f"Consistent:       {abs(mean_a-z) < 0.3}")
    
    summary = {"z":z,"depth":depth,"predicted":float(z),
               "measured":float(np.mean(all_alphas)) if all_alphas else None,
               "std":float(np.std(all_alphas)) if all_alphas else None}
    sf = os.path.join(args.output, f"z{z}_depth{depth}_summary.json")
    with open(sf,'w') as f: json.dump(summary, f, indent=2)
    print(f"Saved: {sf}")

if __name__ == '__main__':
    main()
