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
def vec(rho): return rho.reshape((-1,), order="F")
def unvec(v, d): return v.reshape((d, d), order="F")

def von_neumann_entropy_bits(rho, eps=1e-15):
    w = xp.linalg.eigvalsh(rho)
    w = xp.clip(xp.real(w), eps, 1.0)
    return float(xp.sum(-w * xp.log2(w)))

def partial_trace_twoqubit(rho_4x4, keep):
    r = rho_4x4.reshape((2, 2, 2, 2))
    if keep == 0: return xp.trace(r, axis1=1, axis2=3)
    return xp.trace(r, axis1=0, axis2=2)

def mutual_information_bits(rho_ij):
    rho_i = partial_trace_twoqubit(rho_ij, keep=0)
    rho_j = partial_trace_twoqubit(rho_ij, keep=1)
    return (von_neumann_entropy_bits(rho_i)
            + von_neumann_entropy_bits(rho_j)
            - von_neumann_entropy_bits(rho_ij))

def superop_from_kraus(kraus):
    S = None
    for A in kraus:
        term = kron(A, xp.conj(A))
        S = term if S is None else (S + term)
    return S

def apply_superop(S, rho):
    v = S @ vec(rho)
    d_out = int(math.isqrt(v.size))
    return unvec(v, d_out)

# ── Haar / CNOT / embed ────────────────────────────────────────────────────
def make_haar_unitary_4():
    X = (xp.random.standard_normal((4,4)) + 1j*xp.random.standard_normal((4,4))) / xp.sqrt(2.0)
    Q, R = xp.linalg.qr(X)
    lam = xp.diag(R) / xp.abs(xp.diag(R))
    return Q * lam

def cnot_c0t1():
    U = xp.zeros((4,4), dtype=xp.complex128)
    U[0,0]=1; U[1,1]=1; U[3,2]=1; U[2,3]=1
    return U

def embed_2q_into_3q(U2, which_pair):
    U3 = xp.zeros((8,8), dtype=xp.complex128)
    a, b = which_pair
    for in_state in range(8):
        bits_in = [(in_state >> (2-q)) & 1 for q in (0,1,2)]
        in_pair = (bits_in[a] << 1) | bits_in[b]
        for out_pair in range(4):
            amp = U2[out_pair, in_pair]
            bits_out = bits_in.copy()
            bits_out[a] = (out_pair >> 1) & 1
            bits_out[b] = out_pair & 1
            out_state = (bits_out[0]<<2)|(bits_out[1]<<1)|bits_out[2]
            U3[out_state, in_state] += amp
    return U3

def branching_kraus():
    U_c01 = embed_2q_into_3q(cnot_c0t1(), (0,1))
    U_c02 = embed_2q_into_3q(cnot_c0t1(), (0,2))
    U_haar = embed_2q_into_3q(make_haar_unitary_4(), (1,2))
    U_total = U_haar @ U_c02 @ U_c01
    kraus = []
    for s in (0,1):
        A = xp.zeros((4,2), dtype=xp.complex128)
        for pin in (0,1):
            idx_in = (pin << 2) | 0
            col = U_total[:, idx_in]
            for c1 in (0,1):
                for c2 in (0,1):
                    A[(c1<<1)|c2, pin] = col[(s<<2)|(c1<<1)|c2]
        kraus.append(A)
    return kraus

# ── Tree ───────────────────────────────────────────────────────────────────
@dataclass
class Node:
    left: Optional[int] = None
    right: Optional[int] = None
    kraus: Optional[List] = None
    leaf_id: Optional[int] = None

def build_tree(depth, seed):
    xp.random.seed(seed)
    nodes, next_leaf = [], [0]
    def build(d):
        if d == 0:
            nid = len(nodes)
            nodes.append(Node(leaf_id=next_leaf[0]))
            next_leaf[0] += 1
            return nid
        nid = len(nodes)
        nodes.append(Node(kraus=branching_kraus()))
        nodes[nid].left = build(d-1)
        nodes[nid].right = build(d-1)
        return nid
    root = build(depth)
    return nodes, root

@dataclass
class Obj:
    k: int
    effect: Optional[object] = None
    superop: Optional[object] = None
    targets: Tuple = ()

def contract(nodes, nid, targets):
    nd = nodes[nid]
    if nd.leaf_id is not None:
        if nd.leaf_id in targets:
            return Obj(k=1, superop=kron(eye(2), eye(2)), targets=(nd.leaf_id,))
        return Obj(k=0, effect=eye(2))
    L = contract(nodes, nd.left, targets)
    R = contract(nodes, nd.right, targets)
    tgt = L.targets + R.targets
    k_total = L.k + R.k
    S_E = superop_from_kraus(nd.kraus)

    if k_total == 0:
        X = kron(L.effect, R.effect)
        Fp = xp.zeros((2,2), dtype=xp.complex128)
        for A in nd.kraus:
            Fp += xp.conj(A).T @ X @ A
        return Obj(k=0, effect=Fp)

    def reduce(which_child, G_sib):
        I2 = eye(2)
        T = xp.zeros((4,16), dtype=xp.complex128)
        for m in range(4):
            for n in range(4):
                E = xp.zeros((4,4), dtype=xp.complex128); E[m,n] = 1.0
                if which_child == 0:
                    rho_keep = partial_trace_twoqubit(kron(I2, G_sib) @ E, keep=0)
                else:
                    rho_keep = partial_trace_twoqubit(kron(G_sib, I2) @ E, keep=1)
                T[:, m+4*n] = vec(rho_keep)
        return T @ S_E

    if k_total == 1:
        if L.k==1: return Obj(k=1, superop=L.superop @ reduce(0, R.effect), targets=tgt)
        else:       return Obj(k=1, superop=R.superop @ reduce(1, L.effect), targets=tgt)

    if L.k==1 and R.k==1:
        return Obj(k=2, superop=kron(L.superop, R.superop) @ S_E, targets=tgt)
    if L.k==2 and R.k==0:
        return Obj(k=2, superop=L.superop @ reduce(0, R.effect), targets=tgt)
    if L.k==0 and R.k==2:
        return Obj(k=2, superop=R.superop @ reduce(1, L.effect), targets=tgt)
    raise RuntimeError("Unexpected configuration.")

# ── Distance ───────────────────────────────────────────────────────────────
def graph_dist(i, j, depth):
    if i == j: return 0
    return 2 * (depth - (depth - (i^j).bit_length()))

def graph_dist(i, j, depth):
    if i == j: return 0
    msb = (i ^ j).bit_length()
    return 2 * msb

# ── Root state: the primordial particle ───────────────────────────────────
def root_state():
    return eye(2) / 2.0   # maximally mixed = "maybe"

# ── Main run ───────────────────────────────────────────────────────────────
def run_exhaustive(depth=6, n_trees=20, results_dir="/home/3x-agent/qft/results"):
    os.makedirs(results_dir, exist_ok=True)
    n_leaves = 2**depth
    all_pairs = list(itertools.combinations(range(n_leaves), 2))
    n_pairs = len(all_pairs)
    rho0 = root_state()

    print(f"Backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}")
    print(f"depth={depth}  leaves={n_leaves}  pairs={n_pairs}  trees={n_trees}")
    print(f"Results → {results_dir}\n")

    all_bins = {}   # dG -> list of MI values across all trees
    grand_start = time.time()

    for tree_idx in range(n_trees):
        seed = 1000 + tree_idx
        nodes, root_nid = build_tree(depth, seed)
        tree_bins = {}
        t0 = time.time()

        for k, (i, j) in enumerate(all_pairs):
            dG = graph_dist(i, j, depth)
            obj = contract(nodes, root_nid, (i, j))
            rho_ij = apply_superop(obj.superop, rho0)
            Iij = max(mutual_information_bits(rho_ij), 1e-12)
            tree_bins.setdefault(dG, []).append(Iij)
            all_bins.setdefault(dG, []).append(Iij)

            if (k+1) % 500 == 0 or k == n_pairs-1:
                elapsed = time.time() - t0
                rate = (k+1) / elapsed
                eta = (n_pairs - k - 1) / rate
                print(f"  Tree {tree_idx+1:2d}/{n_trees} | pair {k+1:5d}/{n_pairs} | "
                      f"{rate:.0f} pairs/s | ETA {eta:.0f}s", flush=True)

        # Per-tree summary
        print(f"  Tree {tree_idx+1:2d} done in {time.time()-t0:.1f}s:")
        for dG in sorted(tree_bins):
            vals = np.array(tree_bins[dG])
            print(f"    dG={dG:2d}  n={len(vals):4d}  mean={vals.mean():.6f}  std={vals.std():.6f}")

        # Save per-tree results
        tree_out = {str(k): [float(x) for x in v] for k, v in tree_bins.items()}
        with open(f"{results_dir}/depth{depth}_tree{tree_idx:03d}.json", "w") as f:
            json.dump(tree_out, f)
        print()

    # ── Ensemble summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ENSEMBLE SUMMARY: depth={depth}, {n_trees} trees, {n_pairs} pairs/tree")
    print(f"Total elapsed: {time.time()-grand_start:.1f}s")
    print(f"{'='*60}")

    summary = {}
    for dG in sorted(all_bins):
        vals = np.array(all_bins[dG])
        mean_mi = vals.mean()
        summary[dG] = {"mean": float(mean_mi), "std": float(vals.std()),
                       "n": len(vals), "median": float(np.median(vals))}
        print(f"dG={dG:2d}  n={len(vals):6d}  mean={mean_mi:.6f}  "
              f"std={vals.std():.6f}  median={np.median(vals):.6f}")

    # Fit alpha: MI = A * exp(-alpha * dG)
    distances = np.array(sorted(summary.keys()), dtype=float)
    means = np.array([summary[int(d)]["mean"] for d in distances])
    log_means = np.log(means)
    coeffs = np.polyfit(distances, log_means, 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])
    print(f"\nExponential fit: MI = {A:.4f} * exp(-{alpha:.4f} * dG)")
    print(f"alpha (depth {depth}) = {alpha:.4f}")

    # Monotonicity check
    monotone = all(means[k] > means[k+1] for k in range(len(means)-1))
    print(f"Monotone decay: {monotone}")

    # Save ensemble summary
    with open(f"{results_dir}/depth{depth}_ensemble_summary.json", "w") as f:
        json.dump({"depth": depth, "n_trees": n_trees, "n_pairs": n_pairs,
                   "alpha": alpha, "A": A, "monotone": monotone,
                   "summary": {str(k): v for k, v in summary.items()}}, f, indent=2)
    print(f"\nSaved: {results_dir}/depth{depth}_ensemble_summary.json")
    return all_bins, alpha

if __name__ == "__main__":
    run_exhaustive(depth=6, n_trees=20)
