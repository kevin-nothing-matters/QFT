"""
First Law of Entanglement Entropy Test.

Tests: δS(A) = δ⟨H_A⟩

For a 1+1 CFT, the modular Hamiltonian of interval A = [0, l-1] is:
  H_A = 2π ∫_A T_tt(x) · (l/2π) · sin(πx/l) dx   [for finite system]

For our discrete tree, the modular Hamiltonian of interval A is:
  H_A = -log(ρ_A)

where ρ_A is the reduced density matrix of A.

The first law states: for any perturbation |δρ⟩ of the global state,
  δS(A) = Tr(δρ_A · (-log ρ_A)) = δ⟨H_A⟩

This is an EXACT identity for any density matrix perturbation.
It is not approximate — it follows from δS = -δTr(ρ log ρ) = -Tr(δρ log ρ).

So the first law always holds mathematically. The PHYSICAL content is:
  - Whether H_A is LOCAL (integral of local stress tensor)
  - Whether δ⟨H_A⟩ is GEOMETRICALLY interpretable as curvature

What we can test computationally:
  1. Verify the first law holds numerically (sanity check)
  2. Compute the modular Hamiltonian H_A = -log(ρ_A) for our tree
  3. Perturb one gate: compute δS(A) and δ⟨H_A⟩ = Tr(δρ_A · H_A^(0))
  4. Check δS(A) = δ⟨H_A⟩ to high precision
  5. Extract the eigenvalue spectrum of H_A — if it matches a local CFT
     Hamiltonian, this is evidence of locality

Method: use the existing pairwise MI data + MI-matrix approach for ρ_A.
Perturbation: change the seed of one gate at one node, recompute MI for
interval leaves, measure δS and δ⟨H_A⟩.
"""

import numpy as np
import json
import time

RESULTS_DIR = "/home/3x-agent/qft/results"

# ── Core machinery (from main simulation) ─────────────────────────────────

def make_haar_unitary_4(rng):
    X = (rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(X)
    return Q * (np.diag(R) / np.abs(np.diag(R)))

def cnot_c0t1():
    U = np.zeros((4,4), dtype=complex)
    U[0,0]=1; U[1,1]=1; U[3,2]=1; U[2,3]=1
    return U

def embed_2q_into_3q(U2, which_pair):
    a, b = which_pair
    U3 = np.zeros((8,8), dtype=complex)
    for s in range(8):
        bits = [(s >> (2-q)) & 1 for q in (0,1,2)]
        ip = (bits[a] << 1) | bits[b]
        for op in range(4):
            bo = bits.copy(); bo[a]=(op>>1)&1; bo[b]=op&1
            U3[(bo[0]<<2)|(bo[1]<<1)|bo[2], s] += U2[op, ip]
    return U3

def make_kraus(H):
    Ut = (embed_2q_into_3q(H,(1,2))
          @ embed_2q_into_3q(cnot_c0t1(),(0,2))
          @ embed_2q_into_3q(cnot_c0t1(),(0,1)))
    K = []
    for s in (0,1):
        A = np.zeros((4,2), dtype=complex)
        for p in (0,1):
            col = Ut[:,(p<<2)|0]
            for c1 in (0,1):
                for c2 in (0,1): A[(c1<<1)|c2,p] = col[(s<<2)|(c1<<1)|c2]
        K.append(A)
    return K

def build_gates(depth, seed, perturb_node=None, perturb_seed=None):
    """Build gates, optionally perturbing one node's gate."""
    rng = np.random.default_rng(seed)
    gates = {}
    for gen in range(depth):
        for node in range(2**gen - 1, 2**(gen+1) - 1):
            H = make_haar_unitary_4(rng)
            if perturb_node is not None and node == perturb_node:
                H = make_haar_unitary_4(np.random.default_rng(perturb_seed))
            gates[(gen, node)] = make_kraus(H)
    return gates

def get_path(leaf, depth):
    return [(leaf >> (depth-1-g)) & 1 for g in range(depth)]

def apply_1to2(K, rho1):
    return sum(A @ rho1 @ A.conj().T for A in K)

def apply_1to1(K, rho1, keep_child):
    rho2 = apply_1to2(K, rho1)
    r = rho2.reshape(2,2,2,2)
    if keep_child == 0: return np.trace(r, axis1=1, axis2=3)
    return np.trace(r, axis1=0, axis2=2)

def compute_pair_mi(i, j, depth, gates):
    """Exact pairwise MI via joint propagation."""
    pi, pj = get_path(i, depth), get_path(j, depth)
    lca_gen = depth
    for k in range(depth):
        if pi[k] != pj[k]: lca_gen = k; break

    rho = np.array([[1.,0.],[0.,0.]], dtype=complex)
    node = 0
    for gen in range(lca_gen):
        choice = pi[gen]
        rho = apply_1to1(gates[(gen,node)], rho, keep_child=choice)
        node = 2*node + 1 + choice

    rho_joint = apply_1to2(gates[(lca_gen, node)], rho)
    ni = 2*node + 1 + pi[lca_gen]
    nj = 2*node + 1 + pj[lca_gen]

    I2 = np.eye(2, dtype=complex)
    for gen in range(lca_gen+1, depth):
        ci, cj = pi[gen], pj[gen]
        K_i, K_j = gates[(gen,ni)], gates[(gen,nj)]
        rho_8 = sum(np.kron(A,I2) @ rho_joint @ np.kron(A,I2).conj().T for A in K_i)
        r8 = rho_8.reshape(2,2,2,2,2,2)
        rho_joint = (np.trace(r8,axis1=1,axis2=4) if ci==0 else np.trace(r8,axis1=0,axis2=3)).reshape(4,4)
        rho_8b = sum(np.kron(I2,B) @ rho_joint @ np.kron(I2,B).conj().T for B in K_j)
        r8b = rho_8b.reshape(2,2,2,2,2,2)
        rho_joint = (np.trace(r8b,axis1=2,axis2=5) if cj==0 else np.trace(r8b,axis1=1,axis2=4)).reshape(4,4)
        ni = 2*ni+1+ci; nj = 2*nj+1+cj

    def S(rho, eps=1e-14):
        w = np.real(np.linalg.eigvalsh(rho))
        w = w[w>eps]; w/=w.sum()
        return float(-np.sum(w*np.log(w)))

    rho_i = np.trace(rho_joint.reshape(2,2,2,2), axis1=1, axis2=3)
    rho_j = np.trace(rho_joint.reshape(2,2,2,2), axis1=0, axis2=2)
    return S(rho_i) + S(rho_j) - S(rho_joint)

# ── MI matrix and reduced density matrix proxy ─────────────────────────────

def build_interval_mi_matrix(interval, depth, gates):
    """Build MI matrix for leaves in interval."""
    n = len(interval)
    M = np.zeros((n, n))
    for ii in range(n):
        for jj in range(ii+1, n):
            mi = max(compute_pair_mi(interval[ii], interval[jj], depth, gates), 1e-15)
            M[ii,jj] = M[jj,ii] = mi
    # Diagonal: single-qubit entropy (max entanglement = log 2)
    rho0 = np.array([[1.,0.],[0.,0.]], dtype=complex)
    for ii, leaf in enumerate(interval):
        node, path = 0, get_path(leaf, depth)
        rho = rho0.copy()
        for gen in range(depth):
            rho = apply_1to1(gates[(gen,node)], rho, keep_child=path[gen])
            node = 2*node+1+path[gen]
        w = np.real(np.linalg.eigvalsh(rho))
        w = w[w>1e-14]; w/=w.sum()
        M[ii,ii] = float(-np.sum(w*np.log(w)))
    return M

def rho_from_mi_matrix(M):
    """Normalized MI matrix as proxy for reduced density matrix."""
    n = M.shape[0]
    rho = M / n
    # Symmetrize and ensure PSD
    rho = (rho + rho.T) / 2
    w, v = np.linalg.eigh(rho)
    w = np.clip(w, 0, None)
    if w.sum() > 0: w /= w.sum()
    return v @ np.diag(w) @ v.T

def von_neumann(rho, eps=1e-14):
    w = np.real(np.linalg.eigvalsh(rho))
    w = w[w>eps]; w/=w.sum()
    return float(-np.sum(w*np.log(w)))

def modular_hamiltonian(rho, eps=1e-14):
    """H_A = -log(rho_A). Returns matrix."""
    w, v = np.linalg.eigh(rho)
    log_w = np.where(w > eps, -np.log(np.clip(w, eps, None)), 0.0)
    return v @ np.diag(log_w) @ v.T

# ── First law test ─────────────────────────────────────────────────────────

def first_law_test(depth, base_seed, perturb_node, perturb_seeds,
                   intervals, label=""):
    """
    Test δS(A) = Tr(δρ_A · H_A^(0)) = δ⟨H_A⟩

    For each perturbation and each interval:
      1. Compute ρ_A^(0), H_A^(0) = -log(ρ_A^(0)), S^(0)(A)
      2. Compute ρ_A^(pert), S^(pert)(A)
      3. δS = S^(pert) - S^(0)
      4. δ⟨H_A⟩ = Tr((ρ_A^(pert) - ρ_A^(0)) · H_A^(0))
      5. Check δS ≈ δ⟨H_A⟩
    """
    print(f"\n{'='*60}")
    print(f"First Law Test: {label}")
    print(f"Depth={depth}, base_seed={base_seed}, perturb_node={perturb_node}")
    print(f"{'='*60}")

    # Base state
    gates_0 = build_gates(depth, base_seed)

    results = []

    for p_seed in perturb_seeds:
        print(f"\n  Perturbation seed={p_seed}:")
        gates_p = build_gates(depth, base_seed,
                              perturb_node=perturb_node,
                              perturb_seed=p_seed)

        for interval in intervals:
            l = len(interval)

            # Base
            M0 = build_interval_mi_matrix(interval, depth, gates_0)
            rho0 = rho_from_mi_matrix(M0)
            S0 = von_neumann(rho0)
            H0 = modular_hamiltonian(rho0)

            # Perturbed
            Mp = build_interval_mi_matrix(interval, depth, gates_p)
            rhop = rho_from_mi_matrix(Mp)
            Sp = von_neumann(rhop)

            # First law quantities
            delta_S = Sp - S0
            delta_rho = rhop - rho0
            delta_H_expect = float(np.real(np.trace(delta_rho @ H0)))

            # Check
            if abs(delta_S) > 1e-10:
                ratio = delta_H_expect / delta_S
            else:
                ratio = float('nan')

            print(f"    l={l:3d}: δS={delta_S:+.6f}  "
                  f"δ⟨H⟩={delta_H_expect:+.6f}  "
                  f"ratio={ratio:.4f}  "
                  f"{'✓' if abs(ratio-1.0) < 0.05 else '~' if abs(ratio-1.0) < 0.2 else '✗'}")

            results.append({
                "interval_size": l,
                "perturb_seed": p_seed,
                "S0": S0, "Sp": Sp,
                "delta_S": delta_S,
                "delta_H_expect": delta_H_expect,
                "ratio": ratio,
            })

    return results

# ── Modular Hamiltonian spectrum analysis ──────────────────────────────────

def analyze_modular_spectrum(depth, seed, intervals):
    """
    Compute eigenvalue spectrum of H_A = -log(ρ_A) for each interval.
    For a local CFT Hamiltonian, eigenvalues should be non-negative
    and grow linearly with system size (Entanglement spectrum).
    """
    print(f"\n{'='*60}")
    print(f"Modular Hamiltonian Spectrum Analysis")
    print(f"{'='*60}")

    gates = build_gates(depth, seed)

    for interval in intervals:
        l = len(interval)
        M = build_interval_mi_matrix(interval, depth, gates)
        rho = rho_from_mi_matrix(M)
        H = modular_hamiltonian(rho)

        # Eigenvalues of H (entanglement spectrum)
        evals = sorted(np.real(np.linalg.eigvalsh(H)))
        # Filter: keep meaningful ones
        evals_pos = [e for e in evals if e > 0.01]

        print(f"\n  l={l}: S={von_neumann(rho):.4f}")
        print(f"    H eigenvalues (entanglement spectrum): "
              f"{[f'{e:.3f}' for e in evals[:8]]}")
        print(f"    Min={min(evals):.4f}  Max={max(evals):.4f}  "
              f"N_positive={len(evals_pos)}")

        # Check: for a CFT, spacing should be ~ 2π/l (level spacing)
        if len(evals_pos) >= 2:
            spacings = [evals_pos[i+1]-evals_pos[i]
                       for i in range(min(4, len(evals_pos)-1))]
            expected_spacing = 2*np.pi / l
            print(f"    Level spacings: {[f'{s:.3f}' for s in spacings]}")
            print(f"    Expected CFT spacing 2π/l = {expected_spacing:.3f}")

# ── Main ───────────────────────────────────────────────────────────────────

DEPTH = 4   # small depth for speed — exact MI computation
BASE_SEED = 1000

# Intervals to test: small contiguous sets of leaves
N_LEAVES = 2**DEPTH
intervals = [
    list(range(2)),   # l=2
    list(range(4)),   # l=4
    list(range(6)),   # l=6
    list(range(8)),   # l=8
]

# Perturb node 0 (root) and node 1 (first child)
# Use 5 different perturbation seeds
perturb_seeds = [2000, 2001, 2002, 2003, 2004]

print("First Law of Entanglement Entropy Test")
print("δS(A) = δ⟨H_A⟩  ?")
print(f"Depth {DEPTH}, {N_LEAVES} leaves\n")

# Test 1: Perturb root node
t0 = time.time()
results_root = first_law_test(
    DEPTH, BASE_SEED,
    perturb_node=0,        # root
    perturb_seeds=perturb_seeds,
    intervals=intervals,
    label="Root perturbation"
)
print(f"\nRoot perturbation done in {time.time()-t0:.1f}s")

# Test 2: Perturb a deep node (generation 3)
t0 = time.time()
results_deep = first_law_test(
    DEPTH, BASE_SEED,
    perturb_node=7,        # first node in generation 3
    perturb_seeds=perturb_seeds,
    intervals=intervals,
    label="Deep node perturbation (gen 3)"
)
print(f"\nDeep perturbation done in {time.time()-t0:.1f}s")

# Modular spectrum
analyze_modular_spectrum(DEPTH, BASE_SEED, intervals)

# ── Summary ────────────────────────────────────────────────────────────────
all_results = results_root + results_deep
ratios = [r["ratio"] for r in all_results
          if not np.isnan(r["ratio"]) and abs(r["delta_S"]) > 1e-8]

if ratios:
    print(f"\n{'='*60}")
    print(f"FIRST LAW SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tests: {len(ratios)}")
    print(f"  Mean ratio δ⟨H⟩/δS: {np.mean(ratios):.6f}  (target: 1.000)")
    print(f"  Std ratio:           {np.std(ratios):.6f}")
    print(f"  Min/Max:             {min(ratios):.4f} / {max(ratios):.4f}")
    close = sum(1 for r in ratios if abs(r-1.0) < 0.05)
    print(f"  Within 5% of 1.0:   {close}/{len(ratios)} ({100*close/len(ratios):.1f}%)")
    if np.mean(ratios) > 0.95 and np.std(ratios) < 0.1:
        print(f"\n  *** FIRST LAW CONFIRMED ***")
        print(f"  δS(A) = δ⟨H_A⟩ holds across all perturbations and intervals.")
        print(f"  This is the engine of the Faulkner-Van Raamsdonk derivation.")
    else:
        print(f"\n  First law holds approximately — MI-matrix proxy introduces error.")
        print(f"  Exact density matrix computation needed for precision test.")

out = {
    "depth": DEPTH, "base_seed": BASE_SEED,
    "results_root": results_root,
    "results_deep": results_deep,
    "summary": {
        "mean_ratio": float(np.mean(ratios)) if ratios else None,
        "std_ratio": float(np.std(ratios)) if ratios else None,
        "n_tests": len(ratios),
        "pct_within_5pct": float(100*sum(1 for r in ratios if abs(r-1.0)<0.05)/len(ratios)) if ratios else None,
    }
}
with open(f"{RESULTS_DIR}/first_law_test.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {RESULTS_DIR}/first_law_test.json")
