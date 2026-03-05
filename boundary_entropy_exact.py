"""
Gap 2: Exact boundary entanglement entropy scaling.

Builds the full 2^depth x 2^depth density matrix of all leaves
using a clean statevector approach, then computes S(A) exactly
for contiguous intervals by partial trace.

Tests RT prediction: S(l) = (c/3) * log(l) + const

Uses depth 6 (64 leaves, 64x64 density matrix — fast).
"""

import numpy as np
import json
import time

RESULTS_DIR = "/home/3x-agent/qft/results"

# ── Gate construction (same as main simulation) ────────────────────────────

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

def build_gates(depth, seed):
    rng = np.random.default_rng(seed)
    gates = {}
    for gen in range(depth):
        for node in range(2**gen - 1, 2**(gen+1) - 1):
            gates[(gen, node)] = make_kraus(make_haar_unitary_4(rng))
    return gates

# ── Build full leaf density matrix ─────────────────────────────────────────

def build_full_leaf_dm(depth, gates):
    """
    Build the full 2^depth x 2^depth density matrix over all leaves.

    Strategy: represent state as density matrix over ACTIVE qubits.
    Start: 1 active qubit (root) in state |0><0|.
    At each generation: every active qubit branches into 2 children.
    After depth generations: 2^depth active qubits = all leaves.

    At generation g: 2^g active qubits, rho is 2^(2^g) x 2^(2^g).
    We apply each node's branching channel to its corresponding qubit.

    Key insight: nodes in the same generation act on DIFFERENT qubits,
    so we apply them sequentially without interference.
    """
    # Start: single qubit in |0><0|
    rho = np.array([[1., 0.], [0., 0.]], dtype=complex)

    for gen in range(depth):
        n_nodes = 2**gen          # number of active qubits this generation
        # rho is currently (2^n_nodes x 2^n_nodes)
        dim_in = 2**n_nodes
        dim_out = 2**(2*n_nodes)  # each qubit splits -> double the qubits

        rho_new = np.zeros((dim_out, dim_out), dtype=complex)

        # Apply branching to each qubit one at a time
        # After applying branching to qubit k, the system grows by 1 qubit
        # We process qubits from last to first to keep indexing clean
        # Actually: process left to right, tracking growing dimension

        rho_working = rho.copy()
        n_qubits_working = n_nodes

        for node_idx in range(n_nodes):
            node = (2**gen - 1) + node_idx
            K = gates[(gen, node)]
            # Apply branching channel to qubit node_idx
            # rho_working: (2^n_qubits_working x 2^n_qubits_working)
            rho_working = branch_qubit(rho_working, K, node_idx, n_qubits_working)
            n_qubits_working += 1

        rho = rho_working

    return rho


def branch_qubit(rho, K, q_idx, n_qubits):
    """
    Apply branching channel K to qubit q_idx in an n_qubits system.
    Qubit q_idx is replaced by TWO child qubits (inserted at q_idx, q_idx+1).
    All qubits with index > q_idx shift right by 1.

    Input:  rho (2^n_qubits x 2^n_qubits)
    Output: rho (2^(n_qubits+1) x 2^(n_qubits+1))

    K: list of (4x2) Kraus operators mapping 1 parent -> 2 children.
    """
    dim_in  = 2**n_qubits
    dim_out = 2**(n_qubits + 1)
    n_out   = n_qubits + 1

    rho_out = np.zeros((dim_out, dim_out), dtype=complex)

    for A in K:
        # Build A_full: (dim_out x dim_in) operator in full Hilbert space
        A_full = np.zeros((dim_out, dim_in), dtype=complex)

        for col in range(dim_in):
            # Decompose col into qubit bits (MSB = qubit 0)
            bits_in = [(col >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
            parent_bit = bits_in[q_idx]
            # Bits for qubits other than q_idx (in order)
            other_bits = bits_in[:q_idx] + bits_in[q_idx+1:]

            for c1 in range(2):
                for c2 in range(2):
                    amp = A[(c1 << 1) | c2, parent_bit]
                    if amp == 0: continue
                    # Output bits: insert c1, c2 at positions q_idx, q_idx+1
                    bits_out = other_bits[:q_idx] + [c1, c2] + other_bits[q_idx:]
                    row = sum(b << (n_out - 1 - i) for i, b in enumerate(bits_out))
                    A_full[row, col] += amp

        rho_out += A_full @ rho @ A_full.conj().T

    return rho_out


# ── Partial trace ──────────────────────────────────────────────────────────

def partial_trace(rho, keep, n_qubits):
    """
    Trace out all qubits NOT in keep.
    rho: (2^n_qubits x 2^n_qubits)
    keep: sorted list of qubit indices to keep
    Returns: (2^len(keep) x 2^len(keep))
    """
    keep = sorted(keep)
    trace_out = sorted([i for i in range(n_qubits) if i not in keep])
    n_keep  = len(keep)
    n_trace = len(trace_out)
    dim_keep  = 2**n_keep
    dim_trace = 2**n_trace

    rho_out = np.zeros((dim_keep, dim_keep), dtype=complex)

    for t in range(dim_trace):
        t_bits = [(t >> (n_trace - 1 - i)) & 1 for i in range(n_trace)]

        # Build index arrays for vectorized operation
        rows_full = np.zeros(dim_keep, dtype=int)
        cols_full = np.zeros(dim_keep, dtype=int)

        for ki in range(dim_keep):
            k_bits = [(ki >> (n_keep - 1 - i)) & 1 for i in range(n_keep)]
            bits = [0] * n_qubits
            for pos, q in enumerate(keep):    bits[q] = k_bits[pos]
            for pos, q in enumerate(trace_out): bits[q] = t_bits[pos]
            rows_full[ki] = sum(b << (n_qubits - 1 - i) for i, b in enumerate(bits))

        # rho_out += rho[rows, :][:, rows] for trace indices
        rho_out += rho[np.ix_(rows_full, rows_full)]

    return rho_out


# ── Von Neumann entropy ────────────────────────────────────────────────────

def von_neumann_entropy(rho, eps=1e-14):
    w = np.real(np.linalg.eigvalsh(rho))
    w = w[w > eps]
    w /= w.sum()
    return float(-np.sum(w * np.log(w)))  # nats


# ── Main ───────────────────────────────────────────────────────────────────

DEPTH   = 5
N_TREES = 20
N_LEAVES = 2**DEPTH
interval_sizes = list(range(1, 17))

print(f"Depth {DEPTH} | {N_LEAVES} leaves | {N_TREES} trees")
print(f"Interval sizes: {interval_sizes}\n")

all_S = {l: [] for l in interval_sizes}

for tree_idx in range(N_TREES):
    t0 = time.time()
    print(f"Tree {tree_idx+1}/{N_TREES} ...", flush=True)
    gates = build_gates(DEPTH, seed=1000+tree_idx)

    rho_all = build_full_leaf_dm(DEPTH, gates)
    tr = np.trace(rho_all).real
    print(f"  rho trace={tr:.8f}  shape={rho_all.shape}  ({time.time()-t0:.1f}s)", flush=True)

    for l in interval_sizes:
        keep = list(range(l))
        rho_A = partial_trace(rho_all, keep, N_LEAVES)
        S = von_neumann_entropy(rho_A)
        all_S[l].append(S)

    # Print this tree's results
    for l in interval_sizes:
        print(f"  l={l:3d}  S={all_S[l][-1]:.6f}", flush=True)
    print(f"  Tree done in {time.time()-t0:.1f}s\n", flush=True)

# ── Ensemble average ───────────────────────────────────────────────────────
print(f"{'='*60}")
print("ENSEMBLE AVERAGE")
print(f"{'='*60}")
avg_S = {}
for l in interval_sizes:
    avg_S[l] = float(np.mean(all_S[l]))
    print(f"  l={l:3d}  S={avg_S[l]:.6f}  std={np.std(all_S[l]):.6f}")

# ── RT fit ─────────────────────────────────────────────────────────────────
print(f"\n--- RT Fit: S(l) = (c/3)*log(l) + const ---")
fit_l = [l for l in interval_sizes if l >= 2]
log_l = np.log(np.array(fit_l, dtype=float))
S_arr = np.array([avg_S[l] for l in fit_l])

X = np.column_stack([log_l, np.ones_like(log_l)])
coeffs = np.linalg.lstsq(X, S_arr, rcond=None)[0]
c_over_3, const = coeffs
c = 3 * c_over_3

S_pred = c_over_3 * log_l + const
ss_res = np.sum((S_arr - S_pred)**2)
ss_tot = np.sum((S_arr - S_arr.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"  c/3   = {c_over_3:.6f}")
print(f"  c     = {c:.6f}")
print(f"  const = {const:.6f}")
print(f"  R²    = {r2:.6f}")
print(f"\n  Predicted: {[f'{s:.4f}' for s in S_pred]}")
print(f"  Actual:    {[f'{s:.4f}' for s in S_arr]}")

print(f"\n--- Verdict ---")
if r2 > 0.99:
    print(f"  *** LOG SCALING CONFIRMED: R²={r2:.6f}, c={c:.4f} ***")
    print(f"  Ryu-Takayanagi formula operating in the simulation.")
elif r2 > 0.95:
    print(f"  Marginal: R²={r2:.4f}, c={c:.4f}. RT consistent but not clean.")
else:
    print(f"  R²={r2:.4f}. Log scaling not confirmed.")

out = {
    "depth": DEPTH, "n_trees": N_TREES,
    "avg_S": {str(l): avg_S[l] for l in interval_sizes},
    "c": float(c), "r2": float(r2), "const": float(const),
}
with open(f"{RESULTS_DIR}/boundary_entropy_exact.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {RESULTS_DIR}/boundary_entropy_exact.json")
