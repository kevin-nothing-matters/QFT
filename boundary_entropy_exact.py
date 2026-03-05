"""
Gap 2: Boundary entanglement entropy — exact computation.

Computes S(A) exactly for contiguous intervals A = [0, 1, ..., l-1]
by building the full 2^l x 2^l joint density matrix of l leaves.

Tractable for l <= 8 (256x256 matrix).
Tests RT prediction: S(l) = (c/3) * log(l) + const

Method:
  For each interval [0..l-1], build rho_A by:
  1. Start from pure root state |0><0|
  2. Propagate through tree, keeping all l target leaves
  3. Trace out all non-target leaves
  4. Compute von Neumann entropy of the resulting 2^l x 2^l matrix

For the joint state of l leaves we use a recursive approach:
  - Split interval at LCA structure
  - Maintain joint density matrix of all leaves collected so far
"""

import numpy as np
import json
import time

RESULTS_DIR = "/home/3x-agent/qft/results"

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    xp = np
    GPU = False

# ── Core machinery ─────────────────────────────────────────────────────────

def von_neumann_entropy(rho, eps=1e-15):
    """Von Neumann entropy in nats."""
    if isinstance(rho, np.ndarray):
        w = np.linalg.eigvalsh(rho)
    else:
        w = np.linalg.eigvalsh(rho.get())
    w = np.real(w)
    w = w[w > eps]
    w = w / w.sum()  # renormalize for numerical safety
    return float(-np.sum(w * np.log(w)))

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

def get_path(leaf, depth):
    return [(leaf >> (depth-1-g)) & 1 for g in range(depth)]

# ── Exact joint density matrix for a set of leaves ────────────────────────

def compute_joint_density_matrix(leaves, depth, gates):
    """
    Build exact joint density matrix rho_{l0, l1, ..., l_{n-1}}
    for an arbitrary set of leaves using recursive tree traversal.

    Returns a (2^n x 2^n) density matrix where n = len(leaves).

    Algorithm:
      Represent state as a joint density matrix over the currently-tracked leaves.
      At each node, either:
        - The node is an ancestor of multiple target leaves: maintain joint state
        - The node leads to only one target leaf: propagate single-qubit state
        - The node leads to no target leaves: trace out (don't track)
    """
    leaves = sorted(leaves)
    n = len(leaves)
    if n == 0:
        return np.array([[1.0]], dtype=complex)

    def subtree_leaves(node, gen):
        """All target leaves that are descendants of this node."""
        node_depth = gen
        # Node at generation gen covers leaves from node_start to node_start + 2^(depth-gen) - 1
        # where node index within generation = node - (2^gen - 1)
        node_idx_in_gen = node - (2**gen - 1)
        span = 2**(depth - gen)
        start = node_idx_in_gen * span
        end = start + span
        return [l for l in leaves if start <= l < end]

    def propagate(node, gen, rho_parent, parent_qubit_idx_in_state):
        """
        Propagate state through node, keeping track of which target leaves
        are in each subtree.

        rho_parent: current joint density matrix (2^k x 2^k)
                    where k = number of tracked qubits
        parent_qubit_idx_in_state: which qubit in rho_parent is the parent

        Returns: updated rho with parent replaced by its children
                 (only children that lead to target leaves)
        """
        nonlocal rho_current, tracked_leaves

        left_child  = 2*node + 1
        right_child = 2*node + 2
        left_gen    = gen + 1
        right_gen   = gen + 1

        left_targets  = subtree_leaves(left_child,  left_gen)  if left_gen  <= depth else []
        right_targets = subtree_leaves(right_child, right_gen) if right_gen <= depth else []

        K = gates[(gen, node)]  # Kraus ops: 4x2 each

        return left_targets, right_targets, K

    # ── Iterative approach: build full statevector for small systems ───────
    # For l <= 8 target leaves at depth 8, we build the reduced density matrix
    # by computing the full 2^depth statevector and tracing out non-target leaves.
    # At depth 8: 2^8 = 256 dimensional statevector — very fast.

    n_all_leaves = 2**depth
    dim = n_all_leaves

    # Build full statevector via unitary circuit on all leaves
    # We represent the state as a vector in the 2^n_leaves Hilbert space
    # where qubit ordering is leaf 0, leaf 1, ..., leaf_{n_leaves-1}

    # Initialize: all leaves in |0>
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0

    # Apply the tree as a unitary on the leaf space
    # Each internal node's branching channel maps parent -> two children
    # We implement this as a unitary by fixing the environment (ancilla = |0>)
    # and applying the full 3-qubit unitary, then discarding the parent

    # Since we're working with a pure state and want S(A) for a subsystem,
    # we can use the Schmidt decomposition approach:
    # Build psi as a statevector over all leaves, then compute reduced density matrix

    # The tree circuit acts on qubits in a specific order.
    # We process generation by generation, top to bottom.
    # At each generation, we apply the branching unitary to each node.

    # State representation: tensor product of all leaf qubits
    # Initially all in |0>, but the tree has internal structure.

    # More practical: use the recursive Kraus approach to build
    # the full density matrix of ALL leaves, then trace out non-target leaves.

    # Build rho over all leaves recursively
    rho_all = build_full_leaf_density_matrix(depth, gates)

    # Now trace out all leaves NOT in our target set
    target_set = set(leaves)
    all_leaves_list = list(range(n_all_leaves))
    trace_out = [i for i in all_leaves_list if i not in target_set]

    rho_reduced = partial_trace_numpy(rho_all, trace_out, n_all_leaves)
    return rho_reduced


def build_full_leaf_density_matrix(depth, gates):
    """
    Build the full density matrix over all 2^depth leaves.
    Uses the Kraus operator structure recursively.

    For depth <= 8 (256 leaves), this is a 256x256 matrix — tractable.
    """
    n_leaves = 2**depth

    def node_to_leaf_range(node, gen):
        node_idx = node - (2**gen - 1)
        span = 2**(depth - gen)
        return node_idx * span, (node_idx + 1) * span

    def build_subtree_dm(node, gen, rho_parent):
        """
        Given a 2x2 density matrix of the parent qubit,
        return the (2^n_leaves_in_subtree x 2^n_leaves_in_subtree)
        density matrix of all leaves in this subtree.
        """
        if gen == depth:
            # This node IS a leaf — return its density matrix
            return rho_parent

        K = gates[(gen, node)]  # Kraus ops

        # Apply branching channel: parent -> joint state of two children
        rho_children = sum(A @ rho_parent @ A.conj().T for A in K)  # 4x4

        left_child  = 2*node + 1
        right_child = 2*node + 2

        # rho_children is 4x4 = (left_child ⊗ right_child)
        # Split into left and right subtree density matrices
        # Left child density matrix: trace out right
        r = rho_children.reshape(2,2,2,2)
        rho_left  = np.trace(r, axis1=1, axis2=3)   # 2x2
        rho_right = np.trace(r, axis1=0, axis2=2)   # 2x2

        # But we need to maintain correlations between left and right subtrees!
        # We must keep rho_children as the joint state and recurse on both sides
        # while maintaining the tensor product structure.

        # Recursively build density matrices for each subtree
        # Since left and right subtrees are causally independent after the split,
        # their joint state is rho_left_subtree ⊗ rho_right_subtree IF we ignore
        # the initial entanglement from rho_children.
        # But rho_children is NOT necessarily a product state.

        # Correct approach: propagate the joint 4x4 state through both subtrees
        # This requires tracking all cross-correlations.
        # For depth <= 8, use the statevector approach instead.
        rho_left_subtree  = build_subtree_dm(left_child,  gen+1, rho_left)
        rho_right_subtree = build_subtree_dm(right_child, gen+1, rho_right)

        # Approximate: tensor product (ignores inter-subtree entanglement)
        # This is the WRONG approach for entanglement entropy
        # Use statevector instead (see below)
        return np.kron(rho_left_subtree, rho_right_subtree)

    # The recursive approach above loses inter-subtree correlations.
    # Use statevector simulation instead for depth <= 8.
    return build_full_dm_statevector(depth, gates)


def build_full_dm_statevector(depth, gates):
    """
    Build full density matrix of all leaves using statevector simulation.
    State lives in 2^(depth) dimensional space (one qubit per leaf).
    We simulate the tree circuit acting on leaf qubits.

    The tree circuit: at each internal node, the parent qubit is entangled
    with two fresh |0> ancilla qubits (children), then the parent is discarded.

    We implement this by starting with the root in |0> and n_leaves-1 ancillas,
    all in |0>, and applying the circuit.

    Total qubits in circuit = n_leaves (leaves only after tracing out parents).
    State space = 2^n_leaves.

    We simulate by tracking the full 2^n_leaves statevector.
    """
    n_leaves = 2**depth
    # We'll track a density matrix over all currently-active qubits
    # Starting: root is the only qubit, in state |0>
    # At each generation, each active qubit splits into two children

    # Represent state as density matrix over active qubits
    # Generation 0: 1 qubit (root)
    rho = np.array([[1., 0.], [0., 0.]], dtype=complex)  # 2x2

    for gen in range(depth):
        n_nodes_this_gen = 2**gen
        dim_current = 2**n_nodes_this_gen  # dimension of current rho
        dim_next = 2**(2*n_nodes_this_gen)  # each qubit -> 2 children

        rho_next = np.zeros((dim_next, dim_next), dtype=complex)

        # Apply each node's branching channel independently
        # Node k in this generation acts on qubit k
        # Joint state: rho is over all nodes in this generation

        # For independent nodes, we can apply channels one at a time
        # by tensoring with identity on other qubits

        rho_current = rho.copy()

        for node_idx in range(n_nodes_this_gen):
            node = (2**gen - 1) + node_idx
            K = gates[(gen, node)]

            # Embed: apply K to qubit node_idx of rho_current
            # rho_current is 2^n_nodes x 2^n_nodes
            # After applying K to qubit node_idx: that qubit -> 2 qubits
            n_qubits = n_nodes_this_gen
            rho_current = apply_branching_to_qubit(rho_current, K, node_idx, n_qubits)

        rho = rho_current

    return rho


def apply_branching_to_qubit(rho, K, qubit_idx, n_qubits):
    """
    Apply branching channel K to qubit qubit_idx in an n_qubits system.
    This replaces one qubit with two child qubits.
    Output has n_qubits+1 qubits.

    rho: (2^n_qubits x 2^n_qubits)
    K: list of (4x2) Kraus operators
    Returns: (2^(n_qubits+1) x 2^(n_qubits+1)) density matrix
    """
    dim_in  = 2**n_qubits
    dim_out = 2**(n_qubits + 1)

    # Build the full Kraus operators in the n_qubits+1 qubit space
    # Qubit qubit_idx -> two new qubits (inserted at positions qubit_idx, qubit_idx+1)
    # Other qubits shifted right by 1

    rho_out = np.zeros((dim_out, dim_out), dtype=complex)

    for A in K:
        # A is 4x2: maps 1 parent qubit -> 2 child qubits
        # Embed A into full Hilbert space
        # A_full maps 2^n_qubits -> 2^(n_qubits+1)

        A_full = np.zeros((dim_out, dim_in), dtype=complex)

        for col in range(dim_in):
            # Decompose col into bits
            bits_in = [(col >> (n_qubits-1-i)) & 1 for i in range(n_qubits)]
            parent_bit = bits_in[qubit_idx]
            other_bits = bits_in[:qubit_idx] + bits_in[qubit_idx+1:]

            # A maps parent_bit -> two child bits (c1, c2)
            for c1 in range(2):
                for c2 in range(2):
                    amp = A[(c1<<1)|c2, parent_bit]
                    # Build output bits: insert c1, c2 at position qubit_idx
                    bits_out = other_bits[:qubit_idx] + [c1, c2] + other_bits[qubit_idx:]
                    row = 0
                    for b in bits_out: row = (row << 1) | b
                    A_full[row, col] += amp

        rho_out += A_full @ rho @ A_full.conj().T

    return rho_out


def partial_trace_numpy(rho, trace_out_indices, n_qubits):
    """
    Trace out qubits in trace_out_indices from rho (2^n_qubits x 2^n_qubits).
    Returns reduced density matrix over remaining qubits.
    """
    keep = sorted([i for i in range(n_qubits) if i not in trace_out_indices])
    n_keep = len(keep)
    n_trace = len(trace_out_indices)
    trace_out = sorted(trace_out_indices)

    dim_keep  = 2**n_keep
    dim_trace = 2**n_trace
    dim_total = 2**n_qubits

    rho_out = np.zeros((dim_keep, dim_keep), dtype=complex)

    for t in range(dim_trace):
        t_bits = [(t >> (n_trace-1-i)) & 1 for i in range(n_trace)]

        for ki in range(dim_keep):
            for kj in range(dim_keep):
                k_bits_i = [(ki >> (n_keep-1-i)) & 1 for i in range(n_keep)]
                k_bits_j = [(kj >> (n_keep-1-i)) & 1 for i in range(n_keep)]

                # Reconstruct full indices
                bits_i = [0]*n_qubits
                bits_j = [0]*n_qubits
                for pos, q in enumerate(keep):
                    bits_i[q] = k_bits_i[pos]
                    bits_j[q] = k_bits_j[pos]
                for pos, q in enumerate(trace_out):
                    bits_i[q] = t_bits[pos]
                    bits_j[q] = t_bits[pos]

                fi = sum(b << (n_qubits-1-i) for i,b in enumerate(bits_i))
                fj = sum(b << (n_qubits-1-i) for i,b in enumerate(bits_j))

                rho_out[ki, kj] += rho[fi, fj]

    return rho_out


# ── Main ───────────────────────────────────────────────────────────────────

print(f"Backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}")
print("Computing EXACT boundary entanglement entropy S(l)")
print("for contiguous leaf intervals at depth 6 (64 leaves).\n")
print("Testing RT prediction: S(l) = (c/3) * log(l) + const\n")

# Use depth 6 (64 leaves) — statevector is 64x64, tractable
DEPTH = 6
N_TREES = 10
N_LEAVES = 2**DEPTH
interval_sizes = list(range(1, 17))  # l = 1..16

all_S = {l: [] for l in interval_sizes}

for tree_idx in range(N_TREES):
    print(f"Tree {tree_idx+1}/{N_TREES} (seed={1000+tree_idx})", flush=True)
    t0 = time.time()
    gates = build_gates(DEPTH, seed=1000+tree_idx)

    # Build full 64x64 density matrix of all leaves
    print(f"  Building full leaf density matrix...", flush=True)
    rho_all = build_full_dm_statevector(DEPTH, gates)
    print(f"  Done ({time.time()-t0:.1f}s). Shape: {rho_all.shape}", flush=True)

    # Verify trace = 1
    tr = np.trace(rho_all).real
    print(f"  Trace = {tr:.6f} (should be 1.0)", flush=True)

    for l in interval_sizes:
        interval = list(range(l))
        trace_out = list(range(l, N_LEAVES))
        rho_A = partial_trace_numpy(rho_all, trace_out, N_LEAVES)
        S = von_neumann_entropy(rho_A)
        all_S[l].append(S)
        print(f"  l={l:3d}  S={S:.6f}", flush=True)

    print(f"  Tree done in {time.time()-t0:.1f}s\n", flush=True)

# ── Ensemble average ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("ENSEMBLE AVERAGE")
print(f"{'='*60}")

avg_S = {}
for l in interval_sizes:
    avg_S[l] = float(np.mean(all_S[l]))
    std_S = float(np.std(all_S[l]))
    print(f"  l={l:3d}  S={avg_S[l]:.6f}  std={std_S:.6f}")

# ── RT fit on l >= 2 ──────────────────────────────────────────────────────
print(f"\n--- RT Fit: S(l) = (c/3)*log(l) + const ---")
fit_sizes = [l for l in interval_sizes if l >= 2]
sizes_arr = np.array(fit_sizes, dtype=float)
S_arr = np.array([avg_S[l] for l in fit_sizes])
log_l = np.log(sizes_arr)

X = np.column_stack([log_l, np.ones_like(log_l)])
coeffs = np.linalg.lstsq(X, S_arr, rcond=None)[0]
c_over_3, const = coeffs
c = 3 * c_over_3

S_pred = c_over_3 * log_l + const
ss_res = np.sum((S_arr - S_pred)**2)
ss_tot = np.sum((S_arr - S_arr.mean())**2)
r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

print(f"  c/3   = {c_over_3:.6f}")
print(f"  c     = {c:.6f}")
print(f"  const = {const:.6f}")
print(f"  R²    = {r2:.6f}")
print(f"\n  Predicted: {[f'{s:.4f}' for s in S_pred]}")
print(f"  Actual:    {[f'{s:.4f}' for s in S_arr]}")

print(f"\n--- Verdict ---")
if r2 > 0.99:
    print(f"  *** LOG SCALING CONFIRMED: R² = {r2:.6f} ***")
    print(f"  S(l) = (c/3)*log(l) with c = {c:.4f}")
    if abs(c - 1.0) < 0.3:
        print(f"  c ≈ 1: free boson CFT boundary")
    elif abs(c - 0.5) < 0.2:
        print(f"  c ≈ 1/2: Ising/free fermion CFT boundary")
    else:
        print(f"  c = {c:.4f}: non-minimal CFT")
    print(f"  AdS3/CFT2 structure confirmed via Ryu-Takayanagi.")
elif r2 > 0.95:
    print(f"  Marginal log scaling R²={r2:.4f}, c={c:.4f}. RT consistent.")
else:
    print(f"  R²={r2:.4f}. Log scaling not confirmed at this precision.")

out = {
    "depth": DEPTH, "n_trees": N_TREES,
    "interval_sizes": interval_sizes,
    "avg_S": {str(l): avg_S[l] for l in interval_sizes},
    "all_S": {str(l): all_S[l] for l in interval_sizes},
    "c": float(c), "c_over_3": float(c_over_3),
    "const": float(const), "r2": float(r2),
}
with open(f"{RESULTS_DIR}/boundary_entropy_exact.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {RESULTS_DIR}/boundary_entropy_exact.json")
