"""
Phase C v6: Cosmological Scaling — Pure NumPy Statevector
Author: Kevin Donahue (kevin@3xgenetics.com)

No QuTiP for gate operations. Direct statevector manipulation
via NumPy reshape. Each gate is O(2^n) not O(4^n).

Depth 4 = 30 qubits = 17 GB statevector. Works on 104 GB VM.
"""

import numpy as np
import time, json, os, sys
from datetime import datetime
from itertools import combinations
from scipy.stats import unitary_group, linregress

print("Phase C v6: Cosmological Scaling (Pure NumPy)")
print("Author: Kevin Donahue (kevin@3xgenetics.com)")
print()

DEPTHS = [int(d) for d in sys.argv[1:]] if len(sys.argv) > 1 else [2, 3]
N_ENSEMBLE = {2: 200, 3: 100, 4: 10, 5: 2}
OUTPUT_DIR = os.path.expanduser("~/qft/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

H_GATE = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


# ── GATE OPERATIONS (pure NumPy, O(2^n) each) ──

def apply_single(state, n_qubits, target, gate):
    s = state.reshape([2] * n_qubits)
    s = np.moveaxis(s, target, 0)
    shape = s.shape
    s = gate @ s.reshape(2, -1)
    s = s.reshape(shape)
    s = np.moveaxis(s, 0, target)
    return s.reshape(-1)


def apply_cnot(state, n_qubits, control, target):
    s = state.reshape([2] * n_qubits)
    idx = [slice(None)] * n_qubits
    idx[control] = 1
    sub = s[tuple(idx)].copy()
    t_ax = target if target < control else target - 1
    sub = np.moveaxis(sub, t_ax, 0)
    sub = sub[[1, 0]]
    sub = np.moveaxis(sub, 0, t_ax)
    s[tuple(idx)] = sub
    return s.reshape(-1)


# ── PARTIAL TRACE AND ENTROPY ──

def partial_trace(state, n_qubits, keep):
    keep = sorted(keep)
    trace_out = [q for q in range(n_qubits) if q not in keep]
    psi = np.transpose(state.reshape([2] * n_qubits), keep + trace_out)
    dim_k = 2 ** len(keep)
    dim_t = 2 ** len(trace_out)
    psi = psi.reshape(dim_k, dim_t)
    return psi @ psi.conj().T


def von_neumann_entropy(rho):
    evals = np.real(np.linalg.eigvalsh(rho))
    evals = evals[evals > 1e-12]
    if len(evals) == 0:
        return 0.0
    return float(-np.sum(evals * np.log2(evals)))


def mutual_info_from_state(state, n_qubits, qi, qj):
    ri = partial_trace(state, n_qubits, [qi])
    rj = partial_trace(state, n_qubits, [qj])
    rij = partial_trace(state, n_qubits, [qi, qj])
    return max(0.0, von_neumann_entropy(ri) + von_neumann_entropy(rj) - von_neumann_entropy(rij))


# ── TREE BUILDER ──

def build_tree(depth, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_qubits = 2 ** (depth + 1) - 2
    mem_gb = (2 ** n_qubits) * 16 / 1e9
    if mem_gb > 90:
        raise MemoryError("Would need %.1f GB" % mem_gb)

    # Allocate statevector
    state = np.zeros(2 ** n_qubits, dtype=np.complex128)
    state[0] = 1.0

    # Bell pair on qubits 0, 1
    state = apply_single(state, n_qubits, 0, H_GATE)
    state = apply_cnot(state, n_qubits, 0, 1)

    # Random unitaries on root
    for q in [0, 1]:
        U = unitary_group.rvs(2).astype(np.complex128)
        state = apply_single(state, n_qubits, q, U)

    # Track tree structure
    parents_at_gen = {0: [0, 1]}
    children_of = {}
    next_qubit = 2

    for gen in range(1, depth):
        parents = parents_at_gen[gen - 1]
        gen_children = []
        for p in parents:
            c1, c2 = next_qubit, next_qubit + 1
            next_qubit += 2

            state = apply_cnot(state, n_qubits, p, c1)
            state = apply_cnot(state, n_qubits, p, c2)
            state = apply_single(state, n_qubits, p, H_GATE)

            for c in [c1, c2]:
                U = unitary_group.rvs(2).astype(np.complex128)
                state = apply_single(state, n_qubits, c, U)

            children_of[p] = [c1, c2]
            gen_children.extend([c1, c2])

        parents_at_gen[gen] = gen_children

    state = state / np.linalg.norm(state)
    leaves = parents_at_gen[depth - 1]

    # Build genealogy
    parent_of = {}
    for p, cs in children_of.items():
        for c in cs:
            parent_of[c] = p
    parent_of[0] = "R"
    parent_of[1] = "R"

    def ancestors(q):
        path = [q]
        while q in parent_of and parent_of[q] != "R":
            q = parent_of[q]
            path.append(q)
        path.append("R")
        return path

    leaf_ancestry = {i: ancestors(leaves[i]) for i in range(len(leaves))}

    return state, n_qubits, leaves, len(leaves), leaf_ancestry


def gen_distance(i, j, leaf_ancestry):
    ai = leaf_ancestry[i]
    aj = leaf_ancestry[j]
    sj = set(aj)
    for di, a in enumerate(ai):
        if a in sj:
            return di + aj.index(a)
    return 999


# ── TEST FUNCTIONS ──

def test_kinship(state, n_qubits, leaves, n_leaves, leaf_ancestry, max_pairs=200):
    mi_by_dist = {}
    pairs = list(combinations(range(n_leaves), 2))
    if len(pairs) > max_pairs:
        np.random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    for i, j in pairs:
        ng = gen_distance(i, j, leaf_ancestry)
        mi = mutual_info_from_state(state, n_qubits, leaves[i], leaves[j])
        mi_by_dist.setdefault(ng, []).append(mi)
    results = {}
    for ng in sorted(mi_by_dist):
        results[ng] = {
            "mean": float(np.mean(mi_by_dist[ng])),
            "std": float(np.std(mi_by_dist[ng])),
            "n": len(mi_by_dist[ng]),
        }
    dists = sorted(results)
    means = [results[d]["mean"] for d in dists]
    mono = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    sib = results[min(results)]["mean"] if results else 0
    for ng in results:
        results[ng]["ratio"] = results[ng]["mean"] / sib if sib > 0 else 0
        results[ng]["predicted"] = 0.5 ** ((ng - min(results)) // 2)
    return {"by_distance": results, "monotonic": mono, "sibling_mi": sib}


def test_rt(state, n_qubits, leaves, n_leaves, depth, n_parts=30):
    entropies = []
    boundaries = []
    sizes = []
    for _ in range(n_parts):
        k = np.random.randint(1, n_leaves)
        A_idx = sorted(np.random.choice(n_leaves, k, replace=False))
        A_qubits = [leaves[i] for i in A_idx]
        rho_A = partial_trace(state, n_qubits, A_qubits)
        S = von_neumann_entropy(rho_A)
        bonds = minimal_cut(list(A_idx), n_leaves, depth)
        entropies.append(S)
        boundaries.append(bonds)
        sizes.append(k)
    if len(set(boundaries)) > 1:
        sl, ic, r, p, se = linregress(boundaries, entropies)
        r2_b = r ** 2
    else:
        sl, r2_b, p = 0, 0, 1
    if len(set(sizes)) > 1:
        sl2, ic2, r2, p2, se2 = linregress(sizes, entropies)
        r2_v = r2 ** 2
    else:
        r2_v = 0
    return {
        "boundary_r2": float(r2_b),
        "volume_r2": float(r2_v),
        "holographic": r2_b > r2_v,
        "slope": float(sl),
        "p": float(p),
    }


def minimal_cut(partition_A, n_leaves, depth):
    setA = set(partition_A)
    membership = ["A" if i in setA else "B" for i in range(n_leaves)]
    cuts = 0
    for level in range(depth):
        new_mem = []
        for p in range(0, len(membership), 2):
            l = membership[p]
            r = membership[p + 1] if p + 1 < len(membership) else l
            if l != r:
                cuts += 1
                new_mem.append("M")
            elif l == "M" or r == "M":
                new_mem.append("M")
            else:
                new_mem.append(l)
        membership = new_mem
    return max(1, cuts)


def test_einstein(state, n_qubits, leaves, n_leaves, leaf_ancestry, max_pairs=120):
    pairs = list(combinations(range(n_leaves), 2))
    if len(pairs) > max_pairs:
        np.random.shuffle(pairs)
        pairs = pairs[:max_pairs]
    mi_d = {}
    for i, j in pairs:
        mi = mutual_info_from_state(state, n_qubits, leaves[i], leaves[j])
        mi_d[(i, j)] = mi
        mi_d[(j, i)] = mi
    d_d = {}
    for k, v in mi_d.items():
        d_d[k] = -np.log2(v) if v > 1e-10 else 20.0
    all_leaves = list(set(x for p in pairs for x in p))
    lap = []
    tvs = []
    for i, j in pairs:
        dij = d_d.get((i, j))
        if dij is None:
            continue
        nbrs = [k for k in all_leaves if k != i and k != j and (i, k) in d_d]
        if len(nbrs) < 2:
            continue
        avg = np.mean([d_d[(i, k)] for k in nbrs])
        lap.append(avg - dij)
        tvs.append(mi_d[(i, j)] / dij if dij > 0.01 else 0)
    if len(lap) > 5 and len(set(tvs)) > 1:
        sl, ic, r, p, se = linregress(tvs, lap)
        return {"slope": float(sl), "r2": float(r ** 2), "p": float(p), "n": len(lap)}
    return {"slope": 0, "r2": 0, "p": 1, "n": len(lap)}


# ── MAIN ──

all_results = {"timestamp": datetime.now().isoformat(), "depths": {}}
kinship_data = {}

for depth in DEPTHS:
    n_leaves = 2 ** depth
    n_qubits = 2 ** (depth + 1) - 2
    n_ens = N_ENSEMBLE.get(depth, 5)
    mem_gb = (2 ** n_qubits) * 16 / 1e9
    print("=" * 60)
    print("  DEPTH %d (%d leaves, %d qubits, %.2f GB, %d trees)" % (depth, n_leaves, n_qubits, mem_gb, n_ens))
    print("=" * 60)

    d_res = {"kinship": [], "rt": [], "einstein": [], "times": []}
    k_accum = {}

    for trial in range(n_ens):
        t0 = time.time()
        sys.stdout.write("  Tree %d/%d... " % (trial + 1, n_ens))
        sys.stdout.flush()
        try:
            state, nq, leaves, nl, la = build_tree(depth, seed=trial * 100 + depth)
            sys.stdout.write("built (%.0fs)... " % (time.time() - t0))
            sys.stdout.flush()

            kr = test_kinship(state, nq, leaves, nl, la)
            rt = test_rt(state, nq, leaves, nl, depth)
            ee = test_einstein(state, nq, leaves, nl, la)

            d_res["kinship"].append(kr)
            d_res["rt"].append(rt)
            d_res["einstein"].append(ee)
            for ng, data in kr["by_distance"].items():
                k_accum.setdefault(ng, []).append(data["mean"])

            dt = time.time() - t0
            d_res["times"].append(dt)
            print("done (%.1fs total)" % dt)

            # Free memory
            del state

        except MemoryError as e:
            print("MEMORY ERROR: %s" % str(e))
            break
        except Exception as e:
            print("ERROR: %s" % str(e))
            import traceback
            traceback.print_exc()
            continue

    print("\n  -- Kinship Decay --")
    if k_accum:
        mn = min(k_accum)
        sib = np.mean(k_accum[mn])
        print("  %-6s %-12s %-10s %-10s %s" % ("ng", "MI", "Ratio", "Pred", "N"))
        print("  " + "-" * 50)
        for ng in sorted(k_accum):
            m = np.mean(k_accum[ng])
            r = m / sib if sib > 0 else 0
            pr = 0.5 ** ((ng - mn) // 2)
            print(
                "  %-6d %-12.6f %-10.4f %-10.4f %d"
                % (ng, m, r, pr, len(k_accum[ng]))
            )
    mono_pct = (
        sum(1 for k in d_res["kinship"] if k["monotonic"])
        / max(1, len(d_res["kinship"]))
        * 100
    )
    print("  Monotonic decay: %.0f%%" % mono_pct)

    if d_res["rt"]:
        r2s = [r["boundary_r2"] for r in d_res["rt"]]
        holo = (
            sum(1 for r in d_res["rt"] if r["holographic"])
            / len(d_res["rt"])
            * 100
        )
        print("\n  -- Ryu-Takayanagi --")
        print("  Mean R2: %.4f +/- %.4f" % (np.mean(r2s), np.std(r2s)))
        print("  Holographic: %.0f%%" % holo)

    if d_res["einstein"]:
        r2e = [e["r2"] for e in d_res["einstein"] if e["r2"] > 0]
        if r2e:
            print("\n  -- Einstein Equation --")
            print("  Mean R2: %.4f +/- %.4f" % (np.mean(r2e), np.std(r2e)))

    if d_res["times"]:
        print("\n  Avg time: %.1fs/tree" % np.mean(d_res["times"]))
        print("  Total: %.0fs" % np.sum(d_res["times"]))

    all_results["depths"][str(depth)] = {
        "n_trees": len(d_res["times"]),
        "kinship": {
            str(ng): {
                "mean": float(np.mean(vs)),
                "std": float(np.std(vs)),
                "n": len(vs),
            }
            for ng, vs in k_accum.items()
        },
    }
    kinship_data[depth] = k_accum

print("\n" + "=" * 60)
print("  COSMIC EXPANSION")
print("=" * 60)
print("  %-6s %-8s %-8s %-12s" % ("Gen", "Leaves", "Scale", "Avg MI"))
print("  " + "-" * 40)
base = 2 ** min(DEPTHS)
for d in sorted(kinship_data):
    nl = 2 ** d
    sf = nl / base
    all_mi = [v for vals in kinship_data[d].values() for v in vals]
    avg = np.mean(all_mi) if all_mi else 0
    print("  %-6d %-8d %-8.1f %-12.6f" % (d, nl, sf, avg))

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outf = OUTPUT_DIR + "/phase_c_v6_" + ts + ".json"
with open(outf, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nResults saved to " + outf)
print("\nPHASE C COMPLETE.")
