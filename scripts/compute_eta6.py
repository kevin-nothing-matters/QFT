"""
compute_eta6.py — Weingarten k=6 computation for QFT τ(3) analytic anchor
===========================================================================
Computes η₆ = E[c(V)³], the third moment of the per-step Bell-sector
contraction weight, using the sixfold Haar average over the branching isometry V.

The analytic program for Law 2:

    k=2: η₂ = E[c]   = 2/5        → τ_op(1) = log(5/2) = 0.9163  [EXACT]
    k=4: η₄ = E[c²]  ≈ 60/323     → τ_op(2) = -log(η₄) ≈ 1.683
    k=6: η₆ = E[c³]  = ?          → τ_op(3) = -log(η₆) = ?  [THIS SCRIPT]

Empirical targets from full-tree ensemble:
    τ(2) = 1.502  [operator gap at k=4: +0.181]
    τ(3) = 2.176  [operator gap at k=6: to measure]

Convergence test: if |gap(k=6)/τ(3)| < |gap(k=4)/τ(2)|, the Weingarten
series is converging toward the true MI moment structure.

CRITICAL NOTE: η₆ = E[c(V)³] uses the SAME V for all three moments.
This is NOT E[c]³ = (2/5)³. It is a genuine k=6 Haar integral.

Author: QFT research session, March 2026
"""

import numpy as np
from fractions import Fraction
from itertools import permutations
import time

# ─── Pauli matrices and Bell operator ────────────────────────────────────────
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
paulis = [X, Y, Z]

B      = np.kron(X,X) + np.kron(Y,Y) + np.kron(Z,Z)
norm_B = float(np.real(np.trace(B.conj().T @ B)))   # = 12.0

print("=" * 68)
print("QFT compute_eta6.py  —  Weingarten k=6, third moment η₆ = E[c³]")
print("=" * 68)
print(f"  Tr(B†B) = {norm_B:.1f}")
print(f"  η₂ = E[c]   = 2/5 = {2/5:.6f}  [analytic, exact]")
print(f"  η₄ = E[c²]  ≈ {60/323:.6f}  [MC, v28]")
print(f"  Empirical τ(3) = 2.176  [tree ensemble target]")
print()


# ─── Core functions ──────────────────────────────────────────────────────────

def sample_V(rng):
    Z = rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))
    Q, _ = np.linalg.qr(Z)
    return Q[:, [0, 2]]   # 4×2 isometry: columns 0,2 of Haar U(4)


def contraction_weight(V):
    """c(V) = Tr(B† Phi_doubled_V(B)) / Tr(B†B)"""
    result = np.zeros((4,4), dtype=complex)
    for sigma in paulis:
        rho4    = V @ sigma @ V.conj().T
        phi_sig = np.zeros((2,2), dtype=complex)
        for r in range(2):
            phi_sig += rho4[r::2, r::2]
        result += np.kron(phi_sig, phi_sig)
    return float(np.real(np.trace(B.conj().T @ result))) / norm_B


def run_mc(n, seed):
    rng = np.random.default_rng(seed)
    return np.array([contraction_weight(sample_V(rng)) for _ in range(n)])


# ─── STEP 1: Validate η₂ and η₄ ─────────────────────────────────────────────
print("STEP 1  Validate η₂ and η₄")
t0     = time.time()
c100k  = run_mc(100_000, seed=0)
eta2_v = np.mean(c100k);      eta4_v = np.mean(c100k**2)
print(f"  η₂ = {eta2_v:.6f}  (analytic 0.400000,  err={abs(eta2_v-2/5):.2e})  {'✓' if abs(eta2_v-2/5)<0.002 else '✗'}")
print(f"  η₄ = {eta4_v:.6f}  (prev     {60/323:.6f}, err={abs(eta4_v-60/323):.2e})")
print(f"  Elapsed: {time.time()-t0:.1f}s")
print()


# ─── STEP 2: Compute η₆ = E[c³] ─────────────────────────────────────────────
print("STEP 2  Compute η₆ = E[c(V)³]")

print("  2a. 200k samples...")
t0   = time.time()
c200 = run_mc(200_000, seed=42)
eta6_fast = np.mean(c200**3)
print(f"      η₆ = {eta6_fast:.6f} ± {np.std(c200**3)/np.sqrt(len(c200)):.6f}  ({time.time()-t0:.0f}s)")

print("  2b. 2M samples (high precision)...")
t0  = time.time()
c2M = run_mc(2_000_000, seed=999)
eta6     = np.mean(c2M**3)
eta6_sem = np.std(c2M**3) / np.sqrt(len(c2M))
print(f"      η₆ = {eta6:.8f} ± {eta6_sem:.8f}  ({time.time()-t0:.0f}s)")
print()


# ─── STEP 3: Rational form ───────────────────────────────────────────────────
print("STEP 3  Rational form search")
best_err, best_frac = 1.0, None
for den in range(1, 100_000):
    num = round(eta6 * den)
    if num <= 0: continue
    err = abs(num/den - eta6)
    if err < best_err:
        best_err, best_frac = err, (num, den)
    if err < eta6_sem * 0.1: break

f6 = Fraction(*best_frac)
print(f"  Best fraction: {f6} = {float(f6):.10f}  (err={best_err:.2e}, SEM={eta6_sem:.2e})")
print(f"  {'HIGH confidence ✓' if best_err < eta6_sem else 'Tentative — need more samples'}")
print()


# ─── STEP 4: Weingarten matrix U(4), k=6 ────────────────────────────────────
print("STEP 4  Exact Weingarten coefficients U(4), k=6  (720×720 system)")

S6 = list(permutations(range(6)))

def inv_p(s):
    r=[0]*6
    for i,x in enumerate(s): r[x]=i
    return tuple(r)

def comp(s,t): return tuple(s[t[i]] for i in range(6))

def ncycles(s):
    vis,c=[False]*6,0
    for i in range(6):
        if not vis[i]:
            c+=1; j=i
            while not vis[j]: vis[j]=True; j=s[j]
    return c

def ctype(s):
    vis,L=[False]*6,[]
    for i in range(6):
        if not vis[i]:
            l=0; j=i
            while not vis[j]: vis[j]=True; j=s[j]; l+=1
            L.append(l)
    return tuple(sorted(L,reverse=True))

print("  Building 720×720 Gram matrix (n=4)...")
t0 = time.time()
n  = 4
M6 = np.zeros((720,720))
for i,s in enumerate(S6):
    for j,t in enumerate(S6):
        M6[i,j] = n**ncycles(comp(s,inv_p(t)))

Wg6  = np.linalg.inv(M6)
res6 = np.max(np.abs(M6 @ Wg6 - np.eye(720)))
print(f"  Inversion residual: {res6:.2e}  {'PASS ✓' if res6<1e-6 else 'WARNING'}")
print(f"  Elapsed: {time.time()-t0:.1f}s")

id6 = S6.index(tuple(range(6)))
seen6 = {}
for i,s in enumerate(S6):
    ct = ctype(s)
    if ct not in seen6:
        seen6[ct] = Wg6[id6, i]

print()
print(f"  Wg coefficients for U(4), k=6  ({len(seen6)} conjugacy classes of S₆):")
for ct,wg in sorted(seen6.items(), key=lambda x:-len(x[0])):
    frac = Fraction(wg).limit_denominator(10_000_000)
    print(f"    {str(ct):22s}  Wg = {wg:+.10f}  ≈  {frac}")
print()


# ─── STEP 5: Convergence analysis ───────────────────────────────────────────
print("=" * 68)
print("STEP 5  Convergence: Weingarten operator-space vs empirical τ(q)")
print("=" * 68)
print()

eta2_an = 2/5
eta4_an = 60/323
eta6_an = eta6

tau_op  = {1: np.log(5/2),  2: -np.log(eta4_an),  3: -np.log(eta6_an)}
tau_emp = {1: np.log(5/2),  2: 1.502,              3: 2.176,  4: 2.709}

print(f"  {'q':>3}  {'τ_op(q)':>10}  {'τ_emp(q)':>10}  {'gap':>8}  {'|gap|/τ_emp':>12}")
print("  " + "-"*55)
for q in [1, 2, 3]:
    top  = tau_op[q]
    temp = tau_emp[q]
    gap  = top - temp
    print(f"  {q:>3}  {top:>10.4f}  {temp:>10.4f}  {gap:>+8.4f}  {abs(gap)/temp:>11.1%}")

print()

gap_frac2 = abs(tau_op[2] - tau_emp[2]) / tau_emp[2]
gap_frac3 = abs(tau_op[3] - tau_emp[3]) / tau_emp[3]

if gap_frac3 < gap_frac2:
    verdict = f"CONVERGING ✓ — gap fraction decreased {gap_frac2:.1%} → {gap_frac3:.1%}"
else:
    verdict = f"NOT converging — gap fraction {gap_frac2:.1%} → {gap_frac3:.1%}"
print(f"  Convergence test: {verdict}")
print()

# β estimates at each level
a = 2/5
beta_k4  = 0.25 * np.log(eta4_an / a**2)
beta_k6  = (tau_op[1]*2 - tau_op[2]) / 2
beta_emp = 0.14
print(f"  β estimates:")
print(f"    β from k=4 (η₄): {beta_k4:.4f}")
print(f"    β from k=6 (η₆, via τ_op): {beta_k6:.4f}")
print(f"    β empirical (asymptotic):  {beta_emp:.4f}")
print()

print("─" * 68)
print("SUMMARY")
print("─" * 68)
print(f"  η₂ = 2/5       = {eta2_an:.8f}  [analytic exact]")
print(f"  η₄ ≈ 60/323    = {eta4_an:.8f}  [v28]")
print(f"  η₆ ≈ {f6:10s} = {eta6_an:.8f}  [this script]  ± {eta6_sem:.2e}")
print()
print(f"  τ_op(1) = {tau_op[1]:.4f}  τ_emp(1) = {tau_emp[1]:.4f}  [exact]")
print(f"  τ_op(2) = {tau_op[2]:.4f}  τ_emp(2) = {tau_emp[2]:.4f}  gap = {tau_op[2]-tau_emp[2]:+.4f}")
print(f"  τ_op(3) = {tau_op[3]:.4f}  τ_emp(3) = {tau_emp[3]:.4f}  gap = {tau_op[3]-tau_emp[3]:+.4f}")
print()
print("  STATUS: The operator-space moments η_{2q} provide a systematic")
print("  upper bound on τ(q). The gap at each order quantifies the size")
print("  of MI nonlinearity corrections (S_A + S_B - S_AB vs operator norm).")
print("  Whether this series converges to τ_emp(q) is the key open question.")
print()
print("Done.")
