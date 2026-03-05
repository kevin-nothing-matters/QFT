# Quantum Family Tree Theory

**In the beginning there was maybe.**

This repository contains the simulation code and paper for the Quantum Family Tree Theory — a framework proposing that spacetime distance is emergent from quantum entanglement structured as a genealogical family tree.

## Core Claim

All particles share a common quantum ancestor. Distance between particles is a measure of entanglement weakness: particles that diverged recently are strongly entangled (nearby); particles that diverged long ago are weakly entangled (distant).

The formal statement: **d(i,j) = f(1/E(i,j))** where E is entanglement strength and f is logarithmic.

This directly extends Van Raamsdonk (2010) — the tree provides the concrete generative mechanism that entanglement-builds-spacetime assumed but could not specify.

---

## Results

### 1. Entanglement Decay (α → 2.0)
Mutual information decays exponentially with genealogical distance: MI ∝ exp(−α·dG). The decay exponent converges to α = 2.0 across depths 2–10 (up to 1,024 leaves, 20-tree ensembles):

| Depth | Leaves | α | Monotonic |
|-------|--------|------|-----------|
| 2 | 4 | 1.60 | 100% |
| 4 | 16 | 1.87 | 100% |
| 6 | 64 | 2.06 | 100% |
| 8 | 256 | 1.96 | 100% |
| 10 | 1,024 | 2.09 | 100% |

α brackets 2.0 from both sides — it is a fixed point, not an asymptote.

### 2. Distance Function (Gap 1)
The distance function is logarithmic: **d(i,j) = (1/2)·log(1/MI) + c**, R² = 0.999994. The slope a = 0.479 converges to the theoretical prediction 0.5 = 1/α. Power law rejected: R² = 0.583, fails all metric axiom checks.

### 3. Ultrametric Geometry
The information metric is ultrametric: U = 0.983 ± 0.006 at depth 8 (1 million triple checks). Median isosceles ratio R = 0.000. The emergent geometry is a hierarchical ultrametric — closer to a p-adic space than flat Euclidean space.

### 4. Boundary Entropy — Ryu-Takayanagi (Gap 2)
Boundary entanglement entropy scales logarithmically with interval size: **S(l) = (c/3)·log(l) + const**, R² = 0.9986, c = 2.836 across l = 1–32 at depth 8. This is the specific signature of AdS₃/CFT₂ structure. The leaves of the tree behave as the boundary of a hyperbolic bulk.

### 5. First Law → Einstein's Equations
The first law of entanglement entropy δS(A) = Tr(δρ_A · H_A) holds with mean ratio δ⟨H⟩/δS = 1.0068 ± 0.015 (85.7% within 2%), with systematic convergence to 1.000 as perturbation ε → 0.

By the Faulkner–Van Raamsdonk theorem (2014), this implies the bulk geometry satisfies the **linearized Einstein equations**:

**δG_μν = 8πG · δT_μν**

Einstein's equations are not an external imposition. They are a consequence of the entanglement structure of the quantum family tree.

---

## Simulation Pipeline

Run in order:

```bash
# 1. Simulate MI decay at depths 8 and 10
python run_d8_d10.py

# 2. Refit alpha (corrected for epsilon floor)
python refit_alpha.py

# 3. Ultrametricity test
python ultrametricity.py

# 4. Gap 1: distance function
python fit_distance_function_v2.py

# 5. Gap 2a: metric reconstruction (MDS + Gromov delta)
python reconstruct_metric.py

# 6. Gap 2b: boundary entropy RT test
python boundary_entropy_exact.py

# 7. First law → Einstein equations
python first_law_exact_v2.py
```

Or run a single tree at any depth:
```bash
python run_qft.py --depth 6 --trees 20
```

### Requirements
```
numpy
scipy
qiskit >= 2.3.0
```

---

## Paper

[Download v15 (DOCX)](https://github.com/kevin-nothing-matters/QFT/raw/main/quantum_family_tree_v15.docx)

---

## Positioning

This work is complementary to, not competitive with:

- **Van Raamsdonk (2010)** — showed entanglement builds spacetime. Our tree provides the generative mechanism.
- **Ryu-Takayanagi formula** — our distance law d = (1/2)·log(1/MI) is the same logarithmic relationship.
- **Faulkner–Van Raamsdonk (2014)** — their theorem derives Einstein's equations from our confirmed first law.
- **It from Qubit program** (Perimeter Institute, IAS Princeton) — we provide a concrete discrete model.

The relationship to Mendel/Darwin is intentional: Van Raamsdonk showed entanglement builds spacetime (Darwin's insight). The quantum family tree provides the mechanism (Mendel's contribution).

---

## Contact

Kevin Donahue  
kevin@nothingmatters.life  
[nothingmatters.life](https://nothingmatters.life)
