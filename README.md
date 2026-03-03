# Quantum Family Tree: Emergent Spacetime from Quantum Genealogy

**"In the beginning there was maybe"**

Computational simulations supporting the Quantum Family Tree theory — a framework proposing that spacetime geometry emerges from the genealogical structure of quantum entanglement.

## Key Results

- **Monotonic kinship decay**: Mutual information between particles decreases with genealogical distance (80% at depth 4, 30 qubits)
- **Holographic scaling**: Distance follows d = 0.854 ln(1/MI) + 1.88 (R² = 0.993), consistent with the Ryu-Takayanagi formula
- **Structural universality**: Pattern holds across random quantum dynamics (Haar-random unitaries), demonstrating topological rather than fine-tuned origins

## Theory

A single quantum state undergoes recursive binary splitting — each generation, every particle splits into two entangled offspring. Entanglement strength between any two descendants decays with their genealogical distance (how many generations since their last common ancestor). This kinship structure IS the geometry of spacetime: closer kin = shorter distance.

The framework aligns with and extends:
- Van Raamsdonk (2010) — spacetime from entanglement
- Ryu-Takayanagi formula — holographic entanglement entropy
- ER=EPR conjecture — wormholes as ancestral paths
- MERA tensor networks — binary entanglement renormalization

## Simulation Architecture

### Pure NumPy statevector simulation (no QuTiP for gates)

Gates are applied by reshaping the statevector into a rank-N tensor and contracting on target indices. This achieves O(2^N) per gate instead of O(4^N) for matrix multiplication, enabling 30-qubit simulations on commodity hardware.

| Depth | Leaves | Qubits | Memory  | Trees | Time/Tree |
|-------|--------|--------|---------|-------|-----------|
| 2     | 4      | 6      | <1 MB   | 200   | <1s       |
| 3     | 8      | 14     | <1 MB   | 100   | ~2s       |
| 4     | 16     | 30     | 17 GB   | 10    | ~9 hrs    |

### Requirements

- Python 3.8+
- NumPy
- SciPy
- 104 GB RAM for depth 4 (Google Cloud n1-highmem-16 recommended)

## Usage

```bash
# Run at depth 2 and 3 (local machine)
python phase_c_v6.py 2 3

# Run at depth 4 (requires 104 GB RAM)
python phase_c_v6.py 4

# Run distance-entanglement fit on results
python fit_analysis.py results/phase_c_v6_TIMESTAMP.json
```

## Results (Depth 4)

| n_g | Mean MI | Ratio | Classical | Relationship |
|-----|---------|-------|-----------|--------------|
| 2   | 1.000   | 1.000 | 1.000     | Siblings     |
| 4   | 0.076   | 0.076 | 0.500     | Cousins      |
| 6   | 0.006   | 0.006 | 0.250     | 2nd Cousins  |
| 8   | 0.001   | 0.001 | 0.125     | 3rd Cousins  |

## Files

- `phase_c_v6.py` — Main simulation (tree construction + MI/RT/Einstein analysis)
- `fit_analysis.py` — Distance-entanglement curve fitting
- `results/` — Output JSON files from simulation runs

## Paper

K. Donahue, "Spacetime as Emergent Geometry from Quantum Genealogy: The Quantum Family Tree Theory" (2026). kevin@nothingmatters.life

## License

MIT License
