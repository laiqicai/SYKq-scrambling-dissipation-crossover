# Crossover of Scrambling and Dissipation Timescales in the SYK<sub>q</sub> Model

**Author:** Qicai Lai  
**Paper:** Submitted to Physical Review D (2026)

## Overview

This repository contains the complete reproduction code for all numerical results and figures in the paper. The code performs exact diagonalization (ED) of the Sachdev-Ye-Kitaev model with $q$-body interactions and computes the scrambling time (OTOC half-life) and dissipation time (two-point function half-life) as a function of system size $N$, interaction order $q$, and inverse temperature $\beta$.

## Requirements

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.5  (for figures only)
```

Install via: `pip install numpy scipy matplotlib`

## Usage

All computations are run through a single script:

```bash
# Table III: q=4, N=8-22, βJ=5  (~8 hours)
python syk_crossover.py table3

# Table IV: q=4,6,8 comparison  (~3-5 hours)
python syk_crossover.py table4

# Table V: temperature dependence  (~1-2 hours)
python syk_crossover.py table5

# Table II: Schwarzian analytic corrections  (< 1 min)
python syk_crossover.py schwarzian

# Generate all figures from saved data
python syk_crossover.py figures

# Run everything
python syk_crossover.py all
```

**Checkpoint support:** Each task saves intermediate results to JSON after every completed system size. If interrupted, re-running the same command will skip already-computed data points.

## Output Files

| File | Content |
|------|---------|
| `table3_results.json` | Table III data: $Jt_{\rm OTOC}$, $Jt_{\rm diss}$, $r$ for $q=4$, $N=8$–$22$ |
| `table4_results.json` | Table IV data: $r(N,q)$ for $q=4,6,8$, $N=8$–$16$ |
| `table5_results.json` | Table V data: $r(N,\beta)$ for $\beta J = 2,5,10,20$ |
| `fig_main.png` | Figure 1: timescales, separation ratio, approach to conformal limit |
| `fig_r_corrected.png` | Figure 2: $r(N)$ with conformal and Schwarzian-corrected predictions |
| `fig_corrections.png` | Figure 3: UV correction, corrected correlator, $\beta$-dependence, $N_*(\beta J)$ |

## Computational Cost

| Task | System sizes | Time estimate |
|------|-------------|---------------|
| Table III ($N \leq 16$) | $d = 16$–$256$ | ~2 hours |
| Table III ($N = 18$–$22$) | $d = 512$–$2048$ | ~6 hours |
| Table IV | $d = 16$–$256$ | ~3-5 hours |
| Table V | $d = 32$–$128$ | ~1-2 hours |
| Schwarzian | analytic | < 1 minute |

Estimates are for a single-core modern laptop (e.g., Apple M1/M2 or Intel i7).

## Code Structure

The script is organized into four sections:

1. **SYK Model Implementation** — Majorana operator construction (Kronecker product for $N \leq 16$, bit-string for $N \geq 18$), Hamiltonian building, and correlator computation in the energy eigenbasis.

2. **Schwarzian Analytic Corrections** — Computation of the Kitaev-Suh constants, the dissipation correction $c_d$, and the corrected crossover formula.

3. **Task Runners** — One function per paper table, each with checkpoint support.

4. **Figure Generation** — Reproduces Figures 1–3 from saved JSON data.

## Key Formulas

The crossover scale in the conformal limit:

$$N_*(q) = \exp\!\Big(2\,\mathrm{arccosh}\big(2^{q/2}\big)\Big)$$

With Schwarzian correction:

$$\log N_*(q, \beta J) = 2\,\mathrm{arccosh}(2^{q/2}) \left[1 + \frac{c_d - c_\kappa}{\beta J}\right]$$

where $c_d \approx 0.05$ (analytic) and $c_\kappa \approx 0.31$ (from ED fit) for $q = 4$.

## License

This code is released under the MIT License.

## Citation

If you use this code, please cite:

```bibtex
@article{Lai2026,
  author  = {Lai, Qicai},
  title   = {Crossover of Scrambling and Dissipation Timescales in the {SYK}$_q$ Model},
  journal = {Phys. Rev. D},
  year    = {2026},
  note    = {submitted}
}
```
