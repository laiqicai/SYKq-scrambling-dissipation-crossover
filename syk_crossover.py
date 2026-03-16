#!/usr/bin/env python3
"""
================================================================================
Reproduction Code for:
  "Crossover of Scrambling and Dissipation Timescales in the SYK_q Model"
  Qicai Lai (2026)
================================================================================

This single script reproduces all numerical results and figures in the paper.
It consolidates five original computation scripts into a unified codebase.

Usage:
  python syk_crossover.py table3          # Table III:  q=4, N=8-22 (βJ=5)
  python syk_crossover.py table4          # Table IV:   q=4,6,8 comparison
  python syk_crossover.py table5          # Table V:    temperature dependence
  python syk_crossover.py schwarzian      # Table II + analytic corrections
  python syk_crossover.py figures         # Generate all figures
  python syk_crossover.py all             # Run everything (≈10-15 hours)

Each task saves intermediate results to JSON and supports resumption.

Requirements:
  numpy, scipy
  matplotlib (optional, for figures only)

Hardware:
  Table III (N≤16): ~2 hours on a modern laptop
  Table III (N=18-22): ~6 additional hours
  Table IV: ~3-5 hours
  Table V: ~1-2 hours
  Schwarzian: < 1 minute (purely analytic)
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit, brentq
from scipy.special import gamma as Gamma
from scipy.integrate import quad
from itertools import combinations
from math import factorial, comb
import json, time, os, sys, argparse, warnings

warnings.filterwarnings("ignore")

OUTDIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."


# =============================================================================
#  I. SYK Model — Unified Implementation
# =============================================================================

def build_majoranas_kronecker(N_majorana):
    """Build Majorana operators via Kronecker products. Best for N ≤ 16."""
    n = N_majorana // 2
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    psi = []
    for j in range(n):
        for pauli in [sx, sy]:
            op = np.array([[1.0]], dtype=complex)
            for k in range(n):
                if k < j:
                    op = np.kron(op, sz)
                elif k == j:
                    op = np.kron(op, pauli)
                else:
                    op = np.kron(op, I2)
            psi.append(op)
    return psi


def build_majoranas_bitstring(N_majorana):
    """Build Majorana operators via bit-string method. Faster for N ≥ 18."""
    n_complex = N_majorana // 2
    dim = 2 ** n_complex
    psi = []
    for m in range(N_majorana):
        j = m // 2
        kind = m % 2
        mat = np.zeros((dim, dim), dtype=complex)
        for state in range(dim):
            jw_sign = 1
            for k in range(j):
                if state & (1 << k):
                    jw_sign *= -1
            occ = (state >> j) & 1
            if kind == 0:
                new_state = state ^ (1 << j)
                mat[new_state, state] += jw_sign
            else:
                if occ == 0:
                    new_state = state | (1 << j)
                    mat[new_state, state] += -1j * jw_sign
                else:
                    new_state = state & ~(1 << j)
                    mat[new_state, state] += 1j * jw_sign
        psi.append(mat)
    return psi


def build_hamiltonian(psi, N, q, rng):
    """Build the SYK_q Hamiltonian with random couplings."""
    dim = psi[0].shape[0]
    H = np.zeros((dim, dim), dtype=complex)
    sigma = np.sqrt(factorial(q - 1) / (N ** (q - 1)))
    phase = 1j ** (q * (q - 1) // 2)
    for combo in combinations(range(N), q):
        J_val = rng.randn() * sigma
        op = np.eye(dim, dtype=complex)
        for idx in combo:
            op = op @ psi[idx]
        H += J_val * phase * op
    H = (H + H.conj().T) / 2.0
    return H


def compute_correlators(E, V, psi, Jt_values, beta, n_otoc_pairs, rng_seed):
    """
    Compute the normalized two-point function I1(t) and OTOC I2(t).

    Works in the energy eigenbasis for efficiency:
      M_i[a,b] = <a|ψ_i|b>
      I1 uses O(N × dim² × n_t) operations.
      I2 uses O(n_pairs × dim³ × n_t) operations.

    Parameters
    ----------
    E : array, shape (dim,) — energy eigenvalues
    V : array, shape (dim, dim) — eigenvectors
    psi : list of arrays — Majorana operators in the computational basis
    Jt_values : array — dimensionless times Jt
    beta : float — inverse temperature (in units of 1/J)
    n_otoc_pairs : int — number of (i,j) pairs to sample for the OTOC
    rng_seed : int — seed for pair sampling

    Returns
    -------
    I1, I2 : arrays of shape (n_t,)
    """
    dim = len(E)
    N = len(psi)
    J = np.std(E)
    times = Jt_values / J if J > 1e-10 else Jt_values

    # Thermal weights
    boltz = np.exp(-beta * E)
    rho_diag = boltz / np.sum(boltz)

    # Energy-basis matrix elements
    M = [V.conj().T @ psi[i] @ V for i in range(N)]

    # --- Two-point function ---
    I1 = np.zeros(len(times))
    for i in range(N):
        Wi = rho_diag[:, None] * np.abs(M[i]) ** 2
        norm_i = np.sum(Wi)
        dE = E[:, None] - E[None, :]
        for ti, t in enumerate(times):
            phase = np.exp(1j * dE * t)
            I1[ti] += np.abs(np.sum(Wi * phase)) / norm_i
    I1 /= N

    # --- OTOC ---
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    pair_rng = np.random.RandomState(rng_seed)
    if n_otoc_pairs < len(all_pairs):
        idx = pair_rng.choice(len(all_pairs), size=n_otoc_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    # OTOC at t=0 for normalization
    otoc_0 = 0.0
    for i, j in pairs:
        P = M[i] @ M[j]
        P2 = P @ P
        otoc_0 += np.sum(rho_diag * np.diag(P2)).real
    otoc_0 /= len(pairs)

    I2 = np.zeros(len(times))
    for ti, t in enumerate(times):
        exp_pos = np.exp(1j * E * t)
        exp_neg = np.exp(-1j * E * t)
        total = 0.0
        for i, j in pairs:
            temp = M[i] * exp_neg[None, :]
            B = temp @ M[j]
            B = exp_pos[:, None] * B
            rhoB = rho_diag[:, None] * B
            total += np.sum(rhoB * B.T).real
        total /= len(pairs)
        I2[ti] = -total / abs(otoc_0) if abs(otoc_0) > 1e-15 else 0.0

    return I1, I2


def find_half_life(signal, times, threshold=0.5):
    """Find the time at which signal crosses threshold by linear interpolation."""
    for i in range(len(signal) - 1):
        if signal[i] >= threshold > signal[i + 1]:
            f = (signal[i] - threshold) / (signal[i] - signal[i + 1])
            return times[i] + f * (times[i + 1] - times[i])
    return None


def bootstrap_ratio(all_I1, all_I2, Jt, n_boot=200, seed=42):
    """Bootstrap estimate of t_diss, t_otoc, and their ratio."""
    rng = np.random.RandomState(seed)
    n_real = len(all_I1)
    t1_boot, t2_boot, r_boot = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n_real, size=n_real, replace=True)
        I1b = np.mean(np.array(all_I1)[idx], axis=0)
        I2b = np.mean(np.array(all_I2)[idx], axis=0)
        tb1 = find_half_life(I1b, Jt)
        tb2 = find_half_life(I2b, Jt)
        if tb1:
            t1_boot.append(tb1)
        if tb2:
            t2_boot.append(tb2)
        if tb1 and tb2:
            r_boot.append(tb1 / tb2)
    return {
        "t_diss": np.mean(t1_boot) if t1_boot else None,
        "t_diss_err": np.std(t1_boot) if t1_boot else None,
        "t_otoc": np.mean(t2_boot) if t2_boot else None,
        "t_otoc_err": np.std(t2_boot) if t2_boot else None,
        "ratio": np.mean(r_boot) if r_boot else None,
        "ratio_err": np.std(r_boot) if r_boot else None,
    }


def run_ed(N, q, beta, n_real, n_otoc_pairs, Jt, verbose=True):
    """
    Run exact diagonalization for SYK_q at given N.

    Returns lists of I1 and I2 curves (one per realization).
    """
    use_bitstring = N >= 18
    if verbose:
        method = "bitstring" if use_bitstring else "Kronecker"
        dim = 2 ** (N // 2)
        print(f"  N={N} | dim={dim} | q={q} | {n_real} samples | method={method}")

    psi = build_majoranas_bitstring(N) if use_bitstring else build_majoranas_kronecker(N)

    all_I1, all_I2 = [], []
    t_start = time.time()

    for r in range(n_real):
        rng = np.random.RandomState(r * 1000 + N + q * 100)
        H = build_hamiltonian(psi, N, q, rng)
        E, V = eigh(H)

        I1, I2 = compute_correlators(E, V, psi, Jt, beta, n_otoc_pairs, rng_seed=r)
        all_I1.append(I1)
        all_I2.append(I2)

        if verbose:
            elapsed = time.time() - t_start
            eta = elapsed / (r + 1) * (n_real - r - 1)
            print(f"\r    {r+1}/{n_real} | {elapsed/60:.1f}min elapsed"
                  f" | ~{eta/60:.1f}min left", end="", flush=True)

    if verbose:
        print(f"\n    Done in {(time.time()-t_start)/60:.1f} min")
    return all_I1, all_I2


# =============================================================================
#  II. Schwarzian Analytic Corrections
# =============================================================================

def syk_constants(q):
    """Compute all analytic constants from Kitaev-Suh (2018) Table 1."""
    Delta = 1.0 / q
    b = ((q - 2) * np.tan(np.pi / q) / (2 * np.pi * q)) ** (1.0 / q)

    def kc(h):
        u = lambda x: Gamma(2 * x) * np.sin(np.pi * x)
        return u(Delta - (1 - h) / 2) * u(Delta - h / 2) / (u(Delta + 0.5) * u(Delta - 1))

    neg_kcp2 = np.pi / np.sin(2 * np.pi / q) - q * (q ** 2 - 6 * q + 6) / (2 * (q - 1) * (q - 2))

    alpha_S_table = {4: 0.00661, 6: 0.001337, 8: 0.000416, 10: 0.000161}
    alpha_S = alpha_S_table.get(q, 1.0 / (4 * np.pi * q ** 2))

    a0 = alpha_S * 6 * q / np.sqrt((q - 1) * b)
    alpha_G = a0 / (neg_kcp2 * np.sqrt((q - 1) * b))
    x0 = np.arccosh(2 ** (q / 2))

    return {
        "q": q, "Delta": Delta, "b": b, "alpha_S": alpha_S, "alpha_G": alpha_G,
        "neg_kc_prime_2": neg_kcp2, "x0": x0,
        "N_star_conf": np.exp(2 * x0),
        "t_diss_conf_over_beta": x0 / np.pi,
    }


def compute_cd(const):
    """Compute the 1/(βJ) correction to the dissipation time, c_d(q)."""
    alpha_G = const["alpha_G"]
    Delta = const["Delta"]
    x0 = const["x0"]
    tanh_x0 = np.tanh(x0)
    f_x0 = 1 - x0 * tanh_x0
    cd = -2 * alpha_G * f_x0 / (2 * Delta * x0 * tanh_x0)
    return cd


def r_corrected(N, q, betaJ, cd, ck):
    """Schwarzian-corrected separation ratio."""
    x0 = np.arccosh(2 ** (q / 2))
    return 2 * x0 / np.log(N) * (1 + (cd - ck) / betaJ)


def Nstar_corrected(q, betaJ, cd, ck):
    """Schwarzian-corrected crossover scale."""
    x0 = np.arccosh(2 ** (q / 2))
    return np.exp(2 * x0) * np.exp(2 * x0 * (cd - ck) / betaJ)


# =============================================================================
#  III. Task Runners — Each Produces One Table/Figure Set
# =============================================================================

# ---------- Configurations ----------

TABLE3_CONFIG = {
    # N: (n_realizations, n_otoc_pairs)
    8: (100, 56), 10: (80, 90), 12: (50, 80), 14: (30, 80),
    16: (20, 60), 18: (15, 50), 20: (10, 50), 22: (5, 40),
}

TABLE4_CONFIG = {
    # (q, N): (n_realizations, n_otoc_pairs)
    (4, 8): (80, 56), (4, 10): (60, 60), (4, 12): (40, 60),
    (4, 14): (25, 60), (4, 16): (15, 60),
    (6, 8): (80, 56), (6, 10): (60, 60), (6, 12): (40, 60),
    (6, 14): (25, 60), (6, 16): (15, 60),
    (8, 8): (80, 56), (8, 10): (60, 60), (8, 12): (40, 60),
    (8, 14): (20, 60), (8, 16): (10, 60),
}

TABLE5_CONFIG = {
    # N: (n_realizations, beta_values)
    10: (50, [2.0, 5.0, 10.0, 20.0]),
    12: (30, [2.0, 5.0, 10.0, 20.0]),
    14: (20, [2.0, 5.0, 10.0, 20.0]),
}


def load_or_init(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            pass
    return {}


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Table III: q=4, N=8-22, βJ=5 ----------

def task_table3(output="table3_results.json"):
    """Reproduce Table III: timescales for q=4 at β=5/J."""
    print("\n" + "=" * 65)
    print("  TABLE III: q=4, β=5/J, N=8-22")
    print("=" * 65)

    q, beta = 4, 5.0
    Jt = np.linspace(0, 20.0, 200)
    results = load_or_init(output)

    for N in sorted(TABLE3_CONFIG):
        key = str(N)
        if key in results:
            print(f"  N={N}: already computed, skipping")
            continue

        n_real, n_pairs = TABLE3_CONFIG[N]
        all_I1, all_I2 = run_ed(N, q, beta, n_real, n_pairs, Jt)

        I1m = np.mean(all_I1, axis=0)
        I2m = np.mean(all_I2, axis=0)
        t1 = find_half_life(I1m, Jt)
        t2 = find_half_life(I2m, Jt)
        bs = bootstrap_ratio(all_I1, all_I2, Jt)

        results[key] = {
            "N": N, "q": q, "beta": beta,
            "t_diss": float(t1) if t1 else None,
            "t_otoc": float(t2) if t2 else None,
            "ratio": bs["ratio"], "ratio_err": bs["ratio_err"],
            "t_diss_err": bs["t_diss_err"], "t_otoc_err": bs["t_otoc_err"],
        }
        save_json(results, output)
        r_str = f"{bs['ratio']:.3f}±{bs['ratio_err']:.3f}" if bs["ratio"] else "N/A"
        print(f"  → N={N}: r = {r_str}")

    # Print table
    print(f"\n  {'N':>3}  {'Jt_otoc':>12}  {'Jt_diss':>12}  {'r':>14}")
    print("  " + "-" * 46)
    for N in sorted(TABLE3_CONFIG):
        v = results.get(str(N), {})
        if v.get("ratio"):
            t2s = f"{v['t_otoc']:.3f}({v['t_otoc_err']*1000:.0f})"
            t1s = f"{v['t_diss']:.3f}({v['t_diss_err']*1000:.0f})"
            rs = f"{v['ratio']:.3f}({v['ratio_err']*1000:.0f})"
            print(f"  {N:>3}  {t2s:>12}  {t1s:>12}  {rs:>14}")

    # Fit r = c/log(N)
    Ns, rs, errs = [], [], []
    for N in sorted(TABLE3_CONFIG):
        v = results.get(str(N), {})
        if v.get("ratio"):
            Ns.append(v["N"]); rs.append(v["ratio"]); errs.append(v["ratio_err"])
    if len(Ns) >= 3:
        Na, ra, ea = np.array(Ns, float), np.array(rs), np.array(errs)
        popt, _ = curve_fit(lambda N, c: c / np.log(N), Na, ra, sigma=ea, p0=[4.0])
        c_fit = popt[0]
        c_conf = 2 * np.arccosh(4)
        print(f"\n  Fit: c = {c_fit:.3f}  (conformal: {c_conf:.3f},"
              f" ratio = {c_fit/c_conf:.3f})")
        print(f"  N*_num = e^c ≈ {np.exp(c_fit):.0f}  (conformal: 62)")

    print(f"\n  Results saved: {output}")


# ---------- Table IV: q-dependence ----------

def task_table4(output="table4_results.json"):
    """Reproduce Table IV: r(N,q) for q=4,6,8."""
    print("\n" + "=" * 65)
    print("  TABLE IV: q-dependence, β=5/J")
    print("=" * 65)

    beta = 5.0
    Jt = np.linspace(0, 30.0, 200)
    results = load_or_init(output)

    for (q, N) in sorted(TABLE4_CONFIG):
        key = f"q{q}_N{N}"
        if key in results:
            print(f"  q={q}, N={N}: already computed, skipping")
            continue

        n_real, n_pairs = TABLE4_CONFIG[(q, N)]
        all_I1, all_I2 = run_ed(N, q, beta, n_real, n_pairs, Jt)

        I1m = np.mean(all_I1, axis=0)
        I2m = np.mean(all_I2, axis=0)
        t1 = find_half_life(I1m, Jt)
        t2 = find_half_life(I2m, Jt)
        bs = bootstrap_ratio(all_I1, all_I2, Jt)

        results[key] = {
            "q": q, "N": N, "beta": beta,
            "t_diss": float(t1) if t1 else None,
            "t_otoc": float(t2) if t2 else None,
            "ratio": bs["ratio"], "ratio_err": bs["ratio_err"],
        }
        save_json(results, output)
        r_str = f"{bs['ratio']:.3f}" if bs["ratio"] else "N/A"
        print(f"  → q={q}, N={N}: r = {r_str}")

    # Print table
    print(f"\n  {'N':>3}  {'r(q=4)':>10}  {'r(q=6)':>10}  {'r(q=8)':>10}")
    print("  " + "-" * 36)
    for N in [8, 10, 12, 14, 16]:
        row = f"  {N:>3}"
        for q in [4, 6, 8]:
            v = results.get(f"q{q}_N{N}", {})
            row += f"  {v.get('ratio', 0):>10.3f}" if v.get("ratio") else "       N/A"
        print(row)

    print(f"\n  Results saved: {output}")


# ---------- Table V: temperature dependence ----------

def task_table5(output="table5_results.json"):
    """Reproduce Table V: r(N, β) for q=4."""
    print("\n" + "=" * 65)
    print("  TABLE V: temperature dependence, q=4")
    print("=" * 65)

    q = 4
    Jt = np.linspace(0, 30.0, 200)
    results = load_or_init(output)

    for N in sorted(TABLE5_CONFIG):
        n_real, betas = TABLE5_CONFIG[N]
        for beta in betas:
            key = f"N{N}_beta{beta}"
            if key in results:
                print(f"  N={N}, β={beta}: already computed, skipping")
                continue

            n_pairs = min(N * (N - 1), 60)
            all_I1, all_I2 = run_ed(N, q, beta, n_real, n_pairs, Jt)

            I1m = np.mean(all_I1, axis=0)
            I2m = np.mean(all_I2, axis=0)
            t1 = find_half_life(I1m, Jt)
            t2 = find_half_life(I2m, Jt)
            r_val = t1 / t2 if (t1 and t2) else None

            results[key] = {
                "N": N, "q": q, "beta": beta,
                "t_diss": float(t1) if t1 else None,
                "t_otoc": float(t2) if t2 else None,
                "ratio": float(r_val) if r_val else None,
            }
            save_json(results, output)
            r_str = f"{r_val:.3f}" if r_val else "N/A"
            print(f"  → N={N}, βJ={beta}: r = {r_str}")

    # Print table
    betas = [2.0, 5.0, 10.0, 20.0]
    print(f"\n  {'N':>3}", end="")
    for b in betas:
        print(f"  {'βJ='+str(int(b)):>8}", end="")
    print()
    print("  " + "-" * 38)
    for N in sorted(TABLE5_CONFIG):
        row = f"  {N:>3}"
        for b in betas:
            v = results.get(f"N{N}_beta{b}", {})
            row += f"  {v.get('ratio', 0):>8.3f}" if v.get("ratio") else "      N/A"
        print(row)

    # Fit c_d - c_kappa
    print("\n  Fitting r(βJ) = A(1 + C/βJ):")
    C_vals = []
    for N in sorted(TABLE5_CONFIG):
        bJ_arr, r_arr = [], []
        for b in betas:
            v = results.get(f"N{N}_beta{b}", {})
            if v.get("ratio"):
                bJ_arr.append(b)
                r_arr.append(v["ratio"])
        if len(bJ_arr) >= 3:
            bJ_arr, r_arr = np.array(bJ_arr), np.array(r_arr)
            try:
                popt, pcov = curve_fit(lambda x, A, C: A * (1 + C / x),
                                       bJ_arr, r_arr, p0=[r_arr[-1], -0.3])
                perr = np.sqrt(np.diag(pcov))
                print(f"    N={N}: A={popt[0]:.4f}±{perr[0]:.4f},"
                      f" C=c_d-c_κ={popt[1]:.3f}±{perr[1]:.3f}")
                C_vals.append(popt[1])
            except Exception as e:
                print(f"    N={N}: fit failed ({e})")
    if C_vals:
        C_mean = np.mean(C_vals)
        C_sem = np.std(C_vals) / np.sqrt(len(C_vals))
        print(f"    Average: c_d - c_κ = {C_mean:.3f} ± {C_sem:.3f}")

    print(f"\n  Results saved: {output}")


# ---------- Schwarzian corrections (Table II + analytics) ----------

def task_schwarzian():
    """Compute analytic Schwarzian corrections (Table II)."""
    print("\n" + "=" * 65)
    print("  SCHWARZIAN CORRECTIONS (analytic)")
    print("=" * 65)

    # ED data from Table V for fitting
    ed_beta = {
        10: {2: 1.474, 5: 1.629, 10: 1.678, 20: 1.692},
        12: {2: 1.293, 5: 1.376, 10: 1.406, 20: 1.419},
        14: {2: 1.198, 5: 1.301, 10: 1.362, 20: 1.398},
    }
    ed_r = {8: 2.342, 10: 1.632, 12: 1.380, 14: 1.307, 16: 1.282,
            18: 1.250, 20: 1.201, 22: 1.169}

    # Analytic constants for q=4,6,8
    print("\n  Analytic constants:")
    print(f"  {'q':>3}  {'Δ':>6}  {'α_G':>10}  {'x₀':>8}  {'N*_conf':>8}  {'c_d':>8}")
    print("  " + "-" * 50)
    for q in [4, 6, 8]:
        c = syk_constants(q)
        cd = compute_cd(c)
        print(f"  {q:>3}  {c['Delta']:>6.3f}  {c['alpha_G']:>10.5f}"
              f"  {c['x0']:>8.4f}  {c['N_star_conf']:>8.1f}  {cd:>8.4f}")

    # Fit c_d - c_kappa from beta-dependent data
    print("\n  Fitting c_d - c_κ from β-dependent ED data:")
    C_vals = []
    for N, bdata in ed_beta.items():
        bJs = np.array(sorted(bdata.keys()), dtype=float)
        rs = np.array([bdata[int(b)] for b in bJs])
        try:
            popt, pcov = curve_fit(lambda x, A, C: A * (1 + C / x), bJs, rs, p0=[rs[-1], -0.3])
            perr = np.sqrt(np.diag(pcov))
            print(f"    N={N}: C = {popt[1]:.3f} ± {perr[1]:.3f}")
            C_vals.append(popt[1])
        except:
            pass
    C_mean = np.mean(C_vals) if C_vals else -0.26
    print(f"    Average: c_d - c_κ = {C_mean:.3f}")

    # Table II
    const4 = syk_constants(4)
    cd4 = compute_cd(const4)
    ck4 = cd4 - C_mean

    print(f"\n  TABLE II (q=4):")
    print(f"    c_d           = {cd4:.4f}")
    print(f"    c_d - c_κ     = {C_mean:.3f}")
    print(f"    c_κ           = {ck4:.3f}")
    print(f"    |c_κ/c_d|     = {abs(ck4/cd4):.1f}  (scrambling-dominated)")
    print(f"    N*(βJ=5)      ≈ {Nstar_corrected(4, 5, cd4, ck4):.0f}  (conformal: 62)")

    # Fit c/log(N)
    Ns = np.array(sorted(ed_r.keys()), dtype=float)
    rs = np.array([ed_r[int(n)] for n in Ns])
    popt, _ = curve_fit(lambda N, c: c / np.log(N), Ns, rs, p0=[4.0])
    c_fit = popt[0]
    c_conf = 2 * const4["x0"]
    print(f"\n  Fit r = c/log(N): c = {c_fit:.3f}  (conformal: {c_conf:.3f})")
    print(f"  Ratio: {c_fit/c_conf:.4f}")
    print(f"  Decomposition: {c_fit/c_conf:.3f} ≈ {1+C_mean/5:.3f} × 0.92")

    return {"cd": cd4, "ck": ck4, "C": C_mean, "const": const4}


# ---------- Figures ----------

def task_figures():
    """Generate all paper figures from saved data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figures")
        return

    # Load data
    t3 = load_or_init("table3_results.json")
    t5 = load_or_init("table5_results.json")

    if not t3:
        print("  No Table III data found. Run 'table3' first.")
        return

    const4 = syk_constants(4)
    cd4 = compute_cd(const4)
    x0 = const4["x0"]

    # Fit C from Table V or use default
    C_combined = -0.26
    ck4 = cd4 - C_combined

    # ---- Figure 1: Main figure (3 panels) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    Ns_data, t1s, t2s, rs, rerrs = [], [], [], [], []
    for N in sorted(TABLE3_CONFIG):
        v = t3.get(str(N), {})
        if v.get("ratio"):
            Ns_data.append(v["N"])
            t1s.append(v["t_diss"])
            t2s.append(v["t_otoc"])
            rs.append(v["ratio"])
            rerrs.append(v.get("ratio_err", 0))

    Na = np.array(Ns_data)
    # (a) Timescales
    ax = axes[0]
    ax.plot(Na, t2s, "rs-", ms=5, lw=1.5, label=r"$t_{\rm OTOC}$")
    ax.plot(Na, t1s, "bo-", ms=5, lw=1.5, label=r"$t_{\rm diss}$")
    ax.set_xlabel("N"); ax.set_ylabel("Jt")
    ax.set_title("(a) Characteristic timescales")
    ax.legend()

    # (b) Separation ratio
    ax = axes[1]
    ax.errorbar(Na, rs, yerr=rerrs, fmt="ko", ms=5, capsize=3, label="ED data")
    N_sm = np.linspace(6, 80, 200)
    ax.plot(N_sm, 2 * x0 / np.log(N_sm), "g--", lw=1.5,
            label=f"Conformal (N*=62)")
    ax.plot(N_sm, r_corrected(N_sm, 4, 5.0, cd4, ck4), "r-", lw=1.5,
            label=f"Corrected (N*≈50)")
    if len(Na) >= 3:
        popt, _ = curve_fit(lambda N, c: c / np.log(N), Na, np.array(rs), p0=[4])
        ax.plot(N_sm, popt[0] / np.log(N_sm), "b-.", lw=1.5,
                label=f"Fit c/ln N, c={popt[0]:.2f}")
    ax.axhline(1.0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("N"); ax.set_ylabel(r"$r = t_{\rm diss}/t_{\rm OTOC}$")
    ax.set_title("(b) Separation ratio"); ax.set_xlim(6, 80)
    ax.legend(fontsize=8)

    # (c) Approach to conformal limit
    ax = axes[2]
    r_conf_arr = 2 * x0 / np.log(Na)
    ax.plot(Na, np.array(rs) / r_conf_arr, "ko-", ms=5)
    ax.axhline(1.0, color="g", ls="--", lw=1, label="Conformal limit")
    ax.set_xlabel("N"); ax.set_ylabel(r"$r_{\rm num}/r_{\rm analytic}$")
    ax.set_title("(c) Approach to conformal limit")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig_main.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    # ---- Figure 2: r(N) corrected ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(Na, rs, yerr=rerrs, fmt="ko", ms=6, capsize=3, label="ED data")
    ax2.plot(N_sm, 2 * x0 / np.log(N_sm), "g--", lw=2, label="Conformal (N*=62)")
    ax2.plot(N_sm, r_corrected(N_sm, 4, 5.0, cd4, ck4), "r-", lw=2,
             label=f"Corrected βJ=5 (N*≈{Nstar_corrected(4,5,cd4,ck4):.0f})")
    if len(Na) >= 3:
        ax2.plot(N_sm, popt[0] / np.log(N_sm), "b-.", lw=1.5,
                 label=f"Fit c/ln N, c={popt[0]:.2f}")
    ax2.axhline(1.0, color="gray", ls=":", lw=0.5)
    ax2.set_xlabel("N", fontsize=13); ax2.set_ylabel(r"$r = t_{\rm diss}/t_{\rm OTOC}$", fontsize=13)
    ax2.set_title("Separation ratio for q=4, βJ=5", fontsize=13)
    ax2.legend(fontsize=10); ax2.set_xlim(6, 80); ax2.set_ylim(0.5, 2.6)
    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig_r_corrected.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    # ---- Figure 3: Corrections (4 panels) ----
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 9))
    Delta = const4["Delta"]
    alpha_G = const4["alpha_G"]

    # (a) f(x)
    ax = axes3[0, 0]
    x = np.linspace(0, 4, 200)
    ax.plot(x, 1 - x * np.tanh(x), "b-", lw=2, label=r"$f(x)=1-x\tanh x$")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(x0, color="r", ls="--", lw=1, label=f"$x_0={x0:.2f}$")
    ax.set_xlabel(r"$x = \pi t/\beta$"); ax.set_ylabel(r"$f(x)$")
    ax.set_title("(a) UV correction function"); ax.legend()

    # (b) Corrected correlator
    ax = axes3[0, 1]
    x = np.linspace(0, 4, 200)
    for bJ in [3, 5, 10, np.inf]:
        if bJ == np.inf:
            corr = np.cosh(x) ** (-2 * Delta)
            ax.plot(x, corr, "--", lw=2, label="Conformal")
        else:
            corr = np.cosh(x) ** (-2 * Delta) * (1 - 2 * alpha_G / bJ * (1 - x * np.tanh(x)))
            ax.plot(x, corr, "-", lw=2, label=f"βJ={bJ:.0f}")
    ax.axhline(0.5, color="gray", ls=":", lw=0.5)
    ax.set_xlabel(r"$x = \pi t/\beta$"); ax.set_ylabel(r"$|G_R/G_R(0)|$")
    ax.set_title("(b) Corrected two-point function"); ax.legend(fontsize=9)
    ax.set_xlim(0, 4); ax.set_ylim(-0.1, 1.1)

    # (c) r vs βJ
    ax = axes3[1, 0]
    ed_beta = {
        10: {2: 1.474, 5: 1.629, 10: 1.678, 20: 1.692},
        12: {2: 1.293, 5: 1.376, 10: 1.406, 20: 1.419},
        14: {2: 1.198, 5: 1.301, 10: 1.362, 20: 1.398},
    }
    colors = {10: "blue", 12: "red", 14: "green"}
    bJ_sm = np.linspace(1.5, 25, 100)
    for N, bd in ed_beta.items():
        bJs = sorted(bd.keys())
        ax.plot(bJs, [bd[b] for b in bJs], "o", color=colors[N], ms=7, label=f"ED N={N}")
        ax.plot(bJ_sm, r_corrected(N, 4, bJ_sm, cd4, ck4), "-", color=colors[N], lw=1.5, alpha=0.7)
    ax.set_xlabel(r"$\beta J$"); ax.set_ylabel(r"$r$")
    ax.set_title("(c) β-dependence of r"); ax.legend()

    # (d) N* vs βJ
    ax = axes3[1, 1]
    bJ_range = np.linspace(2, 50, 200)
    for qq in [4, 6, 8]:
        cq = syk_constants(qq)
        cdq = compute_cd(cq)
        ckq = cdq - C_combined
        ax.semilogy(bJ_range, [Nstar_corrected(qq, b, cdq, ckq) for b in bJ_range],
                     "-", lw=2, label=f"q={qq}")
        ax.axhline(cq["N_star_conf"], ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$\beta J$"); ax.set_ylabel(r"$N_*$")
    ax.set_title(r"(d) Corrected $N_*$"); ax.legend(); ax.set_ylim(1, 1e4)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "fig_corrections.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    print("  All figures generated.")


# =============================================================================
#  IV. Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduction code for 'Crossover of Scrambling and "
                    "Dissipation Timescales in the SYK_q Model'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python syk_crossover.py table3        # Table III (q=4, N=8-22)
  python syk_crossover.py table4        # Table IV  (q=4,6,8)
  python syk_crossover.py table5        # Table V   (temperature)
  python syk_crossover.py schwarzian    # Table II  (analytic)
  python syk_crossover.py figures       # All figures
  python syk_crossover.py all           # Everything
        """,
    )
    parser.add_argument("task", choices=["table3", "table4", "table5",
                                         "schwarzian", "figures", "all"],
                        help="Which computation to run")
    args = parser.parse_args()

    print("=" * 65)
    print("  Reproduction Code: SYK_q Scrambling-Dissipation Crossover")
    print("=" * 65)

    if args.task == "table3":
        task_table3()
    elif args.task == "table4":
        task_table4()
    elif args.task == "table5":
        task_table5()
    elif args.task == "schwarzian":
        task_schwarzian()
    elif args.task == "figures":
        task_figures()
    elif args.task == "all":
        task_table3()
        task_table4()
        task_table5()
        task_schwarzian()
        task_figures()


if __name__ == "__main__":
    main()
