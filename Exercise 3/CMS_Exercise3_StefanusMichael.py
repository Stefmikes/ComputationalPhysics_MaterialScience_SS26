import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.optimize import brentq
import os

# ── Constants ────────────────────────────────────────────────────────────────
# All internal calculations use Hartree atomic units:
#   hbar = 1,  m_e = 1,  length in Bohr (a0),  energy in Hartree (Eh),
#   time internally in atomic units; displayed values always in fs
HBAR   = 1.0               # a.u.
m_H    = 1836.15267        # hydrogen mass  [m_e]
m_C    = 12.011 * 1822.888 # carbon mass    [m_e]

# Unit conversions (for display only)
AU_T = 0.024189   # conversion factor: internal time unit → fs
AU_D = 0.529177   # 1 Bohr  → Å
AU_E = 27.21138   # 1 Eh    → eV


# ── 1.  POTENTIAL ────────────────────────────────────────────────────────────
def double_well(x, Vb=0.15, d=2.0):
    """V(x) = Vb*(x²−d²)²/d⁴  [all in a.u.]"""
    return Vb * (x**2 - d**2)**2 / d**4


# ── 2.  INITIAL WAVE PACKET ──────────────────────────────────────────────────
def gaussian_wavepacket(x, x0, sigma):
    """ψ(x,0) = (1/σ√2π)^½ · exp(−(x−x0)²/4σ²)  [no plane-wave, k0=0]"""
    psi = (1.0 / (sigma * np.sqrt(2 * np.pi)))**0.5 \
          * np.exp(-(x - x0)**2 / (4 * sigma**2))
    return psi


# ── 3.  SPLIT-OPERATOR PROPAGATOR ────────────────────────────────────────────
def split_operator_step(psi, exp_V_half, exp_T):
    """One second-order Trotter step."""
    psi   = exp_V_half * psi
    psi_k = np.fft.fft(psi)
    psi_k = exp_T * psi_k
    psi   = np.fft.ifft(psi_k)
    psi   = exp_V_half * psi
    return psi


def run_quantum(x, V, psi0, dt, n_steps, mass, save_every=10):
    """
    Propagate psi0 for n_steps × dt using the split-operator method.
    Returns
    -------
    times   : (n_saved,) in fs
    psi_arr : (n_saved, N)  complex wavefunction snapshots
    """
    dx = x[1] - x[0]
    N  = len(x)
    k  = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    exp_V_half = np.exp(-1j * V * dt / (2 * HBAR))
    exp_T      = np.exp(-1j * HBAR * k**2 / (2 * mass) * dt)

    psi = psi0.astype(complex).copy()
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)  # normalise

    times, psi_arr = [], []
    for step in range(n_steps + 1):
        if step % save_every == 0:
            times.append(step * dt)
            psi_arr.append(psi.copy())
        if step < n_steps:
            psi = split_operator_step(psi, exp_V_half, exp_T)

    return np.array(times), np.array(psi_arr)


# ── 4.  CLASSICAL TRAJECTORY  (Velocity-Verlet) ──────────────────────────────
def classical_force(x, Vb=0.15, d=2.0):
    """−dV/dx for the quartic double-well."""
    return -Vb * 4 * x * (x**2 - d**2) / d**4


def run_classical(x0, v0, dt, n_steps, mass, Vb=0.15, d=2.0):
    """Velocity-Verlet integration of the classical particle."""
    xs = np.zeros(n_steps + 1)
    xs[0] = x0
    x, v = x0, v0
    F = classical_force(x, Vb, d)
    for i in range(1, n_steps + 1):
        x     = x + v * dt + 0.5 * F / mass * dt**2
        F_new = classical_force(x, Vb, d)
        v     = v + 0.5 * (F + F_new) / mass * dt
        F     = F_new
        xs[i] = x
    return xs


# ── 5.  OBSERVABLES ──────────────────────────────────────────────────────────
def expectation_x(psi_arr, x, dx):
    """⟨x⟩(t) for each snapshot [a.u.]."""
    prob = np.abs(psi_arr)**2
    return np.einsum('ti,i->t', prob, x) * dx


def well_populations(psi_arr, x, dx):
    """Fraction of probability in left (x<0) and right (x≥0) wells."""
    prob = np.abs(psi_arr)**2
    PL   = np.sum(prob[:, x < 0],  axis=1) * dx
    PR   = np.sum(prob[:, x >= 0], axis=1) * dx
    return PL, PR


def energy_expectation(psi_arr, x, V, mass, dx):
    """
    ⟨T⟩, ⟨V⟩, ⟨E⟩ for each snapshot [a.u.].

    Kinetic energy via Parseval's theorem in the FFT convention:
        psi_k = FFT(psi)   (numpy unnormalised, length N)
        ⟨T⟩   = Σ_k |psi_k|² T_k · dx / N
    Verified analytically: gives hbar²/(8·m·sigma²) for a free Gaussian.
    """
    N    = len(x)
    k    = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    T_op = HBAR**2 * k**2 / (2 * mass)

    Ek = np.zeros(len(psi_arr))
    Ep = np.zeros(len(psi_arr))
    for i, psi in enumerate(psi_arr):
        psi_k = np.fft.fft(psi)
        Ek[i] = np.real(np.sum(np.conj(psi_k) * T_op * psi_k)) * dx / N
        Ep[i] = np.real(np.sum(np.conj(psi) * V * psi)) * dx

    return Ek, Ep, Ek + Ep


# ── 6.  PLOT HELPERS ─────────────────────────────────────────────────────────
STYLE = dict(fig_bg="white", ax_bg="#f7f9fc", grid_c="#dce3ec", fontsize=11)


def apply_style(ax):
    ax.set_facecolor(STYLE["ax_bg"])
    ax.grid(True, color=STYLE["grid_c"], lw=0.7, zorder=0)
    ax.tick_params(labelsize=STYLE["fontsize"] - 1)


def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"  saved → output/{name}.png")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size":   STYLE["fontsize"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # ── Potential parameters (in a.u.) ───────────────────────────────────
    Vb    = 0.01    # barrier height  [Eh]
    d     = 2.0     # well minima at ±d  [a0]
    sigma = 0.5     # Gaussian width  [a0]
    x0    = -d      # initial packet centre (left well)

    x  = np.linspace(-6, 6, 1024)   # a0
    dx = x[1] - x[0]
    V  = double_well(x, Vb, d)

    psi0   = gaussian_wavepacket(x, x0, sigma)
    psi0  /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)

    # ── Stability check ───────────────────────────────────────────────────
    k_max   = np.pi / dx
    dt_max  = np.pi * 2 * m_H / (HBAR * k_max**2)
    Ek0_H   = HBAR**2 / (8 * m_H * sigma**2)
    Ek0_C   = HBAR**2 / (8 * m_C * sigma**2)
    Ep0     = np.sum(np.abs(psi0)**2 * V) * dx
    E_H     = Ek0_H + Ep0
    E_C     = Ek0_C + Ep0

    print("=" * 62)
    print("Exercise 3 – Wave Packet Dynamics in a Double-Well Potential")
    print("=" * 62)
    print(f"\nParameters (a.u.):")
    print(f"  Vb    = {Vb} Eh  =  {Vb*AU_E:.3f} eV")
    print(f"  d     = {d} a0  =  {d*AU_D:.3f} Å")
    print(f"  sigma = {sigma} a0  =  {sigma*AU_D:.3f} Å")
    print(f"\nEnergies:")
    print(f"  <E>_H = {E_H:.5f} Eh = {E_H*AU_E:.4f} eV  "
          f"(<E>/Vb = {E_H/Vb:.4f}) → {'SUB-BARRIER ✓' if E_H < Vb else 'ABOVE'}")
    print(f"  <E>_C = {E_C:.5f} Eh = {E_C*AU_E:.4f} eV  "
          f"(<E>/Vb = {E_C/Vb:.4f}) → {'SUB-BARRIER ✓' if E_C < Vb else 'ABOVE'}")
    print(f"\nStability: dt_max = {dt_max*AU_T*1000:.2f} as = {dt_max*AU_T:.5f} fs")

    # ── Time-step parameters ──────────────────────────────────────────────
    dt_fine  = 0.10     # internal a.u.  (< dt_max ≈ 0.161 a.u.)
    n_steps  = 500000   # total ≈ 1210 fs
    save_ev  = 500
    print(f"  dt    = {dt_fine*AU_T*1000:.2f} as = {dt_fine*AU_T:.5f} fs  "
          f"[dt/dt_max = {dt_fine/dt_max:.2f}  → stable]")
    print(f"  Total ≈ {n_steps*dt_fine*AU_T:.0f} fs")

    # ─────────────────────────────────────────────────────────────────────
    # PLOT 1: Potential + initial wave packet
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    apply_style(ax)
    ax.plot(x*AU_D, V*AU_E, color="#2c4f8c", lw=2, label="Double-well $V(x)$")
    ax.axhline(Vb*AU_E, color="gray", ls="--", lw=1, alpha=0.7,
               label=f"Barrier $V_b = {Vb*AU_E:.2f}$ eV")
    ax.axhline(E_H*AU_E, color="#e07b39", ls=":", lw=1.5,
               label=rf"$\langle E\rangle_H = {E_H*AU_E:.3f}$ eV")
    prob0 = np.abs(psi0)**2 / np.max(np.abs(psi0)**2) * Vb * AU_E * 0.55
    ax.fill_between(x*AU_D, prob0, alpha=0.35, color="#e07b39",
                    label=r"$|\psi(x,0)|^2$ (scaled)")
    ax.axvline(-d*AU_D, color="#888", ls=":", lw=1)
    ax.axvline( d*AU_D, color="#888", ls=":", lw=1)
    ax.set_xlim(x[0]*AU_D, x[-1]*AU_D)
    ax.set_ylim(-0.05, Vb*AU_E * 1.65)
    ax.set_xlabel("Position $x$ [Å]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title("Double-Well Potential and Initial Wave Packet")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "01_potential_and_initial_state")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION A: Hydrogen, sub-barrier
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[A] Hydrogen, sub-barrier  (dt={dt_fine*AU_T*1000:.2f} as, "
          f"{n_steps*dt_fine*AU_T:.0f} fs total)")
    times, psi_arr = run_quantum(x, V, psi0, dt_fine, n_steps, m_H,
                                  save_every=save_ev)

    # Convert time axis to fs for all plots
    t_fs = times * AU_T

    xexp_H     = expectation_x(psi_arr, x, dx) * AU_D   # Å
    PL_H, PR_H = well_populations(psi_arr, x, dx)
    Ek_H, Ep_H, Etot_H = energy_expectation(psi_arr, x, V, m_H, dx)

    # Classical particle (stays at x=-d, v0=0)
    x_cl_H = run_classical(x0, 0.0, dt_fine, n_steps, m_H, Vb, d) * AU_D
    t_cl   = np.arange(n_steps + 1) * dt_fine * AU_T   # fs

    drift_H = (Etot_H - Etot_H[0]) / np.abs(Etot_H[0]) * 100
    print(f"   <E>_H = {Etot_H[0]*AU_E:.4f} eV,  "
          f"max |drift| = {np.max(np.abs(drift_H)):.2e} %")

    # ── PLOT 2: Quantum ⟨x⟩ vs Classical x(t) ───────────────────────────
    zoom = 100.0   # fs
    mz   = t_fs  <= zoom
    mzc  = t_cl  <= zoom

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_fs[mz],  xexp_H[mz],  color="#2c4f8c", lw=1.8,
            label=r"Quantum $\langle x\rangle(t)$")
    ax.plot(t_cl[mzc], x_cl_H[mzc], color="#c0392b", lw=1.4, ls="--",
            label="Classical $x(t)$  (stationary at $-d$)")
    ax.axhline(-d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.6,
               label=f"Well minima $\\pm${d*AU_D:.2f}$ Å")
    ax.axhline( d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax.set_xlim(0, zoom)
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel("Position [Å]")
    ax.set_title(r"Quantum vs Classical Trajectory — H,  $\langle E\rangle < V_b$  (first 100 fs)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "02_quantum_vs_classical")

    # ── PLOT 3: Well populations ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_fs, PL_H, color="#2c4f8c", lw=1.8, label="$P_L$ (left well)")
    ax.plot(t_fs, PR_H, color="#e07b39", lw=1.8, label="$P_R$ (right well)")
    ax.set_xlim(0, 100.0)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel("Population")
    ax.set_title(r"Well Populations — H,  $\langle E\rangle < V_b$  (first 100 fs)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "03_well_populations")

    # ── PLOT 4: Energy expectation values ────────────────────────────────
    mE = t_fs <= 100.0
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_fs[mE], Ek_H[mE]*AU_E, color="#c0392b", lw=1.8,
            label=r"$\langle T\rangle$ (kinetic)")
    ax.plot(t_fs[mE], Ep_H[mE]*AU_E, color="#2c4f8c", lw=1.8,
            label=r"$\langle V\rangle$ (potential)")
    ax.plot(t_fs[mE], Etot_H[mE]*AU_E, color="black", lw=1.4, ls="--",
            label=r"$\langle E\rangle$ (total, conserved)")
    ax.axhline(Vb*AU_E, color="gray", ls=":", lw=0.9, alpha=0.7,
               label=f"Barrier $V_b = {Vb*AU_E:.2f}$ eV")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title(rf"Energy Expectation Values — H,  "
                 rf"$\langle E\rangle = {Etot_H[0]*AU_E:.3f}$ eV $< V_b = {Vb*AU_E:.2f}$ eV  (first 100 fs)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "04_energy_fine_dt")

    # ── PLOT 5: Timestep comparison ──────────────────────────────────────
    dt_coarse  = 0.15   # internal a.u.  (just inside stability limit)
    n_coarse   = int(n_steps * dt_fine / dt_coarse)
    save_coarse = max(1, n_coarse // len(times))
    print(f"[A'] Coarse timestep  (dt={dt_coarse*AU_T*1000:.2f} as = {dt_coarse*AU_T:.5f} fs)")
    times_c, psi_arr_c = run_quantum(x, V, psi0, dt_coarse, n_coarse, m_H,
                                      save_every=save_coarse)
    _, _, Etot_c = energy_expectation(psi_arr_c, x, V, m_H, dx)
    t_fs_c = times_c * AU_T

    drift_c = (Etot_c - Etot_c[0]) / np.abs(Etot_c[0]) * 100
    print(f"   Max |drift| fine   = {np.max(np.abs(drift_H)):.3e} %")
    print(f"   Max |drift| coarse = {np.max(np.abs(drift_c)):.3e} %")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    apply_style(axes[0]); apply_style(axes[1])
    axes[0].plot(t_fs,   drift_H, color="#2c4f8c", lw=1.4)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Time [fs]"); axes[0].set_ylabel("Relative energy drift [%]")
    axes[0].set_title(rf"Fine:  $\Delta t = {dt_fine*AU_T*1000:.2f}$ as")
    axes[1].plot(t_fs_c, drift_c, color="#c0392b", lw=1.4)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_title(rf"Coarser:  $\Delta t = {dt_coarse*AU_T*1000:.2f}$ as")
    fig.suptitle("Total Energy Drift — Fine vs Coarser Timestep")
    fig.tight_layout()
    save(fig, "05_timestep_comparison")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION B: Above-barrier  (plane-wave kick)
    # ─────────────────────────────────────────────────────────────────────
    k0_hi   = np.sqrt(2 * m_H * 2 * Vb) / HBAR
    v0_cl   = HBAR * k0_hi / m_H
    psi0_hi = gaussian_wavepacket(x, x0, sigma) * np.exp(1j * k0_hi * x)
    psi0_hi /= np.sqrt(np.sum(np.abs(psi0_hi)**2) * dx)

    Ep0_hi  = np.sum(np.abs(psi0_hi)**2 * V) * dx
    Ek0_hi  = HBAR**2 * k0_hi**2 / (2 * m_H)
    E_hi    = Ek0_hi + Ep0_hi
    print(f"\n[B] Above-barrier:  <E> = {E_hi*AU_E:.3f} eV  "
          f"({E_hi/Vb:.2f} Vb)  > Vb = {Vb*AU_E:.3f} eV")

    n_hi    = 100000; save_hi = 100
    times_hi, psi_arr_hi = run_quantum(x, V, psi0_hi, dt_fine, n_hi, m_H,
                                        save_every=save_hi)
    xexp_hi = expectation_x(psi_arr_hi, x, dx) * AU_D
    t_fs_hi = times_hi * AU_T

    x_tp_au = brentq(lambda xv: double_well(xv, Vb, d) - E_hi, d, 6.0)
    x_tp    = x_tp_au * AU_D

    # Fine classical run for left panel (early time)
    n_cl2   = int(50.0 / AU_T / dt_fine)
    x_cl2   = run_classical(x0, v0_cl, dt_fine, n_cl2, m_H, Vb, d) * AU_D
    t_cl2   = np.arange(n_cl2 + 1) * dt_fine * AU_T

    COL_Q = "#2c4f8c"; COL_C = "#c0392b"; YLIM = x_tp * 1.2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    apply_style(axes[0]); apply_style(axes[1])

    ax = axes[0]
    mask_l = t_fs_hi <= 50.0
    ax.plot(t_fs_hi[mask_l], xexp_hi[mask_l], color=COL_Q, lw=2.0, zorder=3,
            label=r"Quantum $\langle x\rangle$")
    ax.plot(t_cl2[t_cl2 <= 50.0], x_cl2[t_cl2 <= 50.0],
            color=COL_C, lw=1.6, ls="--", label="Classical $x(t)$")
    ax.axhline(-d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.55)
    ax.axhline( d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.55)
    ax.set_xlim(0, 50); ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Position [Å]")
    ax.set_title("Early time — first 50 fs", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    ax.fill_between([0, t_fs_hi[-1]], -x_tp, x_tp, color=COL_C, alpha=0.13,
                    label=f"Classical envelope $\\pm{x_tp:.2f}$ Å")
    ax.axhline( x_tp, color=COL_C, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(-x_tp, color=COL_C, lw=1.2, ls="--", alpha=0.6)
    ax.plot(t_fs_hi, xexp_hi, color=COL_Q, lw=2.0, zorder=3,
            label=r"Quantum $\langle x\rangle$  (decoheres $\to 0$)")
    ax.set_xlim(0, t_fs_hi[-1]); ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Position [Å]")
    ax.set_title(f"Full {t_fs_hi[-1]:.0f} fs — decoherence", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        rf"Above-Barrier Regime:  $\langle E\rangle = {E_hi*AU_E:.2f}$ eV"
        rf"  $> V_b = {Vb*AU_E:.2f}$ eV  — Quantum vs Classical (H)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "06_high_energy_regime")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION C: Carbon mass comparison
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C] Carbon mass comparison")
    times_C, psi_arr_C = run_quantum(x, V, psi0, dt_fine, n_steps, m_C,
                                      save_every=save_ev)
    t_fs_C      = times_C * AU_T
    xexp_C      = expectation_x(psi_arr_C, x, dx) * AU_D
    PL_C, PR_C  = well_populations(psi_arr_C, x, dx)
    _, _, Etot_C = energy_expectation(psi_arr_C, x, V, m_C, dx)
    print(f"   <E>_C = {Etot_C[0]*AU_E:.4f} eV")

    smooth_win = max(1, int(50.0 / (dt_fine * AU_T * save_ev)))  # ~50 fs smoothing
    smH_x  = uniform_filter1d(xexp_H, size=smooth_win)
    smC_x  = uniform_filter1d(xexp_C, size=smooth_win)
    smH_PL = uniform_filter1d(PL_H, size=smooth_win)
    smH_PR = uniform_filter1d(PR_H, size=smooth_win)
    smC_PL = uniform_filter1d(PL_C, size=smooth_win)
    smC_PR = uniform_filter1d(PR_C, size=smooth_win)

    early_fs = 100.0; mE3 = t_fs <= early_fs
    C1 = "#2c4f8c"; C1r = "#7fa3d1"; C2 = "#c0392b"; C2r = "#e08070"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.30})

    ax = axes[0, 0]; apply_style(ax)
    ax.plot(t_fs[mE3], xexp_H[mE3], color=C1,  lw=1.6, label="Hydrogen")
    ax.plot(t_fs_C[mE3], xexp_C[mE3], color=C2, lw=1.6, ls="--", label="Carbon")
    ax.axhline(-d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.axhline( d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, early_fs); ax.set_ylim(-d*AU_D*1.3, d*AU_D*1.3)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel(r"$\langle x\rangle$ [Å]")
    ax.set_title(rf"$\langle x\rangle(t)$ — first {early_fs:.0f} fs (raw)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]; apply_style(ax)
    ax.plot(t_fs[mE3],   PL_H[mE3], color=C1,  lw=1.6, label=r"H — $P_L$")
    ax.plot(t_fs[mE3],   PR_H[mE3], color=C1r, lw=1.6, label=r"H — $P_R$")
    ax.plot(t_fs_C[mE3], PL_C[mE3], color=C2,  lw=1.6, ls="--", label=r"C — $P_L$")
    ax.plot(t_fs_C[mE3], PR_C[mE3], color=C2r, lw=1.6, ls="--", label=r"C — $P_R$")
    ax.set_xlim(0, early_fs); ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Population")
    ax.set_title(f"Well populations — first {early_fs:.0f} fs (raw)")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]; apply_style(ax)
    ax.plot(t_fs,   smH_x, color=C1,  lw=1.8, label="Hydrogen")
    ax.plot(t_fs_C, smC_x, color=C2,  lw=1.8, ls="--", label="Carbon")
    ax.axhline(-d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.axhline( d*AU_D, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, t_fs[-1]); ax.set_ylim(-d*AU_D*1.3, d*AU_D*1.3)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel(r"$\langle x\rangle$ [Å]")
    ax.set_title(rf"$\langle x\rangle(t)$ — full {t_fs[-1]:.0f} fs (smoothed)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]; apply_style(ax)
    ax.plot(t_fs,   smH_PL, color=C1,  lw=1.8, label=r"H — $P_L$")
    ax.plot(t_fs,   smH_PR, color=C1r, lw=1.8, label=r"H — $P_R$")
    ax.plot(t_fs_C, smC_PL, color=C2,  lw=1.8, ls="--", label=r"C — $P_L$")
    ax.plot(t_fs_C, smC_PR, color=C2r, lw=1.8, ls="--", label=r"C — $P_R$")
    ax.set_xlim(0, t_fs[-1]); ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Population")
    ax.set_title(f"Well populations — full {t_fs[-1]:.0f} fs (smoothed)")
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "Mass Comparison — Hydrogen vs Carbon\n"
        r"(same $\sigma$, $d$, $V_b$; sub-barrier for both; "
        "top: first 100 fs · bottom: full simulation smoothed)",
        y=1.01, fontsize=12)
    fig.tight_layout()
    save(fig, "07_mass_comparison")

    # ─────────────────────────────────────────────────────────────────────
    # PLOT 8: Wavefunction snapshots
    # ─────────────────────────────────────────────────────────────────────
    snap_fs  = [0, 10, 25, 50, 75, 100]
    snap_idx = [np.argmin(np.abs(t_fs - s)) for s in snap_fs]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.flatten()
    for idx, (si, ax) in enumerate(zip(snap_idx, axes)):
        apply_style(ax)
        prob = gaussian_filter1d(np.abs(psi_arr[si])**2, sigma=2)
        pmax = np.max(prob)
        ax.fill_between(x*AU_D, prob, alpha=0.5, color="#2c4f8c")
        ax.plot(x*AU_D, prob, color="#2c4f8c", lw=1.2)
        Vsc = np.clip(V/Vb * pmax * 0.55, 0, pmax)
        ax.plot(x*AU_D, Vsc, color="gray", lw=1, ls="--", alpha=0.7,
                label="$V(x)$ (scaled)" if idx == 0 else "")
        ax.set_xlim(x[0]*AU_D, x[-1]*AU_D)
        ax.set_ylim(0, pmax * 1.4)
        ax.set_title(f"$t = {t_fs[si]:.0f}$ fs")
        ax.set_xlabel("$x$ [Å]", fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel(r"$|\psi|^2$", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Probability Density Snapshots — H (sub-barrier, first 100 fs)", y=1.01)
    fig.tight_layout()
    save(fig, "08_wavefunction_snapshots")

    # ─────────────────────────────────────────────────────────────────────
    # PARAMETER EXPLORATION: vary d and Vb (2×2 grid)
    # ─────────────────────────────────────────────────────────────────────
    print("\n[NEW] 2×2 parameter exploration — varying d and Vb")

    d_vals  = [1.5, 3.0]    # a0
    Vb_vals = [0.03, 0.15]  # Eh  — low barrier (≈0.82 eV) vs standard (≈4.08 eV)

    n_exp = 200000; save_exp = 200
    results_exp = {}

    for d_e in d_vals:
        for Vb_e in Vb_vals:
            V_e  = double_well(x, Vb_e, d_e)
            Ep_e = np.sum(np.abs(psi0)**2 * V_e) * dx
            E_e  = Ek0_H + Ep_e
            regime = "sub" if E_e < Vb_e else "above"
            print(f"  d={d_e*AU_D:.2f}Å  Vb={Vb_e*AU_E:.2f}eV  "
                  f"<E>/Vb={E_e/Vb_e:.3f}  ({regime}-barrier)")
            t_e, snaps_e = run_quantum(x, V_e, psi0, dt_fine, n_exp,
                                        m_H, save_every=save_exp)
            PL_e, PR_e = well_populations(snaps_e, x, dx)
            results_exp[f"{d_e}_{Vb_e}"] = dict(
                t=t_e*AU_T, PL=PL_e, PR=PR_e, E=E_e, d=d_e, Vb=Vb_e, V=V_e,
                regime=regime)

    # ── PLOT 9: 2×2 populations panel ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5),
                              gridspec_kw={"hspace": 0.50, "wspace": 0.30})
    for i, d_e in enumerate(d_vals):
        for j, Vb_e in enumerate(Vb_vals):
            r  = results_exp[f"{d_e}_{Vb_e}"]
            ax = axes[i, j]; apply_style(ax)
            ax.plot(r["t"], r["PL"], color="#2c4f8c", lw=1.8, label="$P_L$")
            ax.plot(r["t"], r["PR"], color="#e07b39", lw=1.8, label="$P_R$")
            ax.axhline(0.5, color="gray", ls="--", lw=0.9, alpha=0.6)
            ax.set_xlim(0, r["t"][-1]); ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Time [fs]"); ax.set_ylabel("Population")
            ax.set_title(
                f"$d={d_e*AU_D:.2f}$ Å,  $V_b={Vb_e*AU_E:.2f}$ eV\n"
                f"$\\langle E\\rangle/V_b={r['E']/Vb_e:.3f}$ ({r['regime']}-barrier)",
                fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=9, loc="center right")
    fig.suptitle(
        "Parameter Exploration — Spreading Rate vs Barrier Geometry (H)\n"
        r"Rows: well separation $d$;  Columns: barrier height $V_b$",
        y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "09_parameter_exploration_populations")

    # ── PLOT 10: Potential shapes ─────────────────────────────────────────
    xp = (x >= -5) & (x <= 5)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7),
                              gridspec_kw={"hspace": 0.50, "wspace": 0.30})
    for i, d_e in enumerate(d_vals):
        for j, Vb_e in enumerate(Vb_vals):
            r  = results_exp[f"{d_e}_{Vb_e}"]
            ax = axes[i, j]; apply_style(ax)
            Vp  = r["V"][xp] * AU_E; xpa = x[xp] * AU_D
            prob_n = np.abs(psi0[xp])**2 / np.max(np.abs(psi0[xp])**2) \
                     * Vb_e * AU_E * 0.42
            ax.plot(xpa, Vp, color="#2c4f8c", lw=2, label="$V(x)$")
            ax.axhline(Vb_e*AU_E, color="gray", ls="--", lw=1, alpha=0.7,
                       label=f"$V_b={Vb_e*AU_E:.2f}$ eV")
            ax.axhline(r["E"]*AU_E, color="#e07b39", ls=":", lw=1.5,
                       label=rf"$\langle E\rangle={r['E']*AU_E:.3f}$ eV")
            ax.fill_between(xpa, prob_n, alpha=0.30, color="#e07b39",
                            label=r"$|\psi_0|^2$ (scaled)")
            ax.axvline(-d_e*AU_D, color="#aaa", ls=":", lw=0.9)
            ax.axvline( d_e*AU_D, color="#aaa", ls=":", lw=0.9)
            ax.set_xlim(-5*AU_D, 5*AU_D)
            ax.set_ylim(-0.1, Vb_e*AU_E*1.8)
            ax.set_xlabel("$x$ [Å]"); ax.set_ylabel("Energy [eV]")
            ax.set_title(f"$d={d_e*AU_D:.2f}$ Å,  $V_b={Vb_e*AU_E:.2f}$ eV",
                         fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Potential Shapes — 2×2 Parameter Combinations", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "10_parameter_exploration_potentials")

# ─────────────────────────────────────────────────────────────────────
    # ANIMATION — adjustable barrier, correct W-shape, + classical overlay
    # ─────────────────────────────────────────────────────────────────────
    print("\n[Anim] Generating animation (0–100 fs) …")

    # ── Choose regime: 'sub' tunneling only, 'above' classical crosses too ──
    anim_regime = 'sub'   # change to 'above' to see classical crossing

    if anim_regime == 'sub':
        # Sub-barrier: k0=0, packet sits in left well, tunnels quantum-only
        Vb_a   = 0.01           # Eh  → ≈ 0.27 eV  (low enough to see tunneling)
        k0_a   = 0.0            # no momentum kick
        label_regime = "sub-barrier (tunneling only)"
    else:
        # Above-barrier: kick gives enough KE to classically cross
        Vb_a   = 0.15           # Eh  → ≈ 4.08 eV
        k0_a   = np.sqrt(2 * m_H * 2 * Vb_a) / HBAR   # 2×Vb KE
        label_regime = "above-barrier (classical + quantum cross)"

    d_a    = 2.0                # well separation [a0]  — adjust freely
    V_a    = double_well(x, Vb_a, d_a)

    psi0_a = gaussian_wavepacket(x, x0, sigma)
    if k0_a != 0:
        psi0_a = psi0_a * np.exp(1j * k0_a * x)
    psi0_a /= np.sqrt(np.sum(np.abs(psi0_a)**2) * dx)

    # Quantum propagation
    n_a      = int(100.0 / AU_T / dt_fine)
    save_a   = max(1, n_a // 300)
    times_a, psi_arr_a = run_quantum(x, V_a, psi0_a, dt_fine, n_a, m_H,
                                      save_every=save_a)
    t_fs_a  = times_a * AU_T
    xexp_a  = expectation_x(psi_arr_a, x, dx) * AU_D

    # Classical trajectory
    v0_cl_a = HBAR * k0_a / m_H   # 0 if sub-barrier
    x_cl_a  = run_classical(x0, v0_cl_a, dt_fine, n_a, m_H, Vb_a, d_a) * AU_D
    t_cl_a  = np.arange(n_a + 1) * dt_fine * AU_T

    # Subsample classical to match quantum frames
    cl_idx  = np.round(np.linspace(0, n_a, len(t_fs_a))).astype(int)
    xcl_s   = x_cl_a[cl_idx]

    # ── Potential: proper W-shape scaled to fit axes ──────────────────────
    # Scale so barrier peak sits at 0.45 (leaves room for probability density)
    V_a_shifted = V_a - V_a.min()                        # min → 0
    Vpl_a       = V_a_shifted / V_a_shifted.max() * 0.45 # barrier peak → 0.45

    # Energy level line (scaled the same way)
    Ep0_a   = np.sum(np.abs(psi0_a)**2 * V_a) * dx
    Ek0_a   = HBAR**2 * k0_a**2 / (2 * m_H) + HBAR**2 / (8 * m_H * sigma**2)
    E_a     = Ek0_a + Ep0_a
    E_a_sc  = (E_a - V_a.min()) / V_a_shifted.max() * 0.45   # same scaling

    # Well minima positions for vertical guides
    xmin_L  = -d_a * AU_D
    xmin_R  =  d_a * AU_D

    # ── Build figure ──────────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(8, 4.5))
    apply_style(ax_a)

    # Static elements
    ax_a.plot(x * AU_D, Vpl_a, color="#555", lw=1.5, ls="--", alpha=0.8,
              label=f"$V(x)$ scaled  ($V_b={Vb_a*AU_E:.2f}$ eV)")
    ax_a.axhline(E_a_sc, color="#27ae60", lw=1.0, ls=":", alpha=0.8,
                 label=rf"$\langle E\rangle = {E_a*AU_E:.3f}$ eV "
                       f"({'< $V_b$' if E_a < Vb_a else '> $V_b$'})")
    ax_a.axvline(xmin_L, color="#aaa", lw=0.8, ls=":", alpha=0.6)
    ax_a.axvline(xmin_R, color="#aaa", lw=0.8, ls=":", alpha=0.6)
    ax_a.axvline(0,      color="#aaa", lw=0.8, ls=":", alpha=0.4)   # barrier top

    # Dynamic: probability density
    line_q, = ax_a.plot(x * AU_D, np.zeros_like(x),
                         color="#2c4f8c", lw=1.6, zorder=4)

    # Dynamic: quantum <x> marker
    xmark_q, = ax_a.plot([], [], "v", color="#e07b39", ms=9, zorder=5,
                          label=r"Quantum $\langle x\rangle$")

    # Dynamic: classical x(t) marker
    xmark_c, = ax_a.plot([], [], "^", color="#c0392b", ms=9, zorder=5,
                          label="Classical $x(t)$")

    ttext = ax_a.text(0.02, 0.95, "", transform=ax_a.transAxes,
                      fontsize=10, va="top")

    ax_a.set_xlim(x[0] * AU_D, x[-1] * AU_D)
    ax_a.set_ylim(0, 0.70)
    ax_a.set_xlabel("Position $x$ [Å]")
    ax_a.set_ylabel(r"$|\psi(x,t)|^2$  /  $V$ (scaled)")
    ax_a.set_title(f"Wave Packet Dynamics — H  ({label_regime},  0–100 fs)")
    ax_a.legend(loc="upper right", fontsize=8.5, ncol=2)

    # ── Animate ───────────────────────────────────────────────────────────

    def update_a(fr):
        prob = np.abs(psi_arr_a[fr])**2

        line_q.set_ydata(prob)

        # Remove previous fill (keep only the static potential line collection)
        while len(ax_a.collections) > 0:
            ax_a.collections[0].remove()

        ax_a.fill_between(x * AU_D, prob, alpha=0.38, color="#2c4f8c", zorder=3)

        xmark_q.set_data([xexp_a[fr]], [0.018])
        xmark_c.set_data([xcl_s[fr]],  [0.008])
        ttext.set_text(f"$t$ = {t_fs_a[fr]:.1f} fs")
        return line_q, xmark_q, xmark_c, ttext

    anim_a = FuncAnimation(fig_a, update_a, frames=len(t_fs_a),
                            interval=40, blit=False)
    anim_a.save("output/animation.gif", writer="pillow", fps=25, dpi=110)
    plt.close(fig_a)
    print("  saved → output/animation.gif")
    print(f"  Regime: {label_regime}")
    print(f"  Vb = {Vb_a*AU_E:.3f} eV,  <E> = {E_a*AU_E:.3f} eV,  "
          f"<E>/Vb = {E_a/Vb_a:.3f}")