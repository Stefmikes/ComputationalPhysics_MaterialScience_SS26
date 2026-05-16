import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq

# ── Physical constants ──────────────────────────────────────────────────────
HBAR   = 0.6582119569   # eV·fs
AMU_eV = 0.00964853     # 1 amu in eV·fs²/Å²  (= amu_to_kg * Å²/eV·fs²)
m_H    = 1.00794 * AMU_eV   # hydrogen mass [eV·fs²/Å²]
m_C    = 12.011  * AMU_eV   # carbon mass


# 1.  POTENTIAL
def double_well(x, Vb=0.15, d=2.0):
    """
    Symmetric quartic double-well  V(x) = Vb * (x²−d²)² / d⁴
    Minima at ±d,  barrier height Vb at x=0.
    """
    return Vb * (x**2 - d**2)**2 / d**4


# 2.  INITIAL WAVE PACKET (Gaussian, no plane-wave contribution)
def gaussian_wavepacket(x, x0, sigma):
    """
    ψ(x,0) = (1/σ√2π)^(1/2) · exp(−(x−x0)²/4σ²)
    Normalisation ensured analytically; we renormalise numerically for safety.
    """
    psi = (1.0 / (sigma * np.sqrt(2 * np.pi)))**0.5 \
          * np.exp(-(x - x0)**2 / (4 * sigma**2))
    return psi


# 3.  SPLIT-OPERATOR PROPAGATOR
def split_operator_step(psi, exp_V_half, exp_T, dx):
    """
    One time step via second-order Trotter splitting:
        ψ(t+dt) = exp(−i V dt/2ħ) · IFFT[ exp(−i T dt/ħ) · FFT[exp(−i V dt/2ħ)ψ] ]
    exp_V_half and exp_T are pre-computed phase factors.
    """
    psi = exp_V_half * psi                  # half-step in position space
    psi_k = np.fft.fft(psi)                # → momentum space
    psi_k = exp_T * psi_k                   # full step in momentum space
    psi = np.fft.ifft(psi_k)               # → position space
    psi = exp_V_half * psi                  # half-step in position space
    return psi


def run_quantum(x, V, psi0, dt, n_steps, mass, save_every=10):
    """
    Propagate psi0 for n_steps × dt fs using the split-operator method.
    Returns
    -------
    times   : (n_saved,)
    psi_arr : (n_saved, N)  complex wavefunction snapshots
    """
    dx = x[1] - x[0]
    N  = len(x)

    # Momentum grid (FFT convention)
    k  = 2 * np.pi * np.fft.fftfreq(N, d=dx)

    # Pre-compute propagators
    exp_V_half = np.exp(-1j * V * dt / (2 * HBAR))
    exp_T      = np.exp(-1j * HBAR * k**2 / (2 * mass) * dt)

    psi = psi0.astype(complex).copy()
    # Renormalise
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    times   = []
    psi_arr = []

    for step in range(n_steps + 1):
        if step % save_every == 0:
            times.append(step * dt)
            psi_arr.append(psi.copy())
        if step < n_steps:
            psi = split_operator_step(psi, exp_V_half, exp_T, dx)

    return np.array(times), np.array(psi_arr)



# 4.  CLASSICAL TRAJECTORY  (Velocity-Verlet)
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
        x  = x + v * dt + 0.5 * F / mass * dt**2
        F_new = classical_force(x, Vb, d)
        v  = v + 0.5 * (F + F_new) / mass * dt
        F  = F_new
        xs[i] = x
    return xs


# 5.  OBSERVABLES
def expectation_x(psi_arr, x, dx):
    """⟨x⟩(t) for each snapshot."""
    prob = np.abs(psi_arr)**2
    return np.einsum('ti,i->t', prob, x) * dx

def well_populations(psi_arr, x, dx):
    """Fraction of probability in the left (x<0) and right (x>0) wells."""
    prob = np.abs(psi_arr)**2
    PL = np.sum(prob[:, x < 0], axis=1) * dx
    PR = np.sum(prob[:, x >= 0], axis=1) * dx
    return PL, PR

def energy_expectation(psi_arr, x, V, mass, dx):
    """
    Returns arrays ⟨T⟩, ⟨V⟩, ⟨E⟩ for each snapshot.
    Kinetic energy via finite differences in momentum space.
    """
    N = len(x)
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    T_op = HBAR**2 * k**2 / (2 * mass)   # kinetic energy operator in k-space

    Ek = np.zeros(len(psi_arr))
    Ep = np.zeros(len(psi_arr))

    for i, psi in enumerate(psi_arr):
        psi_k = np.fft.fft(psi) * dx / np.sqrt(2 * np.pi)
        Ek[i] = np.real(np.sum(np.conj(psi_k) * T_op * psi_k)) * (2 * np.pi / (N * dx))
        Ep[i] = np.real(np.sum(np.conj(psi) * V * psi)) * dx

    return Ek, Ep, Ek + Ep



# 6.  PLOT HELPERS
STYLE = dict(
    fig_bg   = "white",
    ax_bg    = "#f7f9fc",
    grid_c   = "#dce3ec",
    fontsize = 11,
)

def apply_style(ax):
    ax.set_facecolor(STYLE["ax_bg"])
    ax.grid(True, color=STYLE["grid_c"], lw=0.7, zorder=0)
    ax.tick_params(labelsize=STYLE["fontsize"] - 1)

def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"  saved → output/{name}.png")

if __name__ == "__main__":

    import os
    os.makedirs("output", exist_ok=True)

    plt.rcParams.update({
        "font.family"  : "serif",
        "font.size"    : STYLE["fontsize"],
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })

    # ── Grid & potential parameters ─────────────────────────────────────
    Vb    = 0.15    # barrier height [eV]
    d     = 2.0     # well minima at ±d [Å]
    sigma = 0.5     # Gaussian width [Å]
    x0    = -d      # initial packet centre (left well)

    x  = np.linspace(-6, 6, 1024)
    dx = x[1] - x[0]
    V  = double_well(x, Vb, d)

    # Mean energy of initial packet: purely potential (zero momentum)
    psi0 = gaussian_wavepacket(x, x0, sigma)
    E_mean_low = np.sum(np.abs(psi0)**2 * V) * dx   # ≈ 0 (at minimum)

    print("=" * 60)
    print("Exercise 3 – Double-Well Wave Packet Dynamics")
    print("=" * 60)
    print(f"  Barrier height Vb = {Vb:.3f} eV,  minima at ±{d} Å")
    print(f"  Gaussian σ = {sigma} Å,  centre x0 = {x0} Å")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION A: Hydrogen, E < Vb  (tunnelling regime)
    # ─────────────────────────────────────────────────────────────────────
    dt_fine  = 0.5    # fs
    n_steps  = 60000  # 30 000 fs = 30 ps
    save_ev  = 50

    print(f"\n[A] Hydrogen, E < Vb  (dt={dt_fine} fs, {n_steps*dt_fine/1000:.0f} ps)")
    times, psi_arr = run_quantum(x, V, psi0, dt_fine, n_steps, m_H, save_every=save_ev)
    x_cl_H = run_classical(x0, 0.0, dt_fine, n_steps, m_H)
    t_cl   = np.arange(n_steps + 1) * dt_fine / 1000   # ps

    xexp_H    = expectation_x(psi_arr, x, dx)
    PL_H, PR_H = well_populations(psi_arr, x, dx)
    Ek_H, Ep_H, Etot_H = energy_expectation(psi_arr, x, V, m_H, dx)
    t_ps = times / 1000  # fs → ps

    # ── PLOT 1: Potential + initial state ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    apply_style(ax)
    ax.plot(x, V, color="#2c4f8c", lw=2, label="Double-well $V(x)$")
    ax.axhline(Vb, color="gray", ls="--", lw=1, alpha=0.7, label=f"Barrier $V_b={Vb}$ eV")
    prob0 = np.abs(psi0)**2 / np.max(np.abs(psi0)**2) * Vb * 0.7
    ax.fill_between(x, prob0, alpha=0.35, color="#e07b39", label=r"$|\psi(x,0)|^2$ (scaled)")
    ax.axvline(-d, color="#888", ls=":", lw=1)
    ax.axvline( d, color="#888", ls=":", lw=1)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.02, Vb * 1.6)
    ax.set_xlabel("Position $x$ [Å]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title("Double-Well Potential and Initial Wave Packet")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "01_potential_and_initial_state")

    # ── PLOT 2: Quantum ⟨x⟩ vs Classical x(t) — zoom to 1 ps ───────
    zoom_traj = 1.0
    mask_z    = t_ps <= zoom_traj
    mask_zc   = t_cl <= zoom_traj

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_ps[mask_z],  xexp_H[mask_z],  color="#2c4f8c", lw=1.8,
            label=r"Quantum $\langle x\rangle(t)$")
    ax.plot(t_cl[mask_zc], x_cl_H[mask_zc], color="#c0392b", lw=1.4, ls="--",
            label="Classical $x(t)$")
    ax.axhline(-d, color="gray", ls=":", lw=0.9, alpha=0.6, label=f"Well minima $\\pm${d} Å")
    ax.axhline( d, color="gray", ls=":", lw=0.9, alpha=0.6)
    ax.set_xlim(0, zoom_traj)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Position [Å]")
    ax.set_title("Quantum vs Classical Trajectory — Hydrogen, $\\langle E\\rangle < V_b$  (first 1 ps)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "02_quantum_vs_classical")

    # ── PLOT 3: Well populations — zoom to 1 ps ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_ps[mask_z], PL_H[mask_z], color="#2c4f8c", lw=1.8, label="$P_L$ (left well)")
    ax.plot(t_ps[mask_z], PR_H[mask_z], color="#e07b39", lw=1.8, label="$P_R$ (right well)")
    ax.set_xlim(0, zoom_traj)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Population")
    ax.set_title("Well Populations — Hydrogen, $\\langle E\\rangle < V_b$  (first 1 ps)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "03_well_populations")

    # ── PLOT 4: Energy expectation values — zoom to 1 ps ─────────────
    zoom_E = 1.0   # ps
    mask_E = t_ps <= zoom_E

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_style(ax)
    ax.plot(t_ps[mask_E], Ek_H[mask_E],   color="#c0392b", lw=1.8,
            label=r"$\langle T\rangle$ (kinetic)")
    ax.plot(t_ps[mask_E], Ep_H[mask_E],   color="#2c4f8c", lw=1.8,
            label=r"$\langle V\rangle$ (potential)")
    ax.plot(t_ps[mask_E], Etot_H[mask_E], color="black",   lw=1.4, ls="--",
            label=r"$\langle E\rangle$ (total, conserved)")
    ax.set_xlim(0, zoom_E)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Energy [eV]")
    ax.set_title("Energy Expectation Values — Hydrogen, $\\Delta t = 0.5$ fs  (first 1 ps)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "04_energy_fine_dt")

    # ── PLOT 5: Timestep comparison ───────────────────────────────────
    dt_coarse = 5.0   # fs
    n_coarse  = 6000
    print(f"[A'] Coarse timestep comparison  (dt={dt_coarse} fs)")
    times_c, psi_arr_c = run_quantum(x, V, psi0, dt_coarse, n_coarse, m_H, save_every=1)
    Ek_c, Ep_c, Etot_c = energy_expectation(psi_arr_c, x, V, m_H, dx)
    t_ps_c  = times_c / 1000

    drift_fine   = (Etot_H - Etot_H[0])   / np.abs(Etot_H[0])   * 100
    drift_coarse = (Etot_c - Etot_c[0])   / np.abs(Etot_c[0])   * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    apply_style(axes[0]); apply_style(axes[1])

    axes[0].plot(t_ps,   drift_fine,   color="#2c4f8c", lw=1.4)
    axes[0].axhline(0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("Time [ps]")
    axes[0].set_ylabel("Relative energy drift [%]")
    axes[0].set_title(f"Fine:  $\\Delta t = {dt_fine}$ fs")

    axes[1].plot(t_ps_c, drift_coarse, color="#c0392b", lw=1.4)
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel("Time [ps]")
    axes[1].set_title(f"Coarse:  $\\Delta t = {dt_coarse}$ fs")

    fig.suptitle("Total Energy Drift — Fine vs Coarse Timestep")
    fig.tight_layout()
    save(fig, "05_timestep_comparison")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION B: E > Vb  (above-barrier, quantum vs classical)
    #
    # Energy choice: ⟨Ekin⟩_kick = 2 × Vb  →  k0 = sqrt(2m × 2Vb) / ħ
    # This gives ⟨E⟩ ≈ 2×Vb = 0.30 eV >> Vb = 0.15 eV (clearly above
    # barrier).
    #
    # The classical H atom at this energy oscillates ~300 times per ps —
    # far too fast to resolve at the 0.5 ps scale of the full window.
    # We therefore use a TWO-PANEL approach:
    #
    #   LEFT  (0–0.01 ps, dt_cl = 0.01 fs): fine-resolved classical
    #          trajectory showing 3–4 clean cycles alongside the early
    #          quantum motion.
    #
    #   RIGHT (0–0.5 ps): quantum ⟨x⟩ as a line (shows decoherence
    #          toward 0); classical represented by a ±amplitude shaded
    #          band — the physically honest proxy for the high-frequency
    #          oscillation that cannot be individually resolved at this
    #          timescale.
    # ─────────────────────────────────────────────────────────────────────
    k0_hi   = np.sqrt(2 * m_H * 2 * Vb) / HBAR      # gives ⟨Ekin⟩ = 2×Vb
    v0_cl   = HBAR * k0_hi / m_H                      # matching classical speed [Å/fs]
    psi0_hi = gaussian_wavepacket(x, x0, sigma) * np.exp(1j * k0_hi * x)
    psi0_hi /= np.sqrt(np.sum(np.abs(psi0_hi)**2) * dx)

    Ep0_hi    = np.sum(np.abs(psi0_hi)**2 * V) * dx
    Ek0_hi    = HBAR**2 * k0_hi**2 / (2 * m_H)
    E_mean_hi = Ek0_hi + Ep0_hi

    print(f"\n[B] Above-barrier: ⟨E⟩ ≈ {E_mean_hi:.4f} eV,  Vb = {Vb:.3f} eV  → E > Vb")

    zoom_hi    = 0.5                              # ps — full display window
    n_steps_hi = int(zoom_hi * 1000 / dt_fine)

    # Quantum: save every step for smooth ⟨x⟩
    times_hi, psi_arr_hi = run_quantum(
        x, V, psi0_hi, dt_fine, n_steps_hi, m_H, save_every=1)

    xexp_hi      = expectation_x(psi_arr_hi, x, dx)
    PL_hi, PR_hi = well_populations(psi_arr_hi, x, dx)
    t_ps_hi      = times_hi / 1000

    # ── Fine classical run for LEFT panel (dt=0.01 fs, 0–0.01 ps) ────
    dt_cl_fine = 0.01          # fs — resolves the ~0.003 ps period
    zoom_left  = 0.01          # ps
    n_cl_fine  = int(zoom_left * 1000 / dt_cl_fine)
    x_cl_fine  = run_classical(x0, v0_cl, dt_cl_fine, n_cl_fine, m_H)
    t_cl_fine  = np.arange(n_cl_fine + 1) * dt_cl_fine / 1000   # ps

    # ── Classical turning point for RIGHT panel envelope ──────────────
    # Turning point: solve V(x_tp) = E_mean_hi  (classical energy conservation)
    x_tp = brentq(lambda xv: double_well(xv, Vb, d) - E_mean_hi, d, 6.0)

    # ── PLOT 6 ────────────────────────────────────────────────────────
    COL_Q  = "#2c4f8c"   # blue  — quantum
    COL_C  = "#c0392b"   # red   — classical
    YLIM   = 5.0         # Å — shared y-axis

    mask_left = t_ps_hi <= zoom_left

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    apply_style(axes[0]); apply_style(axes[1])

    # LEFT: 0–0.01 ps — both curves fully resolved
    ax = axes[0]
    ax.plot(t_ps_hi[mask_left], xexp_hi[mask_left],
            color=COL_Q, lw=2.0, zorder=3, label=r"Quantum $\langle x\rangle$")
    ax.plot(t_cl_fine, x_cl_fine,
            color=COL_C, lw=1.6, ls="--", zorder=2, label="Classical $x(t)$")
    ax.axhline(-d, color="gray", ls=":", lw=0.9, alpha=0.55,
               label=f"Well minima $\\pm${d} Å")
    ax.axhline( d, color="gray", ls=":", lw=0.9, alpha=0.55)
    ax.axhline( 0, color="gray", ls="-", lw=0.5, alpha=0.25)
    ax.set_xlim(0, zoom_left)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("Time [ps]", fontsize=11)
    ax.set_ylabel("Position [Å]", fontsize=11)
    ax.set_title(f"Early time — first {zoom_left} ps\n"
                 r"(3–4 classical cycles, initial quantum motion)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.annotate("Classical: ~300 cycles/ps\nfully resolved here",
                xy=(zoom_left * 0.38, x_tp * 0.78), fontsize=8, color=COL_C,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL_C, alpha=0.8))

    # RIGHT: 0–0.5 ps — quantum line + classical ±envelope band
    ax = axes[1]
    t_band = np.array([0.0, zoom_hi])
    ax.fill_between(t_band, -x_tp, x_tp,
                    color=COL_C, alpha=0.13, zorder=1,
                    label=f"Classical envelope $\\pm{x_tp:.1f}$ Å\n(~300 cycles/ps, unresolved)")
    ax.axhline( x_tp, color=COL_C, lw=1.2, ls="--", alpha=0.6)
    ax.axhline(-x_tp, color=COL_C, lw=1.2, ls="--", alpha=0.6)
    ax.plot(t_ps_hi, xexp_hi,
            color=COL_Q, lw=2.0, zorder=3,
            label=r"Quantum $\langle x\rangle$ (decoheres $\to 0$)")
    ax.axhline(-d, color="gray", ls=":", lw=0.9, alpha=0.55)
    ax.axhline( d, color="gray", ls=":", lw=0.9, alpha=0.55)
    ax.axhline( 0, color="gray", ls="-", lw=0.5, alpha=0.25)
    ax.set_xlim(0, zoom_hi)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_xlabel("Time [ps]", fontsize=11)
    ax.set_ylabel("Position [Å]", fontsize=11)
    ax.set_title(f"Full window — first {zoom_hi} ps\n"
                 r"Quantum decoherence vs classical amplitude", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.annotate(r"$\langle x\rangle \to 0$: packet spreads" "\nacross both wells",
                xy=(0.22, 0.55), fontsize=8, color=COL_Q,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COL_Q, alpha=0.8))

    fig.suptitle(
        f"Above-Barrier Regime: $\\langle E\\rangle \\approx {E_mean_hi:.2f}$ eV"
        f" $> V_b = {Vb}$ eV  —  Quantum vs Classical (H atom)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "06_high_energy_regime")

    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION C: Carbon mass comparison
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[C] Carbon mass comparison")
    times_C, psi_arr_C = run_quantum(x, V, psi0, dt_fine, n_steps, m_C, save_every=save_ev)

    xexp_C      = expectation_x(psi_arr_C, x, dx)
    PL_C, PR_C  = well_populations(psi_arr_C, x, dx)

    # ── PLOT 7 (IMPROVED): 4-panel mass comparison ────────────────────
    smooth_window = max(1, int(1.0e3 / (dt_fine * save_ev)))  # ≈ 1 ps worth of snapshots

    smH_x  = uniform_filter1d(xexp_H, size=smooth_window)
    smC_x  = uniform_filter1d(xexp_C, size=smooth_window)
    smH_PL = uniform_filter1d(PL_H,   size=smooth_window)
    smH_PR = uniform_filter1d(PR_H,   size=smooth_window)
    smC_PL = uniform_filter1d(PL_C,   size=smooth_window)
    smC_PR = uniform_filter1d(PR_C,   size=smooth_window)

    early_cut = 2.0   # ps
    mask_E2   = t_ps <= early_cut

    COL_H     = "#2c4f8c"
    COL_H_R   = "#7fa3d1"
    COL_C2    = "#c0392b"
    COL_C2_R  = "#e08070"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30})

    ax = axes[0, 0]
    apply_style(ax)
    ax.plot(t_ps[mask_E2], xexp_H[mask_E2], color=COL_H, lw=1.6, label="Hydrogen")
    ax.plot(t_ps[mask_E2], xexp_C[mask_E2], color=COL_C2, lw=1.6, ls="--", label="Carbon")
    ax.axhline(-d, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.axhline( d, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, early_cut)
    ax.set_ylim(-d * 1.25, d * 1.25)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(r"$\langle x\rangle$ [Å]")
    ax.set_title(r"$\langle x\rangle(t)$ — first 2 ps (raw)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    apply_style(ax)
    ax.plot(t_ps[mask_E2], PL_H[mask_E2], color=COL_H,   lw=1.6, label=r"H — $P_L$")
    ax.plot(t_ps[mask_E2], PR_H[mask_E2], color=COL_H_R, lw=1.6, label=r"H — $P_R$")
    ax.plot(t_ps[mask_E2], PL_C[mask_E2], color=COL_C2,  lw=1.6, ls="--", label=r"C — $P_L$")
    ax.plot(t_ps[mask_E2], PR_C[mask_E2], color=COL_C2_R,lw=1.6, ls="--", label=r"C — $P_R$")
    ax.set_xlim(0, early_cut)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Population")
    ax.set_title("Well populations — first 2 ps (raw)")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    apply_style(ax)
    ax.plot(t_ps, smH_x, color=COL_H,  lw=1.8, label="Hydrogen")
    ax.plot(t_ps, smC_x, color=COL_C2, lw=1.8, ls="--", label="Carbon")
    ax.axhline(-d, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.axhline( d, color="gray", ls=":", lw=0.9, alpha=0.5)
    ax.set_xlim(0, t_ps[-1])
    ax.set_ylim(-d * 1.25, d * 1.25)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(r"$\langle x\rangle$ [Å]")
    ax.set_title(r"$\langle x\rangle(t)$ — full 30 ps (smoothed ~1 ps)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    apply_style(ax)
    ax.plot(t_ps, smH_PL, color=COL_H,   lw=1.8, label=r"H — $P_L$")
    ax.plot(t_ps, smH_PR, color=COL_H_R, lw=1.8, label=r"H — $P_R$")
    ax.plot(t_ps, smC_PL, color=COL_C2,  lw=1.8, ls="--", label=r"C — $P_L$")
    ax.plot(t_ps, smC_PR, color=COL_C2_R,lw=1.8, ls="--", label=r"C — $P_R$")
    ax.set_xlim(0, t_ps[-1])
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time [ps]")
    ax.set_ylabel("Population")
    ax.set_title("Well populations — full 30 ps (smoothed ~1 ps)")
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "Mass Comparison — Hydrogen vs Carbon\n"
        r"(same potential, $\langle E\rangle < V_b$; "
        "top: coherent regime · bottom: long-time trend)",
        y=1.01, fontsize=12
    )
    fig.tight_layout()
    save(fig, "07_mass_comparison")

    # ─────────────────────────────────────────────────────────────────────
    # PLOT 8: Wavefunction snapshots (|ψ|² at selected times)
    # ─────────────────────────────────────────────────────────────────────
    snap_times_ps = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    snap_indices  = [np.argmin(np.abs(t_ps - st)) for st in snap_times_ps]

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes = axes.flatten()

    from scipy.ndimage import gaussian_filter1d

    for idx, (si, ax) in enumerate(zip(snap_indices, axes)):
        apply_style(ax)
        prob_raw = np.abs(psi_arr[si])**2
        prob     = gaussian_filter1d(prob_raw, sigma=2)
        prob_max = np.max(prob)

        ax.fill_between(x, prob, alpha=0.5, color="#2c4f8c")
        ax.plot(x, prob, color="#2c4f8c", lw=1.2)

        V_scaled  = V / Vb * prob_max * 0.6
        V_clipped = np.clip(V_scaled, 0, prob_max * 1.1)
        ax.plot(x, V_clipped, color="gray", lw=1, ls="--", alpha=0.7,
                label="$V(x)$ (scaled)" if idx == 0 else "")

        ax.set_xlim(-6, 6)
        ax.set_ylim(0, prob_max * 1.35)
        ax.set_title(f"$t = {t_ps[si]:.1f}$ ps")
        ax.set_xlabel("$x$ [Å]", fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel(r"$|\psi|^2$", fontsize=9)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Probability Density Snapshots — Hydrogen (Tunnelling)", y=1.01)
    fig.tight_layout()
    save(fig, "08_wavefunction_snapshots")

    # ─────────────────────────────────────────────────────────────────────
    # ANIMATION  (saved as GIF)
    # ─────────────────────────────────────────────────────────────────────
    print("\n[Anim] Generating animation (this may take ~30 s)…")

    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    apply_style(ax_a)
    ax_a.plot(x, V / Vb * 0.25, color="gray", lw=1, ls="--", alpha=0.7, label="$V(x)$ (scaled)")
    fill = ax_a.fill_between(x, np.zeros_like(x), alpha=0.5, color="#2c4f8c")
    line, = ax_a.plot(x, np.zeros_like(x), color="#2c4f8c", lw=1.5)
    xmark, = ax_a.plot([], [], "v", color="#e07b39", ms=8, label=r"$\langle x\rangle$")
    time_text = ax_a.text(0.02, 0.93, "", transform=ax_a.transAxes, fontsize=10)
    ax_a.set_xlim(-6, 6)
    ax_a.set_ylim(0, 0.65)
    ax_a.set_xlabel("Position $x$ [Å]")
    ax_a.set_ylabel(r"$|\psi(x,t)|^2$")
    ax_a.set_title("Wave Packet Dynamics in Double-Well Potential")
    ax_a.legend(loc="upper right", fontsize=9)

    mask_anim = t_ps <= 5.0
    psi_anim  = psi_arr[mask_anim]
    t_anim    = t_ps[mask_anim]
    xexp_anim = xexp_H[mask_anim]

    def update(frame):
        prob = np.abs(psi_anim[frame])**2
        line.set_ydata(prob)
        for coll in ax_a.collections[1:]:
            coll.remove()
        ax_a.fill_between(x, prob, alpha=0.45, color="#2c4f8c")
        xmark.set_data([xexp_anim[frame]], [0.02])
        time_text.set_text(f"$t$ = {t_anim[frame]:.2f} ps")
        return line, xmark, time_text

    anim = FuncAnimation(fig_a, update, frames=len(psi_anim),
                         interval=40, blit=False)
    anim.save("output/animation.gif", writer="pillow", fps=25, dpi=100)
    plt.close(fig_a)
    print("  saved → output/animation.gif")

    # ─────────────────────────────────────────────────────────────────────
    print("\n✓ All done! Results in 'output/'")
    print("  Plots: 01–08 PNG files")
    print("  Animation: animation.gif")