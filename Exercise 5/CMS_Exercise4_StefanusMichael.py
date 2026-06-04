import sys
import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    pass
import os

# ── Optional numba acceleration ──────────────────────────────────────────────
# If numba is unavailable we fall back to a no-op decorator so the script
# still runs (just more slowly for large N).
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):                 # type: ignore
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(f):
            return f
        return deco

# ── Physical constants  (Å / fs / u / eV unit system) ────────────────────────
kB    = 8.617333e-5    # Boltzmann constant                  [eV/K]
ECONV = 103.6427       # 1 u·(Å/fs)² = ECONV eV
#   a  [Å/fs²]  = F [eV/Å]  / (m [u]  * ECONV)
#   KE [eV]     = 0.5 * m [u] * sum(v²[Å/fs]²) * ECONV
#   T  [K]      = 2 * KE / (3 * N * kB)

# ── Na Morse parameters  (Girifalco & Weizer, PhysRev 114, 687, 1959, Table I) ─
# V(r) = D * (exp(-2α(r-r₀)) - 2·exp(-α(r-r₀)))
D_Na     = 0.06334     # well depth              [eV]
r0_Na    = 5.336       # equilibrium distance    [Å]
alpha_Na = 0.58993     # inverse range           [Å⁻¹]
m_Na     = 22.9898     # atomic mass of Na       [u]
r_cut    = 2.5 * r0_Na # cutoff radius (≥ 2r₀)  [Å]

# ── Cu Morse parameters (same paper, Girifalco & Weizer 1959, Table I) ──────
# Stronger well, shorter equilibrium → much higher melting point than Na.
D_Cu     = 0.3429      # well depth              [eV]
r0_Cu    = 2.866       # equilibrium distance    [Å]
alpha_Cu = 1.3588      # inverse range           [Å⁻¹]
m_Cu     = 63.546      # atomic mass of Cu       [u]
r_cut_Cu = 2.5 * r0_Cu # cutoff radius           [Å]


# ── Plot helpers ──────────────────────────────────────────────────────────────
STYLE = dict(fig_bg="white", ax_bg="#f7f9fc", grid_c="#dce3ec", fontsize=11)


def apply_style(ax):
    ax.set_facecolor(STYLE["ax_bg"])
    ax.grid(True, color=STYLE["grid_c"], lw=0.7, zorder=0)
    ax.tick_params(labelsize=STYLE["fontsize"] - 1)


def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"  saved -> output/{name}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 ─ Morse Potential and Periodic Boundary Conditions (1D warmup)
# ═══════════════════════════════════════════════════════════════════════════════

def morse(r, D=D_Na, r0=r0_Na, alpha=alpha_Na):
    """V(r) = D·(exp(-2α(r-r₀)) - 2·exp(-α(r-r₀)))  [eV]"""
    u = np.exp(-alpha * (r - r0))
    return D * (u**2 - 2.0 * u)


def morse_shifted(r, rc=r_cut, D=D_Na, r0=r0_Na, alpha=alpha_Na):
    """Shifted-cutoff Morse: V(r)-V(rc) for r≤rc, 0 for r>rc."""
    V    = morse(r, D, r0, alpha)
    V_rc = morse(rc, D, r0, alpha)
    return np.where(r <= rc, V - V_rc, 0.0)


def min_image_1d(dx, L):
    """Minimum-image displacement in 1D: maps dx into (-L/2, L/2]."""
    return dx - L * np.round(dx / L)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 ─ 3D MD Simulation
# ═══════════════════════════════════════════════════════════════════════════════

# ── System initialization ─────────────────────────────────────────────────────

def init_box(N=27, rho=0.5):
    """
    Cubic box side length for N atoms at density rho [atoms/nm³].
    1 nm³ = 1000 Å³  →  L [Å] = (N / rho * 1000)^(1/3).
    """
    return (N / rho * 1000.0) ** (1.0 / 3.0)


def init_positions_sc(N, L, spacing=None):
    """
    Place N atoms on a compact 3D simple-cubic sub-lattice centred in [0, L)³.

    For N=27 this gives a 3×3×3 cluster. The lattice spacing is set to the
    Morse equilibrium distance r₀ by default so that the atoms actually
    interact at t=0 (a uniform spread over the whole box would put nearest
    neighbours well beyond the cutoff and the cluster would not exist).
    Returns (N, 3) [Å].
    """
    n = round(N ** (1.0 / 3.0))
    assert n ** 3 == N, f"N={N} is not a perfect cube."
    if spacing is None:
        spacing = r0_Na                    # default to Na equilibrium distance
    cluster_size = (n - 1) * spacing
    offset       = 0.5 * (L - cluster_size)
    idx = np.arange(n)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(float)
    pos = pos * spacing + offset
    return pos


def init_velocities(N, T0, mass=m_Na, seed=42):
    """
    Draw velocities from Uniform(-1, 1), zero net momentum, rescale to T0 [K].
    Returns (N, 3) [Å/fs].
    """
    rng = np.random.default_rng(seed)
    vel = rng.uniform(-1.0, 1.0, (N, 3))
    vel -= vel.mean(axis=0)                               # zero net momentum
    KE    = 0.5 * mass * np.sum(vel**2) * ECONV          # [eV]
    T_cur = 2.0 * KE / (3.0 * N * kB)                    # [K]
    vel  *= np.sqrt(T0 / T_cur)                          # rescale to T0
    return vel


# ── XYZ trajectory output (Ovito-compatible) ─────────────────────────────────

def write_xyz_frame(fh, pos, step, dt, comment="", element="Na"):
    """Write one frame in standard XYZ format."""
    N = len(pos)
    fh.write(f"{N}\n")
    fh.write(f"Step={step} t={step * dt:.3f}fs {comment}\n")
    for r in pos:
        fh.write(f"{element}  {r[0]:.6f}  {r[1]:.6f}  {r[2]:.6f}\n")


# ── Force calculation ─────────────────────────────────────────────────────────

def compute_forces_and_energy(pos, L, D=D_Na, r0=r0_Na,
                               alpha=alpha_Na, rc=r_cut):
    """
    Vectorised pairwise Morse forces with minimum-image PBC and shifted cutoff.

    Strategy
    --------
    Build (N, N, 3) displacement matrix dr[i,j] = pos[j] - pos[i], apply
    minimum image, compute scalar distances, select upper-triangle pairs
    within the cutoff, evaluate dV/dr, assemble forces via Newton's 3rd law.

    Returns
    -------
    forces : (N, 3)  [eV/Å]
    epot   : float   [eV]
    """
    N  = len(pos)
    dr = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N, N, 3)
    dr -= L * np.round(dr / L)                            # minimum image

    r2   = np.sum(dr**2, axis=2)                          # (N, N)
    diag = np.eye(N, dtype=bool)
    r2[diag] = 1.0                                        # avoid sqrt(0)
    r    = np.sqrt(r2)

    mask = np.triu(~diag, k=1) & (r < rc)                # upper triangle, cutoff

    r_m  = r[mask]
    u    = np.exp(-alpha * (r_m - r0))

    V_rc = D * (np.exp(-2.0 * alpha * (rc - r0)) - 2.0 * np.exp(-alpha * (rc - r0)))
    epot = float(np.sum(D * (u**2 - 2.0 * u) - V_rc))

    # dV/dr = 2Dα·u·(1 - u)
    # Force on i due to j: F_i = (dV/dr) · r̂_{ij}  (r̂_{ij} points i→j)
    dVdr  = 2.0 * D * alpha * u * (1.0 - u)              # [eV/Å]
    fvec  = (dVdr / r_m)[:, np.newaxis] * dr[mask, :]    # (N_pairs, 3)

    forces = np.zeros((N, 3))
    ii, jj = np.where(mask)
    np.add.at(forces, ii,  fvec)                          # F_i += dV/dr · r̂
    np.add.at(forces, jj, -fvec)                          # F_j -= dV/dr · r̂

    return forces, epot


# ── @njit force kernel (scales much better than NumPy for large N) ───────────
# The vectorised version above builds (N, N, 3) arrays in memory, which
# becomes wasteful and cache-unfriendly when N grows. The double loop below,
# JIT-compiled by numba, is O(N²) in flops but uses O(N) memory and runs
# faster than the NumPy version for N ≳ 100. With HAVE_NUMBA=False this is
# pure Python and *slow* — use the vectorised version in that case.
@njit(cache=True, fastmath=True)
def _forces_njit(pos, L, D, r0, alpha, rc):
    N      = pos.shape[0]
    forces = np.zeros((N, 3))
    epot   = 0.0
    # Shift to enforce V(rc) = 0
    u_rc   = np.exp(-alpha * (rc - r0))
    V_rc   = D * (u_rc * u_rc - 2.0 * u_rc)
    rc2    = rc * rc
    for i in range(N - 1):
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        for j in range(i + 1, N):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            # Minimum-image convention
            dx -= L * round(dx / L)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= rc2:
                continue
            r    = np.sqrt(r2)
            u    = np.exp(-alpha * (r - r0))
            epot += D * (u*u - 2.0*u) - V_rc
            # dV/dr = 2Dα·u·(1-u);  F_i along (r_j - r_i)/r
            dVdr = 2.0 * D * alpha * u * (1.0 - u)
            fac  = dVdr / r
            fx, fy, fz = fac * dx, fac * dy, fac * dz
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz
    return forces, epot


def compute_forces_njit(pos, L, D=D_Na, r0=r0_Na, alpha=alpha_Na, rc=r_cut):
    """Wrapper around the @njit kernel matching the NumPy version's signature."""
    return _forces_njit(pos, L, D, r0, alpha, rc)


# ── Velocity Verlet integrator ────────────────────────────────────────────────

def velocity_verlet_step(pos, vel, forces, L, mass=m_Na, dt=1.0,
                          D=D_Na, r0=r0_Na, alpha=alpha_Na, rc=r_cut,
                          force_fn=None):
    """
    One Velocity-Verlet step.
    pos, vel in Å, Å/fs; forces in eV/Å; dt in fs.
    Acceleration: a = F / (m · ECONV)  [Å/fs²].
    force_fn : callable(pos, L, D, r0, alpha, rc) → (forces, epot).
               Defaults to the NumPy vectorised version.
    """
    if force_fn is None:
        force_fn = compute_forces_and_energy
    acc      = forces / (mass * ECONV)
    pos_new  = pos + vel * dt + 0.5 * acc * dt**2
    pos_new %= L                                          # PBC wrap to [0, L)
    forces_new, epot_new = force_fn(pos_new, L, D, r0, alpha, rc)
    acc_new  = forces_new / (mass * ECONV)
    vel_new  = vel + 0.5 * (acc + acc_new) * dt
    return pos_new, vel_new, forces_new, epot_new


# ── Observables ───────────────────────────────────────────────────────────────

def kinetic_energy(vel, mass=m_Na):
    """KE [eV] from velocities [Å/fs]."""
    return 0.5 * mass * np.sum(vel**2) * ECONV


def temperature(vel, N, mass=m_Na):
    """Instantaneous temperature [K]."""
    return 2.0 * kinetic_energy(vel, mass) / (3.0 * N * kB)


# ── Thermostats ───────────────────────────────────────────────────────────────

def rescale_velocities(vel, T0, N, mass=m_Na):
    """Hard velocity rescaling to enforce T = T0."""
    T_cur = temperature(vel, N, mass)
    if T_cur > 0:
        vel = vel * np.sqrt(T0 / T_cur)
    return vel


def berendsen_scale(T_cur, T0, dt, tau_T):
    """
    Berendsen coupling factor λ = sqrt(1 + dt/τ_T · (T₀/T - 1)).
    Clamped so the sqrt argument is non-negative.
    """
    T_safe = max(T_cur, 1e-10)
    arg    = 1.0 + (dt / tau_T) * (T0 / T_safe - 1.0)
    return np.sqrt(max(arg, 1e-6))


# ── Main MD runner ─────────────────────────────────────────────────────────────

def run_md(pos0, vel0, L, N_steps=10000, dt=1.0, save_every=100,
           thermostat='none', T0=100.0, tau_T=100.0,
           traj_file=None, mass=m_Na,
           D=D_Na, r0=r0_Na, alpha=alpha_Na, rc=r_cut,
           force_fn=None, element="Na"):
    """
    Run an MD simulation.

    Parameters
    ----------
    thermostat : 'none' | 'rescale' | 'berendsen'
    tau_T      : Berendsen coupling time [fs]
    traj_file  : open file handle for XYZ output (or None)
    force_fn   : compute_forces_and_energy (default) or compute_forces_njit
    element    : element symbol written to XYZ trajectory

    Returns
    -------
    times, T_arr, KE_arr, PE_arr  — each (n_saved,)
    """
    if force_fn is None:
        force_fn = compute_forces_and_energy

    N   = len(pos0)
    pos = pos0.copy()
    vel = vel0.copy()
    forces, epot = force_fn(pos, L, D, r0, alpha, rc)

    times, T_arr, KE_arr, PE_arr = [], [], [], []

    for step in range(N_steps + 1):
        if step % save_every == 0:
            KE = kinetic_energy(vel, mass)
            T  = 2.0 * KE / (3.0 * N * kB)
            times.append(step * dt)
            T_arr.append(T)
            KE_arr.append(KE)
            PE_arr.append(epot)
            if traj_file is not None:
                write_xyz_frame(traj_file, pos, step, dt, element=element)

        if step < N_steps:
            pos, vel, forces, epot = velocity_verlet_step(
                pos, vel, forces, L, mass, dt,
                D=D, r0=r0, alpha=alpha, rc=rc, force_fn=force_fn)

            if thermostat == 'rescale' and (step + 1) % 100 == 0:
                vel = rescale_velocities(vel, T0, N, mass)

            elif thermostat == 'berendsen':
                T_cur = temperature(vel, N, mass)
                vel  *= berendsen_scale(T_cur, T0, dt, tau_T)

    return (np.array(times), np.array(T_arr),
            np.array(KE_arr), np.array(PE_arr))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs("output", exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size":   STYLE["fontsize"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    print("=" * 62)
    print("Exercise 4 – Molecular Dynamics of a Sodium Cluster")
    print("=" * 62)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1 — 1D Periodic Boundary Conditions and Morse Potential
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Part 1]  1D PBC demonstration and Morse potential")

    L_1d  = 20.0           # 1D box length [Å]  (ensures r_cut < L/2)
    disp  = np.linspace(-L_1d * 0.9, L_1d * 1.1, 800)  # particle 2 displacement
    x2_raw = disp                                # raw (unwrapped) position
    x2_w   = x2_raw % L_1d                      # wrapped into [0, L_1d)

    dx_min = min_image_1d(x2_raw - 0.0, L_1d)   # min-image from x1=0
    r_1d   = np.abs(dx_min)

    # Protect against r=0 (divide by zero in potential)
    r_1d_safe = np.where(r_1d < 0.5, 0.5, r_1d)

    V_1d_raw = morse(r_1d_safe)
    V_1d_sh  = morse_shifted(r_1d_safe)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    apply_style(axes[0, 0]); apply_style(axes[0, 1])
    apply_style(axes[1, 0]); apply_style(axes[1, 1])

    # (a) Wrapped position of particle 2
    ax = axes[0, 0]
    ax.plot(disp, x2_w, color="#2c4f8c", lw=1.6, label="$x_2$ (wrapped)")
    ax.plot(disp, x2_raw, color="#aaa", lw=1.0, ls="--", label="$x_2$ (unwrapped)")
    ax.axhline(0,     color="#c0392b", ls=":", lw=1, alpha=0.7)
    ax.axhline(L_1d,  color="#c0392b", ls=":", lw=1, alpha=0.7,
               label=f"Box walls (0 and $L={L_1d}$ Å)")
    ax.set_xlabel("Displacement [Å]"); ax.set_ylabel("$x_2$ [Å]")
    ax.set_title("(a) PBC wrap: particle 2 position")
    ax.legend(fontsize=8)

    # (b) Minimum image distance
    ax = axes[0, 1]
    ax.plot(disp, r_1d, color="#2c4f8c", lw=1.6, label=r"$|x_2 - x_1|_\mathrm{min}$")
    ax.axhline(L_1d / 2, color="gray", ls="--", lw=1.0, alpha=0.7,
               label=f"$L/2 = {L_1d/2:.1f}$ Å (max min-image dist.)")
    ax.set_xlabel("Displacement [Å]"); ax.set_ylabel("Min-image distance [Å]")
    ax.set_title("(b) Minimum-image distance")
    ax.legend(fontsize=8)

    # (c/d) Morse potential using min-image distance as particle 2 moves
    ax = axes[1, 0]
    ax.plot(disp, V_1d_raw, color="#2c4f8c", lw=1.6,
            label=r"$V_M(|r|_\mathrm{min})$")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylim(-0.12, 0.25)
    ax.set_xlabel("Displacement of particle 2 [Å]")
    ax.set_ylabel("$V$ [eV]")
    ax.set_title("(c/d) Morse potential via min-image distance")
    ax.legend(fontsize=8)

    # (e) Shifted cutoff potential vs min-image distance
    r_plot = np.linspace(1.5, 11.0, 500)
    ax = axes[1, 1]
    ax.plot(r_plot, morse(r_plot), color="#2c4f8c", lw=2.0,
            label="$V_M(r)$  (raw)")
    ax.plot(r_plot, morse_shifted(r_plot), color="#e07b39", lw=2.0, ls="--",
            label=r"$V_M(r) - V_M(r_\mathrm{cut})$  (shifted)")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(r_cut, color="#c0392b", ls=":", lw=1.2,
               label=f"$r_{{cut}} = {r_cut:.2f}$ Å")
    ax.axvline(r0_Na, color="#27ae60", ls=":", lw=1.2,
               label=f"$r_0 = {r0_Na}$ Å")
    ax.set_xlim(1.5, 11); ax.set_ylim(-0.12, 0.25)
    ax.set_xlabel("$r$ [Å]"); ax.set_ylabel("$V$ [eV]")
    ax.set_title("(e) Shifted-cutoff Morse potential")
    ax.legend(fontsize=8)

    fig.suptitle(
        rf"Part 1: Morse Potential and 1D PBC  "
        rf"($D={D_Na}$ eV, $r_0={r0_Na}$ Å, $\alpha={alpha_Na}$ Å$^{{-1}}$, "
        rf"$r_{{cut}}={r_cut:.2f}$ Å)",
        y=1.01, fontsize=11)
    fig.tight_layout()
    save(fig, "01_morse_1d_pbc")

    print(f"  D    = {D_Na} eV,  r₀ = {r0_Na} Å,  α = {alpha_Na} Å⁻¹")
    print(f"  r_cut = {r_cut:.3f} Å  (= 2.5 × r₀)")
    print(f"  V(r_cut) = {morse(r_cut):.6f} eV  → shift applied to all pairs")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2 — 3D System Initialization
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Part 2]  3D system initialization")

    N          = 27
    T0         = 100.0      # K
    dt         = 1.0        # fs
    N_steps    = 10000
    save_every = 100
    tau_T      = 100.0 * dt  # fs  (Berendsen coupling time)

    L   = init_box(N, rho=0.5)
    pos = init_positions_sc(N, L)
    vel = init_velocities(N, T0, m_Na, seed=42)

    T_init = temperature(vel, N)
    print(f"  Box side  L = {L:.4f} Å  (volume = {L**3/1000:.2f} nm³, "
          f"density = {N / (L**3/1000):.3f} atoms/nm³)")
    print(f"  Initial T = {T_init:.4f} K  (target {T0} K)")

    # Nearest-neighbour distance check
    dr_check = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dr_check -= L * np.round(dr_check / L)
    r_check   = np.sqrt(np.sum(dr_check**2, axis=2))
    np.fill_diagonal(r_check, np.inf)
    print(f"  Min pair distance = {r_check.min():.4f} Å  "
          f"(lattice spacing = {L/round(N**(1/3)):.4f} Å >> r₀ = {r0_Na} Å)")

    # Plot 06: initial 3D configuration
    fig = plt.figure(figsize=(6, 5.5))
    ax3d = fig.add_subplot(111, projection='3d')
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                      c=np.arange(N), cmap="viridis", s=80, zorder=3)
    plt.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.7, label="Atom index")
    # Box edges
    verts = np.array(list(np.ndindex(2, 2, 2)), dtype=float) * L
    for i in range(8):
        for j in range(i + 1, 8):
            diff = verts[i] - verts[j]
            if np.sum(diff != 0) == 1:
                ax3d.plot(*zip(verts[i], verts[j]), 'k-', lw=0.6, alpha=0.4)
    ax3d.set_xlabel("x [Å]"); ax3d.set_ylabel("y [Å]"); ax3d.set_zlabel("z [Å]")
    ax3d.set_title(f"Initial Configuration — {N} Na atoms (3×3×3 SC grid)")
    save(fig, "06_initial_config")

    # ─────────────────────────────────────────────────────────────────────────
    # RUN A — NVE (no thermostat)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Run A]  NVE ({N_steps} steps, dt={dt} fs, T₀={T0} K) …")
    with open("output/traj_nve.xyz", "w") as fh:
        t_nve, T_nve, KE_nve, PE_nve = run_md(
            pos, vel, L, N_steps, dt, save_every,
            thermostat='none', T0=T0, traj_file=fh)

    E_nve  = KE_nve + PE_nve
    drift  = (E_nve - E_nve[0]) / np.abs(E_nve[0]) * 100.0
    print(f"  ⟨T⟩ = {T_nve.mean():.2f} K")
    print(f"  Max |ΔE/E₀| = {np.max(np.abs(drift)):.3e} %")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    apply_style(axes[0]); apply_style(axes[1])

    axes[0].plot(t_nve, T_nve, color="#2c4f8c", lw=1.5, label="$T(t)$")
    axes[0].axhline(T0, color="#c0392b", ls="--", lw=1.0,
                    label=f"Target $T_0 = {T0:.0f}$ K")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("NVE Simulation — Temperature (no thermostat)")
    axes[0].legend(fontsize=9)

    axes[1].plot(t_nve, KE_nve, color="#e07b39", lw=1.4,
                 label=r"$\langle T \rangle$ (kinetic)")
    axes[1].plot(t_nve, PE_nve, color="#2c4f8c", lw=1.4,
                 label=r"$\langle V \rangle$ (potential)")
    axes[1].plot(t_nve, E_nve,  color="black",   lw=1.2, ls="--",
                 label=r"$E_{tot}$ (conserved)")
    axes[1].set_xlabel("Time [fs]"); axes[1].set_ylabel("Energy [eV]")
    axes[1].set_title("NVE Simulation — Energy Conservation")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"NVE Dynamics — Na₂₇, T₀={T0:.0f} K", y=1.01)
    fig.tight_layout()
    save(fig, "02_temperature_nve")

    # ─────────────────────────────────────────────────────────────────────────
    # RUN B — Simple velocity rescaling (every 100 steps)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Run B]  Velocity rescaling thermostat …")
    t_res, T_res, KE_res, PE_res = run_md(
        pos, vel, L, N_steps, dt, save_every,
        thermostat='rescale', T0=T0)
    print(f"  ⟨T⟩ = {T_res.mean():.2f} K  (target {T0} K)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    apply_style(ax)
    ax.plot(t_nve, T_nve, color="#aaa", lw=1.2, alpha=0.7,
            label="NVE (reference)")
    ax.plot(t_res, T_res, color="#c0392b", lw=1.5,
            label="Velocity rescaling (every 100 steps)")
    ax.axhline(T0, color="black", ls="--", lw=1.0, alpha=0.7,
               label=f"Target $T_0 = {T0:.0f}$ K")
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Simple Velocity Rescaling Thermostat — Na₂₇")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "03_temperature_rescale")

    # ─────────────────────────────────────────────────────────────────────────
    # RUN C — Berendsen thermostat
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[Run C]  Berendsen thermostat (τ_T = {tau_T:.0f} fs) …")
    with open("output/traj.xyz", "w") as fh:
        t_ber, T_ber, KE_ber, PE_ber = run_md(
            pos, vel, L, N_steps, dt, save_every,
            thermostat='berendsen', T0=T0, tau_T=tau_T, traj_file=fh)

    equil = len(t_ber) // 4
    print(f"  ⟨T⟩ (all steps)       = {T_ber.mean():.2f} K")
    print(f"  ⟨T⟩ (after equil.)    = {T_ber[equil:].mean():.2f} K  "
          f"(first 25% discarded)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    apply_style(ax)
    ax.plot(t_ber, T_ber, color="#27ae60", lw=1.8,
            label=rf"Berendsen ($\tau_T = {tau_T:.0f}$ fs)")
    ax.axhline(T0, color="black", ls="--", lw=1.0, alpha=0.7,
               label=f"Target $T_0 = {T0:.0f}$ K")
    ax.axhline(T_ber[equil:].mean(), color="#27ae60", ls=":",
               lw=1.2, alpha=0.7,
               label=rf"$\langle T \rangle_{{equil}} = {T_ber[equil:].mean():.1f}$ K")
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Temperature [K]")
    ax.set_title("Berendsen Thermostat — Na₂₇")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "04_temperature_berendsen")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 05 — Thermostat comparison
    # ─────────────────────────────────────────────────────────────────────────
    E_res = KE_res + PE_res
    E_ber = KE_ber + PE_ber

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    apply_style(axes[0]); apply_style(axes[1])

    axes[0].plot(t_nve, T_nve, color="#2c4f8c", lw=1.6, label="NVE")
    axes[0].plot(t_res, T_res, color="#c0392b", lw=1.4, ls="--",
                 label="Velocity rescaling (every 100 steps)")
    axes[0].plot(t_ber, T_ber, color="#27ae60", lw=1.6, ls="-.",
                 label=rf"Berendsen ($\tau_T = {tau_T:.0f}$ fs)")
    axes[0].axhline(T0, color="black", ls=":", lw=1.0, alpha=0.6,
                    label=f"$T_0 = {T0:.0f}$ K")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Thermostat Comparison — Temperature")
    axes[0].legend(fontsize=9)

    axes[1].plot(t_nve, E_nve - E_nve[0], color="#2c4f8c", lw=1.6,
                 label="NVE (flat = conserved)")
    axes[1].plot(t_res, E_res - E_res[0], color="#c0392b", lw=1.4, ls="--",
                 label="Velocity rescaling")
    axes[1].plot(t_ber, E_ber - E_ber[0], color="#27ae60", lw=1.6, ls="-.",
                 label="Berendsen")
    axes[1].axhline(0, color="black", ls=":", lw=0.8, alpha=0.5)
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_ylabel(r"$\Delta E_{tot}$  [eV]")
    axes[1].set_title("Total Energy Change")
    axes[1].legend(fontsize=9)

    fig.suptitle("Thermostat Comparison — Na₂₇ Cluster", y=1.01, fontsize=12)
    fig.tight_layout()
    save(fig, "05_thermostat_comparison")

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS 1 — Berendsen with τ_T = Δt (very stiff coupling)
    # ─────────────────────────────────────────────────────────────────────────
    # The exercise asks: "What happens if τ_T = Δt?" When the coupling time
    # equals the integration step, λ becomes √(T₀/T) every step — the
    # thermostat removes essentially all temperature fluctuations on contact,
    # so the trajectory is no longer truly Newtonian and the system cannot
    # explore the natural NVE-like fluctuations expected of a microcanonical
    # subsystem. We see a near-flat T(t) glued to T₀.
    print(f"\n[Analysis 1]  Berendsen with τ_T = Δt = {dt} fs (very stiff) …")
    tau_stiff = dt
    t_stiff, T_stiff, KE_stiff, PE_stiff = run_md(
        pos, vel, L, N_steps, dt, save_every,
        thermostat='berendsen', T0=T0, tau_T=tau_stiff)
    print(f"  ⟨T⟩ = {T_stiff.mean():.2f} K,  σ_T = {T_stiff.std():.2f} K")
    print(f"  (Berendsen τ_T=100Δt for comparison: σ_T = {T_ber.std():.2f} K)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    apply_style(ax)
    ax.plot(t_ber,   T_ber,   color="#27ae60", lw=1.6,
            label=rf"$\tau_T = 100\,\Delta t$ (proper)")
    ax.plot(t_stiff, T_stiff, color="#c0392b", lw=1.6,
            label=rf"$\tau_T = \Delta t$ (stiff — kills fluctuations)")
    ax.axhline(T0, color="black", ls="--", lw=1.0, alpha=0.7,
               label=f"$T_0 = {T0:.0f}$ K")
    ax.set_xlabel("Time [fs]"); ax.set_ylabel("Temperature [K]")
    ax.set_title(r"Effect of Berendsen coupling time $\tau_T$ — Na$_{27}$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "07_berendsen_tau_comparison")

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS 2 — Temperature sweep (melting study at T=100, 300, 700, 1200 K)
    # ─────────────────────────────────────────────────────────────────────────
    # At low T the cluster stays on its lattice sites (solid-like). As T
    # rises, atoms diffuse and the cluster loses long-range order (liquid-
    # like). At very high T (1200 K) the cluster fully dissociates into a
    # vapour. Each run writes its own XYZ file for Ovito visualisation.
    print("\n[Analysis 2]  Temperature sweep — 100 / 300 / 700 / 1200 K (Berendsen)")
    T_sweep = [100.0, 300.0, 700.0, 1200.0]
    sweep_T_traces = {}
    sweep_KE = {}
    sweep_PE = {}
    for T_target in T_sweep:
        print(f"  → T_target = {T_target:.0f} K …")
        vel_T = init_velocities(N, T_target, m_Na, seed=42)
        fname = f"output/traj_T{int(T_target)}K.xyz"
        with open(fname, "w") as fh:
            t_s, T_s, KE_s, PE_s = run_md(
                pos, vel_T, L, N_steps, dt, save_every,
                thermostat='berendsen', T0=T_target, tau_T=100.0 * dt,
                traj_file=fh, element="Na")
        equil = len(t_s) // 4
        print(f"    ⟨T⟩ after equil. = {T_s[equil:].mean():.2f} K, "
              f"⟨V⟩ = {PE_s[equil:].mean():.3f} eV")
        sweep_T_traces[T_target] = (t_s, T_s)
        sweep_KE[T_target] = KE_s
        sweep_PE[T_target] = PE_s

    # Plot temperature traces for all four runs
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    apply_style(axes[0]); apply_style(axes[1])
    colors = {100.0: "#2c4f8c", 300.0: "#e07b39",
              700.0: "#c0392b", 1200.0: "#7a1e8a"}
    for T_target in T_sweep:
        t_s, T_s = sweep_T_traces[T_target]
        axes[0].plot(t_s, T_s, color=colors[T_target], lw=1.4,
                     label=f"$T_0 = {T_target:.0f}$ K")
        axes[1].plot(t_s, sweep_PE[T_target], color=colors[T_target], lw=1.4,
                     label=f"$T_0 = {T_target:.0f}$ K")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Temperature sweep — Berendsen thermostatted runs")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Time [fs]"); axes[1].set_ylabel(r"$V_{pot}$ [eV]")
    axes[1].set_title("Potential energy — signature of melting")
    axes[1].legend(fontsize=9)
    fig.suptitle("Melting study — Na$_{27}$ cluster", y=1.01, fontsize=12)
    fig.tight_layout()
    save(fig, "08_temperature_sweep")

    print(f"  XYZ files for Ovito:")
    for T_target in T_sweep:
        print(f"    output/traj_T{int(T_target)}K.xyz")

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS 3 — Copper at multiple temperatures (much higher cohesion)
    # ─────────────────────────────────────────────────────────────────────────
    # Same N, same density, same thermostat, just different Morse parameters
    # and mass. Cu has D ≈ 5.4× Na's well depth, so even at T = 1200 K — where
    # the Na cluster has fully dissociated into a vapour — Cu remains a tightly
    # bound solid. Running Cu at both 700 K and 1200 K makes this contrast
    # explicit. Cu's experimental melting point is ≈1358 K, so 1200 K is still
    # below it.
    print("\n[Analysis 3]  Copper at 700 K and 1200 K")
    L_Cu   = init_box(N, rho=0.5)                  # density 0.5 atoms/nm³
    pos_Cu = init_positions_sc(N, L_Cu, spacing=r0_Cu)
    print(f"  L = {L_Cu:.3f} Å, r₀(Cu) = {r0_Cu} Å, "
          f"D(Cu) = {D_Cu} eV  (vs Na: r₀={r0_Na}, D={D_Na})")

    Cu_T_targets = [700.0, 1200.0]
    Cu_traces    = {}   # T_target → (t, T_arr, PE)
    for T_target in Cu_T_targets:
        print(f"  → Cu @ T_target = {T_target:.0f} K …")
        vel_Cu = init_velocities(N, T_target, m_Cu, seed=42)
        fname  = f"output/traj_Cu_{int(T_target)}K.xyz"
        with open(fname, "w") as fh:
            t_c, T_c, KE_c, PE_c = run_md(
                pos_Cu, vel_Cu, L_Cu, N_steps, dt, save_every,
                thermostat='berendsen', T0=T_target, tau_T=100.0 * dt,
                mass=m_Cu, D=D_Cu, r0=r0_Cu, alpha=alpha_Cu, rc=r_cut_Cu,
                traj_file=fh, element="Cu")
        equil = len(t_c) // 4
        print(f"    ⟨T⟩ = {T_c[equil:].mean():.2f} K, "
              f"⟨V⟩ = {PE_c[equil:].mean():.3f} eV  "
              f"(vs Na@{int(T_target)}K: "
              f"{sweep_PE[T_target][equil:].mean():.3f} eV)")
        Cu_traces[T_target] = (t_c, T_c, PE_c)

    # 2-panel comparison plot: temperature (top), potential energy (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    apply_style(axes[0]); apply_style(axes[1])

    # Colour scheme: Na in red/purple, Cu in orange/brown
    style = {
        ('Na',  700.0): dict(color="#c0392b", lw=1.4, ls='-',  label="Na  @ 700 K"),
        ('Cu',  700.0): dict(color="#b07d2b", lw=1.4, ls='-',  label="Cu  @ 700 K"),
        ('Na', 1200.0): dict(color="#7a1e8a", lw=1.4, ls='--', label="Na  @ 1200 K"),
        ('Cu', 1200.0): dict(color="#5d3a18", lw=1.4, ls='--', label="Cu  @ 1200 K"),
    }
    for T_target in Cu_T_targets:
        # Na trace from the sweep
        t_Na, T_Na = sweep_T_traces[T_target]
        PE_Na      = sweep_PE[T_target]
        axes[0].plot(t_Na, T_Na, **style[('Na', T_target)])
        axes[1].plot(t_Na, PE_Na, **style[('Na', T_target)])
        # Cu trace from this analysis
        t_c, T_c, PE_c = Cu_traces[T_target]
        axes[0].plot(t_c, T_c, **style[('Cu', T_target)])
        axes[1].plot(t_c, PE_c, **style[('Cu', T_target)])
        axes[0].axhline(T_target, color="black", ls=":", lw=0.8, alpha=0.4)

    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Na vs Cu — temperature traces (Berendsen, $T_0=700$ and $1200$ K)")
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].set_xlabel("Time [fs]")
    axes[1].set_ylabel(r"$V_\text{pot}$ [eV]")
    axes[1].set_title(r"Na vs Cu — potential energy "
                      r"($D_\text{Cu}\approx 5.4\,D_\text{Na}$)")
    axes[1].legend(fontsize=9, ncol=2)
    fig.suptitle("Sodium vs Copper at 700 K and 1200 K", y=1.01, fontsize=12)
    fig.tight_layout()
    save(fig, "09_Na_vs_Cu")

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS 4 — Timing benchmark: scaling with N and steps
    # ─────────────────────────────────────────────────────────────────────────
    # We benchmark the NumPy vectorised force routine against the @njit
    # version for cubic-cube atom counts N = 27, 64, 125 and (216 if numba
    # available). Forces scale as O(N²); the constant prefactor differs
    # substantially between the two backends.
    print("\n[Analysis 4]  Timing benchmark (force calculation backends)")
    bench_steps = 500   # short run, just to time the integrator
    bench_N_list = [27, 64, 125]
    if HAVE_NUMBA:
        bench_N_list.append(216)
        # JIT-compile once so the first timed run is fair
        _warm = np.array([[0.0, 0.0, 0.0],
                          [r0_Na, 0.0, 0.0],
                          [0.0, r0_Na, 0.0],
                          [0.0, 0.0, r0_Na]])
        _ = compute_forces_njit(_warm, 50.0)
    timings_np  = {}
    timings_jit = {}
    for Nb in bench_N_list:
        Lb = init_box(Nb, rho=0.5)
        pb = init_positions_sc(Nb, Lb)
        vb = init_velocities(Nb, 100.0, m_Na, seed=1)

        t0 = time.perf_counter()
        run_md(pb, vb, Lb, bench_steps, dt, save_every=bench_steps + 1,
               thermostat='none', force_fn=compute_forces_and_energy)
        timings_np[Nb] = time.perf_counter() - t0

        if HAVE_NUMBA:
            t0 = time.perf_counter()
            run_md(pb, vb, Lb, bench_steps, dt, save_every=bench_steps + 1,
                   thermostat='none', force_fn=compute_forces_njit)
            timings_jit[Nb] = time.perf_counter() - t0

        msg = f"  N = {Nb:4d}  NumPy = {timings_np[Nb]:7.3f} s"
        if HAVE_NUMBA:
            speedup = timings_np[Nb] / timings_jit[Nb]
            msg += f"   njit = {timings_jit[Nb]:7.3f} s   speed-up = {speedup:5.2f}×"
        else:
            msg += "   (numba not installed — install for large-N speed-up)"
        print(msg)
    print(f"  (Each timing covers {bench_steps} Velocity-Verlet steps.)")

    # Plot scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    apply_style(ax)
    Ns = np.array(bench_N_list, dtype=float)
    ax.loglog(Ns, [timings_np[n] for n in bench_N_list], 'o-',
              color="#2c4f8c", lw=1.6, ms=7, label="NumPy vectorised")
    if HAVE_NUMBA:
        ax.loglog(Ns, [timings_jit[n] for n in bench_N_list], 's-',
                  color="#27ae60", lw=1.6, ms=7, label="@njit (numba)")
    # O(N²) reference line through the first NumPy point
    ref = timings_np[bench_N_list[0]] * (Ns / Ns[0])**2
    ax.loglog(Ns, ref, 'k--', lw=1.0, alpha=0.6, label=r"$\propto N^2$")
    ax.set_xlabel("Number of atoms $N$")
    ax.set_ylabel(f"Wall time for {bench_steps} steps  [s]")
    ax.set_title("Force-routine scaling")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "10_timing_benchmark")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("All output saved to  output/")
    print("  Plots  : 01–10 .png")
    print("  Ovito  : output/traj.xyz             (Berendsen, 100 K)")
    print("           output/traj_nve.xyz         (NVE)")
    print("           output/traj_T100K.xyz       (melting sweep)")
    print("           output/traj_T300K.xyz       (melting sweep)")
    print("           output/traj_T700K.xyz       (melting sweep)")
    print("           output/traj_T1200K.xyz      (melting sweep)")
    print("           output/traj_Cu_700K.xyz     (copper comparison)")
    print("           output/traj_Cu_1200K.xyz    (copper comparison)")
    print(f"  Numba available: {HAVE_NUMBA}")
    print("=" * 62)