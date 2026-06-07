"""
Computational Physics: Materials Science — Exercise 5 (SS 2026)
NVT ensemble and isochoric heat capacity C_V of a Na cluster — ASE version.

This is the ASE-based implementation recommended by the exercise sheet

Design choice
-------------
The sheet asks for TWO things that pull in different directions:
  * use ASE for reliable, comprehensible dynamics, and
  * keep using numba (@njit) to speed up the force calculation (from sheet 4),
  * AND use the exact Girifalco-Weizer Morse potential with a *shifted* cutoff.

That satisfies the ASE recommendation, the numba requirement, and the exact
Morse + shifted-cutoff specification simultaneously.

Heat capacity (both as in the sheet):
    Method 1 (fluctuations): C_V = (<E^2> - <E>^2) / (kB T^2)
    Method 2 (finite diff) : C_V ~ (<E(T+dT)> - <E(T-dT)>) / Delta T_measured

Units: ASE handles them. Energy [eV], length [Ang], mass [u], time via units.fs.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import write

# ── numba (required by the sheet) ────────────────────────────────────────────
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*a, **k):                                 # type: ignore
        if len(a) == 1 and callable(a[0]):
            return a[0]
        def deco(f):
            return f
        return deco

# ── Na Morse parameters (Girifalco & Weizer, PhysRev 114, 687, 1959) ─────────
# V(r) = D * (exp(-2a(r-r0)) - 2*exp(-a(r-r0)))   [eV]
D_Na     = 0.06334     # well depth              [eV]
r0_Na    = 5.336       # equilibrium distance    [Ang]
alpha_Na = 0.58993     # inverse range           [1/Ang]
m_Na     = 22.9898     # atomic mass             [u]
r_cut    = 2.5 * r0_Na # shifted cutoff (>= 2 r0)[Ang]

kB = units.kB          # 8.617e-5 eV/K  (ASE's own constant)


@njit(cache=True, fastmath=True)
def _morse_forces_njit(pos, L, D, r0, alpha, rc):
    """Pairwise Morse forces + energy, minimum-image PBC, shifted cutoff."""
    N = pos.shape[0]
    forces = np.zeros((N, 3))
    epot = 0.0
    u_rc = np.exp(-alpha * (rc - r0))
    V_rc = D * (u_rc * u_rc - 2.0 * u_rc)
    rc2 = rc * rc
    for i in range(N - 1):
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        for j in range(i + 1, N):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            dx -= L * round(dx / L)          # minimum image (cubic box)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= rc2:
                continue
            r = np.sqrt(r2)
            uu = np.exp(-alpha * (r - r0))
            epot += D * (uu*uu - 2.0*uu) - V_rc       # shifted so V(rc)=0
            dVdr = 2.0 * D * alpha * uu * (1.0 - uu)   # dV/dr
            fac = dVdr / r
            fx, fy, fz = fac*dx, fac*dy, fac*dz
            forces[i, 0] += fx; forces[i, 1] += fy; forces[i, 2] += fz
            forces[j, 0] -= fx; forces[j, 1] -= fy; forces[j, 2] -= fz
    return forces, epot

# Custom ASE Calculator wrapping the numba kernel
class MorseNumba(Calculator):
    """
    Minimal ASE Calculator for the sheet-4 shifted-cutoff Morse potential,
    evaluated by the numba kernel. Assumes a cubic, fully periodic cell.

    ASE calls atoms.get_potential_energy()/get_forces(), which dispatch here;
    we read positions and the (cubic) box length from the Atoms object.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, D=D_Na, r0=r0_Na, alpha=alpha_Na, rc=r_cut, **kwargs):
        super().__init__(**kwargs)
        self.D, self.r0, self.alpha, self.rc = D, r0, alpha, rc

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        L = float(atoms.cell.lengths()[0])             # cubic box
        forces, epot = _morse_forces_njit(
            atoms.get_positions(), L, self.D, self.r0, self.alpha, self.rc)
        self.results['energy'] = epot
        self.results['forces'] = forces


# System construction
def init_box(N=27, rho=0.5):
    """Cubic side length for N atoms at density rho [atoms/nm^3]."""
    return (N / rho * 1000.0) ** (1.0 / 3.0)


def build_cluster(N, L, T0, seed=42, spacing=None):
    """
    Build an N-atom Na cluster on a compact simple-cubic seed centred in a
    cubic periodic box, attach the numba Morse calculator, and draw velocities
    from a Maxwell-Boltzmann distribution at T0 (net momentum removed).
    """
    n = round(N ** (1.0 / 3.0))
    assert n ** 3 == N, "N must be a perfect cube (e.g. 27 = 3^3)."
    if spacing is None:
        spacing = r0_Na
    idx = np.arange(n)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing='ij')
    pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(float)
    pos = pos * spacing + 0.5 * (L - (n - 1) * spacing)

    atoms = Atoms(f'Na{N}', positions=pos, cell=[L, L, L], pbc=True)
    atoms.calc = MorseNumba()
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=rng)
    Stationary(atoms)                                  # zero net momentum (as sheet 4)
    return atoms


# NVT runner using ASE integrators, with production sampling
def run_nvt(atoms, thermostat, T0, n_equil, n_prod, dt=1.0,
            friction=0.01, taut_fs=100.0, sample_interval=1,
            traj_path=None, traj_interval=100):
    """
    thermostat : 'langevin' | 'berendsen'
    dt         : timestep [fs]
    friction   : Langevin friction [1/fs]   (0.01 -> ~100 fs relaxation)
    taut_fs    : Berendsen coupling time [fs]

    Equilibrates for n_equil steps, then records total energy and temperature
    every sample_interval steps for n_prod steps. Returns dict with E [eV] and
    T [K] production arrays.
    """
    if thermostat == 'langevin':
        # fixcm=False: thermostat all 3N DOF -> strictly correct canonical
        # sampling for this small cluster (ASE warns fixcm=True is biased here).
        dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=T0,
                       friction=friction / units.fs, fixcm=False)
    elif thermostat == 'berendsen':
        dyn = NVTBerendsen(atoms, timestep=dt * units.fs, temperature_K=T0,
                           taut=taut_fs * units.fs, fixcm=False)
    else:
        raise ValueError(thermostat)

    dyn.run(n_equil)                                   # discard equilibration

    data = {'E': [], 'T': []}
    def _record():
        data['E'].append(atoms.get_total_energy())     # potential + kinetic
        data['T'].append(atoms.get_temperature())
    dyn.attach(_record, interval=sample_interval)

    frames = []
    if traj_path is not None:
        dyn.attach(lambda: frames.append(atoms.copy()), interval=traj_interval)

    dyn.run(n_prod)

    if traj_path is not None and frames:
        write(traj_path, frames, format='extxyz')      # Ovito-readable

    return dict(E=np.array(data['E']), T=np.array(data['T']))


# Heat-capacity estimators and references
def cv_fluctuation(E, T):
    """Method 1: C_V = Var(E_tot) / (kB T^2)  [eV/K]."""
    return np.asarray(E).var() / (kB * T * T)


def cv_finite_diff(E_plus, E_minus, dT_measured):
    """Method 2: C_V ~ Delta<E> / Delta T_measured  [eV/K]."""
    return (E_plus - E_minus) / dT_measured


def cv_ideal_gas(N):   return 1.5 * N * kB             # translational gas
def cv_dulong_petit(N): return 3.0 * N * kB            # classical harmonic solid


# Plot helpers
def apply_style(ax):
    ax.set_facecolor("#f7f9fc")
    ax.grid(True, color="#dce3ec", lw=0.7, zorder=0)


def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  saved -> output/{name}.png")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs("output", exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    QUICK = os.environ.get("QUICK_TEST", "0") == "1"

    N, dt, rho = 27, 1.0, 0.5
    friction, taut_fs = 0.01, 100.0
    L = init_box(N, rho)

    if QUICK:
        N_EQUIL, N_PROD = 800, 2000
        T_SWEEP = [100.0, 700.0, 5000.0]
    else:
        N_EQUIL, N_PROD = 12000, 5000
        T_SWEEP = [50.0, 100.0, 300.0, 1000.0, 2500.0, 5000.0]

    # warm up njit
    _ = _morse_forces_njit(build_cluster(N, L, 300.0).get_positions(),
                           L, D_Na, r0_Na, alpha_Na, r_cut)

    print("=" * 70)
    print("Exercise 5 (ASE) - NVT heat capacity of a Na cluster")
    print("=" * 70)
    print(f"  ASE Langevin + NVTBerendsen, numba forces (HAVE_NUMBA={HAVE_NUMBA})")
    print(f"  N={N}  L={L:.3f} Ang  dt={dt} fs  friction={friction} 1/fs  "
          f"taut={taut_fs} fs")
    print(f"  equil={N_EQUIL} steps  prod={N_PROD} steps")
    print(f"  ideal gas   C_V = {cv_ideal_gas(N)/(N*kB):.2f} N kB")
    print(f"  Dulong-Petit C_V = {cv_dulong_petit(N)/(N*kB):.2f} N kB")

    # ── (a) Langevin vs Berendsen at 300 K ───────────────────────────────────
    print("\n[5a] Langevin vs Berendsen NVT at 300 K")
    T_a = 300.0
    r_lan = run_nvt(build_cluster(N, L, T_a, 7), 'langevin', T_a,
                    N_EQUIL, N_PROD, dt, friction=friction, sample_interval=1)
    r_ber = run_nvt(build_cluster(N, L, T_a, 7), 'berendsen', T_a,
                    N_EQUIL, N_PROD, dt, taut_fs=taut_fs, sample_interval=1)
    for nm, r in [("Langevin", r_lan), ("Berendsen", r_ber)]:
        print(f"  {nm:9s}: <T>={r['T'].mean():7.2f} K  sigma_T={r['T'].std():6.2f} K"
              f"  | sigma_E={r['E'].std():.4f} eV")
    print(f"  canonical prediction: sigma_T = T sqrt(2/3N) = "
          f"{T_a*np.sqrt(2/(3*N)):.1f} K")

    tt = np.arange(len(r_lan['E']))
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    apply_style(ax[0]); apply_style(ax[1])
    ax[0].plot(tt, r_ber['T'], color="#27ae60", lw=0.9, label="Berendsen")
    ax[0].plot(tt, r_lan['T'], color="#2c4f8c", lw=0.9, alpha=0.7, label="Langevin")
    ax[0].axhline(T_a, color="k", ls="--", lw=1.0, label=f"$T_0={T_a:.0f}$ K")
    ax[0].set_ylabel("Temperature [K]")
    ax[0].legend(fontsize=9)
    ax[1].plot(tt, r_ber['E'] - r_ber['E'].mean(), color="#27ae60", lw=0.9,
               label="Berendsen $\\delta E$")
    ax[1].plot(tt, r_lan['E'] - r_lan['E'].mean(), color="#2c4f8c", lw=0.9,
               alpha=0.7, label="Langevin $\\delta E$")
    ax[1].set_xlabel("Production sample"); ax[1].set_ylabel("$E-\\langle E\\rangle$ [eV]")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); save(fig, "ex5a_ase_thermostats_300K")

    # ── (b) C_V at 5000 K, Method 1, both thermostats ────────────────────────
    print("\n[5b] Method 1 at 5000 K")
    T_b = 5000.0
    rb_lan = run_nvt(build_cluster(N, L, T_b, 11), 'langevin', T_b,
                     N_EQUIL, N_PROD, dt, friction=friction)
    rb_ber = run_nvt(build_cluster(N, L, T_b, 11), 'berendsen', T_b,
                     N_EQUIL, N_PROD, dt, taut_fs=taut_fs)
    print(f"  Langevin : C_V = {cv_fluctuation(rb_lan['E'],T_b)/(N*kB):.3f} N kB")
    print(f"  Berendsen: C_V = {cv_fluctuation(rb_ber['E'],T_b)/(N*kB):.3f} N kB")
    print(f"  ideal gas: C_V = 1.500 N kB  (cluster is fully evaporated at 5000 K)")

    # ── (c)+(e) Temperature sweep, Method 1, Langevin writes Ovito XYZ ───────
    print("\n[5c/5e] Temperature sweep (Method 1)")
    cv_l, cv_b, E_traces = {}, {}, {}
    for Tt_ in T_SWEEP:
        rl = run_nvt(build_cluster(N, L, Tt_, int(Tt_) + 1), 'langevin', Tt_,
                     N_EQUIL, N_PROD, dt, friction=friction,
                     traj_path=f"output/traj_T{int(Tt_)}K.xyz")
        rb = run_nvt(build_cluster(N, L, Tt_, int(Tt_) + 1), 'berendsen', Tt_,
                     N_EQUIL, N_PROD, dt, taut_fs=taut_fs)
        cv_l[Tt_] = cv_fluctuation(rl['E'], Tt_)
        cv_b[Tt_] = cv_fluctuation(rb['E'], Tt_)
        E_traces[Tt_] = rl['E']
        print(f"  T={Tt_:7.0f} | Lang <T>={rl['T'].mean():7.1f} "
              f"C_V={cv_l[Tt_]/(N*kB):5.2f} | Ber <T>={rb['T'].mean():7.1f} "
              f"C_V={cv_b[Tt_]/(N*kB):5.2f}  (N kB)")

    Ts = np.array(T_SWEEP)
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.plot(Ts, [cv_l[t]/(N*kB) for t in T_SWEEP], 'o-', color="#2c4f8c",
            lw=1.6, ms=7, label="Langevin (Method 1)")
    ax.plot(Ts, [cv_b[t]/(N*kB) for t in T_SWEEP], 's--', color="#27ae60",
            lw=1.6, ms=7, label="Berendsen (Method 1)")
    ax.axhline(1.5, color="#c0392b", ls=":", lw=1.5, label="ideal gas $=1.5\\,Nk_B$")
    ax.axhline(3.0, color="#7a1e8a", ls="-.", lw=1.5, label="Dulong-Petit $=3\\,Nk_B$")
    ax.set_xscale("log"); ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("$C_V/(N k_B)$")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex5ce_ase_cv_vs_T")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(T_SWEEP)))
    for c, Tt_ in zip(cmap, T_SWEEP):
        ax.plot(E_traces[Tt_], lw=0.9, color=c, label=f"{Tt_:.0f} K")
    ax.set_xlabel("Production sample"); ax.set_ylabel("$E_{tot}$ [eV]")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    save(fig, "ex5c_ase_energy_vs_time")

    # ── (f) Method 2 at 300 K and 5000 K, both thermostats ──────────────────
    print("\n[5f] Method 2 (finite differences) at 300 K and 5000 K")
    dT = 100.0
    m2 = {}
    for Tc in [300.0, 5000.0]:
        for thermo in ['langevin', 'berendsen']:
            Em, Tm = {}, {}
            for Tp in [Tc - dT, Tc + dT]:
                kw = dict(friction=friction) if thermo == 'langevin' \
                     else dict(taut_fs=taut_fs)
                r = run_nvt(build_cluster(N, L, Tp, int(Tp) + 3), thermo, Tp,
                            N_EQUIL, N_PROD, dt, **kw)
                Em[Tp] = r['E'].mean(); Tm[Tp] = r['T'].mean()
            dT_meas = Tm[Tc + dT] - Tm[Tc - dT]
            m2[(Tc, thermo)] = cv_finite_diff(Em[Tc + dT], Em[Tc - dT], dT_meas)
            print(f"  T={Tc:6.0f} {thermo:9s}: C_V(M2)={m2[(Tc,thermo)]/(N*kB):5.2f} "
                  f"N kB  (measured dT={dT_meas:6.1f} K)")

    print("\n  Summary (C_V in units of N kB):")
    print(f"  {'T [K]':>7} | {'M1 Lang':>8} {'M1 Ber':>8} | {'M2 Lang':>8} {'M2 Ber':>8}")
    for Tc in [300.0, 5000.0]:
        m1l = cv_l.get(Tc); m1b = cv_b.get(Tc)
        if m1l is None:
            rl = run_nvt(build_cluster(N, L, Tc, int(Tc)+1), 'langevin', Tc,
                         N_EQUIL, N_PROD, dt, friction=friction)
            rb = run_nvt(build_cluster(N, L, Tc, int(Tc)+1), 'berendsen', Tc,
                         N_EQUIL, N_PROD, dt, taut_fs=taut_fs)
            m1l = cv_fluctuation(rl['E'], Tc); m1b = cv_fluctuation(rb['E'], Tc)
        print(f"  {Tc:7.0f} | {m1l/(N*kB):8.2f} {m1b/(N*kB):8.2f} | "
              f"{m2[(Tc,'langevin')]/(N*kB):8.2f} {m2[(Tc,'berendsen')]/(N*kB):8.2f}")

    print("\n" + "=" * 70)
    print("Done. output/ holds the plots and traj_T*K.xyz for Ovito.")
    print("=" * 70)