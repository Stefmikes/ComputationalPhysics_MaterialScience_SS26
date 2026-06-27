"""
Computational Physics: Materials Science - Exercise 8 (SS 2026)
Equation of state of a Lennard-Jones gas - ASE version (standalone, based on Ex6).

We add a PRESSURE measurement to the Ex6 LJ MD code and study the equation of state.

    Pressure (sheet eq. 1):   P = (2K + theta) / (3V)
    Virial   (sheet eq. 2):   theta = sum_{i<j} r_ij . f_ij     (minimum image)
    EOS      (sheet eq. 3):   beta P / rho ~= 1 + B2 rho
    B2       (sheet eq. 4):   B2 = -1/2 Integral d^3r ( exp(-beta U(r)) - 1 )
                                 = -2 pi Integral_0^rc ( exp(-beta U(r)) - 1 ) r^2 dr

Production ensemble: Langevin NVT throughout.  Pressure is a STATIC thermodynamic
average, so (unlike the diffusion coefficient in Ex6) it is not biased by a
momentum-damping thermostat; NVT simply holds T = 300 K most stably for a clean <P>.

Units: ASE-native - energy [eV], length [Ang], mass [u], time via units.fs.
Pressure is computed in eV/Ang^3 and converted to bar with the explicit factor
1 eV/Ang^3 = 1.602176634e6 bar.
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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.io import write

# numba (required by the sheet) -----------------------------------------------
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

# LJ parameters (Table 1 of SHEET 8; note sigma CHANGED vs Ex6) ----------------
kB        = units.kB              # 8.617e-5 eV/K
T_REF     = 300.0                 # temperature [K]
EPS_LJ    = 0.3 * kB * T_REF      # well depth = 0.3 kB T -> 7.7553e-3 eV
SIGMA_LJ  = 1.88                  # particle size sigma  [Ang]  (0.188 nm)  <-- Ex8
M_AR      = 39.95                 # mass                 [u]    (argon)
R_CUT     = 2.5 * SIGMA_LJ        # shifted cutoff       [Ang]  (= 4.70 Ang)
DT_FS     = 2.0                   # timestep             [fs]
RHO_START = 0.05                  # starting reduced density rho* = rho sigma^3
FRICTION  = 0.01                  # Langevin friction    [1/fs] (tau = 100 fs)

EV_A3_TO_BAR = 1.602176634e6      # 1 eV/Ang^3 = 1.602176634e6 bar


# ============================================================================
#  Force + energy + VIRIAL kernels
#  Per-pair virial contribution r_ij . f_ij :
#    dx = pos_j - pos_i = -r_ij ;  f_on_i = fac*dx = -fac*r_ij
#    => r_ij . f_ij = (-dx).(fac*dx) = -fac*r^2  = 24 eps (2 s12 - s6)
#  Sign check: attraction (s6 term) -> virial<0 -> lowers P;
#              repulsion (s12 term) -> virial>0 -> raises P.   correct.
# ============================================================================
@njit(cache=True, fastmath=True)
def _lj_kernel_njit(pos, L, eps, sigma, rc):
    N = pos.shape[0]
    forces = np.zeros((N, 3))
    epot = 0.0
    virial = 0.0
    s_rc6 = (sigma / rc) ** 6
    V_rc = 4.0 * eps * (s_rc6 * s_rc6 - s_rc6)         # shift so V(rc)=0
    rc2 = rc * rc
    for i in range(N - 1):
        xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
        for j in range(i + 1, N):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            dx -= L * round(dx / L)                    # minimum image (cubic)
            dy -= L * round(dy / L)
            dz -= L * round(dz / L)
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= rc2:
                continue
            inv_r2 = 1.0 / r2
            s6 = (sigma * sigma * inv_r2) ** 3         # (sigma/r)^6
            s12 = s6 * s6
            epot += 4.0 * eps * (s12 - s6) - V_rc
            fac = 24.0 * eps * (s6 - 2.0 * s12) * inv_r2   # (dV/dr)/r
            fx, fy, fz = fac*dx, fac*dy, fac*dz
            forces[i, 0] += fx; forces[i, 1] += fy; forces[i, 2] += fz
            forces[j, 0] -= fx; forces[j, 1] -= fy; forces[j, 2] -= fz
            virial += -fac * r2                        # r_ij . f_ij summed over i<j
    return forces, epot, virial


def _lj_kernel_vectorized(pos, L, eps, sigma, rc):
    """Vectorised NumPy fallback (numerically identical to the njit kernel)."""
    N = pos.shape[0]
    d = pos[None, :, :] - pos[:, None, :]              # d[i,j] = r_j - r_i
    d -= L * np.round(d / L)
    r2 = np.einsum('ijk,ijk->ij', d, d)
    ii = np.arange(N)
    r2[ii, ii] = np.inf
    within = r2 < rc * rc
    inv_r2 = np.where(within, 1.0 / r2, 0.0)
    s6 = (sigma * sigma * inv_r2) ** 3
    s12 = s6 * s6
    s_rc6 = (sigma / rc) ** 6
    V_rc = 4.0 * eps * (s_rc6 * s_rc6 - s_rc6)
    epot = 0.5 * np.where(within, 4.0 * eps * (s12 - s6) - V_rc, 0.0).sum()
    fac = np.where(within, 24.0 * eps * (s6 - 2.0 * s12) * inv_r2, 0.0)
    forces = (fac[:, :, None] * d).sum(axis=1)
    virial = 0.5 * np.where(within, -fac * r2, 0.0).sum()   # i<j  -> 0.5 * full sum
    return forces, epot, virial


_KERNEL = _lj_kernel_njit if HAVE_NUMBA else _lj_kernel_vectorized


class LJNumba(Calculator):
    """ASE Calculator for the shifted-cutoff LJ potential (cubic PBC)."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, eps=EPS_LJ, sigma=SIGMA_LJ, rc=R_CUT, **kwargs):
        super().__init__(**kwargs)
        self.eps, self.sigma, self.rc = eps, sigma, rc

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        L = float(atoms.cell.lengths()[0])
        forces, epot, virial = _KERNEL(
            atoms.get_positions(), L, self.eps, self.sigma, self.rc)
        self.results['energy'] = epot
        self.results['forces'] = forces
        self.results['virial_scalar'] = virial


# ----------------------------------------------------------------------------
def box_length(N, rho_star, sigma=SIGMA_LJ):
    """Cubic side length [Ang] for N atoms at reduced density rho* = rho sigma^3."""
    return (N * sigma**3 / rho_star) ** (1.0 / 3.0)


def build_fluid(N, L, T0, seed=42):
    n = round(N ** (1.0 / 3.0))
    assert n ** 3 == N, "N must be a perfect cube (125, 216, ...)."
    a = L / n
    idx = np.arange(n)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing='ij')
    pos = (np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) + 0.5) * a
    atoms = Atoms(f'Ar{N}', positions=pos, cell=[L, L, L], pbc=True)
    atoms.calc = LJNumba()
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=rng)
    Stationary(atoms)
    return atoms


def compute_pressure_bar(atoms):
    """Instantaneous pressure P = (2K + theta)/(3V) in bar (sheet eq. 1)."""
    L = float(atoms.cell.lengths()[0])
    _, _, virial = _KERNEL(atoms.get_positions(), L, EPS_LJ, SIGMA_LJ, R_CUT)
    K = atoms.get_kinetic_energy()                     # eV
    V = L ** 3                                         # Ang^3
    P = (2.0 * K + virial) / (3.0 * V)                 # eV/Ang^3
    return P * EV_A3_TO_BAR


# ----------------------------------------------------------------------------
def run_md(atoms, T0, n_eq, n_prod, dt=DT_FS, friction=FRICTION,
           sample_interval=10, traj_path=None, traj_interval=1000,
           record_equil=False):
    """Langevin NVT equilibration, then Langevin NVT production sampling P,T,E."""
    dyn_eq = Langevin(atoms, timestep=dt * units.fs, temperature_K=T0,
                      friction=friction / units.fs, fixcm=False)
    eq = {'T': [], 'E': [], 'P': []}
    if record_equil:
        dyn_eq.attach(lambda: (eq['T'].append(atoms.get_temperature()),
                               eq['E'].append(atoms.get_total_energy()),
                               eq['P'].append(compute_pressure_bar(atoms))),
                      interval=max(1, n_eq // 600))
    dyn_eq.run(n_eq)

    dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=T0,
                   friction=friction / units.fs, fixcm=False)
    data = {'P': [], 'T': [], 'E': []}
    dyn.attach(lambda: (data['P'].append(compute_pressure_bar(atoms)),
                        data['T'].append(atoms.get_temperature()),
                        data['E'].append(atoms.get_total_energy())),
               interval=sample_interval)

    frames = []
    if traj_path is not None:
        dyn.attach(lambda: frames.append(atoms.copy()), interval=traj_interval)

    dyn.run(n_prod)
    if traj_path is not None and frames:
        write(traj_path, frames, format='extxyz')

    out = {k: np.array(v) for k, v in data.items()}
    out['eq_T'] = np.array(eq['T']); out['eq_E'] = np.array(eq['E'])
    out['eq_P'] = np.array(eq['P'])
    return out


# ----------------------------------------------------------------------------
def block_average(x, n_blocks=20):
    """Block-averaged mean and standard error of a (time-correlated) series.
    Samples within a block are correlated; the block MEANS are ~independent,
    so SEM = std(block means)/sqrt(M) is an honest error (sheet part b)."""
    x = np.asarray(x, dtype=float)
    m = len(x) - (len(x) % n_blocks)
    if m < n_blocks:
        return float(x.mean()), float('nan')
    blocks = x[:m].reshape(n_blocks, -1).mean(axis=1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(n_blocks))


def block_error_scan(x, max_blocks=60):
    """SEM vs number of blocks - used to check the error has plateaued."""
    out = []
    for nb in range(4, max_blocks + 1):
        _, sem = block_average(x, nb)
        out.append((nb, sem))
    return np.array(out)


# ----------------------------------------------------------------------------
def mayer_B2_over_sigma3(eps_over_kT, sigma=SIGMA_LJ, rc=R_CUT, n=20000):
    """B2/sigma^3 from the Mayer-f integral of the SHIFTED-cutoff LJ potential
    (consistent with the simulation; the integrand is exactly 0 beyond rc).
        B2 = -2 pi Integral_0^rc ( exp(-beta U) - 1 ) r^2 dr
    beta U depends only on eps/kT, so the result is a function of eps/kT alone."""
    r = np.linspace(1.0e-6, rc, n)
    s6 = (sigma / r) ** 6
    s12 = s6 * s6
    s_rc6 = (sigma / rc) ** 6
    shift = (s_rc6 * s_rc6 - s_rc6)
    betaU = 4.0 * eps_over_kT * (s12 - s6 - shift)     # = U_shifted / kT
    f = np.expm1(-betaU)                               # exp(-betaU) - 1, stable
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    integ = _trapz(f * r * r, r)
    B2 = -2.0 * np.pi * integ
    return B2 / sigma**3


# ----------------------------------------------------------------------------
def apply_style(ax):
    ax.set_facecolor("#f7f9fc")
    ax.grid(True, color="#dce3ec", lw=0.7, zorder=0)


def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  saved -> output/{name}.png")


# ============================================================================
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs("output", exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    QUICK = os.environ.get("QUICK_TEST", "0") == "1"
    N0, T0 = 125, 300.0

    if QUICK:
        N_EQ, N_PROD = 1500, 2500
        RHO_SWEEP = [0.05, 0.20, 0.50]
        SEEDS = [11]
    else:
        N_EQ, N_PROD = 30000, 30000                    # Table 1 (sheet 8)
        RHO_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        SEEDS = [11, 22, 33]

    kT = kB * T0                                        # eV
    eps_over_kT_0 = EPS_LJ / kT                         # = 0.3

    print("=" * 70)
    print("Exercise 8 (ASE) - equation of state of a Lennard-Jones gas")
    print("=" * 70)
    print(f"  numba={HAVE_NUMBA}  N={N0}  T={T0} K")
    print(f"  eps={EPS_LJ*1e3:.4f} meV (=0.3 kB*300K)  sigma={SIGMA_LJ} Ang  "
          f"rc={R_CUT:.2f} Ang  eps/kBT={eps_over_kT_0:.3f}")
    print(f"  dt={DT_FS} fs  Langevin NVT gamma={FRICTION} 1/fs (tau={1/FRICTION:.0f} fs)")
    print(f"  equil={N_EQ}  prod={N_PROD} steps (Table 1)")

    # warm up njit
    _L0 = box_length(N0, RHO_START)
    _ = _lj_kernel_njit(build_fluid(N0, _L0, T0).get_positions(),
                        _L0, EPS_LJ, SIGMA_LJ, R_CUT)

    # -- (a) second virial coefficient B2/sigma^3 vs eps/kBT -----------------
    print("\n[8a] B2/sigma^3 from the Mayer-f integral (shifted-cutoff LJ)")
    eks = np.linspace(0.05, 1.0, 80)
    B2s = np.array([mayer_B2_over_sigma3(e) for e in eks])
    B2_0 = mayer_B2_over_sigma3(eps_over_kT_0)
    # Boyle point: where B2 changes sign (linear interp on the grid)
    sign_change = np.where(np.diff(np.sign(B2s)))[0]
    boyle = None
    if len(sign_change):
        k = sign_change[0]
        boyle = eks[k] - B2s[k] * (eks[k+1]-eks[k]) / (B2s[k+1]-B2s[k])
    print(f"  B2/sigma^3 at eps/kBT={eps_over_kT_0:.3f}  ->  {B2_0:+.4f}")
    if boyle is not None:
        print(f"  Boyle point (B2=0) at eps/kBT ~ {boyle:.3f}  (T* = {1/boyle:.3f})")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(eks, B2s, color="#2c4f8c", lw=1.8)
    ax.plot([eps_over_kT_0], [B2_0], 'o', color="#c0392b", ms=8,
            label=f"our state point: $\\varepsilon/k_BT={eps_over_kT_0:.2f}$, "
                  f"$B_2/\\sigma^3={B2_0:+.3f}$")
    if boyle is not None:
        ax.axvline(boyle, color="#27ae60", ls=":", lw=1.3,
                   label=f"Boyle point $\\varepsilon/k_BT\\approx{boyle:.2f}$")
    ax.set_xlabel("$\\varepsilon / k_B T$")
    ax.set_ylabel("$B_2/\\sigma^3$")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex8a_B2_vs_epskT")

    # -- (b) average pressure at the Table-1 density via block averaging ------
    print(f"\n[8b] average pressure at rho*={RHO_START} via block averaging")
    Lb = box_length(N0, RHO_START)
    runb = run_md(build_fluid(N0, Lb, T0, seed=SEEDS[0]), T0, N_EQ, N_PROD,
                  sample_interval=10, record_equil=True,
                  traj_path="output/traj_rho0.05_300K.xyz", traj_interval=max(1, N_PROD//40))
    P_mean, P_err = block_average(runb['P'], n_blocks=20)
    print(f"  frames={len(runb['P'])}  <T>={runb['T'].mean():.1f} K  "
          f"sigma_T={runb['T'].std():.1f} K")
    print(f"  <P> = {P_mean:.3f} +- {P_err:.3f} bar  (block averaging, 20 blocks)")
    scan = block_error_scan(runb['P'])
    print(f"  block-error scan: SEM ranges {scan[:,1].min():.3f}-{scan[:,1].max():.3f} bar "
          f"(plateau indicates decorrelated blocks)")

    # equilibration / production monitoring
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for a_ in ax: apply_style(a_)
    teq = np.arange(len(runb['eq_T']))
    ax[0].plot(teq, runb['eq_T'], color="#2c4f8c", lw=0.9)
    ax[0].axhline(T0, color="k", ls="--", lw=1.0); ax[0].set_ylabel("T [K]")
    ax[1].plot(teq, runb['eq_E'], color="#c0392b", lw=0.9)
    ax[1].set_ylabel("$E_{tot}$ [eV]")
    ax[2].plot(teq, runb['eq_P'], color="#27ae60", lw=0.9)
    ax[2].set_ylabel("P [bar]"); ax[2].set_xlabel("equilibration sample")
    fig.suptitle(f"Equilibration monitoring (Langevin NVT, 300 K, $\\rho^*$={RHO_START})")
    fig.tight_layout(); save(fig, "ex8_equilibration")

    # -- (c) pressure vs density: measured vs eq.(3) prediction ---------------
    print(f"\n[8c] density sweep rho*=0.05..0.50  ({len(SEEDS)} seed(s) each)")
    rhos, P_meas, P_meas_err = [], [], []
    for rs in RHO_SWEEP:
        Ls = box_length(N0, rs)
        Ps = []
        for s in SEEDS:
            r = run_md(build_fluid(N0, Ls, T0, seed=s), T0, N_EQ, N_PROD,
                       sample_interval=10)
            pm, _ = block_average(r['P'], n_blocks=20)
            Ps.append(pm)
        rhos.append(rs); P_meas.append(np.mean(Ps))
        P_meas_err.append(np.std(Ps, ddof=1)/np.sqrt(len(Ps)) if len(Ps) > 1 else
                          block_average(r['P'], 20)[1])
        print(f"  rho*={rs:.2f}  L={Ls:6.3f} Ang  <P>={np.mean(Ps):8.2f} bar")

    rhos = np.array(rhos); P_meas = np.array(P_meas); P_meas_err = np.array(P_meas_err)
    # eq.(3) prediction:  P = rho kT (1 + (B2/sigma^3) rho*) ,  rho = rho*/sigma^3
    rho_num = rhos / SIGMA_LJ**3                        # Ang^-3
    P_ideal = rho_num * kT * EV_A3_TO_BAR              # bar
    P_pred = P_ideal * (1.0 + B2_0 * rhos)            # bar
    rr = np.linspace(0.02, 0.52, 100)
    rr_num = rr / SIGMA_LJ**3
    P_ideal_c = rr_num * kT * EV_A3_TO_BAR
    P_pred_c = P_ideal_c * (1.0 + B2_0 * rr)

    # P vs rho*
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.errorbar(rhos, P_meas, yerr=P_meas_err, fmt='o', color="#2c4f8c", ms=7,
                capsize=4, label="simulation (block-averaged)")
    ax.plot(rr, P_pred_c, "k--", lw=1.5, label="virial EOS, eq. (3)")
    ax.plot(rr, P_ideal_c, ":", color="#7f8c8d", lw=1.3, label="ideal gas $\\rho k_BT$")
    ax.set_xlabel("reduced density $\\rho^* = \\rho\\sigma^3$")
    ax.set_ylabel("pressure $P$ [bar]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex8c_eos_P_vs_rho")

    # beta P / rho  vs rho*  (cleanest test of eq. 3: line of slope B2/sigma^3)
    betaP_over_rho_meas = (P_meas / EV_A3_TO_BAR) / (rho_num * kT)
    betaP_over_rho_err = (P_meas_err / EV_A3_TO_BAR) / (rho_num * kT)
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.errorbar(rhos, betaP_over_rho_meas, yerr=betaP_over_rho_err, fmt='o',
                color="#2c4f8c", ms=7, capsize=4, label="simulation")
    ax.plot(rr, 1.0 + B2_0 * rr, "k--", lw=1.5,
            label=f"$1 + (B_2/\\sigma^3)\\,\\rho^*$, slope ${B2_0:+.3f}$")
    ax.axhline(1.0, color="#7f8c8d", ls=":", lw=1.3, label="ideal gas")
    ax.set_xlabel("reduced density $\\rho^* = \\rho\\sigma^3$")
    ax.set_ylabel("$\\beta P / \\rho$")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex8c_betaP_over_rho")

    print("\n" + "=" * 70)
    print("Done. output/ holds all plots and traj_rho0.05_300K.xyz for Ovito.")
    print("=" * 70)
