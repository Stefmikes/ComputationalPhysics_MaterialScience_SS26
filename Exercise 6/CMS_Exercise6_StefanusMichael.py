"""
Computational Physics: Materials Science — Exercise 6 (SS 2026)
Self-diffusion coefficient of a Lennard-Jones fluid — ASE version.

    Einstein relation :  MSD(t) = < |r_i(t) - r_i(0)|^2 >_i  ->  D = slope/6
    Green-Kubo        :  D = (1/3) * Integral_0^inf <v_i(0).v_i(t)>_i dt

Units: ASE-native — energy [eV], length [Ang], mass [u], time via units.fs.
Positions are stored in Ang, velocities converted to Ang/fs, so that D comes
out in Ang^2/fs (1 Ang^2/fs = 1e-5 m^2/s).
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

# ── LJ parameters (Table 1 of the sheet; argon-like) ─────────────────────────
# V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
kB       = units.kB              # 8.617e-5 eV/K
T_REF    = 300.0                 # reference temperature [K]
EPS_LJ   = 0.3 * kB * T_REF      # well depth = 0.3 kB T  -> 7.755e-3 eV
SIGMA_LJ = 3.41                  # particle size sigma    [Ang]  (0.341 nm)
M_AR     = 39.95                 # mass                   [u]    (argon)
R_CUT    = 2.5 * SIGMA_LJ        # standard shifted cutoff[Ang]
DT_FS    = 2.0                   # timestep               [fs]
RHO_STAR = 0.5                   # reduced density rho* = rho sigma^3
FRICTION = 0.0002                # Langevin friction      [1/fs] (weak!)

TAU_LJ_FS = SIGMA_LJ * np.sqrt(M_AR / EPS_LJ) / units.fs   # LJ time [fs]


@njit(cache=True, fastmath=True)
def _lj_forces_njit(pos, L, eps, sigma, rc):
    """Pairwise LJ (12-6) forces + energy, minimum-image PBC, shifted cutoff."""
    N = pos.shape[0]
    forces = np.zeros((N, 3))
    epot = 0.0
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
            # dV/dr = (24 eps / r) (s6 - 2 s12);   fac = (dV/dr)/r
            fac = 24.0 * eps * (s6 - 2.0 * s12) * inv_r2
            fx, fy, fz = fac*dx, fac*dy, fac*dz
            forces[i, 0] += fx; forces[i, 1] += fy; forces[i, 2] += fz
            forces[j, 0] -= fx; forces[j, 1] -= fy; forces[j, 2] -= fz
    return forces, epot


class LJNumba(Calculator):
    """ASE Calculator for the shifted-cutoff LJ potential (numba kernel).
    Assumes a cubic, fully periodic cell (same pattern as Ex5 MorseNumba)."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, eps=EPS_LJ, sigma=SIGMA_LJ, rc=R_CUT, **kwargs):
        super().__init__(**kwargs)
        self.eps, self.sigma, self.rc = eps, sigma, rc

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        L = float(atoms.cell.lengths()[0])
        forces, epot = _lj_forces_njit(
            atoms.get_positions(), L, self.eps, self.sigma, self.rc)
        self.results['energy'] = epot
        self.results['forces'] = forces


# ── System construction ──────────────────────────────────────────────────────
def box_length(N, rho_star=RHO_STAR, sigma=SIGMA_LJ):
    """Cubic side length [Ang] for N atoms at reduced density rho*=rho sigma^3."""
    return (N * sigma**3 / rho_star) ** (1.0 / 3.0)


def build_fluid(N, L, T0, seed=42):
    """N-atom simple-cubic lattice filling the periodic box (melts quickly),
    LJ calculator attached, Maxwell-Boltzmann velocities at T0."""
    n = round(N ** (1.0 / 3.0))
    assert n ** 3 == N, "N must be a perfect cube (125, 216, 343, 512, ...)."
    a = L / n                                          # lattice spacing
    idx = np.arange(n)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing='ij')
    pos = (np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]) + 0.5) * a

    atoms = Atoms(f'Ar{N}', positions=pos, cell=[L, L, L], pbc=True)
    atoms.calc = LJNumba()
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=rng)
    Stationary(atoms)
    return atoms


# ── MD runner: Langevin NVT, frequent saving of positions AND velocities ─────
def run_md(atoms, T0, n_equil, n_prod, dt=DT_FS, friction=FRICTION,
           sample_interval=2, traj_path=None, traj_interval=500,
           record_equil=False):
    """
    Equilibrate n_equil steps, then record every sample_interval steps during
    n_prod production steps:
        pos_u : unwrapped positions [Ang]   (ASE never wraps -> raw positions)
        pos_w : wrapped positions   [Ang]   (the deliberate (a) "mistake")
        vel   : velocities          [Ang/fs]
    Returns dict with arrays of shape (n_frames, N, 3) plus t [fs] and
    monitoring traces.
    """
    dyn = Langevin(atoms, timestep=dt * units.fs, temperature_K=T0,
                   friction=friction / units.fs, fixcm=False)

    eq = {'T': [], 'E': []}
    if record_equil:
        dyn.attach(lambda: (eq['T'].append(atoms.get_temperature()),
                            eq['E'].append(atoms.get_total_energy())),
                   interval=10)
    dyn.run(n_equil)
    dyn.observers.clear()

    data = {'pos_u': [], 'pos_w': [], 'vel': [], 'T': []}
    def _record():
        data['pos_u'].append(atoms.get_positions())            # unwrapped
        data['pos_w'].append(atoms.get_positions(wrap=True))   # wrapped
        data['vel'].append(atoms.get_velocities() * units.fs)  # -> Ang/fs
        data['T'].append(atoms.get_temperature())
    dyn.attach(_record, interval=sample_interval)

    frames = []
    if traj_path is not None:
        dyn.attach(lambda: frames.append(atoms.copy()), interval=traj_interval)

    _record()                                          # frame at t=0
    dyn.run(n_prod)

    if traj_path is not None and frames:
        write(traj_path, frames, format='extxyz')      # Ovito-readable

    out = {k: np.array(v) for k, v in data.items()}
    out['t'] = np.arange(out['pos_u'].shape[0]) * sample_interval * dt  # [fs]
    out['eq_T'] = np.array(eq['T']); out['eq_E'] = np.array(eq['E'])
    return out


# ── MSD estimators ───────────────────────────────────────────────────────────
def msd_single_origin(pos):
    """Naive MSD with the single time origin t=0 (statistically noisy):
    MSD(t) = < |r_i(t) - r_i(0)|^2 >_i ."""
    disp = pos - pos[0]
    return (disp ** 2).sum(axis=2).mean(axis=1)


def msd_fft(pos):
    """
    Multiple-time-origin MSD,
        MSD(m) = < |r_i(k+m) - r_i(k)|^2 >_{i,k},
    evaluated with the FFT (Wiener-Khinchin) algorithm in O(T log T)
    instead of the O(T^2) double loop.  Average over all particles.
    """
    T, Np, _ = pos.shape
    # S2(m) = < r(k+m).r(k) >_k  via FFT along the time axis
    f = np.fft.fft(pos, n=2 * T, axis=0)
    acf = np.fft.ifft(f * f.conj(), axis=0)[:T].real.sum(axis=2)   # (T, Np)
    S2 = acf / (T - np.arange(T))[:, None]
    # S1(m) = < |r(k+m)|^2 + |r(k)|^2 >_k  via the standard recursion
    D = (pos ** 2).sum(axis=2)                                     # (T, Np)
    Dext = np.vstack([D, np.zeros((1, Np))])                       # D[T] = 0
    Q = 2.0 * D.sum(axis=0)
    S1 = np.empty((T, Np))
    for m in range(T):
        Q = Q - Dext[m - 1] - Dext[T - m]
        S1[m] = Q / (T - m)
    return (S1 - 2.0 * S2).mean(axis=1)


def fit_diffusion(t, msd, fmin=0.2, fmax=0.5):
    """Linear fit MSD = 6 D t + c in the lag window [fmin, fmax]*t_max
    (long-enough lags to be diffusive, short enough to be well averaged).
    Returns D [Ang^2/fs] and the fit window mask."""
    mask = (t >= fmin * t[-1]) & (t <= fmax * t[-1])
    slope, icept = np.polyfit(t[mask], msd[mask], 1)
    return slope / 6.0, mask, (slope, icept)


def sweep_D(N, L, T0, n_eq, n_pr, seeds):
    """Einstein D averaged over independent seeds (error bars for the report).
    Returns (mean D, std D [Ang^2/fs], mean measured <T> [K])."""
    Ds, Tms = [], []
    for s in seeds:
        r = run_md(build_fluid(N, L, T0, seed=s), T0, n_eq, n_pr,
                   sample_interval=5)
        D_, _, _ = fit_diffusion(r['t'], msd_fft(r['pos_u']))
        Ds.append(D_); Tms.append(r['T'].mean())
    Ds = np.array(Ds)
    return Ds.mean(), Ds.std(), float(np.mean(Tms))


# ── Green-Kubo ───────────────────────────────────────────────────────────────
def vacf_fft(vel):
    """Normalised-per-origin VACF  C(m) = < v_i(k).v_i(k+m) >_{i,k}
    via FFT, averaged over particles. vel shape (T, Np, 3) in Ang/fs."""
    T = vel.shape[0]
    f = np.fft.fft(vel, n=2 * T, axis=0)
    acf = np.fft.ifft(f * f.conj(), axis=0)[:T].real.sum(axis=2)   # (T, Np)
    acf /= (T - np.arange(T))[:, None]
    return acf.mean(axis=1)


def green_kubo_D(vacf, dt_fs):
    """Cumulative D(t) = (1/3) Integral_0^t C(t') dt'  [Ang^2/fs],
    trapezoidal rule."""
    integ = np.concatenate(
        [[0.0], np.cumsum(0.5 * (vacf[1:] + vacf[:-1]) * dt_fs)])
    return integ / 3.0


A2FS_TO_M2S = 1.0e-5          # 1 Ang^2/fs = 1e-5 m^2/s
def in_SI(D):                 # pretty-print helper
    return D * A2FS_TO_M2S


# ── Plot helpers (Ex5 style) ─────────────────────────────────────────────────
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

    N0, T0 = 125, 300.0
    if QUICK:
        N_EQ, N_PR = 500, 1000
        EQ_PR_COMBOS = [(200, 500), (500, 1000)]
        RHO_SWEEP = [0.4, 0.5, 0.7]
        SIZE_SWEEP = [125, 216]
        T_SWEEP = [200.0, 300.0, 500.0]
        SEEDS = [11]
    else:
        N_EQ, N_PR = 10000, 10000                     # Table 1
        EQ_PR_COMBOS = [(1000, 2000), (5000, 5000),
                        (10000, 10000), (10000, 30000)]
        RHO_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        SIZE_SWEEP = [125, 216, 343, 512]
        T_SWEEP = [150.0, 200.0, 250.0, 300.0, 400.0, 500.0]
        SEEDS = [11, 22, 33]                          # error bars (3 seeds)

    L0 = box_length(N0)
    # warm up njit
    _ = _lj_forces_njit(build_fluid(N0, L0, T0).get_positions(),
                        L0, EPS_LJ, SIGMA_LJ, R_CUT)

    print("=" * 70)
    print("Exercise 6 (ASE) - self-diffusion in a Lennard-Jones fluid")
    print("=" * 70)
    print(f"  numba={HAVE_NUMBA}  N={N0}  L={L0:.3f} Ang  rho*={RHO_STAR}")
    print(f"  eps={EPS_LJ*1e3:.4f} meV (=0.3 kB*300K)  sigma={SIGMA_LJ} Ang  "
          f"rc={R_CUT:.2f} Ang")
    print(f"  dt={DT_FS} fs  gamma={FRICTION} 1/fs = {FRICTION*1e3:.2f} 1/ps "
          f"(weak vs collision rate; checked empirically in [6c])")
    print(f"  equil={N_EQ}  prod={N_PR} steps (Table 1)")

    # ── main production run (used for a, c) ──────────────────────────────────
    print("\n[run] main production run at 300 K ...")
    main = run_md(build_fluid(N0, L0, T0, seed=7), T0, N_EQ, N_PR,
                  sample_interval=2, traj_path="output/traj_main_300K.xyz",
                  traj_interval=500, record_equil=True)
    t = main['t']                                       # [fs]
    dt_frame = t[1] - t[0]
    print(f"  frames={len(t)}  <T>={main['T'].mean():.1f} K  "
          f"sigma_T={main['T'].std():.1f} K")

    # equilibration monitoring (judge whether n_eq is sufficient)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for a_ in ax: apply_style(a_)
    teq = np.arange(len(main['eq_T'])) * 10 * DT_FS / 1000.0       # [ps]
    ax[0].plot(teq, main['eq_T'], color="#2c4f8c", lw=0.9)
    ax[0].axhline(T0, color="k", ls="--", lw=1.0)
    ax[0].set_ylabel("T [K]")
    ax[1].plot(teq, main['eq_E'], color="#c0392b", lw=0.9)
    ax[1].set_xlabel("equilibration time [ps]"); ax[1].set_ylabel("$E_{tot}$ [eV]")
    fig.suptitle("Equilibration monitoring (Langevin, 300 K)")
    fig.tight_layout(); save(fig, "ex6_equilibration")

    # ── (a) wrapped-coordinate mistake + correct log-log MSD ─────────────────
    print("\n[6a] MSD: wrapped (wrong) vs unwrapped (correct)")
    msd_w = msd_fft(main['pos_w'])
    msd_u = msd_fft(main['pos_u'])
    msd_u_single = msd_single_origin(main['pos_u'])

    # the wrapped MSD saturates near the geometric plateau ~ L^2/4 per dim
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.plot(t/1000, msd_u, color="#2c4f8c", lw=1.6, label="unwrapped (correct)")
    ax.plot(t/1000, msd_w, color="#c0392b", lw=1.6, label="wrapped (wrong)")
    ax.axhline(L0**2 / 2, color="#c0392b", ls=":", lw=1.2,
               label="$L^2/2$ saturation level")
    ax.set_xlabel("lag time $t$ [ps]"); ax.set_ylabel("MSD [$\\AA^2$]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex6a_wrapped_vs_unwrapped")

    # log-log MSD with ballistic (slope 2) and diffusive (slope 1) guides
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    m = t > 0
    ax.loglog(t[m]/1000, msd_u[m], color="#2c4f8c", lw=1.8,
              label="MSD (multi-origin, FFT)")
    ax.loglog(t[m]/1000, msd_u_single[m], color="#7f8c8d", lw=0.9, alpha=0.7,
              label="MSD (single origin $t_0=0$)")
    tb = t[(t > 0) & (t < 0.2 * TAU_LJ_FS)]
    ax.loglog(tb/1000, msd_u[1]/ (t[1]**2) * tb**2, "k--", lw=1.2,
              label="slope 2 (ballistic)")
    td = t[t > 0.3 * t[-1]]                            # safely diffusive lags
    ref = np.interp(td[0], t, msd_u)
    ax.loglog(td/1000, ref * (td / td[0]), "k:", lw=1.4,
              label="slope 1 (diffusive)")
    ax.set_xlabel("lag time $t$ [ps]"); ax.set_ylabel("MSD [$\\AA^2$]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex6a_msd_loglog")

    # ── (b) Einstein D for several equilibration/production lengths ──────────
    print("\n[6b] Einstein D vs equilibration/production time")
    D_main, mask, (sl, ic) = fit_diffusion(t, msd_u)
    print(f"  main run: D = {D_main:.4e} Ang^2/fs = {in_SI(D_main):.3e} m^2/s")
    res_b = []
    for (ne, npr) in EQ_PR_COMBOS:
        r = run_md(build_fluid(N0, L0, T0, seed=21), T0, ne, npr,
                   sample_interval=5)
        Db, _, _ = fit_diffusion(r['t'], msd_fft(r['pos_u']))
        res_b.append((ne, npr, Db))
        print(f"  n_eq={ne:6d}  n_prod={npr:6d}  ->  D={Db:.4e} Ang^2/fs "
              f"({in_SI(Db):.3e} m^2/s)")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.plot(t/1000, msd_u, color="#2c4f8c", lw=1.6, label="MSD (unwrapped)")
    ax.plot(t[mask]/1000, sl*t[mask]+ic, "k--", lw=1.4,
            label=f"fit: $D={in_SI(D_main):.2e}$ m$^2$/s")
    ax.set_xlabel("lag time $t$ [ps]"); ax.set_ylabel("MSD [$\\AA^2$]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex6b_msd_fit")

    # ── (c) Green-Kubo ───────────────────────────────────────────────────────
    print("\n[6c] Green-Kubo from the VACF")
    C = vacf_fft(main['vel'])
    D_gk_t = green_kubo_D(C, dt_frame)
    # empirical VACF 1/e decay time -> sanity check that the Langevin friction
    # is small compared to the intrinsic collision rate (else D is suppressed)
    i_dec = np.argmax(C < C[0] / np.e)
    t_dec = t[i_dec] if i_dec > 0 else t[-1]
    ratio = 1.0 / (FRICTION * t_dec)
    print(f"  VACF 1/e decay ~ {t_dec:.0f} fs -> collision rate ~ "
          f"{1e3/t_dec:.1f} 1/ps vs gamma = {FRICTION*1e3:.2f} 1/ps "
          f"(ratio {ratio:.0f}x{'': <1}"
          f"{', friction is a weak perturbation' if ratio > 10 else ', WARNING: friction may bias D'})")
    # read off D where the cumulative integral has plateaued: after the VACF
    # has decayed (>> t_dec) but before long-lag noise degrades the integral
    w = (t > max(10 * t_dec, 1000.0)) & (t < 0.3 * t[-1])
    D_gk = D_gk_t[w].mean() if w.any() else D_gk_t[int(0.25 * len(t))]
    print(f"  D(GK) = {D_gk:.4e} Ang^2/fs = {in_SI(D_gk):.3e} m^2/s "
          f"(Einstein: {in_SI(D_main):.3e})")

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    for a_ in ax: apply_style(a_)
    ax[0].plot(t/1000, C / C[0], color="#2c4f8c", lw=1.4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlim(0, min(8 * TAU_LJ_FS / 1000, t[-1]/1000))
    ax[0].set_xlabel("lag time $t$ [ps]")
    ax[0].set_ylabel("VACF $C(t)/C(0)$")
    ax[1].plot(t/1000, in_SI(D_gk_t), color="#27ae60", lw=1.4,
               label="$D_{GK}(t)$ cumulative")
    ax[1].axhline(in_SI(D_main), color="#2c4f8c", ls="--", lw=1.4,
                  label="$D$ (Einstein)")
    ax[1].set_xlabel("upper integration limit $t$ [ps]")
    ax[1].set_ylabel("$D$ [m$^2$/s]")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); save(fig, "ex6c_green_kubo")

    # ── (d) density sweep ────────────────────────────────────────────────────
    print(f"\n[6d] density sweep ({len(SEEDS)} seed(s) each)")
    D_rho, E_rho = [], []
    for rs in RHO_SWEEP:
        Lr = box_length(N0, rho_star=rs)
        Dm, Dsd, _ = sweep_D(N0, Lr, T0, N_EQ, N_PR, SEEDS)
        D_rho.append(Dm); E_rho.append(Dsd)
        print(f"  rho*={rs:.2f}  L={Lr:6.2f} Ang  "
              f"D=({in_SI(Dm):.3e} +- {in_SI(Dsd):.1e}) m^2/s")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.errorbar(RHO_SWEEP, in_SI(np.array(D_rho)), yerr=in_SI(np.array(E_rho)),
                fmt='o-', color="#2c4f8c", lw=1.6, ms=7, capsize=4)
    ax.set_xlabel("reduced density $\\rho^* = \\rho\\sigma^3$")
    ax.set_ylabel("$D$ [m$^2$/s]")
    fig.tight_layout(); save(fig, "ex6d_D_vs_rho")

    # ── (e) box-size sweep at fixed density + 1/L extrapolation ─────────────
    print(f"\n[6e] box-size sweep at fixed rho* "
          f"(Yeh-Hummer 1/L scaling, {len(SEEDS)} seed(s) each)")
    D_L, E_L, L_list = [], [], []
    for Ns in SIZE_SWEEP:
        Ls = box_length(Ns)
        Dm, Dsd, _ = sweep_D(Ns, Ls, T0, N_EQ, N_PR, SEEDS)
        D_L.append(Dm); E_L.append(Dsd); L_list.append(Ls)
        print(f"  N={Ns:4d}  L={Ls:6.2f} Ang  "
              f"D=({in_SI(Dm):.3e} +- {in_SI(Dsd):.1e}) m^2/s")

    invL = 1.0 / np.array(L_list)
    pe = np.polyfit(invL, np.array(D_L), 1)            # D(L) = D_inf + k/L
    D_inf = pe[1]
    print(f"  extrapolated D(L->inf) = {in_SI(D_inf):.3e} m^2/s "
          f"(slope k = {pe[0]:.3e} Ang^3/fs; Yeh-Hummer predicts k<0,"
          f" i.e. D grows with L)")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.errorbar(invL, in_SI(np.array(D_L)), yerr=in_SI(np.array(E_L)),
                fmt='o', color="#2c4f8c", ms=8, capsize=4, label="simulation")
    xx = np.linspace(0, invL.max() * 1.05, 50)
    ax.plot(xx, in_SI(np.polyval(pe, xx)), "k--", lw=1.4,
            label=f"linear fit, $D_\\infty={in_SI(D_inf):.2e}$ m$^2$/s")
    ax.set_xlabel("$1/L$ [$\\AA^{-1}$]"); ax.set_ylabel("$D$ [m$^2$/s]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex6e_D_vs_invL")

    # ── (f) temperature sweep + Arrhenius ────────────────────────────────────
    print(f"\n[6f] temperature sweep + Arrhenius fit ({len(SEEDS)} seed(s) each)")
    D_T, E_T, T_meas = [], [], []
    for Tt in T_SWEEP:
        Dm, Dsd, Tm = sweep_D(N0, L0, Tt, N_EQ, N_PR, SEEDS)
        D_T.append(Dm); E_T.append(Dsd); T_meas.append(Tm)
        print(f"  T={Tt:6.0f} K  <T>={Tm:7.1f} K  "
              f"D=({in_SI(Dm):.3e} +- {in_SI(Dsd):.1e}) m^2/s")

    # fit against the MEASURED mean temperatures, not the nominal targets
    Tm_arr = np.array(T_meas); D_T = np.array(D_T); E_T = np.array(E_T)
    ok = D_T > 0
    pa = np.polyfit(1.0 / Tm_arr[ok], np.log(D_T[ok]), 1)   # ln D = ln D0 - Ea/(kB T)
    Ea = -pa[0] * kB                                        # [eV]
    print(f"  Arrhenius fit (vs <T>): E_A = {Ea*1e3:.2f} meV = {Ea/kB:.0f} K "
          f"({Ea*96.485:.2f} kJ/mol),  D0 = {in_SI(np.exp(pa[1])):.3e} m^2/s")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.errorbar(1000.0 / Tm_arr[ok], in_SI(D_T[ok]), yerr=in_SI(E_T[ok]),
                fmt='o', color="#2c4f8c", ms=8, capsize=4, label="simulation")
    ax.set_yscale("log")
    xx = np.linspace((1000/Tm_arr[ok]).min()*0.95,
                     (1000/Tm_arr[ok]).max()*1.05, 50)
    ax.semilogy(xx, in_SI(np.exp(np.polyval(pa, xx/1000.0))), "k--", lw=1.4,
                label=f"Arrhenius, $E_A={Ea*1e3:.1f}$ meV")
    ax.set_xlabel("$1000/\\langle T\\rangle$ [1/K]"); ax.set_ylabel("$D$ [m$^2$/s]")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex6f_arrhenius")

    print("\n" + "=" * 70)
    print("Done. output/ holds all plots and traj_main_300K.xyz for Ovito.")
    print("=" * 70)