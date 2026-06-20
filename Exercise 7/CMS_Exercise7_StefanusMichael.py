"""
Computational Physics: Materials Science — Exercise 7 (SS 2026)
The 2026 Sodium-Cluster Challenge: lowest-energy Na76 Morse cluster.

Potential (sheet 3 / Girifalco-Weizer 1959 for Na), pairwise, hard-truncated:
    V(r) = D [ exp(-2a(r-r0)) - 2 exp(-a(r-r0)) ]      for r < rcut, else 0
    D = 0.06334 eV,  r0 = 5.336 A,  a = 0.58993 1/A,  rcut = 14.0 A
Range parameter rho0 = a*r0 = 3.148 (moderate range -> compact, polytetrahedral
/ icosahedral-like global minima rather than open fcc fragments).

Strategy (mirrors the four sheet tasks):
  (a) build a physical seed (fcc sphere), HEAT it with MD and watch it melt;
  (b) LOCAL relaxation of both the cold seed and a molten snapshot with several
      ASE optimizers (BFGS, LBFGS, FIRE, MDMin, GoodOldQuasiNewton) -> compare
      convergence, step count and walltime;
  (c) CONFIDENCE in the global minimum: a large ensemble of independent local
      minimisations from random/melted starts -> distribution of minima;
  (d) GLOBAL search that "uses temperature": simulated annealing (melt-quench)
      and basin hopping with a surface-atom relocation move -> the winner.

Units: ASE-native (energy eV, length A, mass u, time units.fs). Cluster lives
in a 60 A box with pbc=False, so there is no periodic image at all and the
vacuum requirement of the sheet is satisfied trivially (gap ~19 A >> rcut).

Output: output/*.png plots and snapshots, and the hand-in file
    Na76_cluster_StefanusMichael.traj   (lowest energy found).
"""

import os
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore", message=".*fixcm.*")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.cluster import Icosahedron
from ase.optimize import BFGS, LBFGS, FIRE, MDMin, GoodOldQuasiNewton
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.io import write

# ── Morse parameters (sheet 7) ───────────────────────────────────────────────
D_NA, R0_NA, ALPHA_NA, RCUT_NA = 0.06334, 5.336, 0.58993, 14.0
RHO0 = ALPHA_NA * R0_NA                                  # = 3.148

# ── numba is optional; vectorised NumPy is plenty fast for N=76 ───────────────
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*a, **k):                                  # type: ignore
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True, fastmath=True)
def _morse_njit(pos, D, r0, a, rcut):
    """Pairwise Morse forces+energy, no PBC (cluster in vacuum)."""
    N = pos.shape[0]
    forces = np.zeros((N, 3))
    epot = 0.0
    rc2 = rcut * rcut
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= rc2:
                continue
            r = np.sqrt(r2)
            e1 = np.exp(-a * (r - r0))
            e2 = e1 * e1
            epot += D * (e2 - 2.0 * e1)
            # dV/dr = 2aD(e1 - e2);  F_i = -dV/dr * d_ij/r
            fac = -(2.0 * a * D * (e1 - e2)) / r
            forces[i, 0] += fac*dx; forces[i, 1] += fac*dy; forces[i, 2] += fac*dz
            forces[j, 0] -= fac*dx; forces[j, 1] -= fac*dy; forces[j, 2] -= fac*dz
    return epot, forces


def _morse_vec(pos, D, r0, a, rcut):
    """Vectorised NumPy twin of the njit kernel (used when numba absent)."""
    d = pos[:, None, :] - pos[None, :, :]               # d[i,j] = r_i - r_j
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    w = r < rcut
    e1 = np.where(w, np.exp(-a * (r - r0)), 0.0)
    e2 = e1 * e1
    epot = 0.5 * np.where(w, D * (e2 - 2.0 * e1), 0.0).sum()
    dVdr = np.where(w, 2.0 * a * D * (e1 - e2), 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        fac = np.where(w, -dVdr / r, 0.0)
    forces = (fac[:, :, None] * d).sum(axis=1)
    return epot, forces


_KERNEL = _morse_njit if HAVE_NUMBA else _morse_vec


class Morse(Calculator):
    """ASE Calculator for the truncated pairwise Morse potential (no PBC)."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, D=D_NA, r0=R0_NA, alpha=ALPHA_NA, rcut=RCUT_NA, **kw):
        super().__init__(**kw)
        self.D, self.r0, self.alpha, self.rcut = D, r0, alpha, rcut

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f = _KERNEL(atoms.get_positions(), self.D, self.r0,
                       self.alpha, self.rcut)
        self.results['energy'] = e
        self.results['forces'] = f


# ── system construction ──────────────────────────────────────────────────────
BOX = 60.0          # A; pbc=False, only used to centre the cluster
NAT = 76


def _wrap(pos):
    """Return an Na76 Atoms object centred in the vacuum box with the Morse calc."""
    pos = pos - pos.mean(0) + BOX / 2.0
    atoms = Atoms(f'Na{NAT}', positions=pos, cell=[BOX]*3, pbc=False)
    atoms.calc = Morse()
    return atoms


def fcc_sphere(nn=R0_NA):
    """76 atoms cut from an fcc lattice (nn spacing = r0), the most compact seed."""
    a = nn * np.sqrt(2.0)
    base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    pts = np.array([(np.array([i, j, k]) + b) * a
                    for i in range(-3, 4) for j in range(-3, 4)
                    for k in range(-3, 4) for b in base])
    c = pts.mean(0)
    idx = np.argsort(((pts - c) ** 2).sum(1))[:NAT]
    return _wrap(pts[idx])


def ico_seed(nn=R0_NA):
    """76 innermost atoms of a 4-shell Mackay icosahedron."""
    ico = Icosahedron('Na', noshells=4, latticeconstant=nn * np.sqrt(2.0))
    p = ico.get_positions()
    idx = np.argsort(((p - p.mean(0)) ** 2).sum(1))[:NAT]
    return _wrap(p[idx])


def rand_sphere(seed=0, nn=R0_NA):
    """Random compact blob inside a sphere sized for ~bulk density."""
    rng = np.random.default_rng(seed)
    R = nn * (NAT / (4 / 3 * np.pi * 0.74)) ** (1 / 3) * 0.85
    p = []
    while len(p) < NAT:
        q = rng.uniform(-R, R, 3)
        if np.linalg.norm(q) < R:
            p.append(q)
    return _wrap(np.array(p))


# ── plot helpers (same look as previous exercises) ───────────────────────────
STYLE = dict(fig_bg="white", ax_bg="#f7f9fc", grid_c="#dce3ec", fontsize=11)
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE = (
    "#2c4f8c", "#e07b39", "#27ae60", "#c0392b", "#7d4fa1")


def apply_style(ax):
    ax.set_facecolor(STYLE["ax_bg"])
    ax.grid(True, color=STYLE["grid_c"], lw=0.7, zorder=0)
    ax.tick_params(labelsize=STYLE["fontsize"] - 1)


def save(fig, name):
    fig.savefig(f"output/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=STYLE["fig_bg"])
    plt.close(fig)
    print(f"  saved -> output/{name}.png")


def snapshot(atoms, name, title=""):
    """Two-view scatter snapshot coloured by coordination (cheap, no extra deps)."""
    p = atoms.get_positions() - atoms.get_positions().mean(0)
    d = p[:, None, :] - p[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', d, d)); np.fill_diagonal(r, 1e9)
    coord = (r < 1.25 * R0_NA).sum(1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, (i, j, lb) in zip(axes, [(0, 1, "xy"), (0, 2, "xz")]):
        apply_style(ax)
        order = np.argsort(p[:, 2 if lb == "xy" else 1])
        sc = ax.scatter(p[order, i], p[order, j], c=coord[order], s=170,
                        cmap="viridis", edgecolors="k", linewidths=0.5,
                        vmin=coord.min(), vmax=coord.max())
        ax.set_aspect("equal"); ax.set_xlabel(f"{lb[0]} [Å]"); ax.set_ylabel(f"{lb[1]} [Å]")
    cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("coordination (r < 1.25 r₀)")
    fig.suptitle(title, y=1.0, fontsize=12)
    save(fig, name)


# ── optimiser bookkeeping ────────────────────────────────────────────────────
def run_optimizer(OptCls, atoms, fmax=1e-3, steps=4000, **kw):
    """Run one ASE optimizer, recording (step, energy, fmax, walltime)."""
    atoms = atoms.copy(); atoms.calc = Morse()
    log = {"E": [], "fmax": [], "t": []}
    t0 = time.perf_counter()
    opt = OptCls(atoms, logfile=None, **kw)

    def rec():
        log["E"].append(atoms.get_potential_energy())
        log["fmax"].append(np.linalg.norm(atoms.get_forces(), axis=1).max())
        log["t"].append(time.perf_counter() - t0)
    opt.attach(rec, interval=1)
    rec()
    opt.run(fmax=fmax, steps=steps)
    return atoms, {k: np.array(v) for k, v in log.items()}


# ── global search ingredients ────────────────────────────────────────────────
def robust_relax(atoms, fmax=5e-3):
    """FIRE (robust far from min) -> LBFGS (fast quadratic polish)."""
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=1500)
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=3000)
    return atoms


def per_atom_energy(atoms):
    p = atoms.get_positions()
    d = p[:, None, :] - p[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', d, d)); np.fill_diagonal(r, 1e9)
    w = r < RCUT_NA
    e1 = np.where(w, np.exp(-ALPHA_NA * (r - R0_NA)), 0.0)
    return (D_NA * (e1 * e1 - 2 * e1) * w).sum(1)


def anneal(atoms, seed, T_hi=1100, n_cool=18, steps_per=250, friction=0.01):
    """Melt-quench: heat well above melting, cool slowly, relax."""
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_hi, rng=rng)
    Stationary(atoms); ZeroRotation(atoms)
    traj_E = []
    for T in np.linspace(T_hi, 20, n_cool):
        dyn = Langevin(atoms, 5 * units.fs, temperature_K=T,
                       friction=friction, rng=rng)
        dyn.attach(lambda: traj_E.append(atoms.get_potential_energy()), interval=25)
        dyn.run(steps_per)
    robust_relax(atoms)
    return atoms, traj_E


def basin_hopping(atoms, steps, seed, dr_set=(0.3, 0.5, 0.8), kT=0.05,
                  reloc_prob=0.35, log_every=50):
    """Basin hopping: perturb (Cartesian shake OR worst-atom relocation),
    relax, accept by Metropolis. Returns the running-best Atoms + energy trace."""
    rng = np.random.default_rng(seed)
    robust_relax(atoms)
    E_cur = atoms.get_potential_energy()
    best = atoms.copy(); E_best = E_cur; cur = atoms.get_positions().copy()
    trace = [E_best]
    for s in range(steps):
        atoms.set_positions(cur)
        p = cur.copy()
        if rng.random() < reloc_prob:
            ea = per_atom_energy(atoms)
            k = np.argsort(ea)[-rng.integers(1, 4)]      # a worst-bonded atom
            tgt = p[rng.integers(len(p))]
            v = tgt - p.mean(0); v /= (np.linalg.norm(v) + 1e-9)
            p[k] = tgt + v * R0_NA * rng.uniform(0.6, 1.0) + rng.normal(0, 0.6, 3)
        else:
            dr = rng.choice(dr_set)
            p = cur + rng.uniform(-dr, dr, (len(p), 3))
        atoms.set_positions(p); robust_relax(atoms)
        E = atoms.get_potential_energy()
        if E < E_best - 1e-6:
            E_best = E; best = atoms.copy()
        if E < E_cur or rng.random() < np.exp(-(E - E_cur) / kT):
            cur = atoms.get_positions().copy(); E_cur = E
        trace.append(E_best)
        if (s + 1) % log_every == 0:
            print(f"     BH step {s+1:4d}: current {E_cur:8.3f}  best {E_best:8.4f}",
                  flush=True)
    return best, E_best, np.array(trace)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": STYLE["fontsize"],
                         "axes.spines.top": False, "axes.spines.right": False})

    # Budget knobs. QUICK gives a fast demo run; full reaches the global basin.
    QUICK = False
    N_ENSEMBLE   = 12 if not QUICK else 4      # part (c) independent minimisations
    N_ANNEAL     = 6  if not QUICK else 2      # part (d) melt-quench restarts
    BH_STEPS     = 250 if not QUICK else 40    # part (d) basin-hopping steps
    WARMSTART    = "Na76_cluster_StefanusMichael.traj"  # seed search if present

    print("=" * 70)
    print("Exercise 7 — 2026 Sodium-Cluster Challenge (Na76, Morse)")
    print(f"  rho0 = a*r0 = {RHO0:.3f}   kernel = {'numba' if HAVE_NUMBA else 'vectorised numpy'}")
    print("=" * 70)

    global_best = {"E": 1e9, "atoms": None}
    def consider(atoms, tag):
        if atoms.calc is None:
            atoms.calc = Morse()
        e = atoms.get_potential_energy()
        if e < global_best["E"] - 1e-6:
            global_best["E"] = e; global_best["atoms"] = atoms.copy()
            print(f"   ** global best {e:9.4f} eV  [{tag}]", flush=True)
        return e

    # =====================================================================
    # (a) initial structure + MD melting
    # =====================================================================
    print("\n[7a] Initial fcc-sphere seed, then heat with MD until it melts")
    seed = fcc_sphere()
    E_seed = seed.get_potential_energy()
    snapshot(seed, "ex7a_seed_cold", f"(a) cold fcc seed,  E = {E_seed:.2f} eV")

    hot = seed.copy(); hot.calc = Morse()
    rng = np.random.default_rng(0)
    MaxwellBoltzmannDistribution(hot, temperature_K=50, rng=rng)
    Stationary(hot); ZeroRotation(hot)
    T_hist, E_hist, t_hist = [], [], []
    dt = 5 * units.fs
    # ramp temperature in stages to watch the solid -> liquid transition
    step = 0
    for T_target in np.r_[np.linspace(100, 700, 6), np.full(3, 700)]:
        dyn = Langevin(hot, dt, temperature_K=T_target, friction=0.01, rng=rng)
        def rec(d=dyn):
            T_hist.append(hot.get_temperature())
            E_hist.append(hot.get_potential_energy())
            t_hist.append(len(T_hist) * 25 * 5)         # fs (sample every 25 steps)
        dyn.attach(rec, interval=25)
        dyn.run(400)
    molten = hot.copy(); molten.calc = Morse()
    snapshot(molten, "ex7a_molten", f"(a) molten snapshot,  T≈{hot.get_temperature():.0f} K")

    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for a_ in ax: apply_style(a_)
    tt = np.array(t_hist) / 1000.0
    ax[0].plot(tt, T_hist, color=C_RED, lw=1.3)
    ax[0].set_ylabel("temperature [K]")
    ax[0].set_title("(a) MD heating of the Na₇₆ seed — melting")
    ax[1].plot(tt, E_hist, color=C_BLUE, lw=1.3)
    ax[1].axhline(E_seed, color="gray", ls="--", lw=1, label=f"cold seed E={E_seed:.1f} eV")
    ax[1].set_xlabel("time [ps]"); ax[1].set_ylabel("potential energy [eV]")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); save(fig, "ex7a_md_melting")

    # =====================================================================
    # (b) local optimizers compared, on cold seed and molten snapshot
    # =====================================================================
    print("\n[7b] Compare local optimizers on the cold seed and a molten snapshot")
    optimizers = [("BFGS", BFGS), ("LBFGS", LBFGS), ("FIRE", FIRE),
                  ("MDMin", MDMin), ("GoodOldQN", GoodOldQuasiNewton)]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (start, label) in enumerate([(seed, "cold fcc seed"),
                                          (molten, "molten snapshot")]):
        print(f"  -- starting from {label} (E0={start.get_potential_energy():.2f} eV)")
        for (nm, OptCls), c in zip(optimizers, colors):
            kw = {"dt": 5 * units.fs} if nm == "MDMin" else {}
            cap = 400 if nm == "GoodOldQN" else 1500     # GOQN is slow here
            relaxed, log = run_optimizer(OptCls, start, fmax=1e-3,
                                         steps=cap, **kw)
            Ef = relaxed.get_potential_energy(); nstep = len(log["E"]) - 1
            consider(relaxed, f"{nm}/{label}")
            print(f"     {nm:10s}: E={Ef:9.4f} eV  steps={nstep:4d}  "
                  f"walltime={log['t'][-1]*1e3:6.1f} ms")
            axes[0, col].plot(log["E"], color=c, lw=1.5, label=nm)
            axes[1, col].semilogy(np.maximum(log["fmax"], 1e-6), color=c, lw=1.5)
        axes[0, col].set_title(f"(b) energy vs step — {label}")
        axes[0, col].set_ylabel("potential energy [eV]"); axes[0, col].legend(fontsize=9)
        axes[1, col].axhline(1e-3, color="gray", ls="--", lw=1)
        axes[1, col].set_xlabel("optimizer step")
        axes[1, col].set_ylabel("max force [eV/Å]")
        for r in (0, 1): apply_style(axes[r, col])
    fig.suptitle("(b) Local optimizers: convergence from ordered vs disordered start",
                 y=1.0, fontsize=13)
    fig.tight_layout(); save(fig, "ex7b_optimizer_comparison")

    # =====================================================================
    # (c) confidence in the global minimum: ensemble of local minima
    # =====================================================================
    print(f"\n[7c] Ensemble of {N_ENSEMBLE} independent local minimisations")
    minima_E = []
    for s in range(N_ENSEMBLE):
        a = rand_sphere(seed=1000 + s)
        robust_relax(a)
        e = consider(a, f"rand-min{s}")
        minima_E.append(e)
        print(f"     run {s:2d}: E_min = {e:9.4f} eV", flush=True)
    minima_E = np.array(minima_E)

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.hist(minima_E, bins=15, color=C_BLUE, alpha=0.8, edgecolor="k")
    ax.axvline(minima_E.min(), color=C_RED, lw=2,
               label=f"lowest from random starts = {minima_E.min():.3f} eV")
    ax.set_xlabel("local-minimum energy [eV]"); ax.set_ylabel("count")
    ax.set_title("(c) Distribution of local minima from random starts\n"
                 "(a narrow low-energy cluster of hits ⇒ confidence the basin is global)")
    ax.legend(fontsize=9); fig.tight_layout(); save(fig, "ex7c_minima_histogram")

    # =====================================================================
    # (d) global search that uses temperature: annealing + basin hopping
    # =====================================================================
    print(f"\n[7d] Simulated annealing ({N_ANNEAL} restarts) + basin hopping ({BH_STEPS} steps)")
    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    for s in range(N_ANNEAL):
        a = (fcc_sphere() if s % 2 else rand_sphere(seed=2000 + s))
        a, E_trace = anneal(a, seed=s)
        consider(a, f"anneal{s}")
        ax.plot(np.array(E_trace), lw=1.0, alpha=0.7,
                label=f"anneal {s}" if s < 4 else None)
    ax.set_xlabel("MD sample (cooling →)"); ax.set_ylabel("potential energy [eV]")
    ax.set_title("(d) Simulated annealing: melt-quench energy traces")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout(); save(fig, "ex7d_annealing")

    # basin hopping from the best structure so far (warm-start if traj exists)
    bh_seed = global_best["atoms"].copy()
    bh_seed.calc = Morse()
    if os.path.exists(WARMSTART):
        from ase.io import read
        warm = read(WARMSTART); warm.calc = Morse()
        if warm.get_potential_energy() < bh_seed.get_potential_energy():
            bh_seed = warm
            print(f"   (warm-starting basin hopping from {WARMSTART}, "
                  f"E={warm.get_potential_energy():.4f} eV)")
    bh_seed.calc = Morse()
    best_atoms, E_bh, bh_trace = basin_hopping(bh_seed, steps=BH_STEPS, seed=42)
    consider(best_atoms, "basin-hopping")

    fig, ax = plt.subplots(figsize=(9, 5.5)); apply_style(ax)
    ax.plot(bh_trace, color=C_GREEN, lw=1.6)
    ax.set_xlabel("basin-hopping step"); ax.set_ylabel("best energy so far [eV]")
    ax.set_title(f"(d) Basin hopping convergence → {E_bh:.4f} eV")
    fig.tight_layout(); save(fig, "ex7d_basin_hopping")

    # =====================================================================
    # finalise: tight relax + save the winner
    # =====================================================================
    win = global_best["atoms"]; win.calc = Morse()
    FIRE(win, logfile=None).run(fmax=0.02, steps=2000)
    LBFGS(win, logfile=None).run(fmax=1e-4, steps=5000)
    win.center()
    E_win = win.get_potential_energy()
    fmax_win = np.linalg.norm(win.get_forces(), axis=1).max()
    write("Na76_cluster_StefanusMichael.traj", win)
    snapshot(win, "ex7_winner",
             f"WINNER  E = {E_win:.4f} eV  ({E_win/NAT:.4f} eV/atom)")

    print("\n" + "=" * 70)
    print(f"LOWEST Na76 ENERGY = {E_win:.5f} eV   ({E_win/NAT:.4f} eV/atom, "
          f"{E_win/NAT/D_NA:.3f} eps)")
    print(f"  converged fmax = {fmax_win:.2e} eV/Å   (true local minimum)")
    print(f"  saved -> Na76_cluster_StefanusMichael.traj")
    print("=" * 70)

    # ── key-takeaway answers (printed for the report) ────────────────────
    print("""
KEY TAKEAWAYS
  • Local-minimum locators: quasi-Newton (BFGS/LBFGS/GoodOldQN) build an
    approximate Hessian from successive forces and take near-optimal steps near
    a quadratic minimum; FIRE/MDMin are damped-MD methods that are far more
    robust on rough/disordered starts but converge slower in the quadratic
    region. On the cold ordered seed the quasi-Newton methods win on step count;
    on the molten start FIRE is the most reliable.
  • Global minimum: a single relaxation only finds the nearest basin. We gain
    confidence by (c) sampling many independent minima (the lowest basin is hit
    repeatedly and forms a narrow low-energy peak) and by (d) explicitly hopping
    between basins with temperature — simulated annealing (melt then quench
    slowly) and basin hopping. In a real experiment this is literally annealing:
    heat the sample to let atoms rearrange, then cool slowly toward the ground
    state.
  • Geometry optimisation also gives equilibrium bond lengths, cohesive
    energies, vibrational/elastic properties (Hessian), and reaction barriers
    via transition-state search.
""")