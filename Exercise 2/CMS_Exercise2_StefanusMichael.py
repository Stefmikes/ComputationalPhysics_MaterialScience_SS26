import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

#  PARAMETERS  (standard H2 values, eV / Å / amu)
m_H  = 1.00794    # mass of hydrogen     [amu]

# Molecule parameters: De [eV], Re [Å], a [Å⁻¹], mu [amu]
molecules = {
    "H2":  (4.52,  0.741, 1.9376, m_H / 2.0),
    "N2":  (9.76,  1.098, 2.688,  14.007 / 2.0),
    "O2":  (5.21,  1.208, 2.870,  16.000 / 2.0),
    "HCl": (4.43,  1.274, 1.869,  (1.008 * 35.45) / (1.008 + 35.45)),
}

# Active molecule — change this to switch between species
De, Re, a, mu = molecules["HCl"]

# Unit conversions
eV_to_J   = 1.60218e-19
amu_to_kg = 1.66054e-27
A_to_m    = 1e-10

time_unit_fs = A_to_m * np.sqrt(amu_to_kg / eV_to_J) * 1e15   # ~10.18 fs

def V_morse(R):
    """Morse potential [eV] as a function of internuclear distance R [Å]."""
    return De * (1 - np.exp(-a * (R - Re)))**2

def F_morse(R):
    """Force on relative coordinate R from Morse potential [eV/Å]."""
    exp_term = np.exp(-a * (R - Re))
    return -2 * De * a * exp_term * (1 - exp_term)

def V_harmonic(R):
    """Harmonic approximation around Re: V = k/2*(R-Re)^2, k = 2*De*a^2."""
    k = 2 * De * a**2
    return 0.5 * k * (R - Re)**2

def velocity_verlet(R, V_vel, dt, mass):
    F1 = F_morse(R)
    R_new  = R + V_vel * dt + 0.5 * F1 / mass * dt**2
    F2 = F_morse(R_new)
    V_new  = V_vel + 0.5 * (F1 + F2) / mass * dt
    return R_new, V_new

def euler(R, V_vel, dt, mass):
    F = F_morse(R)
    R_new  = R + V_vel * dt
    V_new  = V_vel + F / mass * dt
    return R_new, V_new

def run_simulation(delta, dt, n_steps, method="vv", mass=None):
    """
    Run the dimer simulation.
    delta  : initial displacement from Re  [Å]
    dt     : timestep in our natural units (Å*sqrt(amu/eV))
    n_steps: number of integration steps
    method : 'vv' = Velocity-Verlet, 'euler' = Euler
    mass   : reduced mass [amu]; defaults to global mu if not specified
    Returns arrays: t, R, V_vel, E_kin, E_pot, E_tot
    """
    if mass is None:
        mass = mu

    R     = Re + delta
    V_vel = 0.0          # zero initial velocity

    t_arr    = np.zeros(n_steps + 1)
    R_arr    = np.zeros(n_steps + 1)
    Vv_arr   = np.zeros(n_steps + 1)
    Ek_arr   = np.zeros(n_steps + 1)
    Ep_arr   = np.zeros(n_steps + 1)

    R_arr[0]  = R
    Vv_arr[0] = V_vel
    Ep_arr[0] = V_morse(R)
    Ek_arr[0] = 0.5 * mass * V_vel**2

    step_fn = velocity_verlet if method == "vv" else euler

    for i in range(1, n_steps + 1):
        R, V_vel = step_fn(R, V_vel, dt, mass)
        t_arr[i]  = i * dt
        R_arr[i]  = R
        Vv_arr[i] = V_vel
        Ep_arr[i] = V_morse(R)
        Ek_arr[i] = 0.5 * mass * V_vel**2

    E_tot = Ek_arr + Ep_arr
    return t_arr, R_arr, Vv_arr, Ek_arr, Ep_arr, E_tot

def write_ovito_xyz(filename, t_arr, R_arr):
    """
    Write trajectory to extended XYZ file readable by OVITO.
    The dimer is placed symmetrically around the origin on the x-axis.
    """
    with open(filename, "w") as f:
        for i, (t, R) in enumerate(zip(t_arr, R_arr)):
            f.write("2\n")
            f.write(f'Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" '
                    f'Properties=species:S:1:pos:R:3 Time={t:.6f}\n')
            x1 = -R / 2.0
            x2 =  R / 2.0
            f.write(f"H  {x1:.6f}  0.000000  0.000000\n")
            f.write(f"H  {x2:.6f}  0.000000  0.000000\n")
    print(f"OVITO trajectory written to: {filename}")

#  FREQUENCY ESTIMATION  (zero-crossing method)
def estimate_frequency(t_arr, R_arr):
    """Estimate oscillation frequency by counting half-periods via zero crossings of (R - Re)."""
    signal = R_arr - Re
    crossings = []
    for i in range(len(signal) - 1):
        if signal[i] * signal[i+1] < 0:
            # linear interpolation
            t_cross = t_arr[i] - signal[i] * (t_arr[i+1] - t_arr[i]) / (signal[i+1] - signal[i])
            crossings.append(t_cross)
    if len(crossings) < 2:
        return None
    half_periods = np.diff(crossings)
    T = 2 * np.mean(half_periods)
    omega = 2 * np.pi / T
    return omega, T, crossings

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)

    # ── Simulation parameters ──────────────────
    delta  = 0.05     # small initial displacement [Å] (δ << 1, near-harmonic regime)
    dt     = 0.05     # timestep in natural units (≈ 0.5 fs)
    n_steps = 10000   # total steps (~500 natural time units ≈ 5000 fs ≈ 5 ps)

    dt_fs  = dt * time_unit_fs   # timestep in femtoseconds

    print("=" * 60)
    print("H2 Morse Potential Simulation")
    print("=" * 60)
    print(f"Parameters: De={De} eV, Re={Re} Å, a={a} Å⁻¹, μ={mu:.4f} amu")
    print(f"Initial displacement δ = {delta} Å  (R0 = {Re+delta} Å)")
    print(f"Timestep: dt = {dt:.3f} nat. units ≈ {dt_fs:.2f} fs")
    print(f"Total simulation time: {n_steps*dt:.1f} nat. units "
          f"≈ {n_steps*dt_fs/1000:.2f} ps\n")

    # Analytical angular frequency (harmonic limit)
    k_morse  = 2 * De * a**2
    omega_analytic = np.sqrt(k_morse / mu)   # [1 / (Å*sqrt(amu/eV))]
    omega_analytic_THz = omega_analytic / time_unit_fs * 1e3  # to THz (rad/s → THz)
    nu_analytic_THz    = omega_analytic_THz / (2 * np.pi)
    print(f"Analytical harmonic ω = {omega_analytic:.4f} nat. units")
    print(f"                     ≈ {omega_analytic_THz:.2f} rad THz")
    print(f"                     ≈ {nu_analytic_THz:.2f} THz\n")

    # ── 1. Velocity-Verlet run ─────────────────
    t, R, Vv, Ek, Ep, Etot = run_simulation(delta, dt, n_steps, method="vv")

    res = estimate_frequency(t, R)
    if res:
        omega_sim, T_sim, _ = res
        omega_sim_THz = omega_sim / time_unit_fs * 1e3
        print(f"Simulated ω (VV, δ={delta}) = {omega_sim:.4f} nat. units "
              f"≈ {omega_sim_THz:.2f} rad THz")

    # ── 2. Write OVITO trajectory (every 10 steps) ──
    stride = 10
    write_ovito_xyz("output/trajectory.xyz", t[::stride], R[::stride])

    # PLOT 1: Energy conservation + R(t)
    t_fs = t * time_unit_fs   # convert time axis to fs

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("H₂ Dimer – Morse Potential (Velocity-Verlet)", fontsize=13)

    axes[0].plot(t_fs, R, color="steelblue", lw=1)
    axes[0].axhline(Re, color="gray", ls="--", lw=0.8, label=f"$R_e$ = {Re} Å")
    axes[0].set_ylabel("R  [Å]")
    axes[0].legend()
    axes[0].set_title("Internuclear Distance R(t)")

    axes[1].plot(t_fs, Ek, label="Kinetic T",  color="tomato",  lw=1)
    axes[1].plot(t_fs, Ep, label="Potential V", color="royalblue", lw=1)
    axes[1].plot(t_fs, Etot, label="Total E",   color="black",  lw=1.2, ls="--")
    axes[1].set_ylabel("Energy  [eV]")
    axes[1].legend()
    axes[1].set_title("Energy Conservation")

    drift = (Etot - Etot[0]) / np.abs(Etot[0])
    axes[2].plot(t_fs, drift * 100, color="darkgreen", lw=1)
    axes[2].set_ylabel("Rel. drift  [%]")
    axes[2].set_xlabel("Time  [fs]")
    axes[2].set_title("Relative Total Energy Drift")

    plt.tight_layout()
    plt.savefig("output/energy_conservation.png", dpi=150)
    plt.close()
    print("Saved: output/energy_conservation.png")

    # PLOT 2: Euler vs Velocity-Verlet (energy drift)
    t_e, R_e, Vv_e, Ek_e, Ep_e, Etot_e = run_simulation(delta, dt, n_steps, method="euler")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("Euler vs. Velocity-Verlet – Energy Drift Comparison", fontsize=13)

    drift_vv    = (Etot    - Etot[0])    / np.abs(Etot[0])    * 100
    drift_euler = (Etot_e  - Etot_e[0])  / np.abs(Etot_e[0])  * 100

    axes[0].plot(t_fs, drift_vv,    lw=1, label="Velocity-Verlet", color="steelblue")
    axes[0].plot(t_fs, drift_euler, lw=1, label="Euler",           color="tomato")
    axes[0].set_ylabel("Rel. drift  [%]")
    axes[0].set_xlabel("Time  [fs]")
    axes[0].legend()
    axes[0].set_title("Total Energy Drift")

    # Large dt Euler to show instability
    dt_big = 0.5
    t_eb, R_eb, _, _, _, Etot_eb = run_simulation(delta, dt_big, 2000, method="euler")
    t_eb_fs = t_eb * time_unit_fs
    drift_eb = (Etot_eb - Etot_eb[0]) / np.abs(Etot_eb[0]) * 100
    axes[1].plot(t_eb_fs, drift_eb, lw=1, color="tomato", label=f"Euler Δt={dt_big} (×10 larger)")
    axes[1].set_ylabel("Rel. drift  [%]")
    axes[1].set_xlabel("Time  [fs]")
    axes[1].legend()
    axes[1].set_title("Euler with Large Timestep (instability)")

    plt.tight_layout()
    plt.savefig("output/euler_vs_vv.png", dpi=150)
    plt.close()
    print("Saved: output/euler_vs_vv.png")

    # PLOT 3: Morse vs. Harmonic potential + phase space
    R_range = np.linspace(0.3, 2.5, 500)
    V_m   = V_morse(R_range)
    V_h   = V_harmonic(R_range)

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(R_range, V_m, label="Morse",    color="steelblue", lw=2)
    ax1.plot(R_range, V_h, label="Harmonic", color="tomato",    lw=2, ls="--")
    ax1.axhline(De, color="gray", ls=":", lw=1, label=f"$D_e$ = {De} eV")
    E_init = V_morse(Re + delta)
    ax1.axhline(E_init, color="orange", ls="-.", lw=1, label=f"E_total (δ={delta} Å)")
    ax1.set_xlim(0.3, 2.5)
    ax1.set_ylim(-0.5, De * 1.3)
    ax1.set_xlabel("R  [Å]")
    ax1.set_ylabel("V  [eV]")
    ax1.set_title("Morse vs. Harmonic Potential")
    ax1.legend()

    # Phase space: small δ (harmonic-like) vs large δ (anharmonic)
    ax2 = fig.add_subplot(gs[1])

    for delta_ps, col, lbl in [(0.05, "steelblue", "δ=0.05 Å (harmonic)"),
                                (0.40, "tomato",    "δ=0.40 Å (anharmonic)"),
                                (0.80, "darkgreen", "δ=0.80 Å (near-dissociation)")]:
        _, R_ps, Vv_ps, _, _, _ = run_simulation(delta_ps, dt, n_steps, method="vv")
        ax2.plot(R_ps - Re, Vv_ps, lw=0.8, color=col, label=lbl)

    ax2.set_xlabel("R − $R_e$  [Å]")
    ax2.set_ylabel("$\\dot{R}$  [Å / nat. unit]")
    ax2.set_title("Phase Space (R − $R_e$,  $\\dot{R}$)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("output/morse_vs_harmonic_phase_space.png", dpi=150)
    plt.close()
    print("Saved: output/morse_vs_harmonic_phase_space.png")

    # PLOT 4: ω(δ) — frequency vs initial displacement
    deltas = np.linspace(0.02, 1.50, 40)
    omegas_sim  = []
    omegas_kept = []
    deltas_kept = []

    for d in deltas:
        _, R_d, _, _, _, _ = run_simulation(d, dt, n_steps, method="vv")
        res_d = estimate_frequency(t, R_d)
        if res_d:
            omegas_sim.append(res_d[0])
            deltas_kept.append(d)

    # Analytical harmonic ω (constant)
    omega_harm = np.full_like(deltas, omega_analytic)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(deltas_kept, omegas_sim, s=30, color="steelblue", label="Simulation ω(δ)")
    ax.axhline(omega_analytic, color="tomato", ls="--", lw=1.5,
               label=f"Harmonic limit ω = {omega_analytic:.3f}")
    ax.axvline(x=0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Initial displacement δ  [Å]")
    ax.set_ylabel("Angular frequency ω  [nat. units]")
    ax.set_title("Oscillation Frequency ω vs. Initial Displacement δ")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/frequency_vs_delta.png", dpi=150)
    plt.close()
    print("Saved: output/frequency_vs_delta.png")

    print("\n✓ All done! Check the 'output/' folder for plots and trajectory.")
    print(f"  → Load 'output/trajectory.xyz' in OVITO to visualize the dimer motion.")