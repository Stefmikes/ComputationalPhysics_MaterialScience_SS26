#!/usr/bin/env python3
"""
Projectile Dynamics: Analytical vs. Numerical Integration
=========================================================
Compares analytical projectile motion with Explicit Euler and Velocity Verlet
numerical integration schemes.

Analyses:
  1. Trajectory & energy comparison (Euler vs Verlet vs Analytical)
  2. Timestep (Δt) convergence study
  3. Detailed energy conservation analysis
  4. Air resistance effects (linear drag model)
  5. Earth vs Mars gravity comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ────────────────────────────────────────────────────────────
# Physical Constants
# ────────────────────────────────────────────────────────────
G_EARTH = 9.80665   # m/s² (standard gravitational acceleration)
G_MARS  = 3.72076   # m/s²

# ────────────────────────────────────────────────────────────
# Default Simulation Parameters
# ────────────────────────────────────────────────────────────
DEFAULT_V0    = 50.0    # m/s  – initial speed
DEFAULT_THETA = 45.0    # deg  – launch angle
DEFAULT_M     = 1.0     # kg   – projectile mass
DEFAULT_DT    = 0.01    # s    – integration timestep
DEFAULT_B     = 0.5     # kg/s – linear drag coefficient


# ════════════════════════════════════════════════════════════
#  ANALYTICAL SOLUTIONS — No Air Resistance
# ════════════════════════════════════════════════════════════

def analytical_trajectory_no_drag(t, v0, theta_deg, g):
    """Return (x, y, vx, vy) arrays for the drag-free analytical solution."""
    theta = np.radians(theta_deg)
    x  = v0 * np.cos(theta) * t
    y  = v0 * np.sin(theta) * t - 0.5 * g * t**2
    vx = v0 * np.cos(theta) * np.ones_like(t, dtype=float)
    vy = v0 * np.sin(theta) - g * t
    return x, y, vx, vy

def analytical_time_of_flight_no_drag(v0, theta_deg, g):
    """T = 2·v0·sin(θ) / g"""
    return 2.0 * v0 * np.sin(np.radians(theta_deg)) / g

def analytical_range_no_drag(v0, theta_deg, g):
    """R = v0²·sin(2θ) / g"""
    return v0**2 * np.sin(2 * np.radians(theta_deg)) / g

def analytical_max_height_no_drag(v0, theta_deg, g):
    """H = (v0·sin(θ))² / (2g)"""
    return (v0 * np.sin(np.radians(theta_deg)))**2 / (2 * g)


# ════════════════════════════════════════════════════════════
#  ANALYTICAL SOLUTIONS — Linear Drag  (F_drag = −b·v)
# ════════════════════════════════════════════════════════════

def analytical_trajectory_linear_drag(t, v0, theta_deg, g, m, b):
    """
    Closed-form solution with linear drag F = −b·v  (γ = b/m).

        vx(t) = v0x · e^(−γt)
        vy(t) = (v0y + g/γ)·e^(−γt) − g/γ
        x(t)  = (v0x/γ)·(1 − e^(−γt))
        y(t)  = (1/γ)·(v0y + g/γ)·(1 − e^(−γt)) − (g/γ)·t
    """
    theta = np.radians(theta_deg)
    v0x, v0y = v0 * np.cos(theta), v0 * np.sin(theta)
    gamma = b / m

    vx = v0x * np.exp(-gamma * t)
    vy = (v0y + g / gamma) * np.exp(-gamma * t) - g / gamma
    x  = (v0x / gamma) * (1.0 - np.exp(-gamma * t))
    y  = (1.0 / gamma) * (v0y + g / gamma) * (1.0 - np.exp(-gamma * t)) \
         - (g / gamma) * t
    return x, y, vx, vy

def analytical_time_of_flight_linear_drag(v0, theta_deg, g, m, b,
                                           t_max=200.0, tol=1e-12):
    """Solve y(T)=0 by bisection for the linear-drag case."""
    theta = np.radians(theta_deg)
    v0y = v0 * np.sin(theta)
    gamma = b / m

    def y_func(t):
        return (1.0 / gamma) * (v0y + g / gamma) * (1.0 - np.exp(-gamma * t)) \
               - (g / gamma) * t

    lo, hi = 1e-8, t_max
    # Make sure the root is bracketed
    while y_func(hi) > 0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if y_func(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ════════════════════════════════════════════════════════════
#  ACCELERATION FUNCTIONS  (pluggable force models)
# ════════════════════════════════════════════════════════════

def acceleration_no_drag(state, g, m, **kw):
    """a = [0, −g]  (vacuum projectile)."""
    return np.array([0.0, -g])

def acceleration_linear_drag(state, g, m, b=0.1, **kw):
    """a = −(b/m)·v − g·ŷ  (linear drag)."""
    gamma = b / m
    return np.array([-gamma * state[2], -gamma * state[3] - g])


# ════════════════════════════════════════════════════════════
#  NUMERICAL INTEGRATORS
# ════════════════════════════════════════════════════════════

def _ground_interpolation(traj, times, i):
    """Linearly interpolate to find the exact ground-strike point."""
    y_prev, y_curr = traj[i - 1, 1], traj[i, 1]
    if y_prev == y_curr:
        return traj[:i + 1], times[:i + 1]
    frac = y_prev / (y_prev - y_curr)
    traj[i, 0] = traj[i - 1, 0] + frac * (traj[i, 0] - traj[i - 1, 0])
    traj[i, 1] = 0.0
    times[i]   = times[i - 1] + frac * (times[i] - times[i - 1])
    return traj[:i + 1], times[:i + 1]


def simulate_euler(accel_func, v0, theta_deg, g, m, dt,
                   t_max=None, **acc_kw):
    """
    Explicit Euler integration.

        r(t+Δt) = r(t) + v(t)·Δt
        v(t+Δt) = v(t) + a(t)·Δt

    Returns (trajectory, times) with trajectory columns [x, y, vx, vy].
    """
    theta = np.radians(theta_deg)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])

    if t_max is None:
        t_max = analytical_time_of_flight_no_drag(v0, theta_deg, g) * 1.5

    N = int(t_max / dt) + 1
    traj  = np.zeros((N, 4))
    times = np.zeros(N)
    traj[0] = state.copy()

    for i in range(1, N):
        a = accel_func(state, g, m, **acc_kw)
        state = np.array([
            state[0] + state[2] * dt,
            state[1] + state[3] * dt,
            state[2] + a[0] * dt,
            state[3] + a[1] * dt,
        ])
        traj[i]  = state
        times[i] = i * dt
        if state[1] < 0 and i > 1:
            return _ground_interpolation(traj, times, i)

    return traj, times


def simulate_verlet(accel_func, v0, theta_deg, g, m, dt,
                    t_max=None, **acc_kw):
    """
    Velocity Verlet integration.

        r(t+Δt) = r(t) + v(t)·Δt + ½·a(t)·Δt²
        v(t+Δt) = v(t) + ½·[a(t) + a(t+Δt)]·Δt

    For velocity-dependent forces a predictor–corrector step is used:
    v_predicted = v(t) + a(t)·Δt  is used to evaluate a(t+Δt),
    then v is corrected with the average acceleration.

    Returns (trajectory, times) with trajectory columns [x, y, vx, vy].
    """
    theta = np.radians(theta_deg)
    state = np.array([0.0, 0.0, v0 * np.cos(theta), v0 * np.sin(theta)])

    if t_max is None:
        t_max = analytical_time_of_flight_no_drag(v0, theta_deg, g) * 1.5

    N = int(t_max / dt) + 1
    traj  = np.zeros((N, 4))
    times = np.zeros(N)
    traj[0] = state.copy()

    a_cur = accel_func(state, g, m, **acc_kw)

    for i in range(1, N):
        # ── position update ──
        new_x = state[0] + state[2] * dt + 0.5 * a_cur[0] * dt**2
        new_y = state[1] + state[3] * dt + 0.5 * a_cur[1] * dt**2

        # ── predict v to evaluate a(t+Δt) ──
        vx_pred = state[2] + a_cur[0] * dt
        vy_pred = state[3] + a_cur[1] * dt
        a_new = accel_func(np.array([new_x, new_y, vx_pred, vy_pred]),
                           g, m, **acc_kw)

        # ── correct velocity ──
        new_vx = state[2] + 0.5 * (a_cur[0] + a_new[0]) * dt
        new_vy = state[3] + 0.5 * (a_cur[1] + a_new[1]) * dt

        state = np.array([new_x, new_y, new_vx, new_vy])
        a_cur = accel_func(state, g, m, **acc_kw)

        traj[i]  = state
        times[i] = i * dt
        if state[1] < 0 and i > 1:
            return _ground_interpolation(traj, times, i)

    return traj, times


# ════════════════════════════════════════════════════════════
#  ENERGY HELPERS
# ════════════════════════════════════════════════════════════

def compute_energy(traj, m, g):
    """Return (KE, PE, Total) arrays from a trajectory."""
    KE = 0.5 * m * (traj[:, 2]**2 + traj[:, 3]**2)
    PE = m * g * traj[:, 1]
    return KE, PE, KE + PE

def numerical_time_of_flight(times, traj):
    """Extract time of flight (first y ≤ 0 crossing after launch)."""
    for i in range(1, len(traj)):
        if traj[i, 1] <= 0 and times[i] > 0:
            yp, yc = traj[i - 1, 1], traj[i, 1]
            if yp != yc:
                frac = yp / (yp - yc)
                return times[i - 1] + frac * (times[i] - times[i - 1])
            return times[i]
    return times[-1]


# ════════════════════════════════════════════════════════════
#  ANALYSIS 1 — Trajectory & Energy Comparison
# ════════════════════════════════════════════════════════════

def analysis_1_trajectory(v0=DEFAULT_V0, theta=DEFAULT_THETA,
                          g=G_EARTH, m=DEFAULT_M, dt=DEFAULT_DT,
                          save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    # analytical
    T_a = analytical_time_of_flight_no_drag(v0, theta, g)
    t_a = np.linspace(0, T_a, 1000)
    xa, ya, vxa, vya = analytical_trajectory_no_drag(t_a, v0, theta, g)
    E0 = 0.5 * m * v0**2

    # numerical
    tr_e, t_e = simulate_euler(acceleration_no_drag, v0, theta, g, m, dt)
    tr_v, t_v = simulate_verlet(acceleration_no_drag, v0, theta, g, m, dt)
    KE_e, PE_e, TE_e = compute_energy(tr_e, m, g)
    KE_v, PE_v, TE_v = compute_energy(tr_v, m, g)

    T_e = numerical_time_of_flight(t_e, tr_e)
    T_v = numerical_time_of_flight(t_v, tr_v)

    # ── console summary ──
    print("=" * 65)
    print("  ANALYSIS 1 — Trajectory & Energy (No Drag)")
    print("=" * 65)
    print(f"  v0={v0} m/s  θ={theta}°  g={g} m/s²  m={m} kg  Δt={dt} s")
    print(f"\n  {'':20s} {'Analytical':>12s} {'Euler':>12s} {'Verlet':>12s}")
    print(f"  {'Time of flight /s':20s} {T_a:12.6f} {T_e:12.6f} {T_v:12.6f}")
    R_a = analytical_range_no_drag(v0, theta, g)
    print(f"  {'Range /m':20s} {R_a:12.6f} {tr_e[-1,0]:12.6f} {tr_v[-1,0]:12.6f}")
    H_a = analytical_max_height_no_drag(v0, theta, g)
    print(f"  {'Max height /m':20s} {H_a:12.6f} {np.max(tr_e[:,1]):12.6f} "
          f"{np.max(tr_v[:,1]):12.6f}")
    print(f"\n  E0 = {E0:.4f} J")
    print(f"  Euler  final E = {TE_e[-1]:.6f} J   ΔE/E0 = {(TE_e[-1]-E0)/E0:+.6e}")
    print(f"  Verlet final E = {TE_v[-1]:.6f} J   ΔE/E0 = {(TE_v[-1]-E0)/E0:+.6e}")

    # ── figure ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Projectile: v₀={v0} m/s, θ={theta}°, g={g} m/s², "
                 f"m={m} kg, Δt={dt} s", fontsize=14, weight='bold')

    ax = axes[0, 0]
    ax.plot(xa, ya, 'k-', lw=2.5, label='Analytical', zorder=3)
    ax.plot(tr_e[:, 0], tr_e[:, 1], 'r--', lw=1.6, label='Euler')
    ax.plot(tr_v[:, 0], tr_v[:, 1], 'b:', lw=2.2, label='Verlet')
    ax.set(xlabel='x  [m]', ylabel='y  [m]', title='(a) Trajectory')
    ax.legend(); ax.grid(True, alpha=.3); ax.set_ylim(bottom=-2)

    ax = axes[0, 1]
    xa_e = np.interp(t_e, t_a, xa);  ya_e = np.interp(t_e, t_a, ya)
    xa_v = np.interp(t_v, t_a, xa);  ya_v = np.interp(t_v, t_a, ya)
    err_e = np.sqrt((tr_e[:, 0]-xa_e)**2 + (tr_e[:, 1]-ya_e)**2)
    err_v = np.sqrt((tr_v[:, 0]-xa_v)**2 + (tr_v[:, 1]-ya_v)**2)
    ax.semilogy(t_e, err_e+1e-30, 'r-', lw=1.5, label='Euler')
    ax.semilogy(t_v, err_v+1e-30, 'b-', lw=1.5, label='Verlet')
    ax.set(xlabel='Time [s]', ylabel='|Δr|  [m]',
           title='(b) Position Error vs Analytical')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    ax = axes[1, 0]
    ax.plot(t_e, TE_e, 'r-', lw=1.8, label='Euler')
    ax.plot(t_v, TE_v, 'b-', lw=1.8, label='Verlet')
    ax.axhline(E0, color='k', ls='--', lw=1.4, label=f'E₀ = {E0:.1f} J')
    ax.set(xlabel='Time [s]', ylabel='Total Energy [J]',
           title='(c) Total Energy E = T + U')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = axes[1, 1]
    ax.plot(t_e, (TE_e - E0)/E0, 'r-', lw=1.8, label='Euler')
    ax.plot(t_v, (TE_v - E0)/E0, 'b-', lw=1.8, label='Verlet')
    ax.set(xlabel='Time [s]', ylabel='(E − E₀) / E₀',
           title='(d) Relative Energy Error')
    ax.legend(); ax.grid(True, alpha=.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_trajectory_energy.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


# ════════════════════════════════════════════════════════════
#  ANALYSIS 2 — Timestep Convergence
# ════════════════════════════════════════════════════════════

def analysis_2_timestep(v0=DEFAULT_V0, theta=DEFAULT_THETA,
                        g=G_EARTH, m=DEFAULT_M, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    dt_vals = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    T_a = analytical_time_of_flight_no_drag(v0, theta, g)
    R_a = analytical_range_no_drag(v0, theta, g)
    E0  = 0.5 * m * v0**2

    err_T_e, err_T_v = [], []
    err_R_e, err_R_v = [], []
    dE_e, dE_v = [], []

    for dt in dt_vals:
        te, tt_e = simulate_euler(acceleration_no_drag, v0, theta, g, m, dt)
        tv, tt_v = simulate_verlet(acceleration_no_drag, v0, theta, g, m, dt)
        _, _, Ee = compute_energy(te, m, g)
        _, _, Ev = compute_energy(tv, m, g)

        err_T_e.append(abs(numerical_time_of_flight(tt_e, te) - T_a))
        err_T_v.append(abs(numerical_time_of_flight(tt_v, tv) - T_a))
        err_R_e.append(abs(te[-1, 0] - R_a))
        err_R_v.append(abs(tv[-1, 0] - R_a))
        dE_e.append(abs(Ee[-1] - E0) / E0)
        dE_v.append(abs(Ev[-1] - E0) / E0)

    # ── console ──
    print("\n" + "=" * 65)
    print("  ANALYSIS 2 — Timestep Convergence")
    print("=" * 65)
    hdr = (f"  {'Δt':>8s} | {'|ΔT| Euler':>11s} {'|ΔT| Verlet':>11s} | "
           f"{'|ΔR| Euler':>11s} {'|ΔR| Verlet':>11s} | "
           f"{'|ΔE/E₀| Euler':>13s} {'|ΔE/E₀| Verlet':>13s}")
    print(hdr)
    print("  " + "-" * 95)
    for i, dt in enumerate(dt_vals):
        print(f"  {dt:>8.4f} | {err_T_e[i]:>11.3e} {err_T_v[i]:>11.3e} | "
              f"{err_R_e[i]:>11.3e} {err_R_v[i]:>11.3e} | "
              f"{dE_e[i]:>13.3e} {dE_v[i]:>13.3e}")

    # ── figure ──
    dt_arr = np.array(dt_vals)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Timestep Convergence Study", fontsize=14, weight='bold')

    ax = axes[0]
    ax.loglog(dt_arr, err_T_e, 'ro-', lw=2, ms=6, label='Euler')
    ax.loglog(dt_arr, err_T_v, 'bs-', lw=2, ms=6, label='Verlet')
    ax.loglog(dt_arr, dt_arr / dt_arr[0] * err_T_e[0], 'r--', alpha=.4,
              label='O(Δt)')
    ax.loglog(dt_arr, dt_arr**2 / dt_arr[0]**2 * err_T_v[0], 'b--', alpha=.4,
              label='O(Δt²)')
    ax.set(xlabel='Δt [s]', ylabel='|ΔT| [s]',
           title='Time-of-Flight Error')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    ax = axes[1]
    ax.loglog(dt_arr, err_R_e, 'ro-', lw=2, ms=6, label='Euler')
    ax.loglog(dt_arr, err_R_v, 'bs-', lw=2, ms=6, label='Verlet')
    ax.loglog(dt_arr, dt_arr / dt_arr[0] * err_R_e[0], 'r--', alpha=.4,
              label='O(Δt)')
    ax.loglog(dt_arr, dt_arr**2 / dt_arr[0]**2 * err_R_v[0], 'b--', alpha=.4,
              label='O(Δt²)')
    ax.set(xlabel='Δt [s]', ylabel='|ΔR| [m]', title='Range Error')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    ax = axes[2]
    ax.loglog(dt_arr, dE_e, 'ro-', lw=2, ms=6, label='Euler')
    ax.loglog(dt_arr, dE_v, 'bs-', lw=2, ms=6, label='Verlet')
    ax.loglog(dt_arr, dt_arr / dt_arr[0] * dE_e[0], 'r--', alpha=.4,
              label='O(Δt)')
    ax.set(xlabel='Δt [s]', ylabel='|ΔE/E₀|', title='Relative Energy Error')
    ax.legend(); ax.grid(True, alpha=.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_timestep_convergence.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


# ════════════════════════════════════════════════════════════
#  ANALYSIS 3 — Energy Deep-Dive
# ════════════════════════════════════════════════════════════

def analysis_3_energy(v0=DEFAULT_V0, theta=DEFAULT_THETA,
                      g=G_EARTH, m=DEFAULT_M, dt=DEFAULT_DT,
                      save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    E0 = 0.5 * m * v0**2
    T_a = analytical_time_of_flight_no_drag(v0, theta, g)

    tr_e, t_e = simulate_euler(acceleration_no_drag, v0, theta, g, m, dt)
    tr_v, t_v = simulate_verlet(acceleration_no_drag, v0, theta, g, m, dt)
    KE_e, PE_e, TE_e = compute_energy(tr_e, m, g)
    KE_v, PE_v, TE_v = compute_energy(tr_v, m, g)

    # Euler energy drift: ΔE ≈ 0.5·m·g²·Δt·T  (linear in time)
    drift_expected = 0.5 * m * g**2 * dt * T_a

    print("\n" + "=" * 65)
    print("  ANALYSIS 3 — Energy Conservation Deep-Dive")
    print("=" * 65)
    print(f"  E₀ = ½·m·v₀² = {E0:.4f} J")
    print(f"\n  Euler (Δt = {dt} s):")
    print(f"    Per-step energy gain ≈ ½·m·g²·Δt² = {0.5*m*g**2*dt**2:.4e} J")
    print(f"    Expected total drift ≈ ½·m·g²·Δt·T = {drift_expected:.4e} J")
    print(f"    Actual final ΔE       = {TE_e[-1]-E0:.4e} J")
    print(f"    Relative error        = {(TE_e[-1]-E0)/E0:.4e}")
    print(f"\n  Verlet (Δt = {dt} s):")
    print(f"    For constant acceleration Verlet is EXACT → E conserved.")
    print(f"    Actual final ΔE       = {TE_v[-1]-E0:.4e} J")
    print(f"    Relative error        = {(TE_v[-1]-E0)/E0:.4e}")
    print(f"\n  Key insight: Euler adds energy systematically (symplecticity")
    print(f"  broken), while Verlet is symplectic and conserves energy exactly")
    print(f"  for constant-acceleration problems.")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Energy Analysis: Euler vs Verlet (No Drag)", fontsize=14,
                 weight='bold')

    ax = axes[0, 0]
    ax.plot(t_e, KE_e, 'r-', lw=1.4, label='KE')
    ax.plot(t_e, PE_e, 'b-', lw=1.4, label='PE')
    ax.plot(t_e, TE_e, 'k-', lw=2, label='Total E')
    ax.axhline(E0, color='grey', ls='--', alpha=.5)
    ax.set(xlabel='Time [s]', ylabel='Energy [J]',
           title='(a) Euler — Energy Components')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = axes[0, 1]
    ax.plot(t_v, KE_v, 'r-', lw=1.4, label='KE')
    ax.plot(t_v, PE_v, 'b-', lw=1.4, label='PE')
    ax.plot(t_v, TE_v, 'k-', lw=2, label='Total E')
    ax.axhline(E0, color='grey', ls='--', alpha=.5)
    ax.set(xlabel='Time [s]', ylabel='Energy [J]',
           title='(b) Verlet — Energy Components')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = axes[1, 0]
    ax.plot(t_e, TE_e, 'r-', lw=2, label='Euler')
    ax.plot(t_v, TE_v, 'b-', lw=2, label='Verlet')
    ax.axhline(E0, color='k', ls='--', lw=1.4, label=f'E₀={E0:.1f} J')
    ax.set(xlabel='Time [s]', ylabel='Total Energy [J]',
           title='(c) Total Energy Comparison')
    ax.legend(); ax.grid(True, alpha=.3)

    ax = axes[1, 1]
    ax.plot(t_e, (TE_e - E0)/E0, 'r-', lw=2, label='Euler (drift ↑)')
    ax.plot(t_v, (TE_v - E0)/E0, 'b-', lw=2, label='Verlet (conserved)')
    ax.set(xlabel='Time [s]', ylabel='(E − E₀) / E₀',
           title='(d) Relative Energy Error')
    ax.legend(); ax.grid(True, alpha=.3)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_energy_detail.png'), dpi=150,
                bbox_inches='tight')
    plt.show()


# ════════════════════════════════════════════════════════════
#  ANALYSIS 4 — Air Resistance
# ════════════════════════════════════════════════════════════

def analysis_4_air_resistance(v0=DEFAULT_V0, theta=DEFAULT_THETA,
                               g=G_EARTH, m=DEFAULT_M, dt=DEFAULT_DT,
                               b_values=None, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    if b_values is None:
        b_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    E0 = 0.5 * m * v0**2

    print("\n" + "=" * 65)
    print("  ANALYSIS 4 — Air Resistance (Linear Drag F = −b·v)")
    print("=" * 65)
    print(f"  v0={v0} m/s  θ={theta}°  g={g} m/s²  m={m} kg  Δt={dt} s")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Effect of Linear Air Resistance  F = −b·v",
                 fontsize=14, weight='bold')
    colors = plt.cm.viridis(np.linspace(0, .9, len(b_values)))

    for idx, b in enumerate(b_values):
        if b == 0:
            afunc, akw = acceleration_no_drag, {}
            T_a = analytical_time_of_flight_no_drag(v0, theta, g)
            ta = np.linspace(0, T_a, 1000)
            xa, ya, _, _ = analytical_trajectory_no_drag(ta, v0, theta, g)
        else:
            afunc, akw = acceleration_linear_drag, {'b': b}
            T_a = analytical_time_of_flight_linear_drag(v0, theta, g, m, b)
            ta = np.linspace(0, T_a, 1000)
            xa, ya, _, _ = analytical_trajectory_linear_drag(
                ta, v0, theta, g, m, b)

        te, tt_e = simulate_euler(afunc, v0, theta, g, m, dt, **akw)
        tv, tt_v = simulate_verlet(afunc, v0, theta, g, m, dt, **akw)
        _, _, Ee = compute_energy(te, m, g)
        _, _, Ev = compute_energy(tv, m, g)

        lbl = f'b={b}'

        # (a) trajectory
        axes[0, 0].plot(xa, ya, color=colors[idx], ls='-', lw=2.2,
                        label=lbl+' anal', zorder=3)
        axes[0, 0].plot(te[:, 0], te[:, 1], color=colors[idx], ls='--',
                        lw=1.2, alpha=.7)
        axes[0, 0].plot(tv[:, 0], tv[:, 1], color=colors[idx], ls=':',
                        lw=1.6, alpha=.7)

        # (b) total energy
        axes[0, 1].plot(tt_e, Ee, color=colors[idx], ls='--', lw=1.5,
                        label=lbl+' Euler')
        axes[0, 1].plot(tt_v, Ev, color=colors[idx], ls=':', lw=2,
                        label=lbl+' Verlet')

        # (c) speed
        sp_e = np.sqrt(te[:, 2]**2 + te[:, 3]**2)
        sp_v = np.sqrt(tv[:, 2]**2 + tv[:, 3]**2)
        axes[0, 2].plot(tt_e, sp_e, color=colors[idx], ls='--', lw=1.5,
                        label=lbl)
        axes[0, 2].plot(tt_v, sp_v, color=colors[idx], ls=':', lw=2)

        # (d) vy
        axes[1, 0].plot(tt_e, te[:, 3], color=colors[idx], ls='--', lw=1.5,
                        label=lbl)
        axes[1, 0].plot(tt_v, tv[:, 3], color=colors[idx], ls=':', lw=2)

        # (e) position error
        xa_i = np.interp(tt_e, ta, xa);  ya_i = np.interp(tt_e, ta, ya)
        err_e = np.sqrt((te[:, 0]-xa_i)**2 + (te[:, 1]-ya_i)**2)
        axes[1, 1].semilogy(tt_e, err_e+1e-30, color=colors[idx], ls='--',
                            lw=1.5, label=lbl+' Euler')
        xa_i2 = np.interp(tt_v, ta, xa); ya_i2 = np.interp(tt_v, ta, ya)
        err_v = np.sqrt((tv[:, 0]-xa_i2)**2 + (tv[:, 1]-ya_i2)**2)
        axes[1, 1].semilogy(tt_v, err_v+1e-30, color=colors[idx], ls=':',
                            lw=2, label=lbl+' Verlet')

        # console
        T_e = numerical_time_of_flight(tt_e, te)
        T_v = numerical_time_of_flight(tt_v, tv)
        print(f"\n  b = {b:.2f} kg/s:")
        print(f"    T_flight: anal={T_a:.4f}  Euler={T_e:.4f}  Verlet={T_v:.4f}")
        print(f"    Range:    anal={xa[-1]:.4f}  Euler={te[-1,0]:.4f}  "
              f"Verlet={tv[-1,0]:.4f}")
        print(f"    E_final:  Euler={Ee[-1]:.4f}  Verlet={Ev[-1]:.4f}  "
              f"(E₀={E0:.4f})")

    # (f) range bar chart
    ax = axes[1, 2]
    R_anal, R_eul, R_ver = [], [], []
    for b in b_values:
        if b == 0:
            R_anal.append(analytical_range_no_drag(v0, theta, g))
            afunc, akw = acceleration_no_drag, {}
        else:
            T_a = analytical_time_of_flight_linear_drag(v0, theta, g, m, b)
            ta = np.linspace(0, T_a, 1000)
            xa, *_ = analytical_trajectory_linear_drag(ta, v0, theta, g, m, b)
            R_anal.append(xa[-1])
            afunc, akw = acceleration_linear_drag, {'b': b}
        te, _ = simulate_euler(afunc, v0, theta, g, m, dt, **akw)
        tv, _ = simulate_verlet(afunc, v0, theta, g, m, dt, **akw)
        R_eul.append(te[-1, 0]);  R_ver.append(tv[-1, 0])

    x_pos = np.arange(len(b_values))
    w = 0.25
    ax.bar(x_pos - w, R_anal, w, color='black', alpha=.7, label='Analytical')
    ax.bar(x_pos,     R_eul,  w, color='red',   alpha=.7, label='Euler')
    ax.bar(x_pos + w, R_ver,  w, color='blue',  alpha=.7, label='Verlet')
    ax.set_xticks(x_pos); ax.set_xticklabels([f'{b:.1f}' for b in b_values])
    ax.set(xlabel='b [kg/s]', ylabel='Range [m]',
           title='(f) Range vs Drag Coefficient')
    ax.legend(); ax.grid(True, alpha=.3)

    for a, xl, yl, ti in [
        (axes[0,0], 'x [m]',        'y [m]',      '(a) Trajectories'),
        (axes[0,1], 'Time [s]',     'Energy [J]',  '(b) Total Energy'),
        (axes[0,2], 'Time [s]',     '|v| [m/s]',   '(c) Speed'),
        (axes[1,0], 'Time [s]',     'vy [m/s]',    '(d) Vertical Velocity'),
        (axes[1,1], 'Time [s]',     '|Δr| [m]',    '(e) Position Error')]:
        a.set(xlabel=xl, ylabel=yl, title=ti)
        a.legend(fontsize=7, ncol=2); a.grid(True, alpha=.3)
    axes[0, 0].set_ylim(bottom=-5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_air_resistance.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print("""
  DISCUSSION — Air Resistance
  ────────────────────────────
  • Drag reduces range, flight time, and max height.
  • The trajectory becomes asymmetric: the descent is steeper
    because drag opposes velocity in both directions, but on the
    way down gravity and drag partially cancel in the y-equation.
  • Total mechanical energy decreases monotonically (dissipation):
    dE/dt = F_drag · v = −b·|v|² ≤ 0.
  • To add quadratic (realistic) drag, replace  −(b/m)·v  with
    −(C_d·ρ·A)/(2m)·|v|·v  where C_d is the drag coefficient,
    ρ the air density, and A the cross-sectional area.
""")


# ════════════════════════════════════════════════════════════
#  ANALYSIS 5 — Earth vs Mars
# ════════════════════════════════════════════════════════════

def analysis_5_mars(v0=DEFAULT_V0, theta=DEFAULT_THETA,
                    m=DEFAULT_M, dt=DEFAULT_DT, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  ANALYSIS 5 — Earth vs Mars")
    print("=" * 65)
    print(f"  v0={v0} m/s  θ={theta}°  m={m} kg  Δt={dt} s")
    print(f"  g_Earth = {G_EARTH:.5f} m/s²   g_Mars = {G_MARS:.5f} m/s²")
    print(f"  Ratio g_Earth/g_Mars = {G_EARTH/G_MARS:.3f}")

    planets = [("Earth", G_EARTH, 'royalblue'), ("Mars", G_MARS, 'orangered')]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Projectile: Earth vs Mars", fontsize=14, weight='bold')

    for name, g, col in planets:
        T_a = analytical_time_of_flight_no_drag(v0, theta, g)
        R_a = analytical_range_no_drag(v0, theta, g)
        H_a = analytical_max_height_no_drag(v0, theta, g)
        ta = np.linspace(0, T_a, 1000)
        xa, ya, _, _ = analytical_trajectory_no_drag(ta, v0, theta, g)

        tv, tt_v = simulate_verlet(acceleration_no_drag, v0, theta, g, m, dt)
        _, _, Ev = compute_energy(tv, m, g)
        E0 = 0.5 * m * v0**2

        print(f"\n  {name}:  T={T_a:.3f} s   R={R_a:.3f} m   H={H_a:.3f} m")

        axes[0, 0].plot(xa, ya, color=col, lw=2.5, label=f'{name} (anal)')
        axes[0, 0].plot(tv[:, 0], tv[:, 1], color=col, ls=':', lw=1.8,
                        alpha=.7, label=f'{name} (Verlet)')

        axes[0, 1].plot(ta, ya, color=col, lw=2.5, label=name)
        axes[0, 1].plot(tt_v, tv[:, 1], color=col, ls=':', lw=1.5, alpha=.7)

        sp_a = np.sqrt((v0*np.cos(np.radians(theta)))**2 *
                       np.ones_like(ta) +
                       (v0*np.sin(np.radians(theta)) - g*ta)**2)
        sp_v = np.sqrt(tv[:, 2]**2 + tv[:, 3]**2)
        axes[1, 0].plot(ta, sp_a, color=col, lw=2.5, label=name)
        axes[1, 0].plot(tt_v, sp_v, color=col, ls=':', lw=1.5, alpha=.7)

        axes[1, 1].plot(tt_v, Ev, color=col, lw=2, label=f'{name} Verlet')
        axes[1, 1].axhline(E0, color=col, ls='--', alpha=.3)

    axes[0, 0].set(xlabel='x [m]', ylabel='y [m]', title='(a) Trajectory')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=.3)
    axes[0, 0].set_ylim(bottom=-5)

    axes[0, 1].set(xlabel='Time [s]', ylabel='y [m]', title='(b) Height vs Time')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=.3)

    axes[1, 0].set(xlabel='Time [s]', ylabel='|v| [m/s]', title='(c) Speed')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=.3)

    axes[1, 1].set(xlabel='Time [s]', ylabel='Total Energy [J]',
                   title='(d) Energy (Verlet)')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_earth_vs_mars.png'), dpi=150,
                bbox_inches='tight')
    plt.show()

    print(f"""
  DISCUSSION — Earth vs Mars
  ───────────────────────────
  On Mars (g ≈ {G_MARS:.2f} m/s²) vs Earth (g ≈ {G_EARTH:.2f} m/s²):

  • Time of flight is ~{G_EARTH/G_MARS:.1f}× longer on Mars.
  • Range and max height are ~{G_EARTH/G_MARS:.1f}× larger on Mars.
  • Initial kinetic energy E₀ = ½mv₀² is the SAME on both planets;
    gravitational PE = mgy differs, but total E is still conserved
    by Verlet on both.
  • The trajectory on Mars is a "stretched" parabola — same shape
    parameter but larger spatial and temporal scales.
  • Speed at landing equals launch speed (no drag), but the
    flight takes longer on Mars.
""")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PROJECTILE DYNAMICS: ANALYTICAL vs NUMERICAL INTEGRATION")
    print("=" * 65)

    analysis_1_trajectory()
    analysis_2_timestep()
    analysis_3_energy()
    analysis_4_air_resistance()
    analysis_5_mars()

    print("\n" + "=" * 65)
    print("  ALL ANALYSES COMPLETE — figures saved to ./figures/")
    print("=" * 65)


if __name__ == "__main__":
    main()