"""
Projectile Dynamics: Analytical vs. Numerical Integration
Computational Physics: Materials Science - Exercise 1, SS 2026
Universität Freiburg – Institut für Physik
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class SimConfig:
    """Simulation parameters."""
    mass: float       # kg
    v0: float         # m/s  – initial speed
    theta: float      # degrees – launch angle
    g: float          # m/s^2 – gravitational acceleration
    drag_coeff: float # kg/m  – b in F_drag = -b * v  (set 0 for no drag)
    dt: float         # s     – time step
    t_max: float      # s     – max simulation time

    @property
    def theta_rad(self):
        return np.radians(self.theta)

    @property
    def vx0(self):
        return self.v0 * np.cos(self.theta_rad)

    @property
    def vy0(self):
        return self.v0 * np.sin(self.theta_rad)


# Preset environments
EARTH = dict(g=9.81)
MARS  = dict(g=3.72)


# ─────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────

def acceleration(vx, vy, cfg: SimConfig):
    """Return (ax, ay) including optional linear air resistance."""
    speed = np.sqrt(vx**2 + vy**2)
    if speed > 0 and cfg.drag_coeff > 0:
        ax = -cfg.drag_coeff / cfg.mass * vx
        ay = -cfg.g - cfg.drag_coeff / cfg.mass * vy
    else:
        ax = 0.0
        ay = -cfg.g
    return ax, ay


def kinetic_energy(m, vx, vy):
    return 0.5 * m * (vx**2 + vy**2)


def potential_energy(m, g, y):
    return m * g * y


def total_energy(m, g, x, y, vx, vy):
    return kinetic_energy(m, vx, vy) + potential_energy(m, g, y)


# ─────────────────────────────────────────────
# Numerical integrators
# ─────────────────────────────────────────────

def simulate_euler(cfg: SimConfig):
    """Explicit (forward) Euler integration."""
    x, y   = 0.0, 0.0
    vx, vy = cfg.vx0, cfg.vy0

    xs, ys, vxs, vys, ts = [x], [y], [vx], [vy], [0.0]

    t = 0.0
    while t < cfg.t_max:
        ax, ay = acceleration(vx, vy, cfg)
        x  += vx * cfg.dt
        y  += vy * cfg.dt
        vx += ax * cfg.dt
        vy += ay * cfg.dt
        t  += cfg.dt

        if y < 0 and len(xs) > 1:          # stop just below ground
            break

        xs.append(x); ys.append(y)
        vxs.append(vx); vys.append(vy); ts.append(t)

    return np.array(ts), np.array(xs), np.array(ys), np.array(vxs), np.array(vys)


def simulate_verlet(cfg: SimConfig):
    """Velocity Verlet integration."""
    x, y   = 0.0, 0.0
    vx, vy = cfg.vx0, cfg.vy0

    xs, ys, vxs, vys, ts = [x], [y], [vx], [vy], [0.0]

    t = 0.0
    ax, ay = acceleration(vx, vy, cfg)

    while t < cfg.t_max:
        # Position update
        x += vx * cfg.dt + 0.5 * ax * cfg.dt**2
        y += vy * cfg.dt + 0.5 * ay * cfg.dt**2

        # New acceleration
        ax_new, ay_new = acceleration(vx, vy, cfg)

        # Velocity update with averaged acceleration
        vx += 0.5 * (ax + ax_new) * cfg.dt
        vy += 0.5 * (ay + ay_new) * cfg.dt

        ax, ay = ax_new, ay_new
        t += cfg.dt

        if y < 0 and len(xs) > 1:
            break

        xs.append(x); ys.append(y)
        vxs.append(vx); vys.append(vy); ts.append(t)

    return np.array(ts), np.array(xs), np.array(ys), np.array(vxs), np.array(vys)


# ─────────────────────────────────────────────
# Analytical solution (no drag)
# ─────────────────────────────────────────────

def analytical_trajectory(cfg: SimConfig, t_array):
    """x(t), y(t) from closed-form solution (drag=0)."""
    x = cfg.vx0 * t_array
    y = cfg.vy0 * t_array - 0.5 * cfg.g * t_array**2
    return x, y


def analytical_flight_time(cfg: SimConfig):
    return 2 * cfg.vy0 / cfg.g


def analytical_range(cfg: SimConfig):
    return cfg.v0**2 * np.sin(2 * cfg.theta_rad) / cfg.g


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_trajectory(cfg: SimConfig, title="Trajectory"):
    t_flight = analytical_flight_time(cfg)
    t_arr    = np.linspace(0, t_flight, 500)
    x_a, y_a = analytical_trajectory(cfg, t_arr)

    t_e, x_e, y_e, vx_e, vy_e = simulate_euler(cfg)
    t_v, x_v, y_v, vx_v, vy_v = simulate_verlet(cfg)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_a, y_a,  'k-',  lw=2,   label='Analytical')
    ax.plot(x_e, y_e,  'r--', lw=1.5, label='Euler')
    ax.plot(x_v, y_v,  'b-.',  lw=1.5, label='Velocity Verlet')
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')
    ax.set_title(title); ax.legend(); ax.grid(True)
    plt.tight_layout()
    return fig


def plot_energy(cfg: SimConfig, title="Total Energy vs. Time"):
    t_e, x_e, y_e, vx_e, vy_e = simulate_euler(cfg)
    t_v, x_v, y_v, vx_v, vy_v = simulate_verlet(cfg)

    E0 = total_energy(cfg.mass, cfg.g, 0, 0, cfg.vx0, cfg.vy0)
    E_e = total_energy(cfg.mass, cfg.g, x_e, y_e, vx_e, vy_e)
    E_v = total_energy(cfg.mass, cfg.g, x_v, y_v, vx_v, vy_v)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(E0, color='k', lw=1.5, ls='--', label=f'Analytical E₀ = {E0:.2f} J')
    ax.plot(t_e, E_e, 'r-', lw=1.5, label='Euler')
    ax.plot(t_v, E_v, 'b-', lw=1.5, label='Velocity Verlet')
    ax.set_xlabel('t  [s]'); ax.set_ylabel('E  [J]')
    ax.set_title(title); ax.legend(); ax.grid(True)
    plt.tight_layout()
    return fig


def plot_timestep_study(base_cfg: SimConfig, dt_values):
    """Show how range error scales with Δt for both integrators."""
    t_ref  = analytical_flight_time(base_cfg)
    r_ref  = analytical_range(base_cfg)

    err_euler  = []
    err_verlet = []

    for dt in dt_values:
        cfg = SimConfig(**{**base_cfg.__dict__, 'dt': dt,
                           't_max': 2 * t_ref})
        _, x_e, *_ = simulate_euler(cfg)
        _, x_v, *_ = simulate_verlet(cfg)
        err_euler.append(abs(x_e[-1] - r_ref))
        err_verlet.append(abs(x_v[-1] - r_ref))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog(dt_values, err_euler,  'r-o', label='Euler')
    ax.loglog(dt_values, err_verlet, 'b-s', label='Velocity Verlet')
    ax.set_xlabel('Δt  [s]'); ax.set_ylabel('Range error  [m]')
    ax.set_title('Range error vs. time-step size'); ax.legend(); ax.grid(True, which='both')
    plt.tight_layout()
    return fig


def plot_drag_comparison(cfg_nodrag: SimConfig, drag_coeff: float):
    """Compare trajectories with and without linear air resistance."""
    cfg_drag = SimConfig(**{**cfg_nodrag.__dict__, 'drag_coeff': drag_coeff})

    t_flight = analytical_flight_time(cfg_nodrag)
    t_arr    = np.linspace(0, t_flight, 500)
    x_a, y_a = analytical_trajectory(cfg_nodrag, t_arr)

    _, x_nd, y_nd, *_ = simulate_verlet(cfg_nodrag)
    _, x_d,  y_d,  *_ = simulate_verlet(cfg_drag)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_a,  y_a,  'k-',   lw=2,   label='Analytical (no drag)')
    ax.plot(x_nd, y_nd, 'b--',  lw=1.5, label='Verlet (no drag)')
    ax.plot(x_d,  y_d,  'g-',   lw=1.5, label=f'Verlet (drag b={drag_coeff} kg/m)')
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')
    ax.set_title('Effect of air resistance'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    return fig


def plot_planet_comparison(cfg_earth: SimConfig, cfg_mars: SimConfig):
    t_e_a, x_e_a, *_ = simulate_verlet(cfg_earth)
    t_m_a, x_m_a, *_ = simulate_verlet(cfg_mars)

    _, x_e_v, y_e_v, *_ = simulate_verlet(cfg_earth)
    _, x_m_v, y_m_v, *_ = simulate_verlet(cfg_mars)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_e_v, y_e_v, 'b-', lw=2, label=f'Earth  (g={cfg_earth.g} m/s²)')
    ax.plot(x_m_v, y_m_v, 'r-', lw=2, label=f'Mars   (g={cfg_mars.g} m/s²)')
    ax.set_xlabel('x  [m]'); ax.set_ylabel('y  [m]')
    ax.set_title('Earth vs. Mars trajectory (Velocity Verlet)'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ── Base configuration ──────────────────────────────────────────
    base = SimConfig(
        mass       = 1.0,
        v0         = 30.0,
        theta      = 45.0,
        g          = 9.81,
        drag_coeff = 0.0,
        dt         = 0.05,
        t_max      = 10.0,
    )

    # ── 1. Trajectory & flight time ─────────────────────────────────
    print("=== Analytical results ===")
    print(f"  Flight time : {analytical_flight_time(base):.4f} s")
    print(f"  Range       : {analytical_range(base):.4f} m")

    t_e, x_e, y_e, vx_e, vy_e = simulate_euler(base)
    t_v, x_v, y_v, vx_v, vy_v = simulate_verlet(base)

    print("\n=== Numerical flight times ===")
    print(f"  Euler         : {t_e[-1]:.4f} s   (range: {x_e[-1]:.4f} m)")
    print(f"  Velocity Verlet: {t_v[-1]:.4f} s   (range: {x_v[-1]:.4f} m)")

    # ── 2. Trajectory plot ──────────────────────────────────────────
    fig1 = plot_trajectory(base, "Projectile Trajectory (Earth, Δt=0.05 s)")
    fig1.savefig("trajectory.png", dpi=150)

    # ── 3. Energy plot ──────────────────────────────────────────────
    fig2 = plot_energy(base, "Total Mechanical Energy (Earth)")
    fig2.savefig("energy.png", dpi=150)

    # ── 4. Time-step study ──────────────────────────────────────────
    dt_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    fig3 = plot_timestep_study(base, dt_values)
    fig3.savefig("timestep_study.png", dpi=150)

    # ── 5. Air resistance ───────────────────────────────────────────
    fig4 = plot_drag_comparison(base, drag_coeff=0.1)
    fig4.savefig("air_resistance.png", dpi=150)

    # ── 6. Mars comparison ──────────────────────────────────────────
    mars_cfg = SimConfig(
        mass=base.mass, v0=base.v0, theta=base.theta,
        g=3.72, drag_coeff=0.0, dt=base.dt, t_max=25.0,
    )
    fig5 = plot_planet_comparison(base, mars_cfg)
    fig5.savefig("mars_comparison.png", dpi=150)

    print("\nAll plots saved: trajectory.png, energy.png, timestep_study.png,")
    print("                 air_resistance.png, mars_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()