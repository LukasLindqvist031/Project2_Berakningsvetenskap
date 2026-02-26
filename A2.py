import numpy as np
import matplotlib.pyplot as plt

k0 = 4.518e-4
v0 = 400.0
theta = np.radians(45.0)

def gravitational_func(y):
    return 3.986e14 / (6.371e6 + y)**2

def drag_func(y):
    return k0 * np.exp(-1e-4 * y)

def wind_func(t):                              # fixed: t not y
    return -20.0 * np.exp(-((t - 10) / 5)**2)

def full_model(t, u):                          # fixed: renamed, uses t for wind
    x, y, vx, vy = u
    w      = wind_func(t)                      # fixed: pass t
    g      = gravitational_func(y)
    k      = drag_func(y)
    v_norm = np.sqrt((vx - w)**2 + vy**2)     # fixed: relative velocity

    dx  = vx
    dy  = vy
    dvx = -k * v_norm * (vx - w)              # fixed: correct form
    dvy = -k * v_norm * vy - g

    return np.array([dx, dy, dvx, dvy])

def simulate_projectile(theta, dt):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    u   = np.array([0.0, 0.0, vx0, vy0])
    t   = 0.0

    trajectory = [u.copy()]

    while True:
        k1 = full_model(t,        u)
        k2 = full_model(t + dt/2, u + dt/2 * k1)
        k3 = full_model(t + dt/2, u + dt/2 * k2)
        k4 = full_model(t + dt,   u + dt   * k3)
        u += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        trajectory.append(u.copy())

        if u[1] < 0.05:
            break

    return t, u, np.array(trajectory)

# ── Test different step sizes ─────────────────────────────────────────────────
for dt in [0.1, 0.05, 0.01, 0.005, 0.001]:
    t_final, u_final, traj = simulate_projectile(theta, dt)
    print(f"dt: {dt:.3f}, xN: {u_final[0]:.2f} m, yN: {u_final[1]:.4f} m")

# ── Plot best trajectory ──────────────────────────────────────────────────────
t_final, u_final, traj = simulate_projectile(theta, dt=0.01)  # run again for plot

plt.figure(figsize=(10, 5))
plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
plt.plot(0, 0, 'go', markersize=10, label='Launch point')
plt.plot(u_final[0], 0, 'ro', markersize=10, label=f'Landing point x={u_final[0]:.1f} m')
plt.xlabel('Horizontal distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Trajectory with Air Resistance and Wind')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()