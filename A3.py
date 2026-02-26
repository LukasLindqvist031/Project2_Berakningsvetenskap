import A2
import numpy as np
import matplotlib.pyplot as plt

def f(theta):
    t_final, u_final, traj = A2.simulate_projectile(theta, dt=0.01)
    return u_final[0] - 2700.0

def newton_method(theta0, tolerance=0.01):
    theta = theta0

    print(f"{'Iteration':>10} | {'θ (degrees)':>12} | {'xN (m)':>10} | {'f(θ)':>12}")
    print("-" * 55)

    i = 0
    while True:
        f_theta = f(theta)

        print(f"{i:>10} | {np.degrees(theta):>12.6f} | {f_theta + 2700:>10.2f} | {f_theta:>12.6f}")

        if abs(f_theta) < tolerance:
            print(f"\nConverged! θ* = {np.degrees(theta):.6f}°")
            break

        h           = 1e-4
        derivative  = (f(theta + h) - f(theta - h)) / (2 * h)
        theta      -= f_theta / derivative
        i          += 1

    return theta

# ── Run Newton's method ───────────────────────────────────────────────────────
theta0        = np.radians(45.0)
theta_optimal = newton_method(theta0)

# ── Plot ──────────────────────────────────────────────────────────────────────
t_final, u_final, traj = A2.simulate_projectile(theta_optimal, dt=0.01)

plt.figure(figsize=(10, 5))
plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label=f'Trajectory θ={np.degrees(theta_optimal):.4f}°')
plt.plot(0, 0, 'go', markersize=10, label='Launch point')
plt.plot(u_final[0], 0, 'ro', markersize=10, label=f'Landing point x={u_final[0]:.1f} m')
plt.axvline(x=2700, color='gray', linestyle='--', label='Target x=2700m')
plt.xlabel('Horizontal distance (m)')
plt.ylabel('Height (m)')
plt.title("Newton's method — optimal firing angle")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()