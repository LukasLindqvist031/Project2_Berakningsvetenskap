import numpy as np
import matplotlib.pyplot as plt


v0 = 40.0                      
theta_deg = 40.0               
theta = np.deg2rad(theta_deg)  

vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

k = 0.001
a = 0.02
g = 9.81
T = 4.0

start_state = np.array([0.0, 0.0, vx0, vy0])
timesteps = [1.0, 0.5, 0.25, 0.125]

def simplified_model(t, u):
    x, y, vx, vy = u
    dx = vx
    dy = vy
    dvx = -k * vx + a * np.sin(t)
    dvy = -k * vy - g
    return np.array([dx, dy, dvx, dvy])

def x_analytical(t):
    denom = (k**2 + 1.0) * k
    term = ((-k**2 * vx0 - a - vx0) * np.exp(-k * t)
             - np.cos(t) * a * k**2
             - a * k * np.sin(t)
             + (k**2 + 1.0) * (a + vx0))
    return term / denom

def y_analytical(t):
    term = (-vy0 * k - g) * np.exp(-k * t) + (-g * t + vy0) * k + g
    return term / k**2

def runge_kutta_4(f, t0, y0, dt, T):
    y0 = y0.copy()
    t_values = [t0]
    y_values = [y0.copy()]
    n_steps = int(round((T - t0) / dt))

    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + dt/2, y0 + dt/2 * k1)
        k3 = f(t0 + dt/2, y0 + dt/2 * k2)
        k4 = f(t0 + dt, y0 + dt * k3)
        y0 = y0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t0 += dt
        t_values.append(t0)
        y_values.append(y0.copy())

    return np.array(t_values), np.array(y_values)


errors = []

print(f"{'dt':>8} | {'error':>12}")
print("-" * 24)

for dt in timesteps:
    t_vals, y_vals = runge_kutta_4(simplified_model, 0.0, start_state, dt, T)

    xN = y_vals[-1, 0]
    yN = y_vals[-1, 1]
    x_ex = x_analytical(T)
    y_ex = y_analytical(T)
    err = np.sqrt((x_ex - xN)**2 + (y_ex - yN)**2)
    errors.append(err)
    print(f"{dt:>8.3f} | {err:>12.4e}")

print()
for i in range(1, len(timesteps)):
    p = np.log(errors[i-1] / errors[i]) / np.log(timesteps[i-1] / timesteps[i])
    print(f"p = {p:.2f}")


h_arr = np.array(timesteps)
err_arr = np.array(errors)

plt.figure(figsize=(7, 5))
plt.loglog(h_arr, err_arr, 'ko-', linewidth=2, markersize=7, label='RK4 error ||eN||')

styles = [':', '-.', '--', '-']
for order, style in zip([1, 2, 3, 4], styles):
    scale = err_arr[0] / h_arr[0]**order
    plt.loglog(h_arr, scale * h_arr**order, linestyle=style, color='gray', label=f'h^{order}')

plt.xlabel('Step size dt')
plt.ylabel('L2 error at t = T')
plt.title('Convergence study - RK4 on linearised projectile model')
plt.legend()
plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('loglogdiagram.png', dpi=150)
plt.show()


dt_fine = timesteps[-1]
t_vals, y_vals = runge_kutta_4(simplified_model, 0.0, start_state, dt_fine, T)
t_fine = np.linspace(0, T, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(t_vals, y_vals[:, 0], 'b-o', markersize=3, label=f'RK4 (dt={dt_fine})')
axes[0].plot(t_fine, x_analytical(t_fine), 'r--', linewidth=2, label='Analytical')
axes[0].set_xlabel('t [s]')
axes[0].set_ylabel('x [m]')
axes[0].set_title('Horizontal position x(t)')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

axes[1].plot(t_vals, y_vals[:, 1], 'b-o', markersize=3, label=f'RK4 (dt={dt_fine})')
axes[1].plot(t_fine, y_analytical(t_fine), 'r--', linewidth=2, label='Analytical')
axes[1].set_xlabel('t [s]')
axes[1].set_ylabel('y [m]')
axes[1].set_title('Vertical position y(t)')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.suptitle('RK4 vs analytical solution')
plt.tight_layout()
plt.savefig('RK4_vs_analytical.png', dpi=150)
plt.show()