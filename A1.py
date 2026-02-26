import numpy as np
import matplotlib.pyplot as plt


v0 = 40.0                      
theta_deg = 40.0               
theta = np.deg2rad(theta_deg)  

vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

def f(t,y):
    return -4 * np.pi * np.cos(2 * np.pi * t)
#------------------------------------------------------------
# Exact solution for comparison
#------------------------------------------------------------
def exact_solution(t):
    return -2 * np.sin(2 * np.pi * t)

# ── Parameters ────────────────────────────────────────────────────────────────
v0        = 40.0
theta     = np.deg2rad(40.0)
vx0       = v0 * np.cos(theta)
vy0       = v0 * np.sin(theta)

k   = 0.001
a   = 0.02
g   = 9.81
T   = 4.0

start_state = np.array([0.0, 0.0, vx0, vy0])
timesteps   = [1.0, 0.5, 0.25, 0.125]

# ── Linearized ODE (equation 5 in the report) ────────────────────────────────
# State vector: u = [x, y, vx, vy]
def simplified_model(t, u):
    x, y, vx, vy = u
    dx  = vx
    dy  = vy
    dvx = -k * vx + a * np.sin(t)
    dvy = -k * vy - g
    return np.array([dx, dy, dvx, dvy])

# ── Analytical solution (equation 6 in the report) ───────────────────────────
def x_analytical(t):
    denom = (k**2 + 1.0) * k
    term  = ((-k**2 * vx0 - a - vx0) * np.exp(-k * t)
             - np.cos(t) * a * k**2
             - a * k * np.sin(t)
             + (k**2 + 1.0) * (a + vx0))
    return term / denom

def y_analytical(t):
    term = (-vy0 * k - g) * np.exp(-k * t) + (-g * t + vy0) * k + g
    return term / k**2

# ── RK4 integrator ────────────────────────────────────────────────────────────
def runge_kutta_4(f, t0, y0, dt, T):
    """Integrate f from t0 to T with fixed step dt. Returns (times, states)."""
    y0 = y0.copy()
    t_values = [t0]
    y_values = [y0.copy()]
    n_steps  = int(round((T - t0) / dt))

    for _ in range(n_steps):
        k1 = f(t0,          y0)
        k2 = f(t0 + dt/2,   y0 + dt/2 * k1)
        k3 = f(t0 + dt/2,   y0 + dt/2 * k2)
        k4 = f(t0 + dt,     y0 + dt   * k3)
        y0  = y0 + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t0 += dt
        t_values.append(t0)
        y_values.append(y0.copy())

    return np.array(t_values), np.array(y_values)

# ── Convergence study ─────────────────────────────────────────────────────────
def convergence_study():
    errors = []

    print(f"{'Δt':>8} | {'‖eN‖':>12}")
    print("-" * 24)

    for dt in timesteps:
        t_vals, y_vals = runge_kutta_4(simplified_model, 0.0, start_state, dt, T)

        # L2-norm of error at final time T only (as specified in the report)
        xN, yN = y_vals[-1, 0], y_vals[-1, 1]
        x_ex   = x_analytical(T)
        y_ex   = y_analytical(T)
        err    = np.sqrt((x_ex - xN)**2 + (y_ex - yN)**2)
        errors.append(err)
        print(f"{dt:>8.3f} | {err:>12.4e}")

    # Convergence orders p
    print(f"\n{'Δt1':>8} → {'Δt2':>8} | {'p':>6}")
    print("-" * 32)
    for i in range(1, len(timesteps)):
        p = (np.log(errors[i-1] / errors[i])
             / np.log(timesteps[i-1] / timesteps[i]))
        print(f"{timesteps[i-1]:>8.3f} → {timesteps[i]:>8.3f} | {p:>6.2f}")

    # ── Loglog convergence plot ───────────────────────────────────────────────
    h_arr   = np.array(timesteps)
    err_arr = np.array(errors)

    plt.figure(figsize=(7, 5))
    plt.loglog(h_arr, err_arr, 'ko-', linewidth=2, markersize=7, label='RK4 error ‖eN‖')

    # Reference lines h^1 … h^4, scaled to pass near the first data point
    styles = [':', '-.', '--', '-']
    for order, style in zip([1, 2, 3, 4], styles):
        scale = err_arr[0] / h_arr[0]**order
        plt.loglog(h_arr, scale * h_arr**order,
                   linestyle=style, color='gray', label=f'h^{order}')

    plt.xlabel('Step size Δt')
    plt.ylabel('L2 error at t = T')
    plt.title('Convergence study — RK4 on linearised projectile model')
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ── Trajectory plot (finest Δt) ───────────────────────────────────────────
    dt_fine  = timesteps[-1]
    t_vals, y_vals = runge_kutta_4(simplified_model, 0.0, start_state, dt_fine, T)
    t_fine   = np.linspace(0, T, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_vals, y_vals[:, 0], 'b-o', markersize=3, label=f'RK4 (Δt={dt_fine})')
    axes[0].plot(t_fine, x_analytical(t_fine), 'r--', linewidth=2, label='Analytical')
    axes[0].set_xlabel('t [s]'); axes[0].set_ylabel('x [m]')
    axes[0].set_title('Horizontal position x(t)')
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_vals, y_vals[:, 1], 'b-o', markersize=3, label=f'RK4 (Δt={dt_fine})')
    axes[1].plot(t_fine, y_analytical(t_fine), 'r--', linewidth=2, label='Analytical')
    axes[1].set_xlabel('t [s]'); axes[1].set_ylabel('y [m]')
    axes[1].set_title('Vertical position y(t)')
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.suptitle('RK4 vs analytical solution — linearised model')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    convergence_study()

k = 0.001
a = 0.02
g = 9.81
T = 4.0

start_state = np.array([0.0, 0.0, vx0, vy0])

timesteps = [1.0, 0.5, 0.25, 0.125]  

def x_analytical(t, vx=vx0, k=k, a=a):
    denom = (k ** 2 + 1.0) * k
    term = ((-k ** 2 * vx - a - vx) * np.exp(-k * t)) - np.cos(t) * a * k ** 2 - a * k * np.sin(t) + (k ** 2 + 1.0) * (a + vx)
    return term / denom

def y_analytical(t, vy=vy0, k=k, g=g):
    term = (-vy*k - g) * np.exp(-k * t) + (-g*t + vy) * k + g
    return term / (k**2)

def simplified_model(t, u, k=k, a=a, g=g):
    x, y, vx, vy = u
    dx = vx
    dy = vy
    dvx = -k * vx + a * np.sin(t)
    dvy = -k * vy - g
    return np.array([dx, dy, dvx, dvy])






def runge_kutta_4(f, t0, y0, a, b, h):
    t_values = [t0]
    y_values = np.zeros((1, len(y0)))
    y_values[0] = y0
    n_steps = int((b - a) / h)
    for _ in range(n_steps):
        k1 = f(t0, y0)
        k2 = f(t0 + h / 2, y0 + (h / 2) * k1)
        k3 = f(t0 + h / 2, y0 + (h / 2) * k2)
        k4 = f(t0 + h, y0 + h * k3)
        y0 += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += h
        t_values.append(t0)
        y_values = np.vstack([y_values, y0])
    return np.array(t_values), y_values

def convergence_study():
    a = 0
    b = np.pi
    t0 = a
    y0 = np.array([0.0])
    h_values = [b/n for n in [5, 10, 20, 40, 80, 160, 320, 640]]
    errors_rk4 = []
    plt.figure()
    
    for h in h_values:
        # Reset initial conditions for each h
        t_start = a
        y_start = np.array([0.0])
        
        t_values, y_rk4 = runge_kutta_4(f, t_start, y_start, a, b, h)
        plt.plot(t_values, y_rk4[:, 0], label=f'h={h:.2g}')
        
        # Calculate exact solution at the same time points
        y_exact = exact_solution(t_values)
        error = np.max(np.abs(y_rk4[:, 0] - y_exact))
        errors_rk4.append(error)
        print(f"h={h:.2e}: RK4 error={error:.2e}")
    
    
    # Plot exact solution on the solution figure
    y_exact = exact_solution(t_values)
    plt.plot(t_values, y_exact, label='Exact Solution', color='black', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Runge-Kutta 4: Convergence Study')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot convergence (error vs step size)
    plt.figure()
    plt.loglog(h_values, errors_rk4, 'o-', label='RK4 Error')
    plt.loglog(h_values, [h**1 for h in h_values], linestyle='--', label='O(h)')
    plt.loglog(h_values, [h**2 for h in h_values], linestyle='--', label='O(h²)')
    plt.loglog(h_values, [h**3 for h in h_values], linestyle='--', label='O(h³)')
    plt.loglog(h_values, [h**4 for h in h_values], linestyle='--', label='O(h⁴)')
    plt.xlabel('Step size h')
    plt.ylabel('Max Error')
    plt.title('Convergence Study of RK4')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    a = 0
    b = np.pi
    t0 = a
    y0 = np.array([0.0])
    h = 0.1
    t_values, y_rk4 = runge_kutta_4(f, t0, y0, a, b, h)
    # convergence study
    convergence_study()
    # Exact solution for fine plotting
    t_fine = np.linspace(a, b, 100)
    y_ex_fine = exact_solution(t_fine)
    plt.plot(t_values, y_rk4, label='RK4 Method',
    marker='^', markersize=3)
    plt.plot(t_fine, y_ex_fine, label='Exact Solution',
    linestyle='--')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title(f'h={h}')
    plt.legend()
    plt.grid()
    plt.show()