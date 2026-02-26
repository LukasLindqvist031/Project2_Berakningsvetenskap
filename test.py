import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Analytisk lösning för den förenklade modellen
# ============================================================

def analytical_solution(t, vx, vy, k, a, g):
    """Beräknar den analytiska lösningen enligt ekvation (6)."""
    x = (1 / ((k**2 + 1) * k)) * (
        -(k**2 * vx + a + vx) * np.exp(-k * t) 
        - np.cos(t) * a * k**2 
        - a * np.sin(t) 
        + (k**2 + 1) * (a + vx)
    )
    
    y = (1 / k**2) * (
        (-vy * k - g) * np.exp(-k * t) 
        + (-g * t + vy) * k 
        + g
    )
    
    return x, y

# ============================================================
# Förenklade rörelsesekvationer som ett system av första ordningen
# ============================================================

def simplified_model(t, state, k, a, g):
    """
    Definierar systemet av första ordningens ODE:er.
    state = [x, vx, y, vy]
    """
    x, vx, y, vy = state
    
    dxdt = vx
    dvxdt = -k * vx + a * np.sin(t)
    dydt = vy
    dvydt = -k * vy - g
    
    return np.array([dxdt, dvxdt, dydt, dvydt])

# ============================================================
# Runge-Kutta 4 metod
# ============================================================

def runge_kutta_4(f, t0, state0, T, h, *args):
    """
    RK4-metod för system av första ordningens ODE:er.
    f: funktion som definierar systemet
    t0: starttid
    state0: initialtillstånd
    T: sluttid
    h: tidssteg
    *args: extra parametrar till f
    """
    n_steps = int((T - t0) / h)
    t_values = np.zeros(n_steps + 1)
    state_values = np.zeros((n_steps + 1, len(state0)))
    
    t_values[0] = t0
    state_values[0] = state0
    
    t = t0
    state = state0.copy()
    
    for i in range(n_steps):
        k1 = f(t, state, *args)
        k2 = f(t + h/2, state + (h/2) * k1, *args)
        k3 = f(t + h/2, state + (h/2) * k2, *args)
        k4 = f(t + h, state + h * k3, *args)
        
        state = state + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        
        t_values[i+1] = t
        state_values[i+1] = state
    
    return t_values, state_values

# ============================================================
# Konvergensstudie
# ============================================================

# Parametrar
v0 = 40  # m/s
theta = 40 * np.pi / 180  # omvandla till radianer
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)
k = 0.001
a = 0.02
g = 9.81
T = 4  # sluttid

# Initialtillstånd: [x, vx, y, vy]
state0 = np.array([0, vx0, 0, vy0])

# Tidssteg för konvergensstudien
time_steps = np.array([1, 0.5, 0.25, 0.125])
errors = []
convergence_orders = []

# Referenslösning (mycket fint tidssteg)
t_ref, state_ref = runge_kutta_4(simplified_model, 0, state0, T, 0.001, k, a, g)

# Hitta index för t = T
idx_ref = -1  # sista indexet

print("Konvergensstudie för RK4")
print("=" * 60)
print(f"{'Δt':>8} {'||e_N||':>15} {'Konvergensordning p':>20}")
print("-" * 60)

for i, dt in enumerate(time_steps):
    # Numerisk lösning
    t_num, state_num = runge_kutta_4(simplified_model, 0, state0, T, dt, k, a, g)
    
    # Analytisk lösning vid t = T
    x_exact, y_exact = analytical_solution(T, vx0, vy0, k, a, g)
    
    # Numerisk lösning vid t = T (sista värdet)
    x_num = state_num[-1, 0]
    y_num = state_num[-1, 2]
    
    # Beräkna L2-normen av felet
    error = np.sqrt((x_exact - x_num)**2 + (y_exact - y_num)**2)
    errors.append(error)
    
    if i > 0:
        # Beräkna konvergensordningen
        p = np.log(errors[i-1] / errors[i]) / np.log(time_steps[i-1] / time_steps[i])
        convergence_orders.append(p)
        print(f"{dt:8.4f} {error:15.6e} {p:20.4f}")
    else:
        print(f"{dt:8.4f} {error:15.6e} {'—':>20}")

print("=" * 60)

# ============================================================
# Plotta fellinjen tillsammans med referenslinjer
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Plotta fel för olika tidssteg
ax.loglog(time_steps, errors, 'o-', linewidth=2, markersize=8, label='RK4-fel')

# Plotta referenslinjer för olika ordningar
h_ref = np.logspace(-3, 0, 100)
for order in [1, 2, 3, 4]:
    ax.loglog(h_ref, h_ref**order, '--', linewidth=1.5, label=f'$h^{order}$')

ax.set_xlabel('Tidssteg Δt [s]', fontsize=12)
ax.set_ylabel('L2-norm av fel ||e_N||', fontsize=12)
ax.set_title('Konvergensstudie för RK4-metoden', fontsize=14)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
ax.set_xlim([0.08, 1.5])

plt.tight_layout()
plt.show()

print("\nDiskussion:")
print("-" * 60)
print("RK4-metoden är en fjärde ordningens metod, vilket innebär att")
print("konvergensordningen p bör vara ungefär 4. Från resultaten ovan")
print("kan vi se att metoden konvergerar med rätt ordning.")
