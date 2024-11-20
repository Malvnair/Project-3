import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Define constants
mu_e = 2
r_0 = (7.72e8) * mu_e 
m_0 = (5.67e33)
rho_0 = (9.74e5) * mu_e 
K1 = 1
K2 = 1


def gamma_function(x):
    return (x**2 / (3 * np.sqrt(1 + x**2)))
    

# define the ode system
def coupled_system(r,y):
    rho, m = y
    x = rho**(1/3)
    d_rho_dr = -K1 * m * rho / (gamma_function(x) * r**2) if r > 0 else 0
    d_m_dr = K2 * r**2 * rho
    return [d_rho_dr, d_m_dr]


rho_c_values = np.logspace(-1, 6.4, 10) 
results = []


for rho_c in rho_c_values:
    #intial vector
    y0 = [rho_c, 0]  
    r_span = (1e-8, 1e10) 
    solution = solve_ivp(coupled_system, r_span, y0, method='RK45',
                         events=lambda r, y: y[0])  
    r_end = solution.t[-1]
    m_end = solution.y[1, -1]
    results.append((r_end, m_end))


radii = [r * r_0 for r, m in results]  
masses = [m * m_0 for r, m in results]  
    


plt.figure(figsize=(8, 6))
plt.plot(masses, radii, marker='o', label='Mass-Radius Relation')
plt.xlabel('Mass (g)')
plt.ylabel('Radius (cm)')
plt.xscale('log')
plt.yscale('log')
plt.title('Mass-Radius Relation of White Dwarfs')
plt.grid()
plt.legend()
plt.show()







