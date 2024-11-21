
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Define constants
mu_e = 2
r_0 = (7.72e8/6.957e10) * mu_e 
m_0 = (5.67e33/2e33) / (mu_e)**2
rho_0 = (9.74e5) / mu_e 




def gamma_function(x):
    return (x**2 / (3 * np.sqrt(1 + x**2)))
    

# define the ode system
def coupled_system(r,y):
    rho, m = y
    x = rho**(1/3)
    d_rho_dr = (-m * rho) / (gamma_function(x) * r**2) if r > 0 else 0
    d_m_dr =  r**2 * rho
    return [d_rho_dr, d_m_dr]


rho_c_values = np.logspace(-1, 6.4, 10) 
results = []


def events(r, y):
    return y[0]

for rho_c in rho_c_values:
    #intial vector
    y0 = [rho_c, 0]  
    r_span = (1e-8, 1e7) 
    solution = solve_ivp(coupled_system, r_span, y0, method='RK45',
                         events=[events])  
    r_end = solution.t[-1]
    m_end = solution.y[1, -1]
    results.append((r_end, m_end))


results_array = np.array(results) 
radii = results_array[:, 0] * r_0  
masses = results_array[:, 1] * m_0  


    
################
#####PART2######
##############
largest_mass = max(masses)
Ch_theoretical = 5.836 / (mu_e ** 2)

print(f"Theoretical Chandrasekhar limit: {Ch_theoretical:.4f} Msun")
print(f"The largest mass is {largest_mass:.4f} Msun.")
print(f"The difference between these two masses is {(Ch_theoretical - largest_mass):.4f} Msun, "
      f"with a percent error of {abs((Ch_theoretical - largest_mass) / Ch_theoretical) * 100:.2f}%.")



plt.figure(figsize=(8, 6))
plt.plot(masses, radii, marker='o')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Mass-Radius Relation of White Dwarfs')
plt.grid()
plt.show()





