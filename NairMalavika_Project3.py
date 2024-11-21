
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Define constants
mu_e = 2
r_0 = (7.72e8/6.957e10) / mu_e 
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
################


largest_mass = max(masses)
Ch_theoretical = 5.836 / (mu_e ** 2)

print(f"Theoretical Chandrasekhar limit: {Ch_theoretical:.4f} Msun")
print(f"The largest mass is {largest_mass:.4f} Msun.")
print(f"The difference between these two masses is {(Ch_theoretical - largest_mass):.4f} Msun, "
      f"with a percent error of {abs((Ch_theoretical - largest_mass) / Ch_theoretical) * 100:.2f}%.\n")



plt.figure(figsize=(8, 6))
plt.plot(masses, radii, marker='o')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Mass-Radius Relation of White Dwarfs')
plt.grid()
plt.show()


   
################
#####PART3######
################

#pick random rho_c
selected_rho_c_values = np.sort(np.random.choice(rho_c_values, size=3, replace=False)[::-1])


#different integration method
method = 'DOP853'
method_results = []

for rho_c in selected_rho_c_values:
    y0 = [rho_c, 0]  
    r_span = (1e-8, 1e7) 
    solution = solve_ivp(coupled_system, r_span, y0, method=method,
                        events=[events])  
    r_end = solution.t[-1]
    m_end = solution.y[1, -1]
    method_results.append((r_end, m_end))


method_results_array = np.array(method_results) 
radii_new = method_results_array[:, 0] * r_0  
masses_new = method_results_array[:, 1]* m_0  


plt.figure(figsize=(8, 6))
plt.plot(masses_new, radii_new, marker='o', label=f'Method: {method}')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Mass-Radius Relation Using DOP853 Method')
plt.legend()
plt.grid()
plt.show()


#compare result
for i in range(len(selected_rho_c_values)):
    rk45_radius, rk45_mass = radii[i], masses[i]
    dop853_radius, dop853_mass = radii_new[i], masses_new[i]
    
    radius_diff = abs(dop853_radius - rk45_radius)
    mass_diff = abs(dop853_mass - rk45_mass)
    
    print(f"Central Density: {selected_rho_c_values[i]:.4e}")
    print(f"RK45 Radius: {rk45_radius:.4e} Rsun, DOP853 Radius: {dop853_radius:.4e} Rsun")
    print(f"Radius Difference: {radius_diff:.4e} Rsun")
    print(f"RK45 Mass: {rk45_mass:.4e} Msun, DOP853 Mass: {dop853_mass:.4e} Msun")
    print(f"Mass Difference: {mass_diff:.4e} Msun\n")
    
    
################
#####PART4######
################

data = np.loadtxt('wd_mass_radius.csv', delimiter=',', skiprows=1)

#extract columns from the loaded data
masses_obs = data[:, 0]
mass_unc = data[:, 1]
radii_obs = data[:, 2]
radius_unc = data[:, 3]

plt.figure(figsize=(10, 7))
plt.plot(masses, radii, label='Computed Relation', linestyle='-', marker='o', color='blue')
plt.errorbar(masses_obs, radii_obs, xerr=mass_unc, yerr=radius_unc, fmt='o', label='Observational Data', color='red')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Computed and Observed Mass-Radius Relations')
plt.legend()
plt.grid()
plt.show()

