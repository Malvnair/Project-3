
# Import the necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


################
#####PART1######
################


# Define constants and convert to solar units.
mu_e = 2
r_0 = (7.72e8 / 6.957e10) / mu_e   
m_0 = (5.67e33 / 2e33) / (mu_e)**2  



# Define gamma function.
def gamma_function(x):
    """Calculates the value of gamma given a value of x.

    Args:
        x: dimensionless variable.

    Returns:
       The value of gamma.
    """    
    return (x**2 / (3 * np.sqrt(1 + x**2)))
    

# Define the ode systems.
def coupled_system(r,y):
    """Defines the coupled system of ordinary differential equations for a white dwarf planet.
    It takes in a state vector and calculates drho/dr (equation 8) and dm/dr (equation 9).

    Args:
        r: Radius
        y: State vector containing values of rho and m.

    Returns:
        A list containing the derivatives of the rate of change of density wirth resepct to radius and the rate of change of mass with respect to radius.
    """    
    
    # Break up state vector.
    rho, m = y
    
    # Calculate values.
    x = rho**(1/3)
    d_rho_dr = (-m * rho) / (gamma_function(x) * r**2) if r > 0 else 0
    d_m_dr =  r**2 * rho
    
    # Return a list with the derivative values.
    return [d_rho_dr, d_m_dr]


# Create an array of values of rho_c in the logarithm scale, takes 10 points.
rho_c_values = np.logspace(-1, 6.4, 10) 

# Initialize an array for results.
results = []


# Define an events function.
def events(r, y):
    """Event function to stop the integration when the density becomes zero.

    Args:
        r: The current radius value during the integration.
        y: A list containing the current values of the dependent variables.
           Where y[0] provides the density ('rho') and y[1] provides the mass ('m') at the current radius.

    Returns:
        The density at the current radius, the integration will stop when this value reaches zero.
    """    
    return y[0] 


# Loop through the rho_c values. 
for rho_c in rho_c_values:
    
    # Define the state vector.
    y0 = [rho_c, 0]  
    # Define the range or radius values.
    r_span = (1e-8, 1e7) 
    # Solve the ODEs with the RK45 method, an event is added to stop the integration.
    solution = solve_ivp(coupled_system, r_span, y0, method='RK45',
                         events=[events])  
    # Extract the solutions.
    r_end = solution.t[-1]
    m_end = solution.y[1, -1]
    # Append to the results list.
    results.append((r_end, m_end))


# Print the results for the 10 values of rho_c.
print("rho_c (g/cm^3)   Radius (Rsun)    Mass (Msun)")
for i, rho_c in enumerate(rho_c_values):
    radius = results[i][0] * r_0
    mass = results[i][1] * m_0
    print(f"{rho_c:.2e}          {radius:.4f}         {mass:.4f}")



################
#####PART2######
################


# Convert the results into an array.
results_array = np.array(results) 
# Extract the radii and multiply by r_0.
radii = results_array[:, 0] * r_0  
masses = results_array[:, 1] * m_0  

# Determine the largest mass value.
largest_mass = max(masses)
# Calculate the theoritical mass.
Ch_theoretical = 5.836 / (mu_e ** 2)

# Print the masses.
print(f"\nTheoretical Chandrasekhar limit: {Ch_theoretical:.4f} Msun")
print(f"The largest mass is {largest_mass:.4f} Msun.")

# Print the difference and percent error between the two masses.
print(f"The difference between these two masses is {(Ch_theoretical - largest_mass):.4f} Msun, "
      f"with a percent error of {abs((Ch_theoretical - largest_mass) / Ch_theoretical) * 100:.2f}%.\n")


# Plot the figure.
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

# Select three random rho_c values in descending order.
selected_rho_c_values = np.sort(np.random.choice(rho_c_values, size=3, replace=False)[::-1])


# Specify the different integration method.
method = 'DOP853'
# Initialize an array for DOP853_results.
DOP853_results = []

# Loop through the selected rho_c values. 
for rho_c in selected_rho_c_values:
    y0 = [rho_c, 0]  
    r_span = (1e-8, 1e7) 
    # Solve the ODEs with the the DOP853 method, an event is added to stop the integration.
    solution = solve_ivp(coupled_system, r_span, y0, method=method,
                        events=[events])  
    
    # Extract the solutions.
    r_end = solution.t[-1]
    m_end = solution.y[1, -1]
    # Append to the DOP853 results list.
    DOP853_results.append((r_end, m_end))

# Convert the DOP853 results into an array.
DOP853_results_array = np.array(DOP853_results) 

# Extract the radii and multiply by r_0.
radii_DOP853 = DOP853_results_array[:, 0] * r_0  
masses_DOP853 = DOP853_results_array[:, 1]* m_0  

# Plot the figure with the new method.
plt.figure(figsize=(8, 6))
plt.plot(masses_DOP853, radii_DOP853, marker='o', label=f'Method: {method}')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Mass-Radius Relation Using DOP853 Method')
plt.legend()
plt.grid()
plt.show()


# Loop through the 3 values and compare the two methods.
for i in range(len(selected_rho_c_values)):
    rk45_radius, rk45_mass = radii[i], masses[i]
    dop853_radius, dop853_mass = radii_DOP853[i], masses_DOP853[i]
    
    # Calculate the differences between the radii and masses.
    radius_diff = abs(dop853_radius - rk45_radius)
    mass_diff = abs(dop853_mass - rk45_mass)
    
    #Print the desnity, radii, masses, and their differences.
    print(f"Central Density: {selected_rho_c_values[i]:.4e}")
    print(f"RK45 Radius: {rk45_radius:.4e} Rsun, DOP853 Radius: {dop853_radius:.4e} Rsun")
    print(f"Radius Difference: {radius_diff:.4e} Rsun")
    print(f"RK45 Mass: {rk45_mass:.4e} Msun, DOP853 Mass: {dop853_mass:.4e} Msun")
    print(f"Mass Difference: {mass_diff:.4e} Msun\n")

print("The results are close but do appear to have a mass difference of approximately >1 Msun.\n")
    
    
################
#####PART4######
################

# Load the data file, skip the first row, and seperate the data by commas.
data = np.loadtxt('wd_mass_radius.csv', delimiter=',', skiprows=1)

# Extract columns from the loaded data.
masses_obs = data[:, 0]
mass_unc = data[:, 1]
radii_obs = data[:, 2]
radius_unc = data[:, 3]

# Plot the figure.
plt.figure(figsize=(10, 7))
plt.plot(masses, radii, label='Computed Relation', linestyle='-', marker='o', color='blue')
# Plot the error bars from the given data set.
plt.errorbar(masses_obs, radii_obs, xerr=mass_unc, yerr=radius_unc, fmt='o', label='Observational Data', color='red')
plt.xlabel('Mass (Msun)')
plt.ylabel('Radius (Rsun)')
plt.title('Computed and Observed Mass-Radius Relations')
plt.legend()
plt.grid()
plt.show()

# Initialize an empty list to store absolute differences.
absolute_differences = []

# Loop through each observed mass.
for i in range(len(masses_obs)):
    obs_mass = masses_obs[i]
    obs_radius = radii_obs[i]
    
    # Find the index of the closest computed mass.
    closest_index = np.argmin(np.abs(masses - obs_mass))
    
    # Compute the absolute difference in radius.
    absolute_difference = np.abs(obs_radius - radii[closest_index])
    
    # Append the absolute difference to the list.
    absolute_differences.append(absolute_difference)
    
    
# Print the absolute differences for each observation.
print("Absolute Differences Between Observed and Computed Radii:")
for i, diff in enumerate(absolute_differences):
    print(f"Observation {i}: {diff:.4f} Rsun")
    
print("Therefore, the observations agree with the calculations well. ")
