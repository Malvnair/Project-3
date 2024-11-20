import numpy as np
from scipy.integrate import solve_ivp


# Define constants
mu_e = 2
rho_0 = (9.74e5) * mu_e 
K1 = 1
K2 = 1


def gamma_function(rho):
    x = rho**(1/3)
    return (x**2 / (3 * np.sqrt(1 + x**2)))
    

# define the ode system
def coupled_system(r,y):
    rho, m = y
    d_rho_dr = -K1 * m * rho / (gamma_function(rho) * r**2) if r > 0 else 0
    d_m_dr = K2 * r**2 * rho
    return [d_rho_dr, d_m_dr]










