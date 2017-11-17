import numpy as np
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

"""
    Constants
"""

# density
# [kg / m^3]
rho = 1040.

# heat capacity
# [J / (kg degC)]
Cp = 4016.

# coolant temperature
# [K]
Tc = 0 + 273.15

# u = UA / V
# [m^2]
A = 0.188

# [m^3]
V = 0.1

# [J / (m^3 hr degC)]
u = 20E3

# enthalpy
# [J / mol]
H_FG = -91.2E3
H_FM = -226.3E3
H_FN = -361.3E3

# yield coefficients (1 || 3)
# Y_ij = [mol j / mol i]
Y_XG = 0.134
Y_EG = 1.92
Y_CG = 1.97
Y_XM = 0.268
Y_EM = 3.84
Y_CM = 3.94
Y_XN = 0.402
Y_EN = 5.76
Y_CN = 5.91

# activation energy (3)
# [kcal / mol]
Ea_mu_G = 22.6
Ea_mu_M = 11.3
Ea_mu_N = 7.16

Ea_K_G = -68.6
Ea_K_M = -14.4
Ea_K_N = -19.9
Ea_Kp_G = 10.2
Ea_Kp_M = 26.3

# arrhenius preconstant (frequency factor) (3)
# [1 / hr]
mu_G0 = np.exp(35.77)
mu_M0 = np.exp(16.40)
mu_N0 = np.exp(10.59)

# [mol / m^3]
K_G0 = np.exp(-121.3)
K_M0 = np.exp(-19.15)
K_N0 = np.exp(-26.78)
Kp_G0 = np.exp(23.33)
Kp_M0 = np.exp(55.61)

"""
    Initial conditions
"""
# concentrations
# [mol / m^3]
E0 = 0.
X0 = 125.
G0 = 70.
M0 = 220.
N0 = 40.

# temperature
# [K]
T0 = 10.5 + 273.15
T0 = 8 + 273.15

def arrhenius(prek, ea, temp):
    return prek * np.exp(-ea / (1.987E-3 * temp))

def func(x, t):
    E, X, G, M, N, T = x

    # Michaelis-Menten constants
    # p: inhibition constants
    # [mol / m^3]
    K_G = arrhenius(K_G0, Ea_K_G, T)
    Kp_G = arrhenius(Kp_G0, Ea_Kp_G, T)
    K_M = arrhenius(K_M0, Ea_K_M, T)
    Kp_M = arrhenius(Kp_M0, Ea_Kp_M, T)
    K_N = arrhenius(K_N0, Ea_K_N, T)

    # max velocity of formation
    # [1 / hr]
    mu_G = arrhenius(mu_G0, Ea_mu_G, T)
    mu_M = arrhenius(mu_M0, Ea_mu_M, T)
    mu_N = arrhenius(mu_N0, Ea_mu_N, T)

    # glucose inhibition rate
    gluc_in = Kp_G / (Kp_G + G)

    # specific sugar uptake rates
    # [1 / hr]
    mu_1 = (mu_G * G) / (K_G + G)
    mu_2 = ((mu_M * M) / (K_M + M)) * gluc_in
    mu_3 = ((mu_N * N) / (K_N + N)) * gluc_in * (Kp_M / (Kp_M + M))

    # ODEs
    #dE = Y_EG * (G0 - G) + Y_EM * (M0 - M) + Y_EN * (N0 - N)
    dG = -mu_1 * X
    dM = -mu_2 * X
    dN = -mu_3 * X
    dE = -(Y_EG * dG + Y_EM * dM + Y_EN * dN)
    dX = -(Y_XG * dG + Y_XM * dM + Y_XN * dN)
    dT = (1 / (rho * Cp)) * ((H_FG * dG + H_FM * dM + H_FN * dN) - u * (T - Tc))

    return [dE, dX, dG, dM, dN, dT]

t = np.linspace(0, 150)
inits = [E0, X0, G0, M0, N0, T0]
sol = odeint(func, inits, t)

def abv(e):
    return (100 * e * 46.068) / 789E3

fig = plt.figure()
plt.suptitle('Fermentation in a Batch Reactor')
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.plot(t, sol[:, 1])
ax1.set_ylabel('Yeast Concentration [mol / m^3]')

ax2.plot(t, sol[:, -1] - 273.15)
ax2.set_ylabel('Temperature [deg C]')

ax3.plot(t, sol[:, 2:-1])
ax3.legend(['Glucose', 'Maltose', 'Maltotriose'])
ax3.set_ylabel('Concentration [mol / m^3]')

ax4.plot(t, map(abv, sol[:, 0]))
ax4.set_ylabel('Ethanol [% ABV]')
ax4.set_xlabel('Time [hr]')

fig.show()