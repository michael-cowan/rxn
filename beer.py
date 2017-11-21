import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

path = "C:\Users\mcowa\Documents\_Pitt\Fall17\RxnProcesses\Project\Paper\\"
params = ['E',
          'X',
          'G',
          'M',
          'N',
          'CL',
          'CG',
          'L',
          'I',
          'V',
          'IB',
          'IA',
          'MB',
          'P',
          'EA',
          'EC',
          'IAc',
          'VDK',
          'AAI',
          'T'
          ]
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

# empirical yeast growth inhibition constant
# [mol / m^3]
K_X = 3.65E5

# CO2 mass transfer coefficient
# [1 / hr]
K_GL = 0.07


# NOTE: All following constants are estimated for 10.5 deg C


# amino acide 

# yield coefficient
# Yij = [mol j / mol i]
Y_LX = 0.07734
Y_IX = 0.02172
Y_VX = 0.02045

# Michaelis-Menten constant
# [mol / m^3]
K_L = 0.5905
K_I = 0.07191
K_V = 0.02769

# first-order delay time constant
# [1]
tau_d = 23.54

# fusel alcohols

# yield coefficient
Y_IB_E = 0.1607
Y_IA_E = 0.5128
Y_MB_E = 0.3840
Y_P_E = 0.2216

# esters

# yield coefficient
Y_EA_S = 7.520E-4
Y_EC_X = 1.260E-4
Y_IAc = 2.918E-2

# vicinal diketones
# diacetyl (butanedione) & 2,3-pentanedione grouped into one: VDK

# yield coefficient
Y_VDK = 6.730E-5

# effective first-order rate constant
# [m^3 / (mol hr)]
k_VDK = 1.818E-5

# acetaldehyde

# yield coefficient
Y_AAI = 3.889E-3

# effective first-order rate constant
# [m^3 / (mol hr)]
k_AAI = 3.914E-5

"""
    Initial conditions
"""
# concentrations
# [mol / m^3]
E0 = 0.
X0 = 125.
G0 = 125.
M0 = 220.
N0 = 50.
CL0 = 0.
CG0 = 0.
L0 = 1.25
I0 = 0.6
V0 = 1.0
IB0 = 0.008
IA0 = 0.008
MB0 = 0.
P0 = 0.
EA0 = 0.
EC0 = 0.
IAc0 = 0.
VDK0 = 0.
AAI0 = 0.

# temperature
# [K]
T0 = 10.5 + 273.15
#T0 = 20. + 273.15

"""
    Useful functions
"""

def arrhenius(prek, ea, temp):
    return prek * np.exp(-ea / (1.987E-3 * temp))

lhc_length = 27358.8

def abv(e):
    return (100 * e * 46.068) / 789E3

def dim(rad, vol=0.23, days=7, r=False):
    flow = vol / (days*24.)
    area = np.pi * rad**2
    vel = flow / (1000. * area)
    length = vol / area
    print 'Volumetric Flowrate:\n  %.3e m^3/hr' % flow
    print 'Area:\n  %.3e m^2' % area
    print 'Fluid Velocity:\n  %.3e km/hr' % vel
    print '  %.3e mph' %(vel * 0.621371)
    print 'Length:\n  %.3f m' % length
    if r:
        return {'VolFlowrate': flow, 'Velocity': vel, 'Area': area, 'Length': length}

"""
    ODE setup
"""

def func(x, t, isothermal=False):
    E, X, G, M, N, CL, CG, L, I, V, IB, IA, MB, P, EA, EC, IAc, VDK, AAI, T = x

    # Michaelis-Menten constants
    # p: inhibition constants
    # [mol / m^3]
    K_G = 0.7464#arrhenius(K_G0, Ea_K_G, T)
    Kp_G = 5.356#arrhenius(Kp_G0, Ea_Kp_G, T)
    K_M = 40.97#arrhenius(K_M0, Ea_K_M, T)
    Kp_M = 13.17#arrhenius(Kp_M0, Ea_Kp_M, T)
    K_N = 250.#arrhenius(K_N0, Ea_K_N, T)

    # max velocity of formation
    # [1 / hr]
    mu_G = 0.01348#arrhenius(mu_G0, Ea_mu_G, T)
    mu_M = 0.02581#arrhenius(mu_M0, Ea_mu_M, T)
    mu_N = 0.09881#arrhenius(mu_N0, Ea_mu_N, T)

    # inhibition rates
    gluc_in = Kp_G / (Kp_G + G)
    malt_in = Kp_M / (Kp_M + M)

    # specific sugar uptake rates
    # [1 / hr]
    mu_1 = (mu_G * G) / (K_G + G)
    mu_2 = ((mu_M * M) / (K_M + M)) * gluc_in
    mu_3 = ((mu_N * N) / (K_N + N)) * gluc_in * malt_in

    # feedback inhibition mechanism for cell growth
    mu_x = ((Y_XG * mu_1) + (Y_XM * mu_2) + (Y_XN * mu_3)) * (K_X / (K_X + (X - X0)**2))

    # CO2 saturation concentrations
    # polynomial fit of CO2 solubility data as f(T) [R2 = 0.9955]
    # https://www.engineeringtoolbox.com/gases-solubility-water-d_1148.html
    # [mol / m^3]
    C_sat = np.poly1d([0.0194, -12.829, 2135.8])(T)

    # amino acid delay time
    D = 1 - np.exp(-t / tau_d)

    # ODEs
    
    # sugars
    dG = -mu_1 * X
    dM = -mu_2 * X
    dN = -mu_3 * X
    
    # ethanol
    dE = -(Y_EG * dG + Y_EM * dM + Y_EN * dN)
    
    # yeast
    dX = mu_x * X

    # aqueous CO2
    dCL = K_GL * (C_sat - CL) * bool(CL <= C_sat)

    # gas CO2
    dCG = X * (Y_CG * mu_1 + Y_CM * mu_2 + Y_CN * mu_3) - dCL
    
    # amino acids
    dL = -Y_LX * dX * (L / (K_L + L)) * D
    dI = -Y_IX * dX * (I / (K_I + I)) * D
    dV = -Y_VX * dX * (V / (K_V + V)) * D

    # fusel alcohols
    dIB = -Y_IB_E * dV
    dIA = -Y_IA_E * dL
    dMB = -Y_MB_E * dI
    dP = -Y_P_E * (dV + dI)

    # esters
    dEA = Y_EA_S * -(dG + dM + dN)
    dEC = Y_EC_X * mu_x * X
    dIAc = Y_IAc * dIA

    # vicinal diketones
    dVDK = Y_VDK * mu_x * X - k_VDK * VDK * X

    # acetaldehyde
    dAAI = Y_AAI * (mu_1 + mu_2 + mu_3) * X - k_AAI * AAI * X

    # temperature
    # in case isothermal is selected
    dT = bool(not isothermal) * ((1 / (rho * Cp)) * ((H_FG * dG + H_FM * dM + H_FN * dN) - u * (T - Tc)))

    return [dE, dX, dG, dM, dN, dCL, dCG, dL, dI, dV, dIB, dIA, dMB, dP, dEA, dEC, dIAc, dVDK, dAAI, dT]

"""
    Solution and data vis
"""

def main(tmax=168, isothermal=True):
    thr = np.linspace(0, tmax)
    inits = [E0, X0, G0, M0, N0, CL0, CG0, L0, I0, V0, IB0, IA0, MB0, P0, EA0, EC0, IAc0, VDK0, AAI0, T0]
    sol = odeint(func, inits, thr, args=(isothermal,))
    
    # convert temp to degC
    sol[:, -1] -= 273.15
    
    t = thr / float(tmax)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()
    fig8, ax8 = plt.subplots()
    fig9, ax9 = plt.subplots()
    figf, axf = plt.subplots()

    ax1.plot(t, sol[:, 5:7])
    ax1.set_title('CO2')
    ax1.legend(['Aqueous', 'Gas'])
    ax1.set_xlabel('Normalized PFR Length')
    ax1.set_ylabel('Concentration [mol / m^3]')

    ax2.plot(t, sol[:, 1], color='r')
    ax2.set_title('Yeast')
    ax2.set_xlabel('Normalized PFR Length')
    ax2.set_ylabel('Concentration [mol / m^3]')

    ax3.plot(t, sol[:, 2:5])
    ax3.set_title('Sugars')
    ax3.legend(['Glucose', 'Maltose', 'Maltotriose'])
    ax3.set_xlabel('Normalized PFR Length')
    ax3.set_ylabel('Concentration [mol / m^3]')
    
    ax4.plot(t, sol[:, 7:10])
    ax4.set_title('Amino Acids')
    ax4.legend(['Leucine', 'Isoleucine', 'Valine'])
    ax4.set_xlabel('Normalized PFR Length')
    ax4.set_ylabel('Concentration [mol / m^3]')

    ax5.plot(t, sol[:, 10:14])
    ax5.set_title('Fusel Alcohols')
    ax5.legend(['Isobutyl Alcohol', 'Isoamyl Alcohol', '2-methyl-1-butanol', 'n-propanol'])
    ax5.set_xlabel('Normalized PFR Length')
    ax5.set_ylabel('Concentration [mol / m^3')
    
    ax6.plot(t, sol[:, 14:17])
    ax6.set_title('Esters')
    ax6.legend(['Ethyl Acetate', 'Ethyl Caproate', 'Isoamyl Acetate'])
    ax6.set_xlabel('Normalized PFR Length')
    ax6.set_ylabel('Concentration [mol / m^3]')

    ax7.plot(t, sol[:, 17], color='black')
    ax7.set_title('Vicinal Diketones')
    ax7.set_xlabel('Normalized PFR Length')
    ax7.set_ylabel('Concentration [mol / m^3]')
    
    ax8.plot(t, sol[:, 18], color='c')
    ax8.set_title('Acetaldehyde')
    ax8.set_xlabel('Normalized PFR Length')
    ax8.set_ylabel('Concentration [mol / m^3]')
    
    ax9.plot(t, sol[:, -1], color='g')
    ax9.set_title('Temperature')
    ax9.set_xlabel('Normalized PFR Length')
    ax9.set_ylabel('Concentration [mol / m^3]')

    axf.plot(t, map(abv, sol[:, 0]), color='m')
    axf.set_title('Ethanol')
    axf.set_xlabel('Normalized PFR Length')
    axf.set_ylabel('% ABV')
    
    fa = [(fig1, ax1),
          (fig2, ax2),
          (fig3, ax3),
          (fig4, ax4),
          (fig5, ax5),
          (fig6, ax6),
          (fig7, ax7),
          (fig8, ax8),
          (fig9, ax9),
          (figf, axf)
          ]
    
    for f in fa:
        f[1].set_facecolor('none')

    return sol, {n: (i[1].get_title(), i[0], i[1]) for n,i in enumerate(fa)}

if __name__ == '__main__':
    sol, figs = main(168, True)
    plt.close('all')
    #figs[2][1].show()
    if 0:
        for f in figs.values():
            f[1].savefig(path + 'Figures\\' + f[2].get_title().replace(' ', '') + '_PFR.png', dpi=300)