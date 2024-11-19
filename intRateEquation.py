import numpy as np
from scipy.interpolate import PPoly
import matplotlib.pyplot as plt

# subfunctions
def strainHistory(timeD,strD):

    # Calculate slopes (m) for each segment
    slopes = np.diff(strD) / np.diff(timeD)  # (y2 - y1) / (x2 - x1)

    # Arrange coefficients for PPoly
    coefficients = np.vstack((slopes, strD[:-1]))  # Each segment: [m, b]

    # Create a piecewise polynomial object
    pp = PPoly(coefficients, timeD)

    # Compute the first derivative of the piecewise polynomial
    dpp_dt = pp.derivative()

    # Evaluate the piecewise polynomial at a range of points
    t_eval = np.linspace(timeD[0], timeD[-1], 1000)

    # Plot 
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(timeD, strD, 'o')
    ax1.plot(t_eval, pp(t_eval), label='strain')
    ax1.set_xlim([0, t_eval[-1]])
    ax1.set_xlabel('t / s')
    ax1.set_ylabel('str / m/m')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    ax1.set_position([0.1, 0.1, 0.3, 0.8])
    
    # ax2 = ax1.twinx()
    # ax2.plot(t_eval, dpp_dt(t_eval),'--', label='strain rate')
    # ax2.set_ylabel('dstr / 1/s')
    # ax2.set_position([0.1, 0.1, 0.3, 0.8])

    ax1.set_title('strain history')

    return t_eval, pp, dpp_dt, (ax1, ax3)

# *** MATERIAL PROPERTIES ***
E  = 210000     # E-modulus in MPa
Rp = 235        # yield stress in MPa

# *** LOADING HISTORY ***
timeD = np.array([0, 1, 2, 3])                # discrete time in s
strD  = 1E-3*np.array([0.0, 3.0, 0.0, 3.0])       # discrete strain in m/m

# determine and plot strain and strain rate
timeC, strT, dstrT, ax = strainHistory(timeD, strD)

# number of steps in time integration
numStep = len(timeC)

# time step for integration
dt = timeC[1]

# allocate vectors
sig  = np.zeros((numStep, 1))   # stress
str  = np.zeros((numStep, 1))   # total strain
strP = np.zeros((numStep, 1))   # plastic strain

# *** INTEGRATON OF LOADING HISTORY ***
for i in range(1, numStep):

    # strain rate
    dstr = dstrT(timeC[i])

    # elastic trial stress
    sigTr = sig[i-1] + E*dstr*dt

    # *** ELASTIC STEP ***
    if np.abs(sigTr) < Rp: # evaluate yield function
        dstrE = dstr    # elastic strain rate
        dstrP = 0       # plastic strain rate

    # *** PLASTIC STEP ***
    else:
        dstrE = 0
        dstrP = dstr

    # integration (explicit euler)
    str[i]  = str[i-1]  + dstr*dt
    sig[i]  = sig[i-1]  + E*dstrE*dt
    strP[i] = strP[i-1] + dstrP*dt

#ax3 = ax[1]

# plot stress over time
ax[1].plot(str, sig)
ax[1].set_xlabel('str / m/m')
ax[1].set_ylabel('sig / MPa')
ax[1].set_ylim([-1.05*Rp, 1.05*Rp])
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
ax[1].set_position([0.6, 0.1, 0.3, 0.8])
ax[1].set_title('stress history')

ax[0].plot(timeC, strP, label='plastic strain')
ax[0].legend(loc='upper center')
plt.show()




