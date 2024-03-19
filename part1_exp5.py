import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from funcs import *

data = ascii.read("data_part1_exp5.txt")

t = data['t']
V = data['V']


plt.plot(t,V)
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title('Plot of the amplitude modulated waveform measured on the oscilloscope.')
plt.grid()
plt.show()

# Fit

# initial guess
p0 = [0.3, 15600, -1, 2244, 0, 0]
p0_prod = [0.6, 8900, -1.8, 6700, 0, 0]


p_fit, chi_score, err = fitting(p0, t, V, cos_superpos)
p_fit_prod, chi_score_prod, err_prod = fitting(p0_prod, t, V, cos_prod)


plt.plot(t,V, label='Data')
plt.plot(t, cos_superpos(p_fit, t), label='Fit of superposition of cosines')
plt.plot(t, cos_prod(p_fit_prod, t), label='Fit of product of cosines')
plt.title('Plot of the amplitude modulated waveform measured on the oscilloscope with the fitted graph.')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

# Round freqs
omega_av, omega_av_err = round_sig_fig_uncertainty(p_fit_prod[1], err_prod[1])
omega_mod, omega_mod_err = round_sig_fig_uncertainty(p_fit_prod[3], err_prod[3])
omega1, omega1_err = round_sig_fig_uncertainty(p_fit[1], err[1])
omega2, omega2_err = round_sig_fig_uncertainty(p_fit[3], err[3])

print(f'Carrier frequency = {omega_av} \u00B1 {omega_av_err} Hz \nModulation frequency = {omega_mod} \u00B1 {omega_mod_err} Hz')


print(f"omega 1 = {omega1} \u00B1 {omega1_err} Hz \nomega 2 = {omega2} \u00B1 {omega2_err} Hz")
