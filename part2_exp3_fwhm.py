import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from funcs import *

data50 = np.loadtxt("data_part2_exp3_50.txt", unpack=True)
data100 = np.loadtxt("data_part2_exp3_100.txt", unpack=True)
data250 = np.loadtxt("data_part2_exp3_250.txt", unpack=True)
data500 = np.loadtxt("data_part2_exp3_500.txt", unpack=True)

# 50 ks/s
p050 = [60, 10000, 20, -65]

range50 = (data50[0]>=9765) * (data50[0]<=10285)

p50, chi_score, err50 = fitting(p050, data50[0][range50], data50[1][range50], gauss)

#plt.plot(data50[0], data50[1])
#plt.plot(data50[0][range50], gauss(p50, data50[0][range50]))
#plt.show()

# 100 ks/s
p0100 = [60, 10000, 20, -65]

range100 = (data100[0]>=9510) * (data100[0]<=10460)

p100, chi_score, err100 = fitting(p050, data100[0][range100], data100[1][range100], gauss)

#plt.plot(data100[0], data100[1])
#plt.plot(data100[0][range100], gauss(p100, data100[0][range100]))
#plt.show()

# 250 ks/s
p0250 = [60, 25000, 20, -65]

range250 = (data250[0]>=9000) * (data250[0]<=10730)

p250, chi_score, err250 = fitting(p050, data250[0][range250], data250[1][range250], gauss)

#plt.plot(data250[0], data250[1])
#plt.plot(data250[0][range250], gauss(p250, data250[0][range250]))
#plt.show()


# 500 ks/s
p0500 = [60, 50000, 20, -65]

range500 = (data500[0]>=7300) * (data500[0]<=12600)

p500, chi_score, err500 = fitting(p050, data500[0][range500], data500[1][range500], gauss)

#plt.plot(data500[0], data500[1])
#plt.plot(data500[0][range500], gauss(p500, data500[0][range500]))
#plt.show()


# Calculate and graph

fwhms = np.array([])
fwhm_errs = np.array([])

fwhm50 = 2 * np.sqrt(np.log(2) * 2) * p50[2]
fwhm50_err = 2 * np.sqrt(np.log(2) * 2) * err50[2]

fwhms = np.append(fwhms, fwhm50)
fwhm_errs = np.append(fwhm_errs, fwhm50_err)

fwhm50, fwhm50_err = round_sig_fig_uncertainty(fwhm50, fwhm50_err)

print(f'For 50 kS/s FWHM = {fwhm50} \u00B1 {fwhm50_err} Hz')

fwhm100 = 2 * np.sqrt(np.log(2) * 2) * p100[2]
fwhm100_err = 2 * np.sqrt(np.log(2) * 2) * err100[2]

fwhms = np.append(fwhms, fwhm100)
fwhm_errs = np.append(fwhm_errs, fwhm100_err)

fwhm100, fwhm100_err = round_sig_fig_uncertainty(fwhm100, fwhm100_err)

print(f'For 100 kS/s FWHM = {fwhm100} \u00B1 {fwhm100_err} Hz')

fwhm250 = 2 * np.sqrt(np.log(2) * 2) * p250[2]
fwhm250_err = 2 * np.sqrt(np.log(2) * 2) * err250[2]

fwhms = np.append(fwhms, fwhm250)
fwhm_errs = np.append(fwhm_errs, fwhm250_err)

fwhm250, fwhm250_err = round_sig_fig_uncertainty(fwhm250, fwhm250_err)

print(f'For 250 kS/s FWHM = {fwhm250} \u00B1 {fwhm250_err} Hz')

fwhm500 = 2 * np.sqrt(np.log(2) * 2) * p500[2]
fwhm500_err = 2 * np.sqrt(np.log(2) * 2) * err500[2]

fwhms = np.append(fwhms, fwhm500)
fwhm_errs = np.append(fwhm_errs, fwhm500_err)

fwhm500, fwhm500_err = round_sig_fig_uncertainty(fwhm500, fwhm500_err)

print(f'For 500 kS/s FWHM = {fwhm500} \u00B1 {fwhm500_err} Hz')


freqs = [50e3, 100e3, 250e3, 500e3]

linear_fit = stats.linregress(freqs, fwhms)

x = np.linspace(freqs[0], freqs[-1], 1000)

slope, slope_err = round_sig_fig_uncertainty(linear_fit.slope, linear_fit.stderr)

intercept, intercept_err = round_sig_fig_uncertainty(linear_fit.intercept, linear_fit.intercept_stderr)



plt.errorbar(freqs, fwhms, fwhm_errs, fmt='.', label='Data', color='black')
plt.plot(x, linear_fit.slope*x + linear_fit.intercept, label='Fit')

plt.plot([], [], ' ', label='R$^2$='+str(np.round(linear_fit.rvalue, 6)))
plt.plot([], [], ' ', label=r'$\Delta f^2$=m$f_{eff}$+c')
plt.plot([], [], ' ', label='m='+str(slope)+r'$ \pm $'+str(slope_err))
plt.plot([], [], ' ', label='c='+str(intercept)+r'$ \pm $'+str(intercept_err))

plt.title(r'Full width half maxima ($\Delta f$) for different sample rates against $f_{eff}$')
plt.xlabel(r'$f_{eff}$ [Hz]')
plt.ylabel(r'$\Delta f$ [Hz]')
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,5,1,2,3,4]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
plt.show()
