import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from funcs import fitting, horiz_line, round_sig_fig_uncertainty, derivative

data = ascii.read("data_part1_exp1.txt")

t = data['t']
V = data['V']

plt.plot(t, V)
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title('Plot of a square wave measured on the oscilloscope.')
plt.grid()
plt.show()

# Fit the peaks
index1 = (V>3) * (t<-0.5e-6)
peak_t1 = t[index1]
peak_V1 = V[index1]

index2 = (V>3) * (t>-0.5e-6) * (t<1.5e-6)
peak_t2 = t[index2]
peak_V2 = V[index2]

index3 = (V>3) * (t>1.5e-6)
peak_t3 = t[index3]
peak_V3 = V[index3]

# initial guess
p0 = [3]
p_fit_top1, chi_score_top, err_top1 = fitting(p0, peak_t1, peak_V1, horiz_line)
p_fit_top2, chi_score_top, err_top2 = fitting(p0, peak_t2, peak_V2, horiz_line)
p_fit_top3, chi_score_top, err_top3 = fitting(p0, peak_t3, peak_V3, horiz_line)

# Fit the troughs
index1 = (V<0.5) * (t<-1.5e-6)
trough_t1 = t[index1]
trough_V1 = V[index1]

index2 = (V<0.5) * (t>-1.5e-6) * (t<0.5e-6)
trough_t2 = t[index2]
trough_V2 = V[index2]

index3 = (V<0.5) * (t>0.5e-6)
trough_t3 = t[index3]
trough_V3 = V[index3]

# initial guess
p0 = [0]
p_fit_bottom1, chi_score_bottom, err_bottom1 = fitting(p0, trough_t1, trough_V1, horiz_line)
p_fit_bottom2, chi_score_bottom, err_bottom2 = fitting(p0, trough_t2, trough_V2, horiz_line)
p_fit_bottom3, chi_score_bottom, err_bottom3 = fitting(p0, trough_t3, trough_V3, horiz_line)

plt.plot(t, V, label='Data')
plt.plot(peak_t1, horiz_line(p_fit_top1, peak_t1), label='fit of peak', color='black')
plt.plot(peak_t2, horiz_line(p_fit_top2, peak_t2), color='black')
plt.plot(peak_t3, horiz_line(p_fit_top3, peak_t3), color='black')

plt.plot(trough_t1, horiz_line(p_fit_bottom1, trough_t1), label='fit of trough', color='red')
plt.plot(trough_t2, horiz_line(p_fit_bottom2, trough_t2), color='red')
plt.plot(trough_t3, horiz_line(p_fit_bottom3, trough_t3), color='red')

plt.title('Plot of a square wave measured on the oscilloscope with fitted sections.')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.legend()
plt.grid()
plt.show()

err_top = np.mean([err_top1, err_top2, err_top3])
err_bottom = np.mean([err_bottom1, err_bottom2, err_bottom3])
err_Vpp = np.sqrt(err_top**2 + err_bottom**2)

Vpp = np.mean([p_fit_top1, p_fit_top2, p_fit_top3]) - np.mean([p_fit_bottom1, p_fit_bottom2, p_fit_bottom3])

Vpp, err_Vpp = round_sig_fig_uncertainty(Vpp, err_Vpp)

print(f'Peak to peak voltage is {Vpp} \u00B1 {err_Vpp} V')

# find where the voltage increases suddenly, the difference between successive points is the period
a = np.array([])
for i in range(1,len(V)):
    if np.abs(V[i] - V[i-1]) > 0.3 and V[i-1] < 0.5:
        a = np.append(a, t[i-1])

periods = np.array([])
for i in range(1,len(a)):
    periods = np.append(periods, a[i] - a[i-1])

err_periods = np.sqrt(2*(t[1]-t[0])**2)

period = np.mean(periods)

err_period = np.sqrt(np.sum(err_periods**2))/2

period, err_period = round_sig_fig_uncertainty(period, err_period)

print(f'Period is {period} \u00B1 {err_period} s')
