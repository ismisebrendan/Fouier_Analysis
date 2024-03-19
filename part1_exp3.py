import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

data1 = ascii.read("data_part1_exp3_1.txt")
data2 = ascii.read("data_part1_exp3_2.txt")
data3 = ascii.read("data_part1_exp3_3.txt")

data4 = ascii.read("data_part1_exp3_complex.txt")


plt.rc('font', size = 15)



t1 = data1['t']
V1 = data1['V']

t2 = data2['t']
V2 = data2['V']

t3 = data3['t']
V3 = data3['V']

plt.plot(t1, V1, label='Rising, l = 0 V')
plt.plot(t2, V2, label='Rising, l = 192 mV')
plt.plot(t3, V3, label='Falling, l = 192 mV')
plt.hlines(0, min(t1), max(t1), linestyles='dotted', color='red', label='V = 0 V')
plt.hlines(0.192, min(t1), max(t1), linestyles='dashed', color='red', label='V = 192 mV')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title('Plot of the signals for different trigger levels (l) measured on the oscilloscope.')
plt.legend(loc=2)
plt.grid()
plt.show()


t4 = data4['t']
V4 = data4['V']


plt.plot(t4, V4, label='Signal')
plt.axvline(0, color='black', linestyle=':', label='Trigger point, t = 0')
plt.axvline(19.95e-6, color='red', linestyle='--', label=r't = 19.95 $\mu$s')
plt.axvline(29.77e-6, color='red', linestyle='-.', label=r't = 29.77 $\mu$s')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title(r'Plot of the complex repeating signal, holdoff time = 19.95 $\mu$s')
plt.legend(loc=2)
plt.grid()
plt.show()