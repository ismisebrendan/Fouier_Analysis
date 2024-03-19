import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

data1 = ascii.read("data_part1_exp2_ch1.txt")
data2 = ascii.read("data_part1_exp2_ch2.txt")

t1 = data1['t']
V1 = data1['V']

t2 = data2['t']
V2 = data2['V']

plt.plot(t1, V1, label='Channel 1')
plt.plot(t2, V2, label='Channel 2')
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title('Plot of two square waves measured on the oscilloscope.')
plt.legend()
plt.grid()
plt.show()

