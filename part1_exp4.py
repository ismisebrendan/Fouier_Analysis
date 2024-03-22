import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

data = ascii.read("data_part1_exp4.txt")

t = data['t']
V = data['V']

peak = max(V)

print(f'Pulse height: {peak} V')

ind = V > 1.5

t_upper = t[ind]

print(f'Pulse width: {t_upper[-1] - t_upper[0]} s')

print(np.argmax(V))

plt.plot(t,V)
plt.xlabel('t [s]')
plt.ylabel('V [V]')
plt.title('Plot of a single pulse measured on the oscilloscope.')
plt.grid()
plt.show()