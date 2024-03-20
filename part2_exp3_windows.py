import numpy as np
from funcs import *

data_ft = np.loadtxt("data_part2_exp3_flattop.txt", unpack=True)
data_han = np.loadtxt("data_part2_exp3_hanning.txt", unpack=True)
data_rect = np.loadtxt("data_part2_exp3_rectangular.txt", unpack=True)

'''# flattop window
p0_ft = [60, 10000, 20, -60]

p_ft, chi_score, err_ft = fitting(p0_ft, data_ft[0], data_ft[1], gauss)


plt.plot(data_ft[0], data_ft[1])
plt.plot(data_ft[0], gauss(p_ft, data_ft[0]))
plt.title('Frequency spectrum for the flattop window')
plt.ylabel('A [dBV]')
plt.xlabel('f [Hz]')
plt.show()

# Round
p_ft, err_ft = round_sig_fig_uncertainty(p_ft, err_ft)

# Peak
height_ft = np.max(data_ft[1])

diffs_ft = np.abs(np.diff(data_ft[1]))
step_size_ft = np.min(diffs_ft[diffs_ft!=0])

height_ft_err = step_size_ft

height_ft, height_ft_err = round_sig_fig_uncertainty(height_ft, height_ft_err)

rms_ft = 10**(height_ft/20)
rms_ft_err = 1/20 * 10**(height_ft/20) * np.log(10) * height_ft_err

rms_ft, rms_ft_err = round_sig_fig_uncertainty(rms_ft, rms_ft_err)

print('---Flattop window---')
print(f'Peak frequency {p_ft[1]} \u00B1 {err_ft[1]} Hz')
print(f'Peak height {height_ft} \u00B1 {height_ft_err} dBV')
print(f'V_rms {rms_ft} \u00B1 {rms_ft_err} dBV')


# hanning window
p0_han = [60, 10000, 20, -60]

p_han, chi_score, err_han = fitting(p0_han, data_han[0], data_han[1], gauss)


plt.plot(data_han[0], data_han[1])
plt.plot(data_han[0], gauss(p_han, data_han[0]))
plt.title('Frequency spectrum for the Hanning window')
plt.ylabel('A [dBV]')
plt.xlabel('f [Hz]')
plt.show()

# Round
p_han, err_han = round_sig_fig_uncertainty(p_han, err_han)

# Peak
height_han = np.max(data_han[1])

diffs_han = np.abs(np.diff(data_han[1]))
step_size_han = np.min(diffs_han[diffs_han!=0])

height_han_err = step_size_han

height_han, height_han_err = round_sig_fig_uncertainty(height_han, height_han_err)

rms_han = 10**(height_han/20)
rms_han_err = 1/20 * 10**(height_han/20) * np.log(10) * height_han_err

rms_han, rms_han_err = round_sig_fig_uncertainty(rms_han, rms_han_err)

print('---Hanning window---')
print(f'Peak frequency {p_han[1]} \u00B1 {err_han[1]} Hz')
print(f'Peak height {height_han} \u00B1 {height_han_err} dBV')
print(f'V_rms {rms_han} \u00B1 {rms_han_err} dBV')

# rectangular window
p0_rect = [60, 10000, 20, -60]

p_rect, chi_score, err_rect = fitting(p0_rect, data_rect[0], data_rect[1], lorentz)


plt.plot(data_rect[0], data_rect[1])
plt.plot(data_rect[0], lorentz(p_rect, data_rect[0]))
plt.title('Frequency spectrum for the flattop window')
plt.ylabel('A [dBV]')
plt.xlabel('f [Hz]')
plt.show()

# Round
p_rect, err_rect = round_sig_fig_uncertainty(p_rect, err_rect)

# Peak
height_rect = np.max(data_rect[1])

diffs_rect = np.abs(np.diff(data_rect[1]))
step_size_rect = np.min(diffs_rect[diffs_rect!=0])

height_rect_err = step_size_rect

height_rect, height_rect_err = round_sig_fig_uncertainty(height_rect, height_rect_err)

rms_rect = 10**(height_rect/20)
rms_rect_err = 1/20 * 10**(height_rect/20) * np.log(10) * height_rect_err

rms_rect, rms_rect_err = round_sig_fig_uncertainty(rms_rect, rms_rect_err)

print('---Rectangular window---')
print(f'Peak frequency {p_rect[1]} \u00B1 {err_rect[1]} Hz')
print(f'Peak height {height_rect} \u00B1 {height_rect_err} dBV')
print(f'V_rms {rms_rect} \u00B1 {rms_rect_err} dBV')
'''
# Not fitting

peak_ft_loc = np.argwhere(data_ft[1] == np.amax(data_ft[1]))
height_ft = np.mean(data_ft[1][peak_ft_loc])

diffs_ft = np.abs(np.diff(data_ft[1]))
step_size_ft = np.min(diffs_ft[diffs_ft!=0])

height_ft_err = step_size_ft
height_ft, height_ft_err = round_sig_fig_uncertainty(height_ft, height_ft_err)

rms_ft = 10**(height_ft/20)
rms_ft_err = 1/20 * 10**(height_ft/20) * np.log(10) * height_ft_err
rms_ft, rms_ft_err = round_sig_fig_uncertainty(rms_ft, rms_ft_err)


peak_ft = np.mean(data_ft[0][peak_ft_loc])

if np.size(peak_ft_loc) != 1:
    peak_ft_err = np.std(peak_ft)
else:
    peak_ft_err = np.diff(data_ft[0])[0]

peak_ft, peak_ft_err = round_sig_fig_uncertainty(peak_ft, peak_ft_err)

print('--- Flattop window ---')
print(f'Peak frequency = {peak_ft} \u00B1 {peak_ft_err} Hz')
print(f'Peak height = {height_ft} \u00B1 {height_ft_err} dBV')
print(f'V_rms = {rms_ft} \u00B1 {rms_ft_err} V')

peak_han_loc = np.argwhere(data_han[1] == np.amax(data_han[1]))
height_han = np.mean(data_han[1][peak_han_loc])

diffs_han = np.abs(np.diff(data_han[1]))
step_size_han = np.min(diffs_han[diffs_han!=0])

height_han_err = step_size_han
height_han, height_han_err = round_sig_fig_uncertainty(height_han, height_han_err)

rms_han = 10**(height_han/20)
rms_han_err = 1/20 * 10**(height_han/20) * np.log(10) * height_han_err
rms_han, rms_han_err = round_sig_fig_uncertainty(rms_han, rms_han_err)


peak_han = np.mean(data_han[0][peak_han_loc])

if np.size(peak_han_loc) != 1:
    peak_han_err = np.std(peak_han)
else:
    peak_han_err = np.diff(data_han[0])[0]

peak_han, peak_han_err = round_sig_fig_uncertainty(peak_han, peak_han_err)

print('--- Hanning window ---')
print(f'Peak frequency = {peak_han} \u00B1 {peak_han_err} Hz')
print(f'Peak height = {height_han} \u00B1 {height_han_err} dBV')
print(f'V_rms = {rms_han} \u00B1 {rms_han_err} V')

peak_rect_loc = np.argwhere(data_rect[1] == np.amax(data_rect[1]))
height_rect = np.mean(data_rect[1][peak_rect_loc])

diffs_rect = np.abs(np.diff(data_rect[1]))
step_size_rect = np.min(diffs_rect[diffs_rect!=0])

height_rect_err = step_size_rect
height_rect, height_rect_err = round_sig_fig_uncertainty(height_rect, height_rect_err)

rms_rect = 10**(height_rect/20)
rms_rect_err = 1/20 * 10**(height_rect/20) * np.log(10) * height_rect_err
rms_rect, rms_rect_err = round_sig_fig_uncertainty(rms_rect, rms_rect_err)


peak_rect = np.mean(data_rect[0][peak_rect_loc])

if np.size(peak_rect_loc) != 1:
    peak_rect_err = np.std(peak_rect)
else:
    peak_rect_err = np.diff(data_rect[0])[0]

peak_rect, peak_rect_err = round_sig_fig_uncertainty(peak_rect, peak_rect_err)

print('--- Rectangular window ---')
print(f'Peak frequency = {peak_rect} \u00B1 {peak_rect_err} Hz')
print(f'Peak height = {height_rect} \u00B1 {height_rect_err} dBV')
print(f'V_rms = {rms_rect} \u00B1 {rms_rect_err} V')
