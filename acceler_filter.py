from scipy.fftpack import fft, ifft
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Before importing the csv file, remember remove the header manually.
path = r'C:\Users\yyang\PycharmProjects\pythonProject\acelerometro_terra.csv'
data_X = pd.read_csv(path, low_memory=False, usecols=[2])  # column 2 is X-axis in accelerometer
data_Y = pd.read_csv(path, low_memory=False, usecols=[3])  # column 3 is Y-axis in accelerometer
data_z = pd.read_csv(path, low_memory=False, usecols=[4])  # column 4 is Z-axis in accelerometer
m = len(data_X)
print(m)

sample_count = m  # sample points
fs = 108
T = 1 / fs


# Band_pass filter
def fda(x_1, fstop1, fstop2):
    b, a = signal.butter(8, [2.0 * fstop1 / fs, 2.0 * fstop2 / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, x_1, axis=0)
    print('222')
    return filtedData


data_filter = fda(data_X, 48, 52)  # fs > 2*wn

xFFT = fft(data_X)  # FFT

x = np.linspace(0.0, 1269.0, m)
xFreqs = np.linspace(0.0, 1.0 / (2.0 * T), sample_count // 2)

plt.figure(figsize=(15, 5))
plt.plot(x, data_X, color='b')
plt.title("Non-Filtered data accelerometer X-axis ")
plt.xlabel('Measurement time(s)')
plt.ylabel(r'Amplitude(v)')
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(x, data_filter, color='b')
plt.title("Filtered data accelerometer X-axis ")
plt.xlabel('Measurement time(s)')
plt.ylabel(r'Amplitude(v)')
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(xFreqs, abs(xFFT[0:sample_count // 2]), color='b')
plt.title("FFT result for X-axis in accelerometer")
plt.xlabel('Frequency Domain(HZ)')
plt.ylabel(r'Amplitude(v)')
# plt.savefig('time_domain_for_single_chirp.jpg')
# print(int(xFreqs[np.argmax(xFFT[0:sample_count // 2], axis=0)]))
plt.show()
