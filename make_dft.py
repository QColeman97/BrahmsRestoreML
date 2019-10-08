import librosa, math
import numpy as np
import matplotlib.pyplot as plt

# Exceeds recursion limit
# def FFT(waveform):
#     N = waveform.shape[0]
#     X_even = FFT(windowed_waveform[::2])
#     X_odd = FFT(windowed_waveform[1::2])
#     factor = np.exp(-2j * np.pi * np.arange(N) / N)
#     return np.concatenate([X_even + factor[:N / 2] * X_odd,
#                             X_even + factor[N / 2:] * X_odd])

def FFT_vectorized(x):
    N = x.shape[0]
    # N_min = min(N, 32)
    
    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])

    return X.ravel()

# Note: Use AIFF file for original audio
c4_pathname = 'all_notes_ff/Piano.ff.C4.aiff'

# waveform, sr = librosa.load(c4_pathname, sr=None, mono=False) # Completely non-destructive
waveform, sr = librosa.load(c4_pathname, sr=None)
print('C4 - Time Steps:', len(waveform), '\nNative Sampling Rate (Hz):', sr)
print('Sec of recording:', len(waveform) / sr)

trimmed_waveform, idx = librosa.effects.trim(waveform, top_db=30) 
print('Sample trimmed at this index:', idx)
print('Sec of trimmed recording:', len(trimmed_waveform) / sr)

# DO DFT ON THE WAVEFORM
window_size = int(len(trimmed_waveform) / 3)
windowed_waveform = trimmed_waveform[:window_size]
# Pad the waveform w/ zeros until length is power of 2
next_raise = math.ceil(math.log(window_size,2))
deficit = int(math.pow(2, next_raise) - len(windowed_waveform))
windowed_waveform = np.pad(windowed_waveform, (deficit, 0), mode='constant')
print('Sec of window:', len(windowed_waveform) / sr, 'Length of waveform:', len(windowed_waveform))
# print(type(windowed_waveform))
# print('Waveform:')
# print(windowed_waveform.shape)
plt.plot(windowed_waveform)
plt.show()

# Create our x-axis (frequency bins) of the DFT, length k = nyquist limit? (sr / 2)
# freq_bins, num_bins = [], (sr // 2)
# for k in range(num_bins):
#     freq_bin = np.zeros((2))
#     for n in range(len(windowed_waveform)):
#         x = (-1 * 2 * math.pi * k * n) / len(windowed_waveform)
#         # freq_bin += windowed_waveform[n] * complex(math.cos(x), math.sin(x))
#         freq_bin += windowed_waveform[n] * np.array([math.cos(x), math.sin(x)])
#     freq_bins.append(complex(freq_bin[0], freq_bin[1]))
# freq_bins = np.array(freq_bins)

# DFT - Faster than original DFT alg.
# N = windowed_waveform.shape[0]
# n = np.arange(N)
# k = n.reshape((N, 1))
# M = np.exp(-2j * np.pi * k * n / N)
# freq_bins = np.dot(M, windowed_waveform)

# FFT - Faster than my DFT alg.
# freq_bins = FFT(windowed_waveform)
freq_bins = FFT_vectorized(windowed_waveform)

# print('DFT shape:', freq_bins.shape)

mag_freq_bins = np.abs(freq_bins[: (sr // 2)])
print('Length of DFT:', len(mag_freq_bins))
print('Freq. resolution of bins:', sr / len(windowed_waveform))

plt.plot(mag_freq_bins)
plt.show()

# If a = complex(0,-4) that's 0-4j. The magnitude or abs(a) is 4!