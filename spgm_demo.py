from audio_data_processing import EPSILON, PIANO_WDW_SIZE, make_spectrogram
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from librosa.display import specshow
import librosa
from scipy.io import wavfile

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# dt = 0.0005
# t = np.arange(0.0, 20.0, dt)
# s1 = np.sin(2 * np.pi * 100 * t)
# s2 = 2 * np.sin(2 * np.pi * 400 * t)

# # create a transient "chirp"
# s2[t <= 10] = s2[12 <= t] = 0

# # add some noise into the mix
# nse = 0.01 * np.random.random(size=len(t))

# x = s1 + s2 + nse  # the signal
# NFFT = 1024  # the length of the windowing segments
# Fs = int(1.0 / dt)  # the sampling frequency

sr, sig = wavfile.read('brahms.wav')
sig = sig.astype('float64')
sig = np.mean(sig, axis=-1)

# fig, (ax1, ax2) = plt.subplots(nrows=2)
# # fig, ax1 = plt.subplots()
# ax1.plot(t, x)
# Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
# # Pxx, freqs, bins, im = ax1.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
# # print(Pxx)
# # print(freqs)
# # print(bins)
# # print(im)
# # The `specgram` method returns 4 objects. They are:
# # - Pxx: the periodogram
# # - freqs: the frequency vector
# # - bins: the centers of the time bins
# # - im: the .image.AxesImage instance representing the data in the plot
# # plt.show()

spgm, phses = make_spectrogram(sig, PIANO_WDW_SIZE, EPSILON, ova=True)
spgm = spgm.T
# # toy
# spgm = np.arange(0,12).reshape((3,4))
rows, cols = spgm.shape

def frequency_in_hz(x, pos):
    # return '%.1f Hz' % x
    return '%.2f Hz' % ((x * sr)/PIANO_WDW_SIZE)
# https://matplotlib.org/3.1.1/gallery/ticks_and_spines
formatter = FuncFormatter(frequency_in_hz)

# Graph the spectrogram from 2D array (maplotlib specgram doesn't do this)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
ax.title.set_text('Spectrogram')
ax.set_ylabel('Frequency')
ax.set_xlabel('Time Segments')

# y_ticks = [((y * sr) / PIANO_WDW_SIZE) for y in range((PIANO_WDW_SIZE//2)+1)]
# ax.set_yticks(y_ticks)

img = ax.imshow(np.log(spgm), 
                aspect=3, 
                origin='lower',
                extent=[-0.5, cols-0.5, -0.5, (rows//5)-0.5])
# img.set_extent([-0.5, cols-0.5, -0.5, (rows//2)-0.5])
# print(img.aspect)

# # plt.matshow(Pxx)

# # fig = plt.figure()
# # t_ax = fig.add_subplot(121)
# # t_ax.imshow(Pxx, interpolation='nearest', cmap=cm.Greys_r)

# # plt.pcolormesh(t, freqs, np.abs(Pxx))#, cmap=cm.Greys_r)
# plt.pcolormesh(np.abs(Pxx))#, cmap=cm.Greys_r)

# spgm, phses = make_spectrogram(sig, NFFT, EPSILON, ova=True)
# spgm = spgm.T
# spgm = librosa.amplitude_to_db(abs(spgm))
# print('Shape of My Spgm:', spgm.shape)
# print('My Spgm:', spgm[0])
# lib_stft = librosa.stft(sig, n_fft=NFFT, hop_length=NFFT//2)
# lib_spgm = librosa.amplitude_to_db(abs(lib_stft))
# print('Shape of Librosa Spgm:', lib_spgm.shape)
# print('Librosa Spgm:', lib_spgm[0])
# plt.figure(figsize=(15, 5))
# specshow(spgm, sr=sr, hop_length=NFFT//2, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.show()

# plt.figure(figsize=(15, 5))
# specshow(lib_spgm, sr=sr, hop_length=NFFT//2, x_axis='time', y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
plt.show()