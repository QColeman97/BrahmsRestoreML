import wave, struct, math
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Brahms sr, channel # = Piano sr, channel #
# Brahms recording is 16 bit / sample (sample width = 2)
# Piano note recording is 8 bit / sample (sample width = 1)

# "Constants"
std_sr_hz = 44100
piano_wdw_size = 4096 # 2048
res = std_sr_hz / piano_wdw_size

# 1. Get the file path to the included audio example
c4_filepath = 'all_notes_ff/Piano.ff.C4.wav'
brahms_filepath = 'brahms.wav'

# 2. Load the audio as a waveform `signal` & its attributes
# C4 Recording
sr, c4_stereo_sig = wavfile.read(c4_filepath)
c4_num_chnls = len(c4_stereo_sig[0])
c4_num_spls = len(c4_stereo_sig)
# print('Len of c4_sig:', c4_num_spls / std_sr_hz)
# Brahms Recording
sr, bm_stereo_sig = wavfile.read(brahms_filepath)
bm_num_chnls = len(bm_stereo_sig[0])
bm_num_spls = len(bm_stereo_sig)
print('Len of brahms sig:', bm_num_spls / std_sr_hz)

# Convert from stereo to mono (take left channel)
c4_sig = np.array([x[0] for x in c4_stereo_sig])
bm_sig = np.array([x[0] for x in bm_stereo_sig])  

print('first part of brahms sig:', bm_sig[:(bm_num_spls // 60) * 5])

# Not necessary to trim out silence
# Trim silence out - thresh: 0.1 % of max amplitude
# print('Samples:', len(c4_sig))
# print(max(c4_sig))
# amp_thresh = max(c4_sig) * 0.01
# while c4_sig[0] < amp_thresh:
#     c4_sig = c4_sig[1:]
#     c4_num_spls -= 1
# while c4_sig[-1] < amp_thresh:
#     c4_sig = c4_sig[:-1]
#     c4_num_spls -= 1
# print('Samples:', len(c4_sig))
# print(c4_sig[:2], c4_sig[-2:])

# 3. Create the short-time fourier transfrom of 'y'
# C4 Recording
# np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
c4_dfts = []
c4_num_wdws = math.ceil(c4_num_spls / piano_wdw_size)
for i in range(c4_num_wdws):
# for i in range(3): # For printing each dft
    wdw = c4_sig[i * piano_wdw_size: (i + 1) * piano_wdw_size]
    if len(wdw) != piano_wdw_size:
        deficit = piano_wdw_size - len(wdw)
        wdw = np.pad(wdw, (deficit, 0), mode='constant')

    # Take magnitude, and only section that isn't complex conjugate
    dft = np.log((np.abs(np.fft.fft(wdw)))[:(piano_wdw_size // 2) + 1])
    c4_dfts.append(dft)
    # plt.plot(dft)
    # plt.show()
# Spectrogram matrix w/ correct orientation
c4_dfts = np.array(c4_dfts).T
print('C4 Spectrogram Shape (frequency count = (wdw size/2)+1 , c4_num_wdws):', c4_dfts.shape)

# Brahms Recording
# np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
bm_dfts = []
bm_num_wdws = math.ceil(bm_num_spls / piano_wdw_size)
for i in range(bm_num_wdws):
# for i in range(3): # For printing each dft
    wdw = bm_sig[i * piano_wdw_size: (i + 1) * piano_wdw_size]
    if len(wdw) != piano_wdw_size:
        deficit = piano_wdw_size - len(wdw)
        wdw = np.pad(wdw, (deficit, 0), mode='constant')

    # Take magnitude, and only section that isn't complex conjugate
    # dft = np.log((np.abs(np.fft.fft(wdw)))[:(piano_wdw_size // 2) + 1])
    # dft = np.abs(np.fft.fft(wdw))[:(piano_wdw_size // 2) + 1] - 10000

    # Replace values that give -inf values
    dft = np.fft.fft(wdw)
    if i == 0:
        print('complex DFT:', dft)
    dft = np.abs(dft)
    if i == 0:
        print('mag DFT:', dft)
    dft = dft[:(piano_wdw_size // 2) + 1]
    # dft[dft == 0] = 0.0001
    dft = np.log(dft)
    if i == 0:
        print('log mag DFT:', dft)

    bm_dfts.append(dft)
    # plt.plot(dft)
    # plt.show()

#DEBUG:
print('Brahms First DFT:', bm_dfts[0])

# Spectrogram matrix w/ correct orientation
bm_dfts = np.array(bm_dfts).T
print('Brahms Spectrogram Shape (frequency count = (wdw size/2)+1, bm_num_wdws):', bm_dfts.shape)

# TODO: Step frequency labels by window size (2048)?

# 4. Display spectrogram
# C4 Spectrogram
c4_fig, c4_ax = plt.subplots()
c4_ax.title.set_text('C4 Spectrogram')
c4_ax.set_ylabel('Frequency (Hz)')
# c4_ax.set_xlabel('Windows (%.2f s intervals)' % (1 // std_sr_hz) * piano_wdw_size)
# Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
im = c4_ax.imshow(c4_dfts, extent=[0, c4_num_wdws, std_sr_hz / 2, 0])
c4_fig.tight_layout()
# bottom, top = plt.ylim()
# print('Bottom:', bottom, 'Top:', top)
plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
c4_ax.set_aspect(0.08)     # Set a visually nice ratio
plt.show()

# Brahms Spectrogram
bm_fig, bm_ax = plt.subplots()
bm_ax.title.set_text('Brahms Spectrogram')
bm_ax.set_ylabel('Frequency (Hz)')
# ax.set_xlabel('Windows (%.2f s intervals)' % (1 // std_sr_hz) * piano_wdw_size)
# Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
im = bm_ax.imshow(bm_dfts, extent=[0, bm_num_wdws, std_sr_hz / 2, 0])
# im = bm_ax.imshow(bm_dfts)
bm_fig.tight_layout()
# bottom, top = plt.ylim()
# print('Bottom:', bottom, 'Top:', top)
plt.ylim(8000.0, 0.0)  # Crop an axis (to ~double the piano frequency max # (8000.0))
bm_ax.set_aspect(0.13)     # Set a visually nice ratio
plt.show()


# FYI: other heatmap method
# plt.imshow(c4_dfts, cmap='hot', interpolation='nearest')
# plt.show()





