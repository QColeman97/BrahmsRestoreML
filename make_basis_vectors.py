import os, math
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# WAV FILE CREATION CODE IS BAD, USE AIFF
# from pydub import AudioSegment
# base_dir = os.getcwd()
# os.chdir('all_notes_ff')
# for file_name in os.listdir(os.getcwd()):
#     wav_file_name = AudioSegment.from_file(file_name)
#     wav_file_name.export(file_name[:-4] + "wav", format='wav')
# os.chdir(base_dir)

# "Constants"
std_sr_hz = 44100
piano_wdw_size = 16384 # 32768 # 16384 # 8192 # 4096 # 2048
res = std_sr_hz / piano_wdw_size
best_wdw_num = 2 # 10 for 2048 wdw size, 5 for 4096 wdw size, 3 for 8192 size, 2 for 16384 size, 1 for 32768 size (a number < num_smpls_of_smallest_recording / wdw_size)

# best_wdw_num is chosen empirically, careful as to not overshoot minimum recording length (24973 == ~12.19 (2048-size) wdws)
# TEST: From bigger window sizes downward, different window numbers - going up so to catch the note onset
# (What counts as first note to show FF is the start of consistent color change)
# 32768 size, window 1 -> 11th (12 to be safe) note (K) shows FF, window 2 -> breaks (white columns)
# 16384 size, window 1 -> too early (dark columns), window 2 -> 11th (11 safe enough, but def 12, for sure 15) note (K) shows FF, window 3 -> breaks
# 8192 size, window 2 -> too early (dark columns), window 3 -> 16th note (K) shows FF, window 4 -> 13th note (K) shows FF
# 4096 size, window 4 -> too early (dark columns), window 5 -> 18th note (K) shows FF
# 2048 size, window 11 -> 21st

# VARIABLE DECLARATION
# Piano roll goes from A0 to C8
sorted_notes = ["A0", "Bb0", "B0", "C1", 
                "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
                "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
                "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
                "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
                "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
                "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
                "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]
sorted_file_names = ["Piano.ff." + x + ".wav" for x in sorted_notes]

# Piano roll fundamental frequencies (Hz) - for spectrogram acc. measure
sorted_fund_freq = [28, 29, 31, 33, 
                    35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 
                    69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123, 131, 
                    139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 
                    277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 
                    554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 
                    1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 
                    2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186]

mag_stfts, basis_vectors = [], []
# min_len, max_len = None, None

# FUNCTIONS
def make_spectrogram(waveform):
    # print(type(waveform)) # 1D numpy array
    # Make a window that will work for all samples
    dfts, num_spls = [], len(waveform) # No window size, 1000 # 5000  # Min length @ 9216 (at top_db=30 for clip trim)
    num_wdws = math.ceil(num_spls / piano_wdw_size)
    # for i in range(8):  # 8 windows of 1000 timesteps each
    for i in range(num_wdws):
        wdw = waveform[i * piano_wdw_size: (i + 1) * piano_wdw_size]
        if len(wdw) != piano_wdw_size:
            deficit = piano_wdw_size - len(wdw)
            wdw = np.pad(wdw, (deficit, 0), mode='constant')
            # windowed_waveform = waveform[i * piano_wdw_size: (i + 1) * piano_wdw_size]
        
        # print('Lower:', i * piano_wdw_size, 'Upper:', (i + 1) * piano_wdw_size)

        # Pad zeros to wavefrom til length is power of 2
        # next_raise = math.ceil(math.log(piano_wdw_size,2))
        # deficit = int(math.pow(2, next_raise) - len(windowed_waveform))
        # windowed_waveform = np.pad(windowed_waveform, (deficit, 0), mode='constant')

        dft = np.log((np.abs(np.fft.fft(wdw)))[:(piano_wdw_size // 2) + 1])

        dfts.append(dft)

        # For real input, output had negative freq, so cut out redundant freq w/ slice (n / 2)
        # plt.plot(np.abs(dfts[i])[:(1000 // 2) + 1])
        # plt.title('Window ' + str(i + 1))
        # plt.ylabel('Amplitude')
        # plt.xlabel('Frequency (Hz)?')
        # plt.show()

    # plt.plot(np.abs(dft))
    # plt.plot(np.abs(dft[: (std_sr_hz // 2)]) * 2)
    # plt.ylabel('DFT Test')
    # plt.show()
    return np.array(dfts).T


def make_best_dft(waveform):
    # Note: best_wdw_num is naturally-indexed, so we kinda potentially overshoot recording length of small samples
    # wdw = waveform[best_wdw_num * piano_wdw_size: (best_wdw_num + 1) * piano_wdw_size]
    wdw = waveform[(best_wdw_num - 1) * piano_wdw_size: best_wdw_num * piano_wdw_size]
    # Don't need to pad a non-end window (most representative frequencies) OR DO I?
    if len(wdw) != piano_wdw_size:
            deficit = piano_wdw_size - len(wdw)
            wdw = np.pad(wdw, (deficit, 0), mode='constant')
    return np.log((np.abs(np.fft.fft(wdw)))[:(piano_wdw_size // 2) + 1])


def make_basis_vectors():
    # LOGIC
    base_dir = os.getcwd()
    os.chdir('all_notes_ff')
    # audio_files is a list of strings, need to sort it by note
    unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]   # used to be .aiff
    audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))
    for audio_file in audio_files:
    # for i in range(6):
        # audio_file = audio_files[i]
        sr, stereo_sig = wavfile.read(audio_file)

        # Convert to mono signal (take left channel) 
        sig = np.array([x[0] for x in stereo_sig]) 

        # Need to trim beginning/end silence off signal for basis vectors
        # Trim silence out - thresh: 0.1 % of max amplitude
        amp_thresh = max(sig) * 0.01
        while sig[0] < amp_thresh:
            sig = sig[1:]
            # num_spls -= 1
        while sig[-1] < amp_thresh:
            sig = sig[:-1]
            # num_spls -= 1

        # min_len = len(sig) if (min_len is None or len(sig) < min_len) else min_len
        # max_len = len(sig) if (max_len is None or len(sig) > max_len) else max_len

        # Append to collection of spectrograms
        # mag_stfts.append(make_spectrogram(sig))

        # Append to collection of best dfts (basis vectors)
        basis_vectors.append(make_best_dft(sig))

    # print('Min len:', min_len)    # Min len is 24973 at amp_thresh at 0.1% of max amp
    # print('Max len:', max_len)    # Max len is 3342220 at amp_thresh at 0.1% of max amp
    os.chdir(base_dir)

    basis_vectors = np.array(basis_vectors).T

    # Display spectrograms
    # for i, mag_stft in enumerate(mag_stfts):
    #     # if i > 5:
    #         # break

# Take relevant code from make_basis_vectors and put it in here
def make_spectrograms():
    pass
    
#     print('C4 Spectrogram Shape (frequency count = (wdw size/2)+1 , c4_num_wdws):', mag_stft.shape)

#     # Spectrogram
#     fig, ax = plt.subplots()
#     ax.title.set_text(audio_files[i] + ' Power Spectrogram ' + str(i) + '/88 notes, FF = ' + str(sorted_fund_freq[i]) + ' Hz')
#     ax.set_ylabel('Frequency (Hz)')
#     # c4_ax.set_xlabel('Windows (%.2f s intervals)' % (1 // std_sr_hz) * piano_wdw_size)
#     # Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
#     im = ax.imshow(mag_stft, extent=[0, mag_stft.shape[1], std_sr_hz / 2, 0])
#     fig.tight_layout()
#     # bottom, top = plt.ylim()
#     # print('Bottom:', bottom, 'Top:', top)
#     plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
#     ax.set_aspect(0.13)     # Set a visually nice ratio
#     plt.show()



# Display basis vectors
basis_vectors = make_basis_vectors()

print('Basis Vectors Shape (frequency count = (piano wdw size/2)+1 , num notes on piano):', basis_vectors.shape)
fig, ax = plt.subplots()
ax.title.set_text('Basis Vectors')
ax.set_ylabel('Frequency (Hz)')
# c4_ax.set_xlabel('Windows (%.2f s intervals)' % (1 // std_sr_hz) * piano_wdw_size)
# Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
im = ax.imshow(basis_vectors, extent=[0, basis_vectors.shape[1], std_sr_hz / 2, 0])
# im = ax.imshow(basis_vectors)
fig.tight_layout()
# bottom, top = plt.ylim()
# print('Bottom:', bottom, 'Top:', top)
plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
ax.set_aspect(0.01)     # Set a visually nice ratio
plt.show()
