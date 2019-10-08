import os, librosa
from librosa import display
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

# Piano roll goes from A0 to C8
sorted_notes = ["A0", "Bb0", "B0", "C1", 
                "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
                "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
                "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
                "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
                "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
                "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
                "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]
sorted_file_names = ["Piano.ff." + x + ".aiff" for x in sorted_notes]

# Piano roll fundamental frequencies (Hz) - for spectrogram acc. measure
sorted_fund_freq = [28, 29, 31, 33, 
                    35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 
                    69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123, 131, 
                    139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 
                    277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 
                    554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 
                    1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 
                    2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186]

mag_stfts, native_sr = [], None
# min_len, max_len = None, None
base_dir = os.getcwd()
os.chdir('all_notes_ff')
# audio_files is a list of strings, need to sort it by note
unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('aiff')]   # used to be .wav
audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))
for audio_file in audio_files:
# for i in range(6):
    # audio_file = audio_files[i]
    y, sr = librosa.load(audio_file, sr=None)    # Default sr: 22050 samples/s
    native_sr = sr if native_sr is None else None
    # print(audio_file)
    # print('\nWAVEFORM BELOW:')
    # print(y)
    yt, idx = librosa.effects.trim(y, top_db=30)    # 60 leaves too much quiet in, 120 even more
    # print('\nTRIMMED WAVEFORM BELOW:')
    # print(yt)
    # print('Type:', type(yt), 'Shape:', np.shape(yt))
    # print('Len of yt:', len(yt))
    # print('Len of yt[:50000]:', len(yt[:50000]))
    # min_len = len(yt) if (min_len is None or len(yt) < min_len) else min_len
    # max_len = len(yt) if (min_len is None or len(yt) < max_len) else max_len
    # mag_stfts.append(np.abs(librosa.stft(yt[:10000]))) # TODO: Find best standard length (window)
    mag_stfts.append(np.abs(librosa.stft(yt)))
# print('Min len:', min_len)    # Min len is 205946 (~9.3 s) at top_db=60, 9216 (~0.4 s) at top_db=30
# print('Max len:', max_len)    # Max len is  ( s) at top_db=60,  (138240 s) at top_db=30
os.chdir(base_dir)

# print('AUDIO FILE:')
# print(mag_stfts[0])

# basis_vectors = np.empty(shape=(np.shape(mag_stfts[0])[0], 2250 * 88))
# for i in range(len(mag_stfts)):
#     basis_vectors[:, i * 2250] = mag_stfts[i][:, 4500:6750]

# display.specshow(librosa.amplitude_to_db(basis_vectors, ref=np.max),
#                  y_axis='log',
#                  x_axis='time')
# plt.title('Learned Basis Vectors')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()  # Toggle show plot

for i in range(len(audio_files)):
    display.specshow(librosa.amplitude_to_db(mag_stfts[i], ref=np.max),
                     y_axis='log',
                     x_axis='time',
                     sr=native_sr)
    # if i % 2 == 0:
        # plt.title(audio_files[i] + ' SHORT Power Spectrogram ' + str((i // 2) + 1) + '/88 notes, FF = ' + str(sorted_fund_freq[i // 2]) + ' Hz')
    # else:
        # plt.title(audio_files[i] + ' Power Spectrogram ' + str((i // 2) + 1) + '/88 notes, FF = ' + str(sorted_fund_freq[i // 2]) + ' Hz')
    plt.title(audio_files[i] + ' Power Spectrogram ' + str(i + 1) + '/88 notes, FF = ' + str(sorted_fund_freq[i]) + ' Hz')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()  # Toggle show plot



