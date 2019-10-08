import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

# 1. Get the file path to the included audio example
# filename = librosa.util.example_audio_file()
c4_filename = 'Piano.mf.C4.wav'
db4_filename = 'Piano.mf.Db4.wav'
d4_filename = 'Piano.mf.D4.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
# Using native sampling rate f's up spectrogram, so didn't
# c4_y, c4_sr = librosa.load(c4_filename, sr=None)
c4_y, c4_sr = librosa.load(c4_filename)
print('C4 - Waveform length:\n', len(c4_y), '\nSampling Rate (Hz):', c4_sr)

db4_y, db4_sr = librosa.load(db4_filename)
print('Db4 - Waveform length:\n', len(db4_y), '\nSampling Rate (Hz):', db4_sr)

d4_y, d4_sr = librosa.load(d4_filename)
print('D4 - Waveform length:\n', len(d4_y), '\nSampling Rate (Hz):', d4_sr)

# 3. Create the short-time fourier transfrom of 'y'
# np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
c4 = np.abs(librosa.stft(c4_y))
db4 = np.abs(librosa.stft(db4_y))
d4 = np.abs(librosa.stft(d4_y))

# Display spectrogram - librosa display func
display.specshow(librosa.amplitude_to_db(c4, ref=np.max),
                 y_axis='log',
                 x_axis='time')
plt.title('C4 Power Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
# plt.show()  # Toggle show plot

display.specshow(librosa.amplitude_to_db(db4, ref=np.max),
                 y_axis='log',
                 x_axis='time')
plt.title('Db4 Power Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
# plt.show()  # Toggle show plot

display.specshow(librosa.amplitude_to_db(d4, ref=np.max),
                 y_axis='log',
                 x_axis='time')
plt.title('D4 Power Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
# plt.show()  # Toggle show plot

# 4. Check: reconstruct waveform from short-term fourier transfrom
#    Turn reconstructed wavefrom into new WAV file to play
# rec_y = librosa.istft(d)

# reconst_filename = 'recon_0a5cbf90.wav'
# librosa.output.write_wav(reconst_filename, rec_y, sr)

