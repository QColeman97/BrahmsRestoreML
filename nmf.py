import os, math
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# NMF Basics
# V (FxT) = W (FxC) @ H (CxT)
# Dimensions: F = # freq. bins, C = # components(piano keys), T = # windows in time/timesteps
# Matrices: V = Spectrogram, W = Basis Vectors, H = Activations

# Flag, change to False to run
debug = True
# Debug a len=8 array ([01011001]), and the original non-synthesized waveform/spectrogram

# Get rid of np.log and np.exp b/c these just for visualization ease

# Constants
SORTED_NOTES = ["A0", "Bb0", "B0", "C1", 
                "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
                "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
                "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
                "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
                "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
                "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
                "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]

SORTED_FUND_FREQ = [28, 29, 31, 33, 
                    35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 
                    69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123, 131, 
                    139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 
                    277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 
                    554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 
                    1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 
                    2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186]

# Activation Matrix (H) Learning Part
MAX_LEARN_ITER = 100

# Basis Vectors (W) Part
STD_SR_HZ = 44100
piano_wdw_size = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
res = STD_SR_HZ / piano_wdw_size
best_wdw_num = 5
# num_notes = len(SORTED_NOTES)

# Spectrogram (V) Part
brahms_filepath = 'brahms.wav'
synthetic_brahms_filepath = 'synthetic_brahms.wav'


# Functions
def make_best_dft(waveform, wdw_num):
    # Note: wdw_num is naturally-indexed
    wdw = waveform[(wdw_num - 1) * piano_wdw_size: wdw_num * piano_wdw_size]
    if len(wdw) != piano_wdw_size:
            deficit = piano_wdw_size - len(wdw)
            wdw = np.pad(wdw, (deficit, 0), mode='constant')
    return np.log((np.abs(np.fft.fft(wdw)))[:(piano_wdw_size // 2) + 1])
    # return np.abs(np.fft.fft(wdw))[:(piano_wdw_size // 2) + 1]

# Learning optimization
def make_row_sum_matrix(mtx, out_shape):
    row_sums = mtx.sum(axis=1)
    return np.repeat(row_sums, out_shape[1], axis=0)

# W LOGIC
def make_basis_vectors():
    basis_vectors = []
    base_dir = os.getcwd()
    os.chdir('all_notes_ff')
    # audio_files is a list of strings, need to sort it by note
    unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]
    sorted_file_names = ["Piano.ff." + x + ".wav" for x in SORTED_NOTES]
    audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))
    for audio_file in audio_files:
        sr, stereo_sig = wavfile.read(audio_file)

        # Convert to mono signal (take left channel) 
        sig = np.array([x[0] for x in stereo_sig]) 

        # Need to trim beginning/end silence off signal for basis vectors
        amp_thresh = max(sig) * 0.01
        while sig[0] < amp_thresh:
            sig = sig[1:]
        while sig[-1] < amp_thresh:
            sig = sig[:-1]

        basis_vectors.append(make_best_dft(sig, best_wdw_num))

    os.chdir(base_dir)
    return np.array(basis_vectors).T     # T Needed? Yes

# V LOGIC
def make_spectrogram(wav_filepath):
    sr, stereo_sig = wavfile.read(wav_filepath)
    num_chnls = len(stereo_sig[0])
    num_spls = len(stereo_sig)

    sig = np.array([x[0] for x in stereo_sig])

    spectrogram, phases = [], []
    num_wdws = math.ceil(num_spls / piano_wdw_size)
    # print('Original Sig:\n', sig[piano_wdw_size * (num_wdws // 2): (piano_wdw_size * (num_wdws // 2)) + 10])
    for i in range(num_wdws):
        wdw = sig[i * piano_wdw_size: (i + 1) * piano_wdw_size]
        if len(wdw) != piano_wdw_size:
            deficit = piano_wdw_size - len(wdw)
            wdw = np.pad(wdw, (deficit, 0), mode='constant')

        # Replace values that give -inf values
        # dft = np.abs(np.fft.fft(wdw))[:(piano_wdw_size // 2) + 1]
        if i == (num_wdws // 2):
            print('Original window:\n', wdw[:10])
            print('Original window length:', len(wdw))
            dft = np.fft.fft(wdw)
            print('FFT of wdw:\n', dft[:5])
            print('FFT of wdw length:', len(dft))
            dft = np.abs(dft)
            print('mag FFT of wdw:\n', dft[:5])
            phase_of_dft = np.angle(np.fft.fft(wdw))[:(piano_wdw_size // 2) + 1] # +1 Get's the DC = 0 Hz?
            print('phases:\n', phase_of_dft[:5])
            dft = dft[:(piano_wdw_size // 2) + 1] # +1 Get's the DC = 0 Hz?
            print('positive FFT length:', len(dft))
            dft[dft == 0] = 0.0001
            # print('mag FFT (0 -> 0.0001) of wdw:\n', dft[:5])
            dft = np.log(dft)
            print('log of mag FFT of wdw:\n', dft[:5])
            
        else:
            dft = np.fft.fft(wdw)
            dft = np.abs(dft)
            dft = dft[:(piano_wdw_size // 2) + 1]
            dft[dft == 0] = 0.0001
            dft = np.log(dft)

            phase_of_dft = np.angle(np.fft.fft(wdw))[:(piano_wdw_size // 2) + 1]
            # phase_of_dft = np.angle(np.fft.fft(wdw))

        spectrogram.append(dft)
        phases.append(phase_of_dft)

    # Spectrogram matrix w/ correct orientation
    return np.array(spectrogram).T, phases  # T Needed? Yes

# H LOGIC - Learn / Approximate Activation Matrix
# Main dimensions: freq bins = spectrogram.shape[0], piano keys (components?) = num_notes OR basis_vectors.shape[1], windows = bm_num_wdws OR spectrogram.shape[1]
def make_activations(spectrogram, basis_vectors):
    # CHANGE BACK IF BUG:
    # activations = np.random.rand(num_notes, bm_num_wdws)
    activations = np.random.rand(basis_vectors.shape[1], spectrogram.shape[1])
    for i in range(MAX_LEARN_ITER):
        # H +1 = H * ((Wt dot (V / (W dot H)) / (Wt dot 1) )
        ones = np.ones(spectrogram.shape) # so dimenstions match W transpose dot w/ V
        activations *= (basis_vectors.T @ (spectrogram / (basis_vectors @ activations))) / (basis_vectors.T @ ones)
        # UNCOMMENT OPTIMIZATION:
        # denom = make_row_sum_matrix(basis_vectors.T, spectrogram.shape)
        # activations *= (basis_vectors.T @ (spectrogram / (basis_vectors @ activations))) / denom
    return activations

# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases):
    num_wdws = synthetic_spgm.shape[1]
    # For waveform construction
    synthetic_spgm = synthetic_spgm.T     # Get back into orientation we did calculations on
    # Construct synthetic waveform
    synthetic_sig = []
    for i in range(num_wdws):
        if i == (num_wdws // 2):
            # Inverse the log operation
            dft = synthetic_spgm[i]
            # print('mag FFT of wdw:\n', dft[:5])
            print('log of mag FFT of wdw:\n', dft[:5])
            print('log of mag FFT length:', len(dft))
            dft = np.exp(dft)
            print('exp of log of mag FFT of wdw:\n', dft[:5])
            # Restore original values as before
            # dft[dft <= 0.0001] = 0
            # print('mag FFT (0.0001 -> 0) of wdw:\n', dft[:5])
            # Append the mirror of the synthetic magnitudes to itself

            # dft = dft[: piano_wdw_size // 2] # Eliminate extraneous data point (last element)
            # dft = np.append(dft, np.flip(dft, 0), axis=0)
            dft = np.append(dft, np.flip(dft[: piano_wdw_size // 2], 0), axis=0)

            print('mag FFT of wdw:\n', dft[:5])
            print('mag FFT of wdw (mirrored part):\n', dft[-5:])
            print('pos and neg mag FFT length:', len(dft))
            # Multiply this magnitude spectrogram w/ phase

            # phase = phases[i][: piano_wdw_size // 2] # Eliminate extraneous data point
            # phase = np.append(phase, np.flip(phase, 0), axis=0)
            phase = phases[i]
            phase = np.append(phase, np.flip(phase[: piano_wdw_size // 2], 0), axis=0)

            print('phase of wdw:\n', phase[:5])
            # print(wdw.shape, phase.shape)
            # wdw = wdw * phase     # <- too simple and wrong
            dft = dft * np.exp(1j*phase)
            # wdw = np.array([complex(x[0], x[1]) for x in list(zip(wdw.tolist(), phase.tolist()))])
            # wdw = complex(wdw, phase)
            print('FFT of wdw:\n', dft[:5])

            synthetic_wdw = np.fft.ifft(dft)
            imaginaries = synthetic_wdw.imag.tolist()
            synthetic_wdw = synthetic_wdw.real.tolist()
            print('Synthetic imaginaries:\n', imaginaries[:10])
            print('Synthetic window:\n', synthetic_wdw[:10])
        else:
            # dft = synthetic_spgm[i]
            # Inverse the log operation
            dft = np.exp(synthetic_spgm[i])
            # Restore original values as before
            # dft[dft <= 0.0001] = 0
            # Append the mirror of the synthetic magnitudes to itself
            # dft = dft[: piano_wdw_size // 2] # Eliminate extraneous data point
            dft = np.append(dft, np.flip(dft[: piano_wdw_size // 2], 0), axis=0)
            # Multiply this magnitude spectrogram w/ phase
            # phase = phases[i][: piano_wdw_size // 2] # Eliminate extraneous data point
            phase = phases[i]
            phase = np.append(phase, np.flip(phase[: piano_wdw_size // 2], 0), axis=0)
            # print(wdw.shape, phase.shape)
            # wdw = wdw * phase     # <- too simple and wrong
            dft = dft * np.exp(1j*phase)
            # Below doesn't work, b/c subbing magnitude for the real part
            # wdw = np.array([complex(x[0], x[1]) for x in list(zip(wdw.tolist(), phase.tolist()))])
            # wdw = complex(wdw, phase)
            synthetic_wdw = np.fft.ifft(dft)
            # imaginaries = synthetic_wdw.imag.tolist()
            synthetic_wdw = synthetic_wdw.real.tolist()
            
        # Do ifft on the spectrogram -> waveform
        # synthetic_sig += np.fft.ifft(wdw).real.tolist()
        synthetic_sig += synthetic_wdw

    return synthetic_sig


def main():
    basis_vectors = make_basis_vectors()

    spectrogram, phases = make_spectrogram(brahms_filepath)
    bm_num_wdws = spectrogram.shape[1] # Windows/Time Dimension

    print('Shape of Spectrogram V:', spectrogram.shape)
    print('Shape of Basis Vectors W:', basis_vectors.shape)

    print('Learning Activations...')
    activations = make_activations(spectrogram, basis_vectors)
    print('Shape of Activations H:', activations.shape)

    synthetic_spgm = basis_vectors @ activations

    print('---SYNTHETIC SPGM TRANSITION----')
    synthetic_sig = make_synthetic_signal(synthetic_spgm, phases)

    # Make synthetic WAV file
    wavfile.write(synthetic_brahms_filepath, STD_SR_HZ, np.array(synthetic_sig))


if __name__ == '__main__':
    main()