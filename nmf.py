import os, math
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# NMF Basics
# V (FxT) = W (FxC) @ H (CxT)
# Dimensions: F = # freq. bins, C = # components(piano keys), T = # windows in time/timesteps
# Matrices: V = Spectrogram, W = Basis Vectors, H = Activations

# Numpy notes:
# Return value fft and input of ifft below
# a[0] should contain the zero frequency term,
# a[1:n//2] should contain the positive-frequency terms,
# a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.

# Don't include duplicate 0Hz, include n//2 spot

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

STD_SR_HZ = 44100
PIANO_WDW_SIZE = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
DEBUG_WDW_SIZE = 4
RES = STD_SR_HZ / PIANO_WDW_SIZE
BEST_WDW_NUM = 5
# Activation Matrix (H) Learning Part
MAX_LEARN_ITER = 100

# Spectrogram (V) Part
brahms_filepath = 'brahms.wav'
synthetic_brahms_filepath = 'synthetic_brahms.wav'
debug_filepath = 'brahms_debug.wav'


# Functions
# Flag rid_log: get rid of np.log and np.exp b/c these just for visualization ease, supposedly
def make_best_dft(waveform, wdw_num, wdw_size, rid_log=False):
    wdw = waveform[(wdw_num - 1) * wdw_size: wdw_num * wdw_size]    # wdw_num is naturally-indexed
    if len(wdw) != wdw_size:
            deficit = wdw_size - len(wdw)
            wdw = np.pad(wdw, (deficit, 0), mode='constant')
    if rid_log:
        dft = np.abs(np.fft.fft(wdw))
    else:
        dft = np.log(np.abs(np.fft.fft(wdw)))
    return dft[: (wdw_size // 2) + 1]  # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)

# Learning optimization
def make_row_sum_matrix(mtx, out_shape):
    row_sums = mtx.sum(axis=1)
    return np.repeat(row_sums, out_shape[1], axis=0)

# W LOGIC
# Takes a LONG time to run, consider exporting this matrix to a file to read from instead of calling this func
def make_basis_vectors(wdw_num, wdw_size, rid_log=False):
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

        basis_vectors.append(make_best_dft(sig, wdw_num, wdw_size, rid_log))

    os.chdir(base_dir)
    return np.array(basis_vectors).T     # T Needed? Yes

# V LOGIC
def make_spectrogram(signal, wdw_size, rid_log=False):
    num_spls = len(signal)
    if isinstance(signal[0], int):  # Mono signal = 1 channel   
        sig = np.array(signal)
    else:                           # Stereo signal = 2 channels
        sig = np.array([x[0] for x in signal])

    print('Original Sig:\n', sig[:20])
    print('Data type of sig value:', type(sig[0]), sig[0].dtype)

    spectrogram, pos_phases = [], []
    num_wdws = math.ceil(num_spls / wdw_size)
    # print('Original Sig:\n', sig[wdw_size * (num_wdws // 2): (wdw_size * (num_wdws // 2)) + 10])
    for i in range(num_wdws):
        wdw = sig[i * wdw_size: (i + 1) * wdw_size]
        if len(wdw) != wdw_size:
            deficit = wdw_size - len(wdw)
            wdw = np.pad(wdw, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

        # dft = np.abs(np.fft.fft(wdw))[:(wdw_size // 2) + 1]
        if i == 0:
            print('Original window (len =', len(wdw), '):\n', wdw[:5])

            fft = np.fft.fft(wdw)
            print('FFT of wdw (len =', len(fft), '):\n', fft[:5])

            phases_of_fft = np.angle(fft)
            print('phases of FFT of wdw:\n', phases_of_fft[:5])
            mag_fft = np.abs(fft)
            print('mag FFT of wdw:\n', mag_fft[:5])

            print('pos FFT of wdw:\n', fft[: (wdw_size // 2) + 1])

            pos_phases_of_fft = phases_of_fft[: (wdw_size // 2) + 1]
            pos_mag_fft = mag_fft[: (wdw_size // 2) + 1]
            print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_of_fft))
            print('positive mag FFT:\n', pos_mag_fft[:5])
            print('positive phases:\n', pos_phases_of_fft[:5])

            if not rid_log:
                pos_mag_fft[pos_mag_fft == 0] = 0.0001
                pos_mag_fft = np.log(pos_mag_fft)
                print('log of mag FFT of wdw:\n', pos_mag_fft[:5])
            
        else:
            fft = np.fft.fft(wdw)
            phases_of_fft = np.angle(fft)
            mag_fft = np.abs(fft)
            pos_phases_of_fft = phases_of_fft[: (wdw_size // 2) + 1]
            pos_mag_fft = mag_fft[: (wdw_size // 2) + 1]
            if not rid_log:
                pos_mag_fft[pos_mag_fft == 0] = 0.0001
                pos_mag_fft = np.log(pos_mag_fft)

        spectrogram.append(pos_mag_fft)
        pos_phases.append(pos_phases_of_fft)

    # Spectrogram matrix w/ correct orientation
    return np.array(spectrogram).T, pos_phases  # T Needed? Yes

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
        # UNCOMMENT FOR BUGGY OPTIMIZATION:
        # denom = make_row_sum_matrix(basis_vectors.T, spectrogram.shape)
        # activations *= (basis_vectors.T @ (spectrogram / (basis_vectors @ activations))) / denom
    return activations

# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, rid_log=False):
    num_wdws = synthetic_spgm.shape[1]
    # For waveform construction
    synthetic_spgm = synthetic_spgm.T     # Get back into orientation we did calculations on
    # Construct synthetic waveform
    synthetic_sig = []
    for i in range(num_wdws):
        if i == 0:
            if rid_log:
                pos_mag_fft = synthetic_spgm[i]
            else:
                pos_mag_fft = np.exp(synthetic_spgm[i])
            
            pos_phases_of_fft = phases[i]
            print('positive mag FFT:\n', pos_mag_fft[:5])
            print('positive phases:\n', pos_phases_of_fft[:5])
            print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_of_fft))
            
            neg_mag_fft = np.flip(pos_mag_fft[1: wdw_size // 2], 0)
            print('negative mag FFT:\n', neg_mag_fft[:5])
            
            mag_fft = np.append(pos_mag_fft, neg_mag_fft, axis=0)
            print('mag FFT of wdw:\n', mag_fft[:5])

            neg_phases_of_fft = np.flip([-x for x in pos_phases_of_fft[1: wdw_size // 2]], 0)
            print('negative phases:\n', neg_phases_of_fft[:5])
            phases_of_fft = np.append(pos_phases_of_fft, neg_phases_of_fft, axis=0)
            print('phases of FFT of wdw:\n', phases_of_fft[:5])

            fft = mag_fft * np.exp(1j*phases_of_fft)
            print('FFT of wdw (len =', len(fft), '):\n', fft[:5])

            ifft = np.fft.ifft(fft)

            imaginaries = ifft.imag.tolist()
            synthetic_wdw = ifft.real.tolist()
            print('Synthetic imaginaries:\n', imaginaries[:10])
            print('Synthetic window (len =', len(synthetic_wdw), '):\n', synthetic_wdw[:5])

        else:
            if rid_log:
                pos_mag_fft = synthetic_spgm[i]
            else:
                # Inverse the log operation
                pos_mag_fft = np.exp(synthetic_spgm[i])
            
            # Append the mirror of the synthetic magnitudes to itself
            # mir_freq = pos_mag_fft[1: wdw_size // 2]   
            neg_mag_fft = np.flip(pos_mag_fft[1: wdw_size // 2], 0)

            # dft = np.append(dft, np.flip(mir_freq, 0), axis=0)
            mag_fft = np.append(pos_mag_fft, neg_mag_fft, axis=0)

            # phase = phases[i][: wdw_size // 2] # Eliminate extraneous data point
            pos_phases_of_fft = phases[i]
            # phase = np.append(phase, np.flip(phase[1: wdw_size // 2], 0), axis=0)
            # mir_phase = phase[1: wdw_size // 2]
            # mir_phase = [-x for x in phase[1: wdw_size // 2]]

            neg_phases_of_fft = np.flip([-x for x in pos_phases_of_fft[1: wdw_size // 2]], 0)

            # phase = np.append(np.array([phase[(wdw_size // 2) - 1]]), mir_phase, axis=0)
            # phase = np.append(phase, np.flip(mir_phase, 0), axis=0)
            phases_of_fft = np.append(pos_phases_of_fft, neg_phases_of_fft, axis=0)

            # Multiply this magnitude spectrogram w/ phase
            fft = mag_fft * np.exp(1j*phases_of_fft)
            # Do ifft on the spectrogram -> waveform
            ifft = np.fft.ifft(fft)
            imaginaries = ifft.imag.tolist()
            synthetic_wdw = ifft.real.tolist()
            
        # synthetic_sig += [int(x) for x in synthetic_wdw]
        synthetic_sig += synthetic_wdw

    return synthetic_sig


def show_spectrogram(spectrogram, name):
    num_wdws = spectrogram.shape[1]

    fig, ax = plt.subplots()
    ax.title.set_text(name + ' Spectrogram')
    ax.set_ylabel('Frequency (Hz)')
    # Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
    im = ax.imshow(np.log(spectrogram), extent=[0, num_wdws, STD_SR_HZ // 2, 0])
    fig.tight_layout()
    # bottom, top = plt.ylim()
    # print('Bottom:', bottom, 'Top:', top)
    plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
    ax.set_aspect(0.08)     # Set a visually nice ratio
    plt.show()


def main():
    sr, brahms_sig = wavfile.read(brahms_filepath)
    debug_sig = [0,1,1,0]

    # DEBUG BLOCK - True for debug
    if False:
        # Debug a len=4 array ([0110]) sig
        print('\n\n')
        spectrogram, phases = make_spectrogram(debug_sig, DEBUG_WDW_SIZE, rid_log=True)
        print('\n---SYNTHETIC SPGM TRANSITION----\n')
        synthetic_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, rid_log=True)
        print('Debug Synthetic Sig:\n', synthetic_sig[:20])
        # Also try actual sig
        print('\n\n')
        spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, rid_log=True)
        print('\n---SYNTHETIC SPGM TRANSITION----\n')
        synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, rid_log=True)
        print('Actual Synthetic Sig:\n', np.array(synthetic_sig).astype('uint8')[:20])
        # Make synthetic WAV file
        wavfile.write(debug_filepath, STD_SR_HZ, np.array(synthetic_sig).astype('uint8'))

    else:
        basis_vectors = make_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, rid_log=True)
        spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, rid_log=True)
        # show_spectrogram(spectrogram, name="Original")

        print('Shape of Spectrogram V:', spectrogram.shape)
        print('Shape of Basis Vectors W:', basis_vectors.shape)
        print('Learning Activations...')
        activations = make_activations(spectrogram, basis_vectors)
        print('Shape of Activations H:', activations.shape)

        synthetic_spgm = basis_vectors @ activations
        # show_spectrogram(synthetic_spgm, name="Synthetic")

        print('---SYNTHETIC SPGM TRANSITION----')
        synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, rid_log=True)
        print('Synthesized signal:\n', np.array(synthetic_sig).astype('uint8')[:20])
        # Make synthetic WAV file
        wavfile.write(synthetic_brahms_filepath, STD_SR_HZ, np.array(synthetic_sig).astype('uint8'))


if __name__ == '__main__':
    main()
