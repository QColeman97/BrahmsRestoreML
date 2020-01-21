# restore_audio.py - Quinn Coleman - Senior Research Project / Master's Thesis 2019-20
# Restore audio using NMF. Input is audio, and restored audio file is written to cwd.

# Note Space:
# NMF Basics
# V (FxT) = W (FxC) @ H (CxT)
# Dimensions: F = # freq. bins, C = # components/sources(piano keys), T = # windows in time/timesteps
# Matrices: V = Spectrogram, W = Basis Vectors, H = Activations

# Numpy / Frequency Circle notes:
# Return value fft and input of ifft below
# a[0] should contain the zero frequency term,
# a[1:n//2] should contain the positive-frequency terms,
# a[n//2 + 1:] should contain the negative-frequency terms, in increasing order starting from the most negative frequency.

# Don't include duplicate 0Hz, include n//2 spot

# Mary notes = E4, D4, C4

# Make spectrogram w/ ova resource: https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/

import sys, os, math, librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
# import errno - use commented out

# Constants
# TODO - move to a file
SORTED_NOTES = ["A0", "Bb0", "B0", "C1", 
                "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
                "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
                "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
                "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
                "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
                "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
                "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]

# TODO - move to a file
SORTED_FUND_FREQ = [28, 29, 31, 33, 
                    35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 
                    69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123, 131, 
                    139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 
                    277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 
                    554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 
                    1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 
                    2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186]

MARY_START_INDEX, MARY_STOP_INDEX = 39, 44

# TODO: Move some of these to tests.py?
STD_SR_HZ = 44100
MARY_SR_HZ = 16000
PIANO_WDW_SIZE = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
DEBUG_WDW_SIZE = 4
RES = STD_SR_HZ / PIANO_WDW_SIZE
BEST_WDW_NUM = 5
NUM_NOISE_BV = 5
# Activation Matrix (H) Learning Part
MAX_LEARN_ITER = 100
BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
ACTIVATION_RATIO = 0.08
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008

# Functions
# Learning optimization
def make_row_sum_matrix(mtx, out_shape):
    row_sums = mtx.sum(axis=1)
    return np.repeat(row_sums, out_shape[1], axis=0)

def make_basis_vector(waveform, sgmt_num, wdw_size, ova=False, avg=False):
    if avg:
        num_sgmts = math.floor(len(waveform) / wdw_size) # Including incomplete windows throws off averaging
        all_sgmts = np.array([waveform[i * wdw_size: (i + 1) * wdw_size] for i in range(num_sgmts)])
        sgmt = np.mean(all_sgmts, axis=0)
    
    else:
        sgmt = waveform[(sgmt_num - 1) * wdw_size: sgmt_num * wdw_size]    # sgmt_num is naturally-indexed
        # print("Type of elem in piano note sig:", type(sgmt[0]))
        if len(sgmt) != wdw_size:
                deficit = wdw_size - len(sgmt)
                sgmt = np.pad(sgmt, (deficit, 0), mode='constant')
        
    if ova:
        sgmt *= np.hanning(wdw_size)
    # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
    return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]


# Time/#segments is irrelevant to # of basis vectors made (so maximize)
def make_noise_basis_vectors(wdw_size, ova=False, eq=False, debug=False, precise_noise=False, bv_thresh=800000):
    sr, brahms_sig = wavfile.read('../brahms.wav')
    # Convert to mono signal (avg left & right channels) 
    brahms_sig = np.array([((x[0] + x[1]) / 2) for x in brahms_sig.astype('float64')])
    
    # noise_sig_len = (NUM_NOISE_BV - 2) if ova else NUM_NOISE_BV
    # # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
    # noise_sgmt_num = math.ceil((STD_SR_HZ * 2) / wdw_size)
    # noise_sig = brahms_sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size]    # sgmt_num is naturally-indexed
    
    noise_sig_len = 2 if ova else 1 # The 1 is a guess, 2 is empircally derived
    # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
    noise_sgmt_num = math.ceil((STD_SR_HZ * 2.2) / wdw_size)    # 2.2 seconds (24rd window to (not including) 26th window)
    # print('Noise segment num:', noise_sgmt_num)
    if precise_noise:
        noise_sig = brahms_sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
    else:
    # All noise from beginning of clip
        noise_sig = brahms_sig[: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 

    print('\n----Making Noise Spectrogram--\n')
    spectrogram, phases = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    print('\n----Learning Noise Basis Vectors--\n')
    _, noise_basis_vectors = nmf_learn(spectrogram, num_components=NUM_NOISE_BV, debug=debug)
    if debug:
        print('Shape of Noise Spectogram V:', spectrogram.shape)
        print('Shape of Learned Noise Basis Vectors W:', noise_basis_vectors.shape)

    if False:  # Make louder # if eq:
        new_bvs = []
        for bv in noise_basis_vectors:
            while np.max(bv[1:]) < bv_thresh:
                bv *= 1.1
            new_bvs.append(bv)
        noise_basis_vectors = np.array(new_bvs)

    return list(noise_basis_vectors.T)    # List format is for use in get_basis_vectors(), transpose into similar format

# W LOGIC
# Basis vectors in essence are the "best" dft of a sound w/ constant pitch (distinct freq signature)
def get_basis_vectors(wdw_num, wdw_size, ova=False, mary=False, noise=False, avg=False, eq=False, debug=False, precise_noise=False):
    # To get threshold
    # max_val = None
    bv_thresh = 800000 # Based on max_val (not including first freq bin) - (floor) is 943865
    
    # To make a csv file for dennis
    filename = 'basis_vectors'
    if mary:
        filename += '_mary'
    if ova:
        filename += '_ova'
    if noise:
        filename += '_noise'
    if avg:
        filename += '_avg'
    if eq:
        filename += ('_eq_piano' + str(bv_thresh))
    filename += '.csv'

    try:
        # Line to bypass read from file - no need
        # raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'foo')

        with open(filename, 'r') as bv_f:
            print('FILE FOUND - READING IN BASIS VECTORS')
            basis_vectors = [[float(sub) for sub in string.split(',')] for string in bv_f.readlines()]
    except FileNotFoundError:
        print('FILE NOT FOUND - MAKING BASIS VECTORS')
        with open(filename, 'w') as bv_f:
            basis_vectors = []
            base_dir = os.getcwd()
            os.chdir('all_notes_ff')
            # audio_files is a list of strings, need to sort it by note
            unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]
            sorted_file_names = ['Piano.ff.' + x + '.wav' for x in SORTED_NOTES]
            audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))

            if noise:   # Retrieve noise from brahms sig
                print('\n----Making Noise Basis Vectors--\n')
                # sr, brahms_sig = wavfile.read('../brahms.wav')
                # # Convert to mono signal (avg left & right channels) 
                # brahms_sig = np.array([((x[0] + x[1]) / 2) for x in brahms_sig.astype('float64')])
                # # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
                # noise_wdw = math.ceil((STD_SR_HZ * 2) / wdw_size)
                # noise_basis_vector = make_basis_vector(brahms_sig, noise_wdw, wdw_size, ova=ova)

                # if eq:  # Make louder
                #     while np.max(noise_basis_vector[1:]) < bv_thresh:
                #         noise_basis_vector *= 1.1

                # basis_vectors.append(noise_basis_vector)
                basis_vectors += make_noise_basis_vectors(wdw_size, ova=ova, eq=eq, debug=debug, precise_noise=precise_noise, bv_thresh=800000)
                for basis_vector in basis_vectors:
                    bv_f.write(','.join([str(x) for x in basis_vector]) + '\n')

                # if max_val is None or np.max(noise_basis_vector[1:]) > max_val:
                #     max_val = np.max(noise_basis_vector)

            if mary:
                start, stop = MARY_START_INDEX, MARY_STOP_INDEX
            else:
                start, stop = 0, len(audio_files)
            
            for i in range(start, stop):   # Range of notes in Mary.wav
                audio_file = audio_files[i]

                sr, stereo_sig = wavfile.read(audio_file)
                # Convert to mono signal (avg left & right channels) 
                sig = np.array([((x[0] + x[1]) / 2) for x in stereo_sig])

                # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
                amp_thresh = max(sig) * 0.01
                while sig[0] < amp_thresh:
                    sig = sig[1:]
                while sig[-1] < amp_thresh:
                    sig = sig[:-1]

                basis_vector = make_basis_vector(sig, wdw_num, wdw_size, ova=ova, avg=avg)

                if eq:  # Make it louder
                    while np.max(basis_vector[1:]) < bv_thresh:
                        basis_vector *= 1.1

                basis_vectors.append(basis_vector)
                bv_f.write(','.join([str(x) for x in basis_vector]) + '\n')

                # if max_val is None or np.max(basis_vector[1:]) > max_val:
                #     max_val = np.max(basis_vector)

            os.chdir(base_dir)

        # print('\nMAX BASIS VECTOR VAL:', max_val, '\n')

    basis_vectors = np.array(basis_vectors).T   # T Needed? Yes
    if debug:
        print('Shape of built basis vectors:', basis_vectors.shape)
        plot_matrix(basis_vectors, name="Built Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)

    return basis_vectors


# V LOGIC
def make_spectrogram(signal, wdw_size, ova=False, debug=False):
    num_spls = len(signal)
    if isinstance(signal[0], np.ndarray):   # Stereo signal = 2 channels
        sig = np.array([((x[0] + x[1]) / 2) for x in signal.astype('float64')])
    else:                                   # Mono signal = 1 channel    
        sig = np.array(signal).astype('float64')
    # Keep signal value types float64 so nothing lost? (needed for stereo case, so keep consistent?)

    if debug:
        print('Original Sig:\n', sig[:20])
        # print('Type of elem in orig sig:', type(sig[0]), sig[0].dtype)

    hop_size = int(math.floor(wdw_size / 2)) if (ova and len(sig) >= (wdw_size + int(math.floor(wdw_size / 2)))) else wdw_size   # Half-length of window if ova
    spectrogram, pos_phases = [], []
    if ova:
        # Probably fine, but broken (makes 3 sgmts for a 6 elem sig w/ 4 elem wdwsize)
        # num_sgmts = (math.ceil(num_spls / wdw_size) * 2) - 1
        # if len(sig) == 6:
        #     num_sgmts = 2
        num_sgmts = math.ceil(num_spls / (wdw_size // 2)) - 1
    else:
        num_sgmts = math.ceil(num_spls / wdw_size)

    if debug:
        print('Hop size:', hop_size)
        print('Num segments:', num_sgmts)

    for i in range(num_sgmts):
        # TODO: Does slicing a list make a copy? B/c appears not to
        sgmt = sig[i * hop_size: (i * hop_size) + wdw_size].copy()

        # print("Type of elem in spectrogram:", type(wdw[0]))
        if len(sgmt) != wdw_size:
            deficit = wdw_size - len(sgmt)
            sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

        if debug and (i == 0 or i == 1):
            print('Original segment (len =', len(sgmt), '):\n', sgmt[:5])

        if ova: # Perform lobing on ends of segment
            sgmt *= np.hanning(wdw_size)
        fft = np.fft.fft(sgmt)
        phases_of_fft = np.angle(fft)
        mag_fft = np.abs(fft)
        pos_phases_of_fft = phases_of_fft[: (wdw_size // 2) + 1]
        pos_mag_fft = mag_fft[: (wdw_size // 2) + 1]

        if debug and (i == 0 or i == 1):
            if ova:
                print('hanning mult segment:\n', sgmt[:5])
            print('FFT of wdw (len =', len(fft), '):\n', fft[:5])
            print('phases of FFT of wdw:\n', phases_of_fft[:5])
            print('mag FFT of wdw:\n', mag_fft[:5])
            print('pos FFT of wdw:\n', fft[: (wdw_size // 2) + 1])
            print('\nType of elem in spectrogram:', type(pos_mag_fft[0]), pos_mag_fft[0].dtype, '\n')
            print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_of_fft))
            print('positive mag FFT:\n', pos_mag_fft[:5])
            print('positive phases:\n', pos_phases_of_fft[:5])
            print('\nEnd of Segment -> FT\n')

        spectrogram.append(pos_mag_fft)
        pos_phases.append(pos_phases_of_fft)

    # Spectrogram matrix w/ correct orientation
    spectrogram = np.array(spectrogram).T   # T Needed? Yes
    if debug:
        plot_matrix(spectrogram, name='Built Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)

    return spectrogram, pos_phases

# H LOGIC - Learn / Approximate Activation Matrix
# NMF Basics
# V (FxT) = W (FxC) @ H (CxT)
# Dimensions: F = # freq. bins, C = # components(piano keys), T = # windows in time/timesteps
# Matrices: V = Spectrogram, W = Basis Vectors, H = Activations
# Main dimensions: freq bins = spectrogram.shape[0], piano keys (components?) = num_notes OR basis_vectors.shape[1], windows = bm_num_wdws OR spectrogram.shape[1]

# To debug NMF
def nmf_learn(spectrogram, basis_vectors=None, num_components=None, debug=False):
# def nmf_learn(spectrogram, num_components, debug=False):
    # CHANGE BACK IF BUG:
    # activations = np.random.rand(num_notes, bm_num_wdws)
    # learned_activations = np.random.rand(len(SORTED_NOTES), spectrogram.shape[1])
    # learned_basis_vectors = np.random.rand(spectrogram.shape[0], len(SORTED_NOTES))
    learned_activations = np.random.rand(num_components, spectrogram.shape[1])
    learned_basis_vectors = np.random.rand(spectrogram.shape[0], num_components)
    ones = np.ones(spectrogram.shape) # so dimenstions match W transpose dot w/ V

    # How do we use basis vectors, if we haven't learned them yet?
    for i in range(MAX_LEARN_ITER):
        # H +1 = H * ((Wt dot (V / (W dot H))) / (Wt dot 1) )
        # learned_activations *= ((basis_vectors.T @ (spectrogram / (basis_vectors @ learned_activations))) / (basis_vectors.T @ ones))
        
        # To debug NMF
        if basis_vectors is not None:
            learned_activations *= ((basis_vectors.T @ (spectrogram / (basis_vectors @ learned_activations))) / (basis_vectors.T @ ones))
        else:
            learned_activations *= ((learned_basis_vectors.T @ (spectrogram / (learned_basis_vectors @ learned_activations))) / (learned_basis_vectors.T @ ones))

        # UNCOMMENT FOR BUGGY OPTIMIZATION:
        # denom = make_row_sum_matrix(basis_vectors.T, spectrogram.shape)
        # learned_activations *= (basis_vectors.T @ (spectrogram / (basis_vectors @ learned_activations))) / denom

        # Make learned basis vectors
        # W +1 = W * (((V / (W dot H)) dot Ht) / (1 dot Ht) )
        learned_basis_vectors *= (((spectrogram / (learned_basis_vectors @ learned_activations)) @ learned_activations.T) / (ones @ learned_activations.T))

    if debug:
        print('In Learn - Shape of Learned Activations H:', learned_activations.shape)
        # print('First rows of activations (components):\n', learned_activations[0,:], '\n', learned_activations[1,:], '\n', learned_activations[2,:], '\n')
        # print('First columns of activations (windows):\n', learned_activations[:,0], '\n', learned_activations[:,1], '\n', learned_activations[:,2], '\n')
        plot_matrix(learned_activations, name="Learned Activations", ylabel='Components', ratio=ACTIVATION_RATIO)

        print('In Learn - Shape of Learned Basis Vectors W:', learned_basis_vectors.shape)
        # print('First rows of basis vectors (freq bins):\n', learned_activations[0,:], '\n', learned_activations[1,:], '\n', learned_activations[2,:], '\n')
        # print('First columns of basis vectors (components):\n', learned_activations[:,0], '\n', learned_activations[:,1], '\n', learned_activations[:,2], '\n')
        plot_matrix(learned_basis_vectors, name="Learned Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)

    # if noise: # Remove noise component so no noise in synthetic spectrogram
    #     basis_vectors = basis_vectors.T[1:].T
    #     learned_activations = learned_activations[1:]
    #     if debug:
    #         # print('Shape of De-noised Basis Vectors W:', activations.shape)
    #         plot_matrix(basis_vectors, name="De-noised Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
    #         # print('Shape of De-noised Activations H:', activations.shape)
    #         plot_matrix(learned_activations, name="De-noised Activations", ratio=ACTIVATION_RATIO)

    return learned_activations, learned_basis_vectors


def remove_noise_vectors(activations, basis_vectors, debug=False):
    basis_vectors = basis_vectors.T[NUM_NOISE_BV:].T
    activations = activations[NUM_NOISE_BV:]
    if debug:
        plot_matrix(basis_vectors, name="De-noised Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
        plot_matrix(activations, name="De-noised Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
    return activations, basis_vectors


# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, ova=False, debug=False):
    num_sgmts = synthetic_spgm.shape[1]
    # For waveform construction
    synthetic_spgm = synthetic_spgm.T     # Get back into orientation we did calculations on
    # Construct synthetic waveform
    synthetic_sig = []
    for i in range(num_sgmts):
        pos_mag_fft = synthetic_spgm[i]
        
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
        synthetic_sgmt = ifft.real.tolist()

        if debug and i == 0:
            print('positive mag FFT:\n', pos_mag_fft[:5])
            print('positive phases:\n', pos_phases_of_fft[:5])
            print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_of_fft))
            print('negative mag FFT:\n', neg_mag_fft[:5])
            print('mag FFT of wdw:\n', mag_fft[:5])
            print('negative phases:\n', neg_phases_of_fft[:5])
            print('phases of FFT of wdw:\n', phases_of_fft[:5])
            print('FFT of wdw (len =', len(fft), '):\n', fft[:5])
            print('Synthetic imaginaries:\n', imaginaries[:10])
            print('Synthetic segment (len =', len(synthetic_sgmt), '):\n', synthetic_sgmt[:5])

        # Do overlap-add operations if ova (but only if list has atleast 1 element)
        if ova and len(synthetic_sig):
            ova_sgmt = synthetic_sgmt[: wdw_size // 2].copy()   # First half
            end_sgmt = synthetic_sgmt[wdw_size // 2:].copy()    # Second half

            end_sig = synthetic_sig[-(wdw_size // 2):].copy()   # Last part of sig
            end_sum = [sum(x) for x in zip(ova_sgmt, end_sig)]  # Summed last part w/ first half
            if debug and i == 1:
                print('ova_sgmt:\n', ova_sgmt[-10:], '\nend_sgmt:\n', end_sgmt[-10:], '\nend_sig:\n', end_sig[-10:], '\nend_sum:\n', end_sum[-10:])

            synthetic_sig = synthetic_sig[: -(wdw_size // 2)] + end_sum + end_sgmt
            if debug and i == 1:
                print('End of synth sig:', synthetic_sig[-20:])

        else:
            synthetic_sig += synthetic_sgmt
            if debug and i == 0:
                print('End of synth sig:', synthetic_sig[-20:])

    if debug:
        print('Synthetic Sig - bad type for brahms (not uint8):\n', np.array(synthetic_sig)[:20])

    return np.array(synthetic_sig)


def make_mary_bv_test_activations():
    activations = []
    for j in range(5):
        # 8 divisions of 6 timesteps
        comp = []
        if j == 0: # lowest note
            comp = [0.0001 if ((2*6) <= i < (3*6)) else 0.0000 for i in range(48)]
        elif j == 2:
            comp = [0.0001 if (((1*6) <= i < (2*6)) or ((3*6) <= i < (4*6))) else 0.0000 for i in range(48)]
        elif j == 4:
            comp = [0.0001 if ((0 <= i < (1*6)) or ((4*6) <= i < (7*6))) else 0.0000 for i in range(48)]
        else:
            comp = [0.0000 for i in range(48)]
        activations.append(comp)
    return np.array(activations)


def plot_matrix(matrix, name, ylabel, ratio=0.08):
    num_wdws = matrix.shape[1]

    fig, ax = plt.subplots()
    ax.title.set_text(name)
    ax.set_ylabel(ylabel)
    # Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
    im = ax.imshow(np.log(matrix), extent=[0, num_wdws, STD_SR_HZ // 2, 0])
    fig.tight_layout()
    # bottom, top = plt.ylim()
    # print('Bottom:', bottom, 'Top:', top)
    plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
    ax.set_aspect(ratio)    # Set a visually nice ratio
    plt.show()


def reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, segment=False, write_file=False, debug=False):
    print('--Initiating Reconstruct Mode--')

    if segment:
        # TEMP SO WE CAN FIND MAX NOISE SEGMENT - noise from 1.8 ro 2.1 seconds
        noise_sig_len = 2
        # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
        noise_sgmt_num = math.ceil((STD_SR_HZ * 2.2) / wdw_size)    # 2.1 seconds (23rd window to (not including) 25th window)
        # print('Noise segment num:', noise_sgmt_num)
        noise_sig = sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
        # noise_sig = sig[23 * wdw_size: 25 * wdw_size] 
        # 23 25
        out_filepath = 'practice_' + out_filepath
        print('\n--Making Signal Spectrogram--\n')
        spectrogram, phases = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    else:
        print('\n--Making Signal Spectrogram--\n')
        spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)
    
    print('\n--Making Synthetic Signal--\n')
    synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=ova, debug=debug)
    
    if write_file:
        # Make synthetic WAV file - defaults to original sampling rate, TODO: Does that change things?
        # Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
        wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(sig.dtype))

    return synthetic_sig


def restore_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, marybv=False, noisebv=False, avgbv=False, write_file=False, debug=False, nohanbv=False, prec_noise=False, eqbv=False):
    print('--Initiating Restore Mode--')
    print('\n--Making Piano Basis Vectors--\n')

    # Temporary for testing:
    if nohanbv:
        basis_vectors = get_basis_vectors(BEST_WDW_NUM, wdw_size, ova=False, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise)
    else:
        basis_vectors = get_basis_vectors(BEST_WDW_NUM, wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise, eq=eqbv)
    
    print('\n--Making Signal Spectrogram--\n')
    spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)

    if debug:
        print('Shape of Original Piano Basis Vectors W:', basis_vectors.shape)
        print('Shape of Signal Spectrogram V:', spectrogram.shape)
        
    print('\n--Learning Piano Activations--\n')
    num_components = len(SORTED_NOTES[MARY_START_INDEX: MARY_STOP_INDEX]) if marybv else len(SORTED_NOTES)
    if noisebv:
        num_components += NUM_NOISE_BV

    # For debug NMF - basis vector param
    activations, _ = nmf_learn(spectrogram, basis_vectors, num_components, debug=debug)
    # activations, _ = nmf_learn(spectrogram, num_components, debug=debug)
    if noisebv:
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug)
    if debug:
        if noisebv:
            print('Shape of De-noised Piano Basis Vectors W:', basis_vectors.shape)
            print('Shape of De-noised Piano Activations H:', activations.shape)
        else:
            print('Shape of Piano Activations H:', activations.shape)
    
    print('\n--Making Synthetic Spectrogram--\n')
    synthetic_spgm = basis_vectors @ activations
    if debug:
        print('Shape of Synthetic Signal Spectrogram V\':', synthetic_spgm.shape)
        plot_matrix(synthetic_spgm, name='Synthetic Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)

    print('\n--Making Synthetic Signal--\n')
    synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, wdw_size, ova=ova, debug=debug)
    
    if write_file:
        # Make synthetic WAV file - Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
        wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(sig.dtype))

    return synthetic_sig


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('\nUsage: restore_audio.py <mode> <signal> <debug> [window_size]')
        print('Parameter options:')
        print('Mode             "RECONST"       - Reconstructs signal from spectrogram')
        print('                 "RESTORE"       - Synthesizes restored signal via NMF')
        print('Signal           filepath        - String denoting a WAV filepath')
        print('                 list            - Signal represented by list formatted like "[0,1,1,0]"')
        print('                 natural number  - Random element signal of this length')
        print('Debug            "TRUE"/"FALSE"  - Option to print & plot NMF matrices in restore mode')
        print('Window Size      natural number (power of 2 preferably, default is for piano: 4096)\n')
        print('Currently not editable: Sampling Rate, Best Window & size of Basis Vectors\n')
        sys.exit(1)

    # Pre-configured params
    noisebv_flag = True
    avgbv_flag = True
    ova_flag = True
    marybv_flag = False     # Special case for Mary.wav - basis vectors size optimization test

    # TODO: Use argparse library
    # Configure params
    # Mode - RECONST or RESTORE
    mode = sys.argv[1]
    out_filename = 'restored_' if mode == 'RESTORE' else 'reconst_'
    # Signal - comes as a list, filepath or a length
    sig_sr = STD_SR_HZ # Initialize sr to default
    if sys.argv[2].startswith('['):
        sig = np.array([int(num) for num in sys.argv[2][1:-1].split(',')])
        out_filename += 'my_sig'
    elif not sys.argv[2].endswith('.wav'):  # Work around for is a number
        sig = np.random.rand(int(sys.argv[2].replace(',', '')))
        out_filename += 'rand_sig'
    else:
        sig_sr, sig = wavfile.read(sys.argv[2])
        if sig_sr != STD_SR_HZ:
            sig, _ = librosa.load(sys.argv[2], sr=STD_SR_HZ)  # Upsample to 44.1kHz if necessary
        start_index = (sys.argv[2].rindex('/') + 1) if (sys.argv[2].find('/') != -1) else 0
        out_filename += sys.argv[2][start_index: -4]
    # Debug-print/plot option
    debug_flag = True if sys.argv[3] == 'TRUE' else False
    # Window Size
    wdw_size = int(sys.argv[4]) if (len(sys.argv) == 5) else PIANO_WDW_SIZE
    # Overlap-Add is Necessary & Default
    if ova_flag:
        out_filename += '_ova'

    if mode == 'RECONST': # RECONSTRUCT BLOCK
        out_filename += '.wav'
        reconstruct_audio(sig, wdw_size, out_filename, sig_sr, ova=ova_flag, segment=False, 
                          write_file=True, debug=debug_flag)
    else:   # MAIN RESTORE BLOCK
        if noisebv_flag:
            out_filename += '_noisebv'
        if avgbv_flag:
            out_filename += '_avgbv'
        out_filename += '.wav'
        restore_audio(sig, wdw_size, out_filename, sig_sr, ova=ova_flag, marybv=marybv_flag, 
                      noisebv=noisebv_flag, avgbv=avgbv_flag, write_file=True, debug=debug_flag)


if __name__ == '__main__':
    main()
