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
from copy import deepcopy

# Constants
# TODO - Read this in from the file instead! Then delete
SORTED_NOTES = ["A0", "Bb0", "B0", "C1", 
                "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
                "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
                "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
                "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
                "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
                "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
                "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]

# TODO - Read this in from the file instead! Then delete
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
NUM_NOISE_BV = 5 # 50 # 20 # 3 # 10 # 5 # 10000 is when last good # 100000 is when it gets bad
# BUT 1000 sounds bad in tests.py
# Activation Matrix (H) Learning Part
MAX_LEARN_ITER = 100
BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
ACTIVATION_RATIO = 0.08
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008

WDW_NUM_AFTER_VOICE = 77

# L1_PENALTY = 1000000000000000000 # Quintillion
L1_PENALTY = 0 # 10 ** 19 # 10^9 = 1Bill, 12 = trill, 15 = quad, 18 = quin, 19 = max for me

# TODO: Read notes (& freq if we ever do) in from file, not our program (RAM)
# # Did this for safety
# def write_notes_to_file():
#     with open('piano_notes_and_fund_freqs.csv', 'w') as nf:
#         for i in range(len(SORTED_NOTES)):
#             nf.write(SORTED_NOTES[i] + ',' + str(SORTED_FUND_FREQ[i]) + '\n')


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
def make_noise_basis_vectors(wdw_size, ova=False, eq=False, debug=False, precise_noise=False, bv_thresh=800000, num=NUM_NOISE_BV):
    # sr, brahms_sig = wavfile.read('../brahms.wav')
    _, brahms_sig = wavfile.read('/Users/quinnmc/Desktop/AudioRestore/brahms.wav')
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
    spectrogram, _ = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    print('\n----Learning Noise Basis Vectors--\n')
    _, noise_basis_vectors = nmf_learn(spectrogram, num_components=num, debug=debug)
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


def make_basis_vectors(wdw_num, wdw_size, filepath, ova=False, avg=False, mary_flag=False, eq=False, bv_thresh=800000):
    # bv_thresh = 800000  # Based on max_val (not including first freq bin) - (floor) is 943865
    # max_val = None      # To get threshold
    basis_vectors = []
    with open(filepath, 'w') as bv_f:
        base_dir = os.getcwd()
        os.chdir('all_notes_ff_wav')
        # audio_files is a list of strings, need to sort it by note
        unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]
        sorted_file_names = ['Piano.ff.' + x + '.wav' for x in SORTED_NOTES]
        audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))

        if mary_flag:
            start, stop = MARY_START_INDEX, MARY_STOP_INDEX
        else:
            start, stop = 0, len(audio_files)
        
        for i in range(start, stop):
            audio_file = audio_files[i]
            _, stereo_sig = wavfile.read(audio_file)
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
    return basis_vectors


# We don't save bvs w/ noise anymnore, 
# we just calc noise and pop it on top of restored-from-file piano bvs

# W LOGIC
# Basis vectors in essence are the "best" dft of a sound w/ constant pitch (distinct freq signature)
def get_basis_vectors(wdw_num, wdw_size, ova=False, mary=False, noise=False, avg=False, debug=False, precise_noise=False, eq=False, num_noise=NUM_NOISE_BV):
    # Save/load basis vectors (w/o noise) to/from CSV files
    filepath = 'csv_saves_bv/basis_vectors'
    if mary:
        filepath += '_mary'
    if ova:
        filepath += '_ova'
    if avg:
        filepath += '_avg'
    if eq:
        filepath += ('_eq_piano' + str(bv_thresh))
    filepath += '.csv'

    try:
        with open(filepath, 'r') as bv_f:
            print('FILE FOUND - READING IN BASIS VECTORS:', filepath)
            basis_vectors = [[float(sub) for sub in string.split(',')] for string in bv_f.readlines()]

    except FileNotFoundError:
        print('FILE NOT FOUND - MAKING BASIS VECTORS:', filepath)
        basis_vectors = make_basis_vectors(wdw_num, wdw_size, filepath, ova=ova, avg=avg, mary_flag=mary, eq=eq, bv_thresh=800000)

    if debug:
        print('Basis Vectors Length:', len(basis_vectors))

    # Make and add noise bv's if necessary
    if noise:
        noise_basis_vectors = make_noise_basis_vectors(wdw_size, ova=ova, eq=eq, debug=debug, 
                                                    precise_noise=precise_noise, bv_thresh=800000, num=num_noise)
        basis_vectors = (noise_basis_vectors + basis_vectors)
        if debug:
            print('Noise Basis Vectors Length:', len(noise_basis_vectors))
            print('Basis Vectors Length After putting together:', len(basis_vectors))

    basis_vectors = np.array(basis_vectors).T   # T Needed? Yes
    if debug:
        print('Shape of built basis vectors:', basis_vectors.shape)
        plot_matrix(basis_vectors, name="Built Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)

    return basis_vectors

# Returns magnitude & phases of a DFT, given a signal segment
def fourier_transform(sgmt, wdw_size, ova=False, debug_flag=False):
    if len(sgmt) != wdw_size:
        deficit = wdw_size - len(sgmt)
        sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

    if debug_flag:
        print('Original segment (len =', len(sgmt), '):\n', sgmt[:5])

    if ova: # Perform lobing on ends of segment
        sgmt *= np.hanning(wdw_size)
    fft = np.fft.fft(sgmt)
    phases_fft = np.angle(fft)
    mag_fft = np.abs(fft)
    pos_phases_fft = phases_fft[: (wdw_size // 2) + 1]
    pos_mag_fft = mag_fft[: (wdw_size // 2) + 1]

    if debug_flag:
        if ova:
            print('hanning mult segment:\n', sgmt[:5])
        print('FFT of wdw (len =', len(fft), '):\n', fft[:5])
        print('phases of FFT of wdw:\n', phases_fft[:5])
        print('mag FFT of wdw:\n', mag_fft[:5])
        print('pos FFT of wdw:\n', fft[: (wdw_size // 2) + 1])
        print('\nType of elem in spectrogram:', type(pos_mag_fft[0]), pos_mag_fft[0].dtype, '\n')
        print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_fft))
        print('positive mag FFT:\n', pos_mag_fft[:5])
        print('positive phases:\n', pos_phases_fft[:5])
        print('\nEnd of Segment -> FT\n')
    
    return pos_mag_fft, pos_phases_fft

# V LOGIC
def make_spectrogram(signal, wdw_size, ova=False, debug=False):
    num_spls = len(signal)
    # Keep signal value types float64 so nothing lost? (needed for stereo case, so keep consistent?)
    if isinstance(signal[0], np.ndarray):   # Stereo signal = 2 channels
        sig = np.array([((x[0] + x[1]) / 2) for x in signal.astype('float64')])
    else:                                   # Mono signal = 1 channel    
        sig = np.array(signal).astype('float64')

    if debug:
        print('Original Sig:\n', sig[:20])

    # Hop size is half-length of window if OVA, else it's just window length
    hop_size = int(math.floor(wdw_size / 2)) if (ova and len(sig) >= (wdw_size + int(math.floor(wdw_size / 2)))) else wdw_size
    # Number of segments depends on if OVA implemented
    num_sgmts = (math.ceil(num_spls / (wdw_size // 2)) - 1) if ova else math.ceil(num_spls / wdw_size)

    if debug:
        print('Hop size:', hop_size)
        print('Num segments:', num_sgmts)
    
    spectrogram, pos_phases = [], []
    for i in range(num_sgmts):
        # Slicing a numpy array makes a view, so explicit copy
        sgmt = sig[i * hop_size: (i * hop_size) + wdw_size].copy()
        
        debug_flag = ((i == 0) or (i == 1)) if debug else False
        pos_mag_fft, pos_phases_fft = fourier_transform(sgmt, wdw_size, ova=ova, debug_flag=debug_flag)
        
        if len(sgmt) != wdw_size:
            deficit = wdw_size - len(sgmt)
            sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

        spectrogram.append(pos_mag_fft)
        pos_phases.append(pos_phases_fft)

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

# NMF can be - supervised (given W or H, 
#                          learn H or W)            - check
#            - unsupervised (not given W nor H, 
#                            learn W and H)         - check
#            - semi-supervised (given part of W or part of H, 
#                               learn other part of W and H or other part of H and W)   

# Assumption - for supervised case, activations & basis_vectors never both not null

# TODO: Semi-supervised
#   Case 1: given part of W, learn other part of W and all of H
#       1.1: Specifically, given built Wpiano, learn Wnoise and Hbrahms
#       1.2: Specifically, given built Wnoise, learn Wpiano and Hbrahms
#   Case 2: given part of H, learn other part of H and all of W
#       No specific need for this case yet, but implement if easy

# TODO: Mess w/ different initializations of learned
#   Example, if Wpiano being learned, start with rand matrix or already made Wpiano
#       Initialization case 1: rand matrix - check
#       Initialization case 2: made Wpiano

# TODO: Have a param to specify when to NOT learn voice in our basis vectors (we shouldn't) 
# For now, no param and we just shorten the brahms sig before this call

# Semi-supervised NMF helper function
def partition_matrices(learn_index, basis_vectors, activations, madeinit=False):
    if learn_index > 0:     # Fixed part is left side (Wfix = noise)
        # So I don't make a memory mistake
        basis_vectors_fixed = basis_vectors[:, :learn_index].copy()
        if madeinit:
            basis_vectors_learn = basis_vectors[:, learn_index:].copy()
        else:
            basis_vectors_learn = np.random.rand(basis_vectors[:, learn_index:].shape[0], 
                                                    basis_vectors[:, learn_index:].shape[1])
        activations_for_fixed = activations[:learn_index, :].copy()
        activations_for_learn = activations[learn_index:, :].copy()
    
    else:                   # Fixed part is right side (Wfix = piano)
        # Modify learn index as a result of my failure to combine a flag w/ logic
        learn_index *= -1
        
        basis_vectors_fixed = basis_vectors[:, learn_index:].copy()
        if madeinit:
            basis_vectors_learn = basis_vectors[:, :learn_index].copy()
        else:
            basis_vectors_learn = np.random.rand(basis_vectors[:, :learn_index].shape[0], 
                                                 basis_vectors[:, :learn_index].shape[1])
        activations_for_fixed = activations[learn_index:, :].copy()
        activations_for_learn = activations[:learn_index, :].copy()

        learn_index *= -1
    
    return basis_vectors_fixed, basis_vectors_learn, activations_for_fixed, activations_for_learn

# NMF Learning step formulas:
    # H +1 = H * ((Wt dot (V / (W dot H))) / (Wt dot 1) )
    # W +1 = W * (((V / (W dot H)) dot Ht) / (1 dot Ht) )

# General case NMF algorithm
def nmf_learn(input_matrix, num_components, basis_vectors=None, learn_index=0, madeinit=False, debug=False, incorrect=False, 
              learn_iter=MAX_LEARN_ITER, l1_penalty=L1_PENALTY, pen='Both'):
    activations = np.random.rand(num_components, input_matrix.shape[1])
    ones = np.ones(input_matrix.shape) # so dimensions match W transpose dot w/ V

    if basis_vectors is not None:
        if debug:
            print('In Sup or Semi-Sup Learn - Shape of Given Basis Vectors W:', basis_vectors.shape)
        if learn_index == 0:
            # Sup Learning - Do NMF w/ whole W, only H learn step, get H
            for _ in range(learn_iter):
                activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / ((basis_vectors.T @ ones) + l1_penalty))
        else:
            # Semi-Sup Learning - Do NMF w/ part of W, part of W and H learn steps, get W and H
            (basis_vectors_fixed, basis_vectors_learn, 
            activations_for_fixed, activations_for_learn) = partition_matrices(learn_index, basis_vectors, 
                                                                                 activations, madeinit=madeinit)
            if debug:
                print('Semi-Sup Learning', 'Piano' if (learn_index > 0) else 'Noise')
                print('In Semi-Sup Learn - Shape of Wfix:', basis_vectors_fixed.shape)
                print('In Semi-Sup Learn - Shape of Wlearn:', basis_vectors_learn.shape)
                print('In Semi-Sup Learn - Shape of Hfromfix:', activations_for_fixed.shape)
                print('In Semi-Sup Learn - Shape of Hfromlearn:', activations_for_learn.shape)
                plot_matrix(basis_vectors_fixed, name="Fixed BV Before Learn", ylabel='Components', ratio=BASIS_VECTOR_FULL_RATIO)
                plot_matrix(basis_vectors_learn, name="Learned BV Before Learn", ylabel='Components', ratio=BASIS_VECTOR_FULL_RATIO)
                plot_matrix(activations_for_fixed, name="Activations of Fixed Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)
                plot_matrix(activations_for_learn, name="Activations of Learned Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)

            # Note: No L1-Penalty support for incorrect version
            if incorrect:   # For results of bug
                # Don't fix the fixed part - W = Wfix and Wlearn concatenated together, same w/ H
                if learn_index > 0:
                    activations = np.concatenate((activations_for_fixed, activations_for_learn), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_fixed, basis_vectors_learn), axis=1)
                else:
                    activations = np.concatenate((activations_for_learn, activations_for_fixed), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_learn, basis_vectors_fixed), axis=1)

                for _ in range(learn_iter):
                    activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / ((basis_vectors.T @ ones) + l1_penalty))
                    basis_vectors *= (((input_matrix / (basis_vectors @ activations)) @ activations.T) / (ones @ activations.T))

            else:
                if l1_penalty != 0 and (((pen == 'Both' or pen == 'Piano') and learn_index < 0) or 
                                        ((pen == 'Both' or pen == 'Noise') and learn_index > 0)):
                    if debug:
                        print('Applying L1-Penalty of', str(l1_penalty), 'to', 'Noise' if (learn_index > 0) else 'Piano', '(Fixed) Activations')
                    # Do NMF w/ Wfix (W given subset), only H learn step, get H
                    for _ in range(learn_iter):
                        activations_for_fixed *= ((basis_vectors_fixed.T @ (input_matrix / (basis_vectors_fixed @ activations_for_fixed))) / ((basis_vectors_fixed.T @ ones) + l1_penalty))
                else:
                    # Do NMF w/ Wfix (W given subset), only H learn step, get H
                    for _ in range(learn_iter):
                        activations_for_fixed *= ((basis_vectors_fixed.T @ (input_matrix / (basis_vectors_fixed @ activations_for_fixed))) / (basis_vectors_fixed.T @ ones))
                
                # Make copy of Hlearn to be used by ONLY Wlearn - this prevents W from "making up" for penalized H, vice-verse else bug happens
                activations_for_learn_use = deepcopy(activations_for_learn)
                basis_vectors_learn_use = deepcopy(basis_vectors_learn)

                if l1_penalty != 0 and (((pen == 'Both' or pen == 'Noise') and learn_index < 0) or 
                                        ((pen == 'Both' or pen == 'Piano') and learn_index > 0)):
                    if debug:
                        print('Applying L1-Penalty of', str(l1_penalty), 'to', 'Piano' if (learn_index > 0) else 'Noise', '(Learned) Activations')

                    # Do NMF w/ Wlearn (W given subset OR random mtx), both W and H learn steps, get W and H
                    for _ in range(learn_iter):
                        # activations_for_learn *= ((basis_vectors_learn.T @ (input_matrix / (basis_vectors_learn @ activations_for_learn))) / ((basis_vectors_learn.T @ ones) + l1_penalty))
                        # basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn)) @ activations_for_learn.T) / (ones @ activations_for_learn.T))
                        
                        activations_for_learn *= ((basis_vectors_learn_use.T @ (input_matrix / (basis_vectors_learn_use @ activations_for_learn))) / ((basis_vectors_learn_use.T @ ones) + l1_penalty))
                        basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn_use)) @ activations_for_learn_use.T) / (ones @ activations_for_learn_use.T))

                else:
                    # Do NMF w/ Wlearn (W given subset OR random mtx), both W and H learn steps, get W and H
                    for _ in range(learn_iter):
                        # activations_for_learn *= ((basis_vectors_learn.T @ (input_matrix / (basis_vectors_learn @ activations_for_learn))) / (basis_vectors_learn.T @ ones))
                        # basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn)) @ activations_for_learn.T) / (ones @ activations_for_learn.T))

                        activations_for_learn *= ((basis_vectors_learn_use.T @ (input_matrix / (basis_vectors_learn_use @ activations_for_learn))) / (basis_vectors_learn_use.T @ ones))
                        basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn_use)) @ activations_for_learn_use.T) / (ones @ activations_for_learn_use.T))

                if debug:
                    print('(Penalty Present) Hlearn Sum:' if (l1_penalty != 0) else 'Hlearn Sum:', np.sum(activations_for_learn))
                    print('(Penalty Present) Wlearn Sum:' if (l1_penalty != 0) else 'Wlearn Sum:', np.sum(basis_vectors_learn), '-- for thoroughness')
                    print('(Penalty Present) Hfix Sum:' if (l1_penalty != 0) else 'Hfix Sum:', np.sum(activations_for_fixed))
                    print('(Penalty Present) Wfix Sum:' if (l1_penalty != 0) else 'Wfix Sum:', np.sum(basis_vectors_fixed), '-- for thoroughness')
                    print('In Semi-Sup Learn - after learn - Shape of Wfix:', basis_vectors_fixed.shape)
                    print('In Semi-Sup Learn - after learn - Shape of Wlearn:', basis_vectors_learn.shape)
                    print('In Semi-Sup Learn - after learn - Shape of Hfromfix:', activations_for_fixed.shape)
                    print('In Semi-Sup Learn - after learn - Shape of Hfromlearn:', activations_for_learn.shape)
                    plot_matrix(basis_vectors_fixed, name="Fixed BV After Learn", ylabel='Components', ratio=BASIS_VECTOR_FULL_RATIO)
                    plot_matrix(basis_vectors_learn, name="Learned BV After Learn", ylabel='Components', ratio=BASIS_VECTOR_FULL_RATIO)
                    plot_matrix(activations_for_fixed, name="Activations of Fixed After Learn", ylabel='Components', ratio=ACTIVATION_RATIO)
                    plot_matrix(activations_for_learn, name="Activations of Learned After Learn", ylabel='Components', ratio=ACTIVATION_RATIO)

                # Finally, W = Wfix and Wlearn concatenated together, same w/ H
                if learn_index > 0:
                    activations = np.concatenate((activations_for_fixed, activations_for_learn), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_fixed, basis_vectors_learn), axis=1)
                else:
                    activations = np.concatenate((activations_for_learn, activations_for_fixed), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_learn, basis_vectors_fixed), axis=1)
    
    else:
        # Unsup learning - Do NMF, both W and H learn steps, get W and H
        basis_vectors = np.random.rand(input_matrix.shape[0], num_components)
        if debug:
            print('In Unsup Learn - Shape of Learn Basis Vectors W:', basis_vectors.shape)
            print('In Unsup Learn - Shape of Learn Activations H:', activations.shape)

        # For L1-Penalty, supply a copy of H for W to learn from so W doesn't "make up" for penalized H, vice-verse else bug happens
        activations_use = deepcopy(activations)
        basis_vectors_use = deepcopy(basis_vectors)

        for _ in range(learn_iter):
            # activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / ((basis_vectors.T @ ones) + l1_penalty))
            # basis_vectors *= (((input_matrix / (basis_vectors @ activations)) @ activations.T) / (ones @ activations.T))

            activations *= ((basis_vectors_use.T @ (input_matrix / (basis_vectors_use @ activations))) / ((basis_vectors_use.T @ ones) + l1_penalty))
            basis_vectors *= (((input_matrix / (basis_vectors @ activations_use)) @ activations_use.T) / (ones @ activations_use.T))

    if debug:
        print('(Penalty Present) H Sum:' if (l1_penalty != 0) else 'H Sum:', np.sum(activations))
        print('In Learn - Shape of Learned Activations H:', activations.shape)
        plot_matrix(activations, name="Learned Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
        print('In Learn - Shape of Learned Basis Vectors W:', basis_vectors.shape)
        plot_matrix(basis_vectors, name="Learned Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)

    return activations, basis_vectors


def remove_noise_vectors(activations, basis_vectors, debug=False, num_noisebv=NUM_NOISE_BV):
    basis_vectors = basis_vectors.T[num_noisebv:].T
    activations = activations[num_noisebv:]
    if debug:
        plot_matrix(basis_vectors, name="De-noised Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
        plot_matrix(activations, name="De-noised Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
    return activations, basis_vectors


# Returns real signal, given positive magnitude & phases of a DFT
def inverse_fourier_transform(pos_mag_fft, pos_phases_fft, wdw_size, ova=False, end_sig=None, debug_flag=False):
    # Append the mirrors of the synthetic magnitudes and phases to themselves
    neg_mag_fft = np.flip(pos_mag_fft[1: wdw_size // 2], 0)
    mag_fft = np.append(pos_mag_fft, neg_mag_fft, axis=0)

    neg_phases_fft = np.flip([-x for x in pos_phases_fft[1: wdw_size // 2]], 0)
    phases_fft = np.append(pos_phases_fft, neg_phases_fft, axis=0)

    # Multiply this magnitude fft w/ phases fft
    fft = mag_fft * np.exp(1j*phases_fft)
    # Do ifft on the fft -> waveform
    ifft = np.fft.ifft(fft)
    imaginaries = ifft.imag
    synthetic_sgmt = ifft.real

    if debug_flag:
        print('positive mag FFT:\n', pos_mag_fft[:5])
        print('positive phases:\n', pos_phases_fft[:5])
        print('positive mag FFT and phase lengths:', len(pos_mag_fft), len(pos_phases_fft))
        print('negative mag FFT:\n', neg_mag_fft[:5])
        print('mag FFT of wdw:\n', mag_fft[:5])
        print('negative phases:\n', neg_phases_fft[:5])
        print('phases of FFT of wdw:\n', phases_fft[:5])
        print('FFT of wdw (len =', len(fft), '):\n', fft[:5])
        print('Synthetic imaginaries:\n', imaginaries[:10])
        print('Synthetic segment (len =', len(synthetic_sgmt), '):\n', synthetic_sgmt[:5])

    if ova:
        sgmt_halves = np.split(synthetic_sgmt, 2)
        ova_sgmt, end_sgmt = sgmt_halves[0], sgmt_halves[1] # First, then second half

        if end_sig is None:
            end_sig = np.zeros(wdw_size // 2)

        end_sum = ova_sgmt + end_sig    # Numpy element-wise addition of OVA parts
        synthetic_sgmt = np.concatenate((end_sum, end_sgmt), axis=0)    # Concatenate OVA part, with trailing end

        if debug_flag:
            print('ova_sgmt (len =', len(ova_sgmt), '):\n', ova_sgmt[-10:], 
                  '\nend_sgmt (len =', len(end_sgmt), '):\n', end_sgmt[-10:], 
                  '\nend_sig (len =', len(end_sig), '):\n', end_sig[-10:], 
                  '\nend_sum (len =', len(end_sum), '):\n', end_sum[-10:])

    return synthetic_sgmt.tolist()


# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, ova=False, debug=False):
    num_sgmts = synthetic_spgm.shape[1]
    synthetic_spgm = synthetic_spgm.T     # Get spectrogram back into orientation we did calculations on
    synthetic_sig = []
    for i in range(num_sgmts):
        debug_flag = (i == 0 or i == 1) if debug else False

        # Do overlap-add operations if ova (but only if list already has >= 1 element)
        if ova and len(synthetic_sig):
            synthetic_sgmt = inverse_fourier_transform(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                       end_sig=synthetic_sig[-(wdw_size // 2):].copy())
            synthetic_sig = synthetic_sig[: -(wdw_size // 2)] + synthetic_sgmt
        else:
            synthetic_sig += inverse_fourier_transform(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag)

        if debug_flag:
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
    _ = ax.imshow(np.log(matrix), extent=[0, num_wdws, STD_SR_HZ // 2, 0])
    fig.tight_layout()
    # bottom, top = plt.ylim()
    # print('Bottom:', bottom, 'Top:', top)
    plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
    ax.set_aspect(ratio)    # Set a visually nice ratio
    plt.show()


def reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, segment=False, write_file=False, debug=False):
    print('--Initiating Reconstruct Mode--')

    # if segment:
    #     # TEMP SO WE CAN FIND MAX NOISE SEGMENT - noise from 1.8 ro 2.1 seconds
    #     noise_sig_len = 2
    #     # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
    #     noise_sgmt_num = math.ceil((STD_SR_HZ * 2.2) / wdw_size)    # 2.1 seconds (23rd window to (not including) 25th window)
    #     # print('Noise segment num:', noise_sgmt_num)
    #     noise_sig = sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
    #     # noise_sig = sig[23 * wdw_size: 25 * wdw_size] 
    #     # 23 25
    #     out_filepath = 'practice_' + out_filepath
    #     print('\n--Making Signal Spectrogram--\n')
    #     spectrogram, phases = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    if segment:
        # TEMP SO WE CAN FIND NO VOICE SEGMENT
        # no_voice_sig = sig[(77 * wdw_size):]# + (wdw_size // 2):] 
        
        # out_filepath = 'novoice_' + out_filepath
        print('\n--Making Signal Spectrogram--\n')
        spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)

        # TODO
        # spectrogram = spectrogram[:, 154:]
        # phases = [x[154:] for x in phases]

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


def restore_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, marybv=False, noisebv=False, avgbv=False, semisuplearn='None', 
                  semisupmadeinit=False, write_file=False, debug=False, nohanbv=False, prec_noise=False, eqbv=False, incorrect_semisup=False,
                  learn_iter=MAX_LEARN_ITER, num_noisebv=NUM_NOISE_BV, l1_penalty=L1_PENALTY):
    print('--Initiating Restore Mode--')
    print('\n--Making Piano Basis Vectors--\n')

    # Temporary branch for testing:
    if nohanbv:
        # basis_vectors = get_basis_vectors(BEST_WDW_NUM, wdw_size, ova=False, mary=marybv, noise=noisebv, avg=avgbv, semisuplearn=semisuplearn, semisupmadeinit=semisupmadeinit, debug=debug, precise_noise=prec_noise)
        basis_vectors = get_basis_vectors(BEST_WDW_NUM, wdw_size, ova=False, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise, num_noise=num_noisebv)
    else:
        basis_vectors = get_basis_vectors(BEST_WDW_NUM, wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise, eq=eqbv, num_noise=num_noisebv)

    if 'brahms' in out_filepath:
        # Take out voice from brahms sig for now, should take it from nmf_learn from spectrogram later
        sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]

    print('\n--Making Signal Spectrogram--\n')
    spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)

    if debug:
        print('Shape of Original Piano Basis Vectors W:', basis_vectors.shape)
        print('Shape of Signal Spectrogram V:', spectrogram.shape)
        
    print('\n--Learning Piano Activations--\n')
    # num_components = basis_vectors.shape[1]   # BAD
    num_components = len(SORTED_NOTES[MARY_START_INDEX: MARY_STOP_INDEX]) if marybv else len(SORTED_NOTES)
    if noisebv:
        num_components += num_noisebv

    if semisuplearn == 'Piano':     # Semi-Supervised Learn (learn Wpiano too)
        activations, basis_vectors = nmf_learn(spectrogram, num_components, basis_vectors=basis_vectors, learn_index=num_noisebv, 
                                               madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
                                               l1_penalty=l1_penalty)
    elif semisuplearn == 'Noise':   # Semi-Supervised Learn (learn Wnoise too)
        activations, basis_vectors = nmf_learn(spectrogram, num_components, basis_vectors=basis_vectors, learn_index=(-1 * num_noisebv), 
                                               madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
                                               l1_penalty=l1_penalty)
    else:                           # Supervised Learn
        activations, _ = nmf_learn(spectrogram, num_components, basis_vectors=basis_vectors, debug=debug, learn_iter=learn_iter, 
                                   l1_penalty=l1_penalty)

    if noisebv:
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug, num_noisebv=num_noisebv)
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
    noisebv_flag = True     # Confirmed helps - to be kept true
    avgbv_flag = True       # Confirmed helps - to be kept true
    ova_flag = True         # Confirmed helps - to be kept true
    marybv_flag = False     # Special case for Mary.wav - basis vectors size optimization test

    # Ternary flag - 'Piano', 'Noise', or 'None' (If not 'None', noisebv_flag MUST BE TRUE)
    semi_sup_learn = 'None'
    semi_sup_made_init = True   # Only considered when semi_sup_learn != 'None'
    # Not advised - semi_sup_init=False & semi_sup_learn='Piano'

    l1pen_flag = True if (L1_PENALTY != 0) else False

    # TODO: Use argparse library
    # Configure params
    # Mode - RECONST or RESTORE
    mode = sys.argv[1]
    out_filepath = 'output_restored_wav_v2/' if mode == 'RESTORE' else 'output_reconstructed_wav/'
    # Signal - comes as a list, filepath or a length
    sig_sr = STD_SR_HZ # Initialize sr to default
    if sys.argv[2].startswith('['):
        sig = np.array([int(num) for num in sys.argv[2][1:-1].split(',')])
        out_filepath += 'my_sig'
    elif not sys.argv[2].endswith('.wav'):  # Work around for is a number
        sig = np.random.rand(int(sys.argv[2].replace(',', '')))
        out_filepath += 'rand_sig'
    else:
        sig_sr, sig = wavfile.read(sys.argv[2])
        if sig_sr != STD_SR_HZ:
            sig, _ = librosa.load(sys.argv[2], sr=STD_SR_HZ)  # Upsample to 44.1kHz if necessary
        start_index = (sys.argv[2].rindex('/') + 1) if (sys.argv[2].find('/') != -1) else 0
        out_filepath += sys.argv[2][start_index: -4]
    # Debug-print/plot option
    debug_flag = True if sys.argv[3] == 'TRUE' else False
    # Window Size
    wdw_size = int(sys.argv[4]) if (len(sys.argv) == 5) else PIANO_WDW_SIZE
    # Overlap-Add is Necessary & Default
    if ova_flag:
        out_filepath += '_ova'

    if mode == 'RECONST': # RECONSTRUCT BLOCK
        # FOR TESTING
        no_voice = False
        if no_voice:
            out_filepath += '_novoice'
        out_filepath += '.wav'
        reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=ova_flag, segment=no_voice, 
                          write_file=True, debug=debug_flag)
    else:   # MAIN RESTORE BLOCK
        
        # out_filepath += '_messedup'
        
        if semi_sup_learn == 'Piano':
            out_filepath += '_sslrnpiano'
            if semi_sup_made_init:
                out_filepath += '_madeinit'
        elif semi_sup_learn == 'Noise':
            out_filepath += '_sslrnnoise'
            if semi_sup_made_init:
                out_filepath += '_madeinit'
        if noisebv_flag:
            out_filepath += ('_' + str(NUM_NOISE_BV) + 'noisebv')
        if avgbv_flag:
            out_filepath += '_avgbv'
        if l1pen_flag:
            out_filepath += ('_l1pen' + str(L1_PENALTY))
        out_filepath += '.wav'
        restore_audio(sig, wdw_size, out_filepath, sig_sr, ova=ova_flag, marybv=marybv_flag, noisebv=noisebv_flag, 
                      avgbv=avgbv_flag, semisuplearn=semi_sup_learn, semisupmadeinit=semi_sup_made_init, write_file=True, debug=debug_flag)


if __name__ == '__main__':
    main()
