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

# SORTED_NOTES = ["A0", "Bb0", "B0", "C1", 
#                 "Db1", "D1", "Eb1", "E1", "F1", "Gb1", "G1", "Ab1", "A1", "Bb1", "B1", "C2", 
#                 "Db2", "D2", "Eb2", "E2", "F2", "Gb2", "G2", "Ab2", "A2", "Bb2", "B2", "C3", 
#                 "Db3", "D3", "Eb3", "E3", "F3", "Gb3", "G3", "Ab3", "A3", "Bb3", "B3", "C4", 
#                 "Db4", "D4", "Eb4", "E4", "F4", "Gb4", "G4", "Ab4", "A4", "Bb4", "B4", "C5", 
#                 "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C6", 
#                 "Db6", "D6", "Eb6", "E6", "F6", "Gb6", "G6", "Ab6", "A6", "Bb6", "B6", "C7", 
#                 "Db7", "D7", "Eb7", "E7", "F7", "Gb7", "G7", "Ab7", "A7", "Bb7", "B7", "C8"]

# SORTED_FUND_FREQ = [28, 29, 31, 33, 
#                     35, 37, 39, 41, 44, 46, 49, 52, 55, 58, 62, 65, 
#                     69, 73, 78, 82, 87, 93, 98, 104, 110, 117, 123, 131, 
#                     139, 147, 156, 165, 175, 185, 196, 208, 220, 233, 247, 262, 
#                     277, 294, 311, 330, 349, 370, 392, 415, 440, 466, 494, 523, 
#                     554, 587, 622, 659, 698, 740, 784, 831, 880, 932, 988, 1047, 
#                     1109, 1175, 1245, 1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 
#                     2217, 2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186]

# Make spectrogram w/ ova resource: https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/

import sys, os, math, librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Constants
STD_SR_HZ = 44100
MARY_SR_HZ = 16000
PIANO_WDW_SIZE = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
DEBUG_WDW_SIZE = 4
# Resolution (Windows per second) = STD_SR_HZ / PIANO_WDW_SIZE

MARY_START_INDEX, MARY_STOP_INDEX = 39, 44  # Mary notes = E4, D4, C4
BEST_PIANO_BV_SGMT = 5
WDW_NUM_AFTER_VOICE = 77
NUM_PIANO_NOTES = 88
NUM_MARY_PIANO_NOTES = MARY_STOP_INDEX - MARY_START_INDEX
MAX_LEARN_ITER = 100

BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
ACTIVATION_RATIO = 8.0
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008


# Functions
# Learning optimization
def make_row_sum_matrix(mtx, out_shape):
    row_sums = mtx.sum(axis=1)
    return np.repeat(row_sums, out_shape[1], axis=0)

# Idea - rank-1 approx = take avg of the pos. mag. spectrogram NOT the signal
def make_basis_vector(waveform, wf_type, wf_sr, num, wdw_size, ova=False, avg=False, debug=False):
    # if debug:
    #     print('In make bv')
    spectrogram, phases = make_spectrogram(waveform, wdw_size, ova=ova)
    # if debug:
    #     print('Made the V')
    if avg:
        # OLD WAY - Averaged the signal, not a spectrogram
        # num_sgmts = math.floor(len(waveform) / wdw_size) # Including incomplete windows throws off averaging
        # all_sgmts = np.array([waveform[i * wdw_size: (i + 1) * wdw_size] for i in range(num_sgmts)])
        # sgmt = np.mean(all_sgmts, axis=0)
        # if ova:
        #     sgmt *= np.hanning(wdw_size)
        # # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
        # return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

        basis_vector = np.mean(spectrogram, axis=1) # Actually the bv that makes best rank-1 approx. of V (piano note spectrogram) - the avg

    else:
        # OLD WAY - Made a single pos mag fft
        # sgmt = waveform[(BEST_PIANO_BV_SGMT - 1) * wdw_size: BEST_PIANO_BV_SGMT * wdw_size]    # BEST_PIANO_BV_SGMT is naturally-indexed
        # # print("Type of elem in piano note sig:", type(sgmt[0]))
        # if len(sgmt) != wdw_size:
        #         deficit = wdw_size - len(sgmt)
        #         sgmt = np.pad(sgmt, (deficit, 0), mode='constant')
        # if ova:
        #     sgmt *= np.hanning(wdw_size)
        # # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
        # return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

        basis_vector = spectrogram[:, BEST_PIANO_BV_SGMT].copy()

    if debug:
        if wf_sr > 0 and avg and ova:   # Temp test - success!
            avg_spgm = np.array([basis_vector for _ in range(spectrogram.shape[1])]).T
            avg_sig = make_synthetic_signal(avg_spgm, phases, wdw_size, wf_type, ova=ova, debug=False)
            wavfile.write('/Users/quinnmc/Desktop/AudioRestore/avged_ova_notes/avged_ova_note_' + str(num) + '.wav', 
                          wf_sr, avg_sig.astype(wf_type))

        # print('Shape of note spectrogram:', spectrogram.shape)
        # print('Shape of basis vector made from this:', basis_vector.shape, '\n')

    return basis_vector


def make_basis_vector_old(waveform, wdw_size, ova=False, avg=False):
    if avg:
        num_sgmts = math.floor(len(waveform) / wdw_size) # Including incomplete windows throws off averaging
        all_sgmts = np.array([waveform[i * wdw_size: (i + 1) * wdw_size] for i in range(num_sgmts)])
        sgmt = np.mean(all_sgmts, axis=0)
    
    else:
        sgmt = waveform[(BEST_PIANO_BV_SGMT - 1) * wdw_size: BEST_PIANO_BV_SGMT * wdw_size].copy()  # BEST_PIANO_BV_SGMT is naturally-indexed
        # print("Type of elem in piano note sig:", type(sgmt[0]))
        if len(sgmt) != wdw_size:
                deficit = wdw_size - len(sgmt)
                sgmt = np.pad(sgmt, (deficit, 0), mode='constant')
        
    if ova:
        sgmt *= np.hanning(wdw_size)
    # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
    return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1].copy()


# Time/#segments is irrelevant to # of basis vectors made (so maximize)
def make_noise_basis_vectors(num, wdw_size, ova=False, eq=False, debug=False, precise_noise=False, eq_thresh=800000,
                             start=0, stop=25):
    # sr, brahms_sig = wavfile.read('../brahms.wav')
    _, brahms_sig = wavfile.read('/Users/quinnmc/Desktop/AudioRestore/brahms.wav')
    # Convert to mono signal (avg left & right channels) 
    # brahms_sig = np.array([((x[0] + x[1]) / 2) for x in brahms_sig.astype('float64')])

    # Precise noise is pointless -> b/c we want to mximize what we draw noise from
    # noise_sig_len = 2 if ova else 1 # The 1 is an educated guess, 2 is empircally derived
    # # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
    # noise_sgmt_num = math.ceil((STD_SR_HZ * 2.2) / wdw_size)    # 2.2 seconds (24rd window to (not including) 26th window)
    # if precise_noise:
    #     noise_sig = brahms_sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
    # else:
    # # All noise from beginning of clip
    #     noise_sig = brahms_sig[: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 

    noise_sig = brahms_sig[(start * wdw_size): (stop * wdw_size)].copy()

    # Equalize noise bv's? - no doesnt make sense to
    # if eq:  # Make it louder
    #     while np.max(np.abs(sig)) < sig_thresh:
    #         sig *= 1.1

    print('\n----Making Noise Spectrogram--\n')
    spectrogram, _ = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    print('\n----Learning Noise Basis Vectors--\n')
    _, noise_basis_vectors = nmf_learn(spectrogram, num, debug=debug)
    if debug:
        print('Shape of Noise Spectogram V:', spectrogram.shape, np.sum(spectrogram))
        print('Shape of Learned Noise Basis Vectors W:', noise_basis_vectors.shape)

    # if False:  # Make louder # if eq:
    #     new_bvs = []
    #     for bv in noise_basis_vectors:
    #         while np.max(bv[1:]) < bv_thresh:
    #             bv *= 1.1
    #         new_bvs.append(bv)
    #     noise_basis_vectors = np.array(new_bvs)

    return list(noise_basis_vectors.T)    # List format is for use in get_basis_vectors(), transpose into similar format


def make_basis_vectors(wdw_size, filepath, ova=False, avg=False, mary_flag=False, eq=False, eq_thresh=800000, debug=False):
    # bv_thresh = 800000  # Based on max_val (not including first freq bin) - (floor) is 943865
    sig_thresh = 410    # 137 150.0 - Actual median max, 410 448.8 - Actual mean max, 11000 11966.0 - Actual max
    # max_val = None      # To get threshold
    basis_vectors, sorted_notes = [], []
    # Read in ordered piano notes
    with open('/Users/quinnmc/Desktop/AudioRestore/piano_notes_and_fund_freqs.csv', 'r') as notes_f:
        for line in notes_f.readlines():
            sorted_notes.append(line.split(',')[0])
    
    with open(filepath, 'w') as bv_f:
        base_dir = os.getcwd()
        os.chdir('/Users/quinnmc/Desktop/AudioRestore/all_notes_ff_wav')
        # audio_files is a list of strings, need to sort it by note
        unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]
        sorted_file_names = ['Piano.ff.' + x + '.wav' for x in sorted_notes]
        audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))

        if mary_flag:
            start, stop = MARY_START_INDEX, MARY_STOP_INDEX
        else:
            start, stop = 0, len(audio_files)
        
        for i in range(start, stop):
            audio_file = audio_files[i]
            note_sr, stereo_sig = wavfile.read(audio_file)
            orig_note_sig_type = stereo_sig.dtype
            # Convert to mono signal (avg left & right channels) 
            sig = np.array([((x[0] + x[1]) / 2) for x in stereo_sig.astype('float64')])

            # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
            amp_thresh = max(sig) * 0.01
            while sig[0] < amp_thresh:
                sig = sig[1:]
            while sig[-1] < amp_thresh:
                sig = sig[:-1]

            if eq:  # Make it louder
                while np.mean(np.abs(sig)) < sig_thresh:
                    sig *= 1.1
                # Manual override for fussy basis vectors (85, 86, 88)
                if i == 85 or i == 87:
                    sig *= 1.5
                elif i == 84:
                    sig *= 1.75

            # Write trimmed piano note signals to WAV - check if trim is good
            # if i == 40 or i == 43 or i == 46 or i < 10:
            #     wavfile.write('/Users/quinnmc/Desktop/AudioRestore/trimmed_notes/trimmed_note_' + str(i) + '.wav', 
            #                   note_sr, sig.astype(orig_note_sig_type))
            
            basis_vector = make_basis_vector(sig, orig_note_sig_type, note_sr, i, wdw_size, ova=ova, avg=avg, debug=debug)
            # else:   # Testing temp block - success!
                # basis_vector = make_basis_vector(sig, orig_note_sig_type, -1, i, wdw_size, ova=ova, avg=avg, debug=debug)

            # Old - adjust signal instead
            # if eq:  # Make it louder
            #     while np.max(basis_vector[1:]) < bv_thresh:
            #         basis_vector *= 1.1

            basis_vectors.append(basis_vector)
            bv_f.write(','.join([str(x) for x in basis_vector]) + '\n')

            # if max_val is None or np.median(np.abs(sig)) > max_val:
            #     max_val = np.median(np.abs(sig))

        os.chdir(base_dir)
    # print('\nMAX SIG VAL:', max_val, '\n')
    return basis_vectors


# We don't save bvs w/ noise anymnore, 
# we just calc noise and pop it on top of restored-from-file piano bvs

# W LOGIC
# Basis vectors in essence are the "best" dft of a sound w/ constant pitch (distinct freq signature)
def get_basis_vectors(wdw_size, ova=False, mary=False, noise=False, avg=False, debug=False, precise_noise=False, eq=False, 
                      num_noise=0, noise_start=6, noise_stop=25):
    # Save/load basis vectors (w/o noise) to/from CSV files
    filepath = '/Users/quinnmc/Desktop/AudioRestore/csv_saves_bv/basis_vectors'
    if mary:
        filepath += '_mary'
    if ova:
        filepath += '_ova'
    if avg:
        filepath += '_avg'
    if eq:
        filepath += '_eqsig' # '_eqmeansig' '_eqmediansig'
    filepath += '.csv'

    try:
        with open(filepath, 'r') as bv_f:
            print('FILE FOUND - READING IN BASIS VECTORS:', filepath)
            basis_vectors = [[float(sub) for sub in string.split(',')] for string in bv_f.readlines()]

    except FileNotFoundError:
        print('FILE NOT FOUND - MAKING BASIS VECTORS:', filepath)
        basis_vectors = make_basis_vectors(wdw_size, filepath, ova=ova, avg=avg, mary_flag=mary, eq=eq, eq_thresh=800000, debug=debug)

    if debug:
        print('Basis Vectors Length:', len(basis_vectors))

    # Make and add noise bv's if necessary
    if noise:
        noise_basis_vectors = make_noise_basis_vectors(num_noise, wdw_size, ova=ova, eq=eq, debug=debug, 
                                                    precise_noise=precise_noise, eq_thresh=800000,
                                                    start=noise_start, stop=noise_stop)
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
def signal_to_pos_fft(sgmt, wdw_size, ova=False, debug_flag=False):
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
    pos_phases_fft = phases_fft[: (wdw_size // 2) + 1].copy()
    pos_mag_fft = mag_fft[: (wdw_size // 2) + 1].copy()

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
    # If 8-bit PCM, convert to 16-bit PCM (signed to unsigned)
    if signal.dtype == 'uint8':
        signal = convert_sig_8bit_to_16bit(signal)

    num_spls = len(signal)
    if isinstance(signal[0], np.ndarray):   # Stereo signal = 2 channels
        sig = np.array([((x[0] + x[1]) / 2) for x in signal.astype('float64')])
    else:                                   # Mono signal = 1 channel    
        sig = np.array(signal).astype('float64')

    if debug:
        print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', sig[(wdw_size // 2): (wdw_size // 2) + 20]) if len(sig) > 20 else print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', sig)

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
        pos_mag_fft, pos_phases_fft = signal_to_pos_fft(sgmt, wdw_size, ova=ova, debug_flag=debug_flag)
        
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

# L1-Penalize fix - only do when basis vectors fixed, and (probably) only when Wfixed is piano cause piano H good

# General case NMF algorithm
def nmf_learn(input_matrix, num_components, basis_vectors=None, learn_index=0, madeinit=False, debug=False, incorrect=False, 
              learn_iter=MAX_LEARN_ITER, l1_penalty=0, mutual_use_update=True):
    activations = np.random.rand(num_components, input_matrix.shape[1])
    basis_vectors_on_pass = basis_vectors   # For use in debug print for l1-penalty
    ones = np.ones(input_matrix.shape) # so dimensions match W transpose dot w/ V
    if debug:
        print('Made activations:\n', activations)

    if debug:
        print('In NMF Learn, input_matrix sum:', np.sum(input_matrix))

    if basis_vectors is not None:
        if debug:
            print('In Sup or Semi-Sup Learn - Shape of Given Basis Vectors W:', basis_vectors.shape)
        if learn_index == 0:
            if debug:
                print('Applying L1-Penalty of', str(l1_penalty), 'to Activations')
            # Sup Learning - Do NMF w/ whole W, only H learn step, get H
            for _ in range(learn_iter):
                activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / ((basis_vectors.T @ ones) + l1_penalty))
        else:
            # Semi-Sup Learning - Do NMF w/ part of W, part of W and H learn steps, get W and H
            # No L1-Penalty in Unsupervised Learning Part (learning both W and H)
            (basis_vectors_fixed, basis_vectors_learn, 
            activations_for_fixed, activations_for_learn) = partition_matrices(learn_index, basis_vectors, 
                                                                                 activations, madeinit=madeinit)
            if debug:
                print('Semi-Sup Learning', 'Piano' if (learn_index > 0) else 'Noise')
                print('In Semi-Sup Learn - Shape of Wfix:', basis_vectors_fixed.shape)
                print('In Semi-Sup Learn - Shape of Wlearn:', basis_vectors_learn.shape)
                print('In Semi-Sup Learn - Shape of Hfromfix:', activations_for_fixed.shape)
                print('In Semi-Sup Learn - Shape of Hfromlearn:', activations_for_learn.shape)
                plot_matrix(basis_vectors_fixed, name="Fixed BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                plot_matrix(basis_vectors_learn, name="Learned BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                plot_matrix(activations_for_fixed, name="Activations of Fixed Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)
                plot_matrix(activations_for_learn, name="Activations of Learned Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)

            if incorrect:   # For results of bug
                # Don't fix the fixed part - W = Wfix and Wlearn concatenated together, same w/ H
                # No L1-Penalty in Incorrect Approach
                if learn_index > 0:
                    activations = np.concatenate((activations_for_fixed, activations_for_learn), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_fixed, basis_vectors_learn), axis=1)
                else:
                    activations = np.concatenate((activations_for_learn, activations_for_fixed), axis=0)
                    basis_vectors = np.concatenate((basis_vectors_learn, basis_vectors_fixed), axis=1)

                if mutual_use_update:
                    for _ in range(learn_iter):
                        activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / (basis_vectors.T @ ones))
                        basis_vectors *= (((input_matrix / (basis_vectors @ activations)) @ activations.T) / (ones @ activations.T))
                else:
                    activations_use = deepcopy(activations)
                    basis_vectors_use = deepcopy(basis_vectors)
                    for _ in range(learn_iter):
                        activations *= ((basis_vectors_use.T @ (input_matrix / (basis_vectors_use @ activations))) / (basis_vectors_use.T @ ones))
                        basis_vectors *= (((input_matrix / (basis_vectors @ activations_use)) @ activations_use.T) / (ones @ activations_use.T))

            else:
                # if l1_penalty != 0 and ((pen == 'Piano' and learn_index < 0) or (pen == 'Noise' and learn_index > 0)): # or pen == 'Both'):
                if debug:
                    print('Applying L1-Penalty of', str(l1_penalty), 'to', 'Noise' if (learn_index > 0) else 'Piano', '(Fixed) Activations')
                # Do NMF w/ Wfix (W given subset), only H learn step, get H
                for learn_i in range(learn_iter):
                    activations_for_fixed *= ((basis_vectors_fixed.T @ (input_matrix / (basis_vectors_fixed @ activations_for_fixed))) / ((basis_vectors_fixed.T @ ones) + l1_penalty))

                    if debug and (learn_i % 5 == 0):
                        # Strange - activations seem tobe the same for cmoponents in groups of almost 3 (2.75)
                        #     (only see 32 distinguishable rows), try to see difference in first 8 components - look the same
                        # Lets look at first 5ish components (rows) of activations
                        print('Last 10 (components) rows of activations:')
                        print(np.mean(activations_for_fixed[-1]))
                        print(np.mean(activations_for_fixed[-2]))
                        print(np.mean(activations_for_fixed[-3]))
                        print(np.mean(activations_for_fixed[-4]))
                        print(np.mean(activations_for_fixed[-5]))
                        print(np.mean(activations_for_fixed[-6]))
                        print(np.mean(activations_for_fixed[-7]))
                        print(np.mean(activations_for_fixed[-8]))
                        print(np.mean(activations_for_fixed[-9]))
                        print(np.mean(activations_for_fixed[-10]))
                        print()

                        print('Last 10 columns of basis vectors:')
                        print(np.mean(basis_vectors_fixed[:, -1]))
                        print(np.mean(basis_vectors_fixed[:, -2]))
                        print(np.mean(basis_vectors_fixed[:, -3]))
                        print(np.mean(basis_vectors_fixed[:, -4]))
                        print(np.mean(basis_vectors_fixed[:, -5]))
                        print(np.mean(basis_vectors_fixed[:, -6]))
                        print(np.mean(basis_vectors_fixed[:, -7]))
                        print(np.mean(basis_vectors_fixed[:, -8]))
                        print(np.mean(basis_vectors_fixed[:, -9]))
                        print(np.mean(basis_vectors_fixed[:, -10]))
                        print()

                        plot_matrix(activations_for_fixed, 'Fixed Activations', 'Components', ACTIVATION_RATIO)
                        # plot_matrix(activations_for_fixed[:11], 'Fixed Activations (Components 1-11)', 'Components', ACTIVATION_RATIO)
                        # plot_matrix(activations_for_fixed[:5], 'Fixed Activations (Components 1-5)', 'Components', ACTIVATION_RATIO)

                        plot_matrix(basis_vectors_fixed, 'Fixed Basis Vectors', 'Frequency (Hz)', BASIS_VECTOR_FULL_RATIO)

                        

                # else:
                #     # Do NMF w/ Wfix (W given subset), only H learn step, get H
                #     for _ in range(learn_iter):
                #         activations_for_fixed *= ((basis_vectors_fixed.T @ (input_matrix / (basis_vectors_fixed @ activations_for_fixed))) / (basis_vectors_fixed.T @ ones))

                # if l1_penalty != 0 and ((pen == 'Noise' and learn_index < 0) or (pen == 'Piano' and learn_index > 0) or pen == 'Both'):
                #     if debug:
                #         print('Applying L1-Penalty of', str(l1_penalty), 'to', 'Piano' if (learn_index > 0) else 'Noise', '(Learned) Activations')

                #     if mutual_use_update:
                #         # Do NMF w/ Wlearn (W given subset OR random mtx), both W and H learn steps, get W and H
                #         for _ in range(learn_iter):
                #             activations_for_learn *= ((basis_vectors_learn.T @ (input_matrix / (basis_vectors_learn @ activations_for_learn))) / ((basis_vectors_learn.T @ ones) + l1_penalty))
                #             basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn)) @ activations_for_learn.T) / (ones @ activations_for_learn.T))
                #     else:
                #         # Make copy of Hlearn to be used by ONLY Wlearn - this prevents W from "making up" for penalized H, vice-verse else bug happens
                #         activations_for_learn_use = deepcopy(activations_for_learn)
                #         basis_vectors_learn_use = deepcopy(basis_vectors_learn)
                #         for _ in range(learn_iter):
                #             activations_for_learn *= ((basis_vectors_learn_use.T @ (input_matrix / (basis_vectors_learn_use @ activations_for_learn))) / ((basis_vectors_learn_use.T @ ones) + l1_penalty))
                #             basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn_use)) @ activations_for_learn_use.T) / (ones @ activations_for_learn_use.T))

                # else:
                if mutual_use_update:
                    # Do NMF w/ Wlearn (W given subset OR random mtx), both W and H learn steps, get W and H
                    for learn_i in range(learn_iter):

                        if debug and (learn_i == 0):
                            pass
                            # print('Before Hlearn update:')
                            # print('Input Matrix:\n', input_matrix)
                            # print('Basis Vectors:\n', basis_vectors_learn)
                            # print('Activations:\n', activations_for_learn)
                            # print('H Calc 1:\n', basis_vectors_learn @ activations_for_learn)
                            # print('H Calc 2:\n', input_matrix / (basis_vectors_learn @ activations_for_learn))
                            # print('H Calc 3:\n', basis_vectors_learn.T @ (input_matrix / (basis_vectors_learn @ activations_for_learn)))
                            # print('H Calc 4:\n', basis_vectors_learn.T @ ones)

                        activations_for_learn *= ((basis_vectors_learn.T @ (input_matrix / (basis_vectors_learn @ activations_for_learn))) / (basis_vectors_learn.T @ ones))
                        
                        if debug and (learn_i == 0):
                            pass
                            # print('After Hlearn, Before Wlearn update:')
                            # print('Input Matrix:\n', input_matrix)
                            # print('Basis Vectors:\n', basis_vectors_learn)
                            # print('Activations:\n', activations_for_learn)
                            # print('W Calc 1:\n', basis_vectors_learn @ activations_for_learn)
                            # print('W Calc 2:\n', input_matrix / (basis_vectors_learn @ activations_for_learn))
                            # print('W Calc 3:\n', (input_matrix / (basis_vectors_learn @ activations_for_learn)) @ activations_for_learn.T)
                            # print('W Calc 4:\n', ones @ activations_for_learn.T)

                        basis_vectors_learn *= (((input_matrix / (basis_vectors_learn @ activations_for_learn)) @ activations_for_learn.T) / (ones @ activations_for_learn.T))
                else:
                    # Make copy of Hlearn to be used by ONLY Wlearn - this prevents W from "making up" for penalized H, vice-verse else bug happens
                    activations_for_learn_use = deepcopy(activations_for_learn)
                    basis_vectors_learn_use = deepcopy(basis_vectors_learn)
                    for _ in range(learn_iter):
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
                    plot_matrix(basis_vectors_fixed, name="Fixed BV After Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                    plot_matrix(basis_vectors_learn, name="Learned BV After Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
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
        # No L1-Penalty in Unsupervised Learning
        basis_vectors = np.random.rand(input_matrix.shape[0], num_components)
        if debug:
            print('Made basis_vectors:\n', basis_vectors)

        if debug:
            print('In Unsup Learn - Shape of Learn Basis Vectors W:', basis_vectors.shape, 'Sum:', np.sum(basis_vectors))
            print('In Unsup Learn - Shape of Learn Activations H:', activations.shape, 'Sum:', np.sum(activations))

        if mutual_use_update:
            for learn_i in range(learn_iter):
                # if debug and (learn_i == 0):
                #     print('Before H update:')
                #     print('Input Matrix:\n', input_matrix)
                #     print('Basis Vectors:\n', basis_vectors)
                #     print('Activations:\n', activations)
                #     print('H Calc 1:\n', basis_vectors @ activations)
                #     print('H Calc 2:\n', input_matrix / (basis_vectors @ activations))
                #     print('H Calc 3:\n', basis_vectors.T @ (input_matrix / (basis_vectors @ activations)))
                #     print('H Calc 4:\n', basis_vectors.T @ ones)

                activations *= ((basis_vectors.T @ (input_matrix / (basis_vectors @ activations))) / (basis_vectors.T @ ones))

                # if debug and (learn_i == 0):
                #     print('After H, Before W update:')
                #     print('Input Matrix:\n', input_matrix)
                #     print('Basis Vectors:\n', basis_vectors)
                #     print('Activations:\n', activations)
                #     print('W Calc 1:\n', basis_vectors @ activations)
                #     print('W Calc 2:\n', input_matrix / (basis_vectors @ activations))
                #     print('W Calc 3:\n', (input_matrix / (basis_vectors @ activations)) @ activations.T)
                #     print('W Calc 4:\n', ones @ activations.T)
                
                
                basis_vectors *= (((input_matrix / (basis_vectors @ activations)) @ activations.T) / (ones @ activations.T))
                if debug:
                    pass
                    # print('H Sum during learn:', np.sum(activations))
                    # print('W Sum during learn:', np.sum(basis_vectors))
                    # print('V Sum during learn:', np.sum(input_matrix))
        else:
            # For L1-Penalty, supply a copy of H for W to learn from so W doesn't "make up" for penalized H, vice-verse else bug happens
            activations_use = deepcopy(activations)
            basis_vectors_use = deepcopy(basis_vectors)
            for _ in range(learn_iter):
                activations *= ((basis_vectors_use.T @ (input_matrix / (basis_vectors_use @ activations))) / (basis_vectors_use.T @ ones))
                basis_vectors *= (((input_matrix / (basis_vectors @ activations_use)) @ activations_use.T) / (ones @ activations_use.T))

    # Report activation sums for penalty check (sup and semisup)
    if basis_vectors_on_pass is not None:
        # Report fixed activation sum for penalty check in case of semi-sup
        if learn_index != 0:
            print('(Penalty Present) Hfix Sum:' if (l1_penalty != 0) else 'Hfix Sum:', np.sum(activations_for_fixed))
        print('(Penalty Present) H Sum:' if (l1_penalty != 0) else 'H Sum:', np.sum(activations))

    if debug:
        print('In Learn - Shape of Learned Activations H:', activations.shape)
        plot_matrix(activations, name="Learned Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
        print('In Learn - Shape of Learned Basis Vectors W:', basis_vectors.shape)
        plot_matrix(basis_vectors, name="Learned Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)

    return activations, basis_vectors


def noise_split_matrices(activations, basis_vectors, num_noisebv, debug=False):
    # piano_basis_vectors = basis_vectors.T[num_noisebv:].T
    piano_basis_vectors = basis_vectors[:, num_noisebv:].copy()
    piano_activations = activations[num_noisebv:].copy()
    noise_basis_vectors = basis_vectors[:, :num_noisebv].copy()
    noise_activations = activations[:num_noisebv].copy()
    if debug:
        plot_matrix(piano_basis_vectors, name="De-noised Piano Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
        plot_matrix(piano_activations, name="De-noised Piano Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
        plot_matrix(noise_basis_vectors, name="Sep. Noise Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
        plot_matrix(noise_activations, name="Sep. Noise Piano Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
    return noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors


# Returns real signal, given positive magnitude & phases of a DFT
def pos_fft_to_signal(pos_mag_fft, pos_phases_fft, wdw_size, ova=False, end_sig=None, debug_flag=False):
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
        synthetic_sgmt = np.concatenate((end_sum, end_sgmt), axis=0)    # Concatenate OVA part with trailing end part

        if debug_flag:
            print('ova_sgmt (len =', len(ova_sgmt), '):\n', ova_sgmt[-10:], 
                  '\nend_sgmt (len =', len(end_sgmt), '):\n', end_sgmt[-10:], 
                  '\nend_sig (len =', len(end_sig), '):\n', end_sig[-10:], 
                  '\nend_sum (len =', len(end_sum), '):\n', end_sum[-10:])

    return synthetic_sgmt.tolist()


# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_type, ova=False, debug=False):
    num_sgmts = synthetic_spgm.shape[1]
    # If both noise and piano in spgm, reuse phases in synthesis
    if num_sgmts != len(phases):   
        phases += phases
    synthetic_spgm = synthetic_spgm.T     # Get spectrogram back into orientation we did calculations on
    synthetic_sig = []
    for i in range(num_sgmts):
        debug_flag = (i == 0 or i == 1) if debug else False

        # Do overlap-add operations if ova (but only if list already has >= 1 element)
        if ova and len(synthetic_sig):
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                       end_sig=synthetic_sig[-(wdw_size // 2):].copy())
            synthetic_sig = synthetic_sig[: -(wdw_size // 2)] + synthetic_sgmt
        else:
            synthetic_sig += pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag)

        if debug_flag:
            print('End of synth sig:', synthetic_sig[-20:])

    synthetic_sig = np.array(synthetic_sig)

    if debug:
    # sig_copy = synthetic_sig.copy()
    # Adjust by factor if I want to compare clearly w/ orig sig (small wdw sizes)
        print_synth_sig = np.around(synthetic_sig).astype('float64')
        (print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', print_synth_sig[(wdw_size // 2): (wdw_size // 2) + 20]) 
            if len(synthetic_sig) > 20 else 
                print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', print_synth_sig))

    # Adjust volume to original signal level
    # Multiplicative factor works for wdw_size = 4, experimenting....
    # TEMP - KEEP!!! - (4/3) works for wdw_size = 4!
    # synthetic_sig = synthetic_sig * 0.996275
    # synthetic_sig = np.around(synthetic_sig).astype('float64')

    if orig_type == 'uint8':
        # synthetic_sig = np.around(synthetic_sig).astype('int16')    # Careful: round floats, before converting to int
        # synthetic_sig = np.clip(synthetic_sig, -32768, 32767)       # Safety measure
        synthetic_sig = convert_sig_16bit_to_8bit(synthetic_sig)
        # TEMPORARY - Don't convert 8-bit PCM back, instead write to 16-bit PCM
        # Confirms that soundfile's 16-bit PCM conv, is same conv as ours
        # return synthetic_sig.astype('int16')

    # TEMP, change to just return processed sig, instead or returning it and sig_copy
    return synthetic_sig.astype(orig_type)


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
    num_comp = matrix.shape[0]


    # plt.matshow(matrix)
    # plt.colorbar()
    # plt.title(name)

    # # x_pos = np.arange(matrix.shape[0])
    # # plt.xticks(x_pos)
    
    # # y_pos = np.arange(matrix.shape[1])
    # # plt.yticks(y_pos)

    # if ylabel == 'Frequency (Hz)':
    #     # plt.ylim(top=STD_SR_HZ // 2)
    #     plt.ylim(top=8000.0)
    # else:
    #     plt.ylim(top=num_comp)


    fig, ax = plt.subplots()
    ax.title.set_text(name)
    ax.set_ylabel(ylabel)
    if ylabel == 'Frequency (Hz)':
        # Map the axis to a new correct frequency scale, something in imshow() 0 to 44100 / 2, step by window size
        _ = ax.imshow(np.log(matrix), extent=[0, num_wdws, STD_SR_HZ // 2, 0])    
        fig.tight_layout()
        # bottom, top = plt.ylim()
        # print('Bottom:', bottom, 'Top:', top)
        plt.ylim(8000.0, 0.0)   # Crop an axis (to ~double the piano frequency max)
    else:
        _ = ax.imshow(matrix, extent=[0, num_wdws, num_comp, 0])
        fig.tight_layout()
        # bottom, top = plt.ylim()
        # print('Bottom:', bottom, 'Top:', top)
        plt.ylim(num_comp, 0.0)   # Crop an axis (to ~double the piano frequency max)
    ax.set_aspect(ratio)    # Set a visually nice ratio
    plt.show()


def convert_sig_8bit_to_16bit(sig):
    sig = sig.astype('int16')
    sig = sig - 128     # Bring to range [-128, 127]
    # sig = sig / 128     # Bring to range [-1.0, 0.99] ~ [-1.0, 1.0]
    # sig = sig * 32768   # Bring to range [-32768, 32512] ~ [-32768, 32767]
    sig = sig * 256     # Bring to range [-32768, 32512] ~ [-32768, 32767], no more info loss (no div, and trunc)
    # return np.around(sig).astype('int16')   # Careful: round floats, before converting to int
    return sig          # No need to round, since int preserved

# Badly named function, actually converts from type output by DSP (float64)
def convert_sig_16bit_to_8bit(sig):
    # sig = sig.astype('int16')   # Should be ok, cause floats all have zeros after decimal (empirically), but empirically worse overall results
    sig = np.around(sig).astype('int16')    # Careful: round floats, before converting to int
    sig = np.clip(sig, -32768, 32767)         # Safety measure
    sig = sig / 256     # Bring to range [-128, 127]
    # sig = sig / 32768   # Bring to range [-1.0, 0.99] ~ [-1.0, 1.0]
    # sig = sig * 128     # Bring to range [-128, 127]
    sig = sig.astype('int16')
    sig = sig + 128     # Bring to range [0, 255]
    return sig.astype('uint8')


def write_partial_sig(sig, wdw_size, start_index, end_index, out_filepath, sig_sr):
    # To get all noise part in brahms, rule of thumb = 25 windows
    sig = sig[(start_index * wdw_size): (end_index * wdw_size)]
    wavfile.write(out_filepath, sig_sr, sig)


def reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, segment=False, write_file=False, debug=False):
    print('--Initiating Reconstruct Mode--')
    orig_sig_type = sig.dtype

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
    synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, orig_sig_type, ova=ova, debug=debug)
    
    # Delt w/ in prev method # Correct way to convert back to 8-bit PCM (unsigned -> signed)
    # if orig_sig_type == 'uint8':
    #     # Bring to range [-1, 1]
    #     signal = signal / 32768
    #     # Bring to range [0, 255]
    #     signal = signal * 128
    #     # Signed to unsigned
    #     signal = signal + 128
    #     signal = signal.astype('uint8')

    if write_file:
        # Make synthetic WAV file - defaults to original sampling rate, TODO: Does that change things?
        # Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
        # wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(orig_sig_type))
        wavfile.write(out_filepath, sig_sr, synthetic_sig)

    return synthetic_sig


def restore_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, marybv=False, noisebv=False, avgbv=False, semisuplearn='None', 
                  semisupmadeinit=False, write_file=False, debug=False, nohanbv=False, prec_noise=False, eqbv=False, incorrect_semisup=False,
                  learn_iter=MAX_LEARN_ITER, num_noisebv=0, noise_start=6, noise_stop=83, l1_penalty=0, write_noise_sig=False):
    print('--Initiating Restore Mode--')
    orig_sig_type = sig.dtype

    print('\n--Making Piano Basis Vectors--\n')

    # Temporary branch for testing:
    if nohanbv:
        basis_vectors = get_basis_vectors(wdw_size, ova=False, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise, 
                                          num_noise=num_noisebv, noise_start=noise_start, noise_stop=noise_stop)
    else:
        basis_vectors = get_basis_vectors(wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, precise_noise=prec_noise, eq=eqbv, 
                                          num_noise=num_noisebv, noise_start=noise_start, noise_stop=noise_stop)

    if 'brahms' in out_filepath:
        # Take out voice from brahms sig for now, should take it from nmf_learn from spectrogram later
        # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]
        sig = sig[WDW_NUM_AFTER_VOICE * wdw_size: -(20 * wdw_size)]     
        # Temp - 0 values cause nan matrices, TODO: Find optimal point to cut off sig

    orig_sig_len = len(sig)
    print('\n--Making Signal Spectrogram--\n')
    spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)

    if debug:
        print('Shape of Original Piano Basis Vectors W:', basis_vectors.shape)
        print('Shape of Signal Spectrogram V:', spectrogram.shape)
        
    print('\n--Learning Piano Activations--\n')
    num_components = NUM_MARY_PIANO_NOTES if marybv else NUM_PIANO_NOTES
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

    # Update: Keep, but separate the noise matrices. Use all matrices to create a single spectrogram.
    if noisebv:
        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noisebv, debug=debug)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spgm = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)
        if debug:
            print('Shape of De-noised Piano Basis Vectors W:', piano_basis_vectors.shape)
            print('Shape of De-noised Piano Activations H:', piano_activations.shape)
            print('Shape of De-pianoed Noise Basis Vectors W:', noise_basis_vectors.shape)
            print('Shape of De-pianoed Noise Activations H:', noise_activations.shape)
            print('Shape of Synthetic Signal Spectrogram V\':', synthetic_spgm.shape)
            plot_matrix(synthetic_piano_spgm, name='Synthetic Piano Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)
            plot_matrix(synthetic_noise_spgm, name='Synthetic Noise Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)
            plot_matrix(synthetic_spgm, name='Synthetic Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)
    else:
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_spgm = basis_vectors @ activations
        if debug:
            print('Shape of Piano Activations H:', activations.shape)
            print('Shape of Synthetic Signal Spectrogram V\':', synthetic_spgm.shape)
            plot_matrix(synthetic_spgm, name='Synthetic Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)

    print('\n--Making Synthetic Signal--\n')
    synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_sig_type, ova=ova, debug=debug)
    
    if noisebv:
        noise_synthetic_sig = synthetic_sig[:orig_sig_len].copy()
        # Update: Remove first half of signal (noise half)
        # Sun update for L1-Pen: write whole file in case wavfile.write does normalizing
        # synthetic_sig = synthetic_sig[orig_sig_len:]
        
        if write_file:
            # Make synthetic WAV file - Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
            # wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(orig_sig_type))
            wavfile.write(out_filepath, sig_sr, synthetic_sig)
            if write_noise_sig:
                wavfile.write(out_filepath[:-4] + 'noisepart.wav', sig_sr, noise_synthetic_sig)

        # return synthetic_sig, noise_synethetic_sig
    else:

        if write_file:
            # Make synthetic WAV file - Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
            # wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(orig_sig_type))
            wavfile.write(out_filepath, sig_sr, synthetic_sig)

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

    l1_penalty = 0 # 10 ** 19 # 10^9 = 1Bill, 12 = trill, 15 = quad, 18 = quin, 19 = max for me
    l1pen_flag = True if (l1_penalty != 0) else False

    learn_iter = 100

    num_noise_bv = 5 # 50 # 20 # 3 # 10 # 5 # 10000 is when last good # 100000 is when it gets bad, but 1000 sounds bad in tests.py

    # TODO: Use argparse library
    # Configure params
    # Mode - RECONST or RESTORE
    mode = sys.argv[1]
    out_filepath = ('/Users/quinnmc/Desktop/AudioRestore/output_restored_wav_v3/' if mode == 'RESTORE' else 
                    '/Users/quinnmc/Desktop/AudioRestore/output_reconstructed_wav/')
    # Signal - comes as a list, filepath or a length
    sig_sr = STD_SR_HZ # Initialize sr to default
    if sys.argv[2].startswith('['):
        sig = np.array([int(num) for num in sys.argv[2][1:-1].split(',')])
        out_filepath += 'my_sig'
    elif not sys.argv[2].endswith('.wav'):  # Work around for is a number
        sig = np.random.rand(int(sys.argv[2].replace(',', '')))
        out_filepath += 'rand_sig'
    else:
        sig_sr, sig = wavfile.read('/Users/quinnmc/Desktop/AudioRestore/' + sys.argv[2])
        if sig_sr != STD_SR_HZ:
            sig, sig_sr = librosa.load(sys.argv[2], sr=STD_SR_HZ)  # Upsample to 44.1kHz if necessary
        start_index = (sys.argv[2].rindex('/') + 1) if (sys.argv[2].find('/') != -1) else 0
        out_filepath += sys.argv[2][start_index: -4]
    
    # print('INPUT SIGNAL DATA TYPES:', sig.dtype)
    # print('FOR uint8:')
    # print('IS INPUT FILE POSITIVE INTS?', 'Yes' if (np.sum(np.abs(sig)) == np.sum(sig)) 
    #     else 'No -- that\'s bad')
    # print('WHAT ARE INPUT SIGNALS MIN & MAX VALS? MIN (should be >= 0):', np.min(sig), 'MAX (should be <= 255):', np.max(sig))
    
    # Debug-print/plot option
    debug_flag = True if sys.argv[3] == 'TRUE' else False
    # Window Size
    wdw_size = int(sys.argv[4]) if (len(sys.argv) == 5) else PIANO_WDW_SIZE
    # Overlap-Add is Necessary & Default
    if ova_flag:
        out_filepath += '_ova'

    if mode == 'RECONST':   # RECONSTRUCT BLOCK
        # FOR TESTING
        no_voice = False
        if no_voice:
            out_filepath += '_novoice'
        
        out_filepath += '.wav'
        reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=ova_flag, segment=no_voice, 
                          write_file=True, debug=debug_flag)
    
    else:                   # MAIN RESTORE BLOCK
        if semi_sup_learn == 'Piano':
            out_filepath += '_sslrnpiano'
            if semi_sup_made_init:
                out_filepath += '_madeinit'
        elif semi_sup_learn == 'Noise':
            out_filepath += '_sslrnnoise'
            if semi_sup_made_init:
                out_filepath += '_madeinit'
        if noisebv_flag:
            out_filepath += ('_' + str(num_noise_bv) + 'noisebv')
        if avgbv_flag:
            out_filepath += '_avgbv'
        if l1pen_flag:
            out_filepath += ('_l1pen' + str(l1_penalty))
        out_filepath += '.wav'
        restore_audio(sig, wdw_size, out_filepath, sig_sr, ova=ova_flag, marybv=marybv_flag, noisebv=noisebv_flag, num_noisebv=num_noise_bv,
                      avgbv=avgbv_flag, semisuplearn=semi_sup_learn, semisupmadeinit=semi_sup_made_init, write_file=True, learn_iter=learn_iter,
                      l1_penalty=l1_penalty, debug=debug_flag)


if __name__ == '__main__':
    main()
