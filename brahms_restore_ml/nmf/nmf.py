# restore_audio.py - Quinn Coleman - Senior Research Project / Master's Thesis 2019-20
# Restore audio using NMF. Input is audio, and restored audio file is written to cwd.

# Compared to new - this is new
# from BrahmsRestoreDSP.audio_data_processing import *
# from basis_vectors import *

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

from .basis_vectors import *
from ..audio_data_processing import *

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
NUM_SCORE_NOTES = 73
NUM_PIANO_NOTES_RANGE_HEARD = 61
SCORE_IGNORE_BOTTOM_NOTES = 12
SCORE_IGNORE_TOP_NOTES = 10
IGNORE_BOTTOM_NOTES = 15
IGNORE_TOP_NOTES = 12
NUM_MARY_PIANO_NOTES = MARY_STOP_INDEX - MARY_START_INDEX
MAX_LEARN_ITER = 100

BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
ACTIVATION_RATIO = 8.0
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008


# Functions

# # Learning optimization - for ones matrix?
# def make_row_sum_matrix(mtx, out_shape):
#     row_sums = mtx.sum(axis=1)
#     return np.repeat(row_sums, out_shape[1], axis=0)

# NMF Learning step formulas:
    # H +1 = H * ((Wt matmul (V / (W matmul H))) / (Wt matmul 1) )
    # W +1 = W * (((V / (W matmul H)) matmul Ht) / (1 matmul Ht) )

# Unused (experimental) params:
# L1-Penalize fix - only penalize H corresponding to basis vectors fixed (supervised), 
# and (probably) only when Wfixed is piano cause piano H good
# SSLRN fix - incorrect sslrn w/ "incorrect" (defined by all W & H updated), 
# is more like unsupervised NMF, which doesn't make a drastic improvement if any
# Non-mutual update fix - no difference is seen when W & H aren't updated using each other

# Unsuccessful idea - if learning any W - restrict it from learning from voice part of V
# Tried - (only let W & H access later part of V, let unsupervised technique cover remainder of V)
# Old note below, bad idea
# Have a param to specify when to NOT learn voice in our basis vectors (we shouldn't) 
# For now, no param and we just shorten the brahms sig before this call

def updateH(H, W, V, n, l1_pen=0):
    # ones = np.ones(V.shape)
    # W.T @ ones = broadcasted W.T row-sums
    WT_mult_ones = np.tile(np.sum(W.T, axis=-1)[np.newaxis].T, (1, n)) # (W.T @ ones)
    H *= ((W.T @ (V / (W @ H))) / (WT_mult_ones + l1_pen))

def updateW(W, H, V, m):
    # ones = np.ones(V.shape)
    # print('ONES SHAPE:', ones.shape, 'HT SHAPE:', H.T.shape)
    # ones @ H.T = broadcasted H.T column-sums
    ones_mult_HT = np.tile(np.sum(H.T, axis=0)[np.newaxis], (m, 1))   # (ones @ H.T)
    W *= (((V / (W @ H)) @ H.T) / ones_mult_HT)

# With any supervision W is returned unchanged. No supervision, W is made & returned
def extended_nmf(V, k, W=None, sslrn='None', split_index=0, l1_pen=0, debug=False, incorrect=False, 
        learn_iter=MAX_LEARN_ITER, mutual_update=True, pen_all=False):
    m, n = V.shape
    H = np.random.rand(k, n) + 1
    print('H VALUES:', H[0, :100])
    # ones = np.ones(V.shape)
    if debug:
        print('IN NMF, V shape:', V.shape, 'W shape:', W if (W is None) else W.shape, 'H shape:', H.shape, 'ones shape:', ones.shape)
        print('Sum of input V:', np.sum(V))
        plot_matrix(H, 'H Before Learn', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)

    if W is not None:
        if debug:
            plot_matrix(W, 'W Before Learn', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
            print('Applying L1-Penalty of', l1_pen, 'to H corres. to fixed W')
            print('Supervised Learning') if (sslrn == 'None') else print('Semi-Supervised Learning', sslrn)

        if sslrn == 'None':
            # Supervised Learning
            for _ in range(learn_iter):
                # H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
                updateH(H, W, V, n, l1_pen=l1_pen)
        # SemiSup - Looks like only use the sections of W & H, & same V & ones, in multiplications for updates to sections of W & H
        elif sslrn == 'Piano':
            # Semi-supervised Learning Piano
            for _ in range(learn_iter):
                # if pen_all:
                #     H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
                # else:
                # H[:split_index] *= ((W[:, :split_index].T @ (V / (W[:, :split_index] @ H[:split_index]))) / 
                #                         ((W[:, :split_index].T @ ones) + l1_pen))      # only penalize corr. to fixed
                # H[split_index:] *= ((W[:, split_index:].T @ (V / (W[:, split_index:] @ H[split_index:]))) / 
                #                         (W[:, split_index:].T @ ones))
                # W[:, split_index:] *= (((V / (W[:, split_index:] @ H[split_index:])) @ H[split_index:].T) / 
                #                         (ones @ H[split_index:].T))
                updateH(H[:split_index], W[:, :split_index], V, n, l1_pen=l1_pen)
                updateH(H[split_index:], W[:, split_index:], V, n)     # only penalize H corresponding to fixed
                updateW(W[:, split_index:], H[split_index:], V, m)
        else:
            # Semi-supervised Learning Noise
            if V.shape[1] > 1000:   # For Brahms recording
                # NEW
                k_voice = 10
                Vvoice = V[:, :WDW_NUM_AFTER_VOICE].copy()
                Vrest = V[:, WDW_NUM_AFTER_VOICE:].copy()
                Hvoice = np.random.rand(k_voice, WDW_NUM_AFTER_VOICE) + 1
                Wvoice = np.random.rand(m, k_voice) + 1
                # ones_voice = np.ones((m, WDW_NUM_AFTER_VOICE))
                for _ in range(learn_iter):
                    # Hvoice *= ((Wvoice.T @ (Vvoice / (Wvoice @ Hvoice))) / (Wvoice.T @ ones_voice))
                    # Wvoice *= (((Vvoice / (Wvoice @ Hvoice)) @ Hvoice.T) / (ones_voice @ Hvoice.T))
                    updateH(Hvoice, Wvoice, Vvoice, WDW_NUM_AFTER_VOICE)
                    updateW(Wvoice, Hvoice, Vvoice, m)

                Hrest = np.random.rand(k, n - WDW_NUM_AFTER_VOICE) + 1
                # ones_rest = np.ones((m, n - WDW_NUM_AFTER_VOICE))
                for _ in range(learn_iter):
                    # Hrest[split_index:] *= ((W[:, split_index:].T @ (Vrest / (W[:, split_index:] @ Hrest[split_index:]))) / 
                    #                         ((W[:, split_index:].T @ ones_rest) + l1_pen))      # only penalize corr. to fixed
                    # Hrest[:split_index] *= ((W[:, :split_index].T @ (Vrest / (W[:, :split_index] @ Hrest[:split_index]))) / 
                    #                         (W[:, :split_index].T @ ones_rest))
                    # W[:, :split_index] *= (((Vrest / (W[:, :split_index] @ Hrest[:split_index])) @ Hrest[:split_index].T) / 
                    #                         (ones_rest @ Hrest[:split_index].T))
                    updateH(Hrest[split_index:], W[:, split_index:], Vrest, n - WDW_NUM_AFTER_VOICE, l1_pen=l1_pen)  # only penalize H corr. to fixed
                    updateH(Hrest[:split_index], W[:, :split_index], Vrest, n - WDW_NUM_AFTER_VOICE)
                    updateW(W[:, :split_index], Hrest[:split_index], Vrest, m)

                Wpiano = W[:, split_index:]
                Wnoise = W[:, :split_index]
                W = np.concatenate((Wnoise, Wvoice, Wpiano), axis=-1)

                Hvoice = np.concatenate((Hvoice, np.zeros((k_voice, n - WDW_NUM_AFTER_VOICE))), axis=-1)
                Hrest = np.concatenate((np.zeros((k, WDW_NUM_AFTER_VOICE)), Hrest), axis=-1)
                
                Hpiano = Hrest[split_index:]
                Hnoise = Hrest[:split_index]
                H = np.concatenate((Hnoise, Hvoice, Hpiano))
            else:
                # OLD
                for _ in range(learn_iter):
                    # # if pen_all:
                    # #     H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
                    # # else:
                    # H[split_index:] *= ((W[:, split_index:].T @ (V / (W[:, split_index:] @ H[split_index:]))) / 
                    #                         ((W[:, split_index:].T @ ones) + l1_pen))      # only penalize corr. to fixed
                    # H[:split_index] *= ((W[:, :split_index].T @ (V / (W[:, :split_index] @ H[:split_index]))) / 
                    #                         (W[:, :split_index].T @ ones))
                    # W[:, :split_index] *= (((V / (W[:, :split_index] @ H[:split_index])) @ H[:split_index].T) / 
                    #                         (ones @ H[:split_index].T))
                    updateH(H[split_index:], W[:, split_index:], V, n, l1_pen=l1_pen)
                    updateH(H[:split_index], W[:, :split_index], V, n)
                    updateW(W[:, :split_index], H[:split_index], V, m)
    else:
        W = np.random.rand(m, k) + 1
        if debug:
            print('Unsupervised Learning')
        # Unsupervised Learning
        for _ in range(learn_iter):
            # if pen_all:
            #     H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
            # else:
            # H *= ((W.T @ (V / (W @ H))) / (W.T @ ones))
            # W *= (((V / (W @ H)) @ H.T) / (ones @ H.T))
            updateH(H, W, V, n, l1_pen=l1_pen)
            updateW(W, H, V, m)
    if debug:
        plot_matrix(W, 'W After Learn', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(H, 'H After Learn', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
    
    return W, H


def source_split_matrices(activations, basis_vectors, num_noisebv, debug=False):
    piano_basis_vectors = basis_vectors[:, num_noisebv:]
    piano_activations = activations[num_noisebv:]
    noise_basis_vectors = basis_vectors[:, :num_noisebv]
    noise_activations = activations[:num_noisebv]
    if debug:
        print('In split')
        print('De-noised Piano Basis Vectors (first):', piano_basis_vectors.shape, piano_basis_vectors[:][0])
        print('De-noised Piano Activations (first):', piano_activations.shape, piano_activations[0])
        print('Sep. Noise Basis Vectors (first):', noise_basis_vectors.shape, noise_basis_vectors[:][0])
        print('Sep. Noise Activations (first):', noise_activations.shape, noise_activations[0])
        plot_matrix(piano_basis_vectors, 'De-noised Piano Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(piano_activations, 'De-noised Piano Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
        plot_matrix(noise_basis_vectors, 'Sep. Noise Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(noise_activations, 'Sep. Noise Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
    return noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors


def make_mary_bv_test_activations(vol_factor=1):
    activations = []
    for j in range(5):
        # 8 divisions of 6 timesteps
        comp = []
        if j == 0: # lowest note
            comp = [0.0001 * vol_factor if ((2*6) <= i < (3*6)) else 0.0000 for i in range(48)]
        elif j == 2:
            comp = [0.0001 * vol_factor if (((1*6) <= i < (2*6)) or ((3*6) <= i < (4*6))) else 0.0000 for i in range(48)]
        elif j == 4:
            comp = [0.0001 * vol_factor if ((0 <= i < (1*6)) or ((4*6) <= i < (7*6))) else 0.0000 for i in range(48)]
        else:
            comp = [0.0000 for i in range(48)]
        activations.append(comp)
    return np.array(activations)


# - Don't change params, will mess up tests that could help narrow down extreme noise cutout NMF bug
# - Data Rules
#   - convert all sigs to float64 for calculations, then back to original type on return
# - Matrix Rules
#   - before NMF, in natural rows of vectors orientation
#   - in NMF, converted to orientation that makes sense for NMF W=(feats, k), H=(k, timesteps)
#   - after NMF, orient back to rows of vectors when necessary

def restore_with_nmf(sig, wdw_size, out_filepath, sig_sr, ova=True, marybv=False, noisebv=True, avgbv=True, semisuplearn='None', 
                  semisupmadeinit=False, write_file=True, debug=False, nohanbv=False, prec_noise=False, eqbv=False, incorrect_semisup=False,
                  learn_iter=MAX_LEARN_ITER, num_noisebv=10, noise_start=6, noise_stop=83, l1_penalty=0, write_noise_sig=False,
                  a430hz_bv=False, scorebv=False, audible_range_bv=False):
    orig_sig_type = sig.dtype

    print('\n--Making Piano & Noise Basis Vectors--\n')
    # Are in row orientation - natural
    basis_vectors = get_basis_vectors(wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, num_noise=num_noisebv, 
                                      precise_noise=prec_noise, eq=eqbv, noise_start=noise_start, noise_stop=noise_stop,
                                      randomize='None' if (semisupmadeinit and semisuplearn != 'None') else semisuplearn, 
                                      a430hz=a430hz_bv, score=scorebv, audible_range=audible_range_bv)
    # # Currently to avoid silly voices
    # if 'brahms' in out_filepath:
    #     if debug:
    #         print('Sig before cut (bgn,end):', sig[:10], sig[-10:])
    #     # # New, keeps ending in
    #     # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]  
    #     # Take out voice from brahms sig for now, should take it from nmf from spectrogram later
    #     # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]
    #     sig = sig[WDW_NUM_AFTER_VOICE * wdw_size: -(20 * wdw_size)]     
    #     # 0 values cause nan matrices, ?? TODO: Find optimal point to cut off sig
    #     if debug:
    #         print('Sig after cut (bgn,end):', sig[:10], sig[-10:])

    orig_sig_len = len(sig)
    print('\n--Making Brahms Spectrogram--\n')
    spectrogram, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=ova, debug=debug)

    if debug:
        print('Given Basis Vectors W (first):', basis_vectors.shape, basis_vectors[0])
        print('Given Brahms Spectrogram V (first timestep):', spectrogram.shape, spectrogram[0])
        # # TEMP
        # plot_matrix(spectrogram, 'Brahms Spectrogram', 'frequency', 'time segments', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        
    print('\nGoing into NMF--Learning Activations--\n') if semisuplearn == 'None' else print('\n--Going into NMF--Learning Activations & Basis Vectors--\n')
    k = NUM_SCORE_NOTES if scorebv else (NUM_MARY_PIANO_NOTES if marybv else NUM_PIANO_NOTES)
    if audible_range_bv:
        k -= ((SCORE_IGNORE_TOP_NOTES + SCORE_IGNORE_BOTTOM_NOTES) if scorebv else (IGNORE_TOP_NOTES + IGNORE_BOTTOM_NOTES))
    if noisebv:
        k += num_noisebv
    # Transpose W and V from natural orientation to NMF-liking orientation
    basis_vectors, spectrogram = basis_vectors.T, spectrogram.T
    basis_vectors, activations = extended_nmf(spectrogram, k, W=basis_vectors, split_index=num_noisebv, 
                                                debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
                                                l1_pen=l1_penalty, sslrn=semisuplearn)
    # spectrogram, basis_vectors = spectrogram.T, basis_vectors.T
    # activations, _ = nmf_learn(spectrogram, k, basis_vectors=basis_vectors, debug=debug, learn_iter=learn_iter, 
    #                             l1_penalty=l1_penalty)

    # if semisuplearn == 'Piano':     # Semi-Supervised Learn (learn Wpiano too)
    #     basis_vectors, activations = extended_nmf(spectrogram, k, basis_vectors=basis_vectors, learn_index=num_noisebv, 
    #                                            madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
    #                                            l1_pen=l1_penalty, sslrn=semisuplearn)

    #     # basis_vectors, activations = nmf(spectrogram, k, basis_vectors=basis_vectors, learn_index=num_noisebv, 
    #     #                                        madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
    #     #                                        l1_penalty=l1_penalty)
    # elif semisuplearn == 'Noise':   # Semi-Supervised Learn (learn Wnoise too)
    #     basis_vectors, activations = extended_nmf(spectrogram, k, basis_vectors=basis_vectors, learn_index=num_noisebv, 
    #                                            madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
    #                                            l1_penalty=l1_penalty, sslrn=semisuplearn)
    #     # basis_vectors, activations = nmf(spectrogram, k, basis_vectors=basis_vectors, learn_index=(-1 * num_noisebv), 
    #     #                                        madeinit=semisupmadeinit, debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
    #     #                                        l1_penalty=l1_penalty)
    # else:                           # Supervised Learn
    #     _, activations = extended_nmf(spectrogram, k, basis_vectors=basis_vectors, debug=debug, learn_iter=learn_iter, 
    #                                l1_penalty=l1_penalty)

    # Update: Keep, but separate the noise matrices. Use all matrices to create a single spectrogram.
    if noisebv:
        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = source_split_matrices(activations, basis_vectors, num_noisebv, debug=debug)
        print('\n--Making Synthetic Brahms Spectrogram (Source-Separating)--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations

        if debug:
            plot_matrix(synthetic_piano_spgm, 'Synthetic Piano Spectrogram (BEFORE MASKING)', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
        # New - apply filter tf mask - no difference seen
        synthetic_spgm = basis_vectors @ activations
        piano_mask = synthetic_piano_spgm / synthetic_spgm
        synthetic_piano_spgm = piano_mask * spectrogram

        # Include noise within result to battle any normalizing wavfile.write might do
        synthetic_spgm = np.concatenate((synthetic_piano_spgm, synthetic_noise_spgm), axis=-1)
        if debug:
            print('Synthetic Signal Spectrogram V\' (first):', synthetic_spgm.shape, synthetic_spgm[:][0])
            plot_matrix(synthetic_piano_spgm, 'Synthetic Piano Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
            plot_matrix(synthetic_noise_spgm, 'Synthetic Noise Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
            plot_matrix(synthetic_spgm, 'Synthetic Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
    else:
        # Unused branch
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_spgm = basis_vectors @ activations
        if debug:
            print('Shape of Piano Activations H:', activations.shape)
            print('Shape of Synthetic Signal Spectrogram V\':', synthetic_spgm.shape)
            plot_matrix(synthetic_spgm, 'Synthetic Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)

    print('\n--Making Synthetic Brahms Signal--\n')
    # Artifact of my NMF - orient spgm correctly back
    synthetic_spgm = synthetic_spgm.T
    synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_sig_type, ova=ova, debug=debug)
    if debug:
        print('Gotten synthetic sig:', synthetic_sig[:10])

    # Important - writing functionality isn't change for noise or not
    if noisebv:
        noise_synthetic_sig = synthetic_sig[orig_sig_len:]
        # # Update: Remove first half of signal (noise half)
        # # Sun update for L1-Pen: write whole file in case wavfile.write does normalizing
        # # synthetic_sig = synthetic_sig[orig_sig_len:]
        # synthetic_sig = synthetic_sig[:orig_sig_len]
        
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
