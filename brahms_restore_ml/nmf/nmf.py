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
from numpy.lib.type_check import real
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random

from .basis_vectors import *
from ..audio_data_processing import *

# Constants
STD_SR_HZ = 44100
MARY_SR_HZ = 16000
PIANO_WDW_SIZE = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
DEBUG_WDW_SIZE = 4
# Resolution (Windows per second) = STD_SR_HZ / PIANO_WDW_SIZE
BRAHMS_TSTEPS = 1272

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
BRAHMS_SILENCE_WDWS = 15


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

def updateH(H, W, V, n, l1_pen=0, wholeW=None, wholeH=None):
    Vpart = (V / (W @ H)) if (wholeW is None and wholeH is None) else (V / (wholeW @ wholeH))
    # W.T @ ones = broadcasted W.T row-sums
    WT_mult_ones = np.tile(np.sum(W.T, axis=-1)[np.newaxis].T, (1, n)) # replaced (W.T @ ones)
    H *= ((W.T @ Vpart) / (WT_mult_ones + l1_pen))

def updateW(W, H, V, m, wholeW=None, wholeH=None):
    Vpart = (V / (W @ H)) if (wholeW is None and wholeH is None) else (V / (wholeW @ wholeH))
    # ones @ H.T = broadcasted H.T column-sums
    ones_mult_HT = np.tile(np.sum(H.T, axis=0)[np.newaxis], (m, 1))   # replaced (ones @ H.T)
    W *= ((Vpart @ H.T) / ones_mult_HT)

# With any supervision W is returned unchanged. No supervision, W is made & returned
def extended_nmf(V, k, W=None, sslrn='None', split_index=0, l1_pen=0, debug=False, incorrect=False, 
        learn_iter=MAX_LEARN_ITER, made_init=False, madeinit_learn_factor=1): # temp useless param
    # Expmt - if madeinit: heighten/lower to try to (enocurage noise to learn)/(prevent piano from learning) noise
    # madeinit_learn_limit = round(learn_iter * madeinit_learn_factor)
    m, n = V.shape
    H = np.random.rand(k, n) + 1
    if debug:
        print('IN NMF, V shape:', V.shape, 'W shape:', W if (W is None) else W.shape, 'H shape:', H.shape) #, 'ones shape:', ones.shape)
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
                updateH(H[:split_index], W[:, :split_index], V, n, wholeW=W, wholeH=H)                  # noise H
                updateH(H[split_index:], W[:, split_index:], V, n, l1_pen=l1_pen, wholeW=W, wholeH=H)   # piano H

        # SemiSup - Looks like only use the sections of W & H, & same V & ones, in multiplications for updates to sections of W & H
        # # Tested to see if sound difference, learning voice or not, no difference heard
        # elif V.shape[1] > 1000:
        #     # Don't learn from voice part of V - assumes Brahms recording
        #     k_voice = 10
        #     real_wdw_num_after_voice = WDW_NUM_AFTER_VOICE + 80
        #     Vvoice = V[:, :real_wdw_num_after_voice].copy()
        #     Vrest = V[:, real_wdw_num_after_voice:].copy()
        #     Hvoice = np.random.rand(k_voice, real_wdw_num_after_voice) + 1
        #     Wvoice = np.random.rand(m, k_voice) + 1
        #     # Don't supply whole W or H, b/c this isn't semi-sup, this is its own approximation (voice part)
        #     for _ in range(learn_iter):
        #         updateH(Hvoice, Wvoice, Vvoice, real_wdw_num_after_voice)
        #         updateW(Wvoice, Hvoice, Vvoice, m)
        #     Hrest = np.random.rand(k, n - real_wdw_num_after_voice) + 1

        #     if sslrn == 'Piano':
        #         for i in range(learn_iter):
        #             updateH(Hrest[:split_index], W[:, :split_index], Vrest, n - real_wdw_num_after_voice, l1_pen=l1_pen, wholeW=W, wholeH=Hrest)    # only penalize H corr. to fixed
        #             updateH(Hrest[split_index:], W[:, split_index:], Vrest, n - real_wdw_num_after_voice, wholeW=W, wholeH=Hrest)        
        #             # TEMP - to find difference among semi-sup made-init (too much learn overwrites?)
        #             if not made_init or (made_init and i < madeinit_learn_limit):
        #                 updateW(W[:, split_index:], Hrest[split_index:], Vrest, m, wholeW=W, wholeH=Hrest)
        #     else:   # learn noise
        #         for i in range(learn_iter):
        #             updateH(Hrest[split_index:], W[:, split_index:], Vrest, n - real_wdw_num_after_voice, l1_pen=l1_pen, wholeW=W, wholeH=Hrest)    # only penalize H corr. to fixed
        #             updateH(Hrest[:split_index], W[:, :split_index], Vrest, n - real_wdw_num_after_voice, wholeW=W, wholeH=Hrest)
        #             # TEMP - to find difference among semi-sup made-init (too much learn overwrites?)
        #             if not made_init or (made_init and i < madeinit_learn_limit):
        #                 updateW(W[:, :split_index], Hrest[:split_index], Vrest, m, wholeW=W, wholeH=Hrest)

        #     Wpiano = W[:, split_index:]
        #     Wnoise = W[:, :split_index]
        #     W = np.concatenate((Wnoise, Wvoice, Wpiano), axis=-1)

        #     Hvoice = np.concatenate((Hvoice, np.zeros((k_voice, n - real_wdw_num_after_voice))), axis=-1)
        #     Hrest = np.concatenate((np.zeros((k, real_wdw_num_after_voice)), Hrest), axis=-1)
            
        #     Hpiano = Hrest[split_index:]
        #     Hnoise = Hrest[:split_index]
        #     H = np.concatenate((Hnoise, Hvoice, Hpiano))
        else:
            # Learn from whole V
            if sslrn == 'Piano':
                for i in range(learn_iter):
                    updateH(H[:split_index], W[:, :split_index], V, n, l1_pen=l1_pen, wholeW=W, wholeH=H)
                    updateH(H[split_index:], W[:, split_index:], V, n, wholeW=W, wholeH=H)     # only penalize H corresponding to fixed
                    # # TEMP - to find difference among semi-sup made-init (too much learn overwrites?)
                    # if not made_init or (made_init and i < madeinit_learn_limit):
                    #     updateW(W[:, split_index:], H[split_index:], V, m, wholeW=W, wholeH=H)
                    updateW(W[:, split_index:], H[split_index:], V, m, wholeW=W, wholeH=H)
            else:   # learn noise
                for i in range(learn_iter):
                    updateH(H[split_index:], W[:, split_index:], V, n, l1_pen=l1_pen, wholeW=W, wholeH=H)
                    updateH(H[:split_index], W[:, :split_index], V, n, wholeW=W, wholeH=H)
                    # TEMP - to find difference among semi-sup made-init (too much learn overwrites?)
                    # if not made_init or (made_init and i < madeinit_learn_limit):
                    updateW(W[:, :split_index], H[:split_index], V, m, wholeW=W, wholeH=H)  
                    # # TEMP force 1 update
                    # if i == 0:
                    #     updateW(W[:, :split_index], H[:split_index], V, m, wholeW=W, wholeH=H)
    else:
        W = np.random.rand(m, k) + 1
        if debug:
            print('Unsupervised Learning')
        # Unsupervised Learning
        for _ in range(learn_iter):
            # H *= ((W.T @ (V / (W @ H))) / (W.T @ ones))
            # W *= (((V / (W @ H)) @ H.T) / (ones @ H.T))
            updateH(H, W, V, n)
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
        plot_matrix(piano_basis_vectors, 'Piano Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(piano_activations, 'Piano Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
        plot_matrix(noise_basis_vectors, 'Noise Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(noise_activations, 'Noise Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
    return noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors


def make_mary_bv_test_activations(vol_factor=1):
    activations = []
    actvn_amount = 5
    note_len = 48
    zero_val = 0 # EPSILON
    # nl_factor = 20
    for j in range(5):
        # 8 divisions of 6 timesteps
        comp = []
        if j == 0: # lowest note
            comp = [actvn_amount * vol_factor if ((2*6) <= i < (3*6)) else zero_val for i in range(note_len)]
        elif j == 2:
            comp = [actvn_amount * vol_factor if (((1*6) <= i < (2*6)) or ((3*6) <= i < (4*6))) else zero_val for i in range(note_len)]
        elif j == 4:
            comp = [actvn_amount * vol_factor if ((0 <= i < (1*6)) or ((4*6) <= i < (7*6))) else zero_val for i in range(note_len)]
        else:
            comp = [zero_val for i in range(note_len)]
        activations.append(comp)
    return np.array(activations)


def pick_top_acts(H, top=2, score_range=False):
    k, n = H.shape
    topH = np.zeros((k, n))

    if score_range:
        # zero out out-of-range activations
        # H[:SCORE_IGNORE_BOTTOM_NOTES, :] = np.zeros((SCORE_IGNORE_BOTTOM_NOTES, n))
        # H[-SCORE_IGNORE_TOP_NOTES:, :] = np.zeros((SCORE_IGNORE_TOP_NOTES, n))
        # zero out bottom-of-range activations
        H[:2, :] = np.zeros((2, n))
    for i in range(n):      # iterate thru all columns (tsteps)
        Hcol = H[:, i]
        top_indices = np.argpartition(Hcol, -top)[-top:].tolist()
        for j in range(k):  # iterate thru all actvns in col
            if j in top_indices:
                topH[j, i] = Hcol[j]
    return topH

# - Don't change params, will mess up tests that could help narrow down extreme noise cutout NMF bug
# - Data Rules
#   - convert all sigs to float64 for calculations, then back to original type on return
# - Matrix Rules
#   - before NMF, in natural rows of vectors orientation
#   - in NMF, converted to orientation that makes sense for NMF W=(feats, k), H=(k, timesteps)
#   - after NMF, orient back to rows of vectors when necessary

def restore_with_nmf(sig, wdw_size, out_filepath, sig_sr, ova=True, marybv=False, noisebv=True, avgbv=True, semisuplearn='None', 
                  semisupmadeinit=False, write_file=True, debug=False, nohanbv=False, prec_noise=True, eqbv=False, incorrect_semisup=False,
                  learn_iter=MAX_LEARN_ITER, num_noisebv=10, noise_start=6, noise_stop=25, l1_penalty=0, write_noise_sig=False,
                  a430hz_bv=False, scorebv=True, audible_range_bv=False, dmged_pianobv=False, num_pbv_unlocked=None, 
                  top_acts=None, top_acts_score=False):
    orig_sig_type = sig.dtype
        
    print('\n--Making Piano & Noise Basis Vectors--\n')
    # Are in row orientation - natural
    basis_vectors = get_basis_vectors(wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, num_noise=num_noisebv, 
                                      precise_noise=prec_noise, eq=eqbv, noise_start=noise_start, noise_stop=noise_stop,
                                      randomize='None' if (semisupmadeinit and semisuplearn != 'None') else semisuplearn, 
                                      a430hz=a430hz_bv, score=scorebv, audible_range=audible_range_bv, 
                                      dmged_piano=dmged_pianobv, unlocked_piano_count=num_pbv_unlocked)
    if dmged_pianobv:
        # # Supervised case, include noise
        # hq_piano_basis_vectors = get_basis_vectors(wdw_size, ova=ova, 
        #                                            noise=noisebv,
        #                                            avg=avgbv, debug=debug, 
        #                                            num_noise=num_noisebv, randomize='None' if (semisupmadeinit and semisuplearn != 'None') else semisuplearn,
        #                                            a430hz=a430hz_bv, score=scorebv,
        #                                            audible_range=audible_range_bv)
        # No more masking -> only piano now
        hq_piano_basis_vectors = get_basis_vectors(wdw_size, ova=ova, 
                                                   noise=False, # change
                                                   avg=avgbv, debug=debug, 
                                                   num_noise=num_noisebv, randomize='None' if (semisupmadeinit and semisuplearn != 'None') else semisuplearn,
                                                   a430hz=a430hz_bv, score=scorebv,
                                                   audible_range=audible_range_bv)

    # # Currently to avoid silly voices
    # if 'brahms' in out_filepath:
    #     if debug:
    #         print('Sig before cut (bgn,end):', sig[:10], sig[-10:])
    #     # # New, keeps ending in
    #     # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]  
    #     # Take out voice from brahms sig for now, should take it from nmf from spectrogram later
    #     # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]
    #     sig = sig[WDW_NUM_AFTER_VOICE * wdw_size: -(20 * wdw_size)]     
    #     # 0 values cause nan matrices, ?? Found optimal point to cut off sig (WDW_NUM_AFTER_VOICE + 80)
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
    k = (num_pbv_unlocked if (num_pbv_unlocked is not None) else 
        (NUM_SCORE_NOTES if scorebv else (NUM_MARY_PIANO_NOTES if marybv else NUM_PIANO_NOTES)))
    if audible_range_bv:
        k -= ((SCORE_IGNORE_TOP_NOTES + SCORE_IGNORE_BOTTOM_NOTES) if scorebv else (IGNORE_TOP_NOTES + IGNORE_BOTTOM_NOTES))
    num_pianobv = k
    # basis_vectors_save = basis_vectors.T.copy()   # for debugging after NMF
    if noisebv:
        k += num_noisebv
    # Transpose W and V from natural orientation to NMF-liking orientation
    basis_vectors, spectrogram = basis_vectors.T, spectrogram.T
    basis_vectors, activations = extended_nmf(spectrogram, k, W=basis_vectors, split_index=num_noisebv, 
                                                debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
                                                l1_pen=l1_penalty, sslrn=semisuplearn, made_init=semisupmadeinit)

    # Update: Keep, but separate the noise matrices. Use all matrices to create a single spectrogram.
    if noisebv:
        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = source_split_matrices(activations, basis_vectors, num_noisebv, debug=debug)
        print('\n--Making Synthetic Brahms Spectrogram (Source-Separating)--\n')
        if top_acts is not None:
            piano_activations = pick_top_acts(piano_activations, top=top_acts, score_range=top_acts_score)
            plot_matrix(piano_activations, 
                str(top_acts) + ' Highest-Value Piano Activations' + ('(Except Bottom Notes)' if top_acts_score else ''), 
                'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)

        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations

        # FOR L1-penalty - show the average # non-zero activations per timestep - piano
        max_notes, avg = None, 0 
        rand_activation = random.randint(0, piano_activations.shape[1] - 1)
        for i in range(piano_activations.shape[1]):
            # avg += np.count_nonzero(activations[:, i])
            # Callobrated at 0.1 for supervised l1-pen=75,000,000 - barely any notes heard (0.3 per timestep)
            # Callobrated at 0.15 for semi-supervised-learn-noise
            notes_per_thresh = 0.2
            t_step_num_notes = len([0 for j in range(piano_activations.shape[0]) if piano_activations[:, i][j] > notes_per_thresh])
            avg += t_step_num_notes
            if max_notes is None or (t_step_num_notes > max_notes):
                max_notes = t_step_num_notes
        avg = avg / piano_activations.shape[1]
        print('\nAVERAGE # NON-ZERO PIANO ACTIVATIONS PER TIMESTEP:', avg)
        print('MAX NON-ZERO PIANO ACTIVATIONs IN A TIMESTEP:', max_notes)
        print('PIANO ACTIVATIONS AT A TIME STEP:', piano_activations[:, rand_activation], '\n')

        # # FOR L1-penalty - show the average # non-zero activations per timestep - noise
        # max_notes, avg = None, 0 
        # rand_activation = random.randint(0, noise_activations.shape[1] - 1)
        # for i in range(noise_activations.shape[1]):
        #     # avg += np.count_nonzero(activations[:, i])
        #     # Callobrated at 0.1 for supervised l1-pen=75,000,000 - barely any notes heard (0.3 per timestep)
        #     # Callobrated at 0.15 for semi-supervised-learn-noise
        #     notes_per_thresh = 0 # 0.2
        #     t_step_num_notes = len([0 for j in range(noise_activations.shape[0]) if piano_activations[:, i][j] > notes_per_thresh])
        #     avg += t_step_num_notes
        #     if max_notes is None or (t_step_num_notes > max_notes):
        #         max_notes = t_step_num_notes
        # avg = avg / piano_activations.shape[1]
        # print('\nAVERAGE # NON-ZERO PIANO ACTIVATIONS PER TIMESTEP:', avg)
        # print('MAX NON-ZERO PIANO ACTIVATIONs IN A TIMESTEP:', max_notes)
        # print('PIANO ACTIVATIONS AT A TIME STEP:', piano_activations[:, rand_activation], '\n')

        if dmged_pianobv:   # For dmged piano W in learning, masked spectrogram needs to be made from high-qual piano
            hq_piano_basis_vectors = hq_piano_basis_vectors.T
            if semisuplearn != 'None' and spectrogram.shape[1] > 1000:
                if debug:
                    plot_matrix(hq_piano_basis_vectors, 'High-Quality Piano BVs (BEFORE)', 'k', 'frequency', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)
                    # print('W in NMF shape:', basis_vectors.shape, 'HQ Piano W shape:', hq_piano_basis_vectors.shape)
                    # print('Noise & Voice W shape:', basis_vectors[:, :(-1 * num_pianobv)].shape, 'HQ Piano W shape:', hq_piano_basis_vectors[:, (-1 * num_pianobv):].shape)
                # hq_piano_basis_vectors = np.concatenate((basis_vectors[:, :(-1 * num_pianobv)], hq_piano_basis_vectors[:, (-1 * num_pianobv):]), axis=-1)
                # no more masking -> only retrieve non-noise part = voice (if applicable) + piano
                    print('HQ PIANO BVs - SHAPE OF VOICE PART:', basis_vectors[:, num_noisebv:(-1 * num_pianobv)].shape)
                hq_piano_basis_vectors = np.concatenate((basis_vectors[:, num_noisebv:(-1 * num_pianobv)], hq_piano_basis_vectors[:, (-1 * num_pianobv):]), axis=-1)
                if debug:
                    plot_matrix(hq_piano_basis_vectors, 'High-Quality Piano BVs (AFTER)', 'k', 'frequency', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)
                    plot_matrix(activations, 'All Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
            # spectrogram = hq_piano_basis_vectors @ activations
            # no more masking -> overwrite synthetic piano spgm w/ hq piano
            synthetic_piano_spgm = hq_piano_basis_vectors @ piano_activations

        # if debug:
        #     plot_matrix(synthetic_piano_spgm, 'Synthetic Piano Spectrogram (BEFORE MASKING)', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
        # # In order to incorporate hq piano basis vectors (manipulate the spectrogram) - get rid of tf-masking??
        # # Apply filter tf soft mask (no difference seen from before)
        # synthetic_spgm = basis_vectors @ activations
        # piano_mask = synthetic_piano_spgm / synthetic_spgm
        # synthetic_piano_spgm = piano_mask * spectrogram
        # # synthetic_piano_spgm = piano_mask * synthetic_piano_spgm    # mask piano source # doesn't make sense
        # noise_mask = synthetic_noise_spgm / synthetic_spgm
        # synthetic_noise_spgm = noise_mask * spectrogram
        # # synthetic_noise_spgm = noise_mask * synthetic_noise_spgm    # mask noise source # doesn't make sense     

        # # FOR EVAL - SPECTROGRAM PLOTS - not during testing
        # if not write_noise_sig:
        #     restore_plot_path = os.getcwd() + '/brahms_restore_ml/nmf/eval_spgm_plots/'
        #     eval_name, plot_name = 'sup_a436hz', 'Supervised NMF, Piano Basis Vectors A4 = 436Hz, 2 NBVs'
        #     plot_matrix(synthetic_piano_spgm[:, BRAHMS_SILENCE_WDWS:-BRAHMS_SILENCE_WDWS], name=plot_name, 
        #         xlabel='time (4096-sample windows)', ylabel='frequency', plot_path=(restore_plot_path + eval_name + '.png'), show=False)

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
            else:   # FOR EVAL - WAV SAMPLES - not during testing
                eval_smpl_path = os.getcwd() + '/brahms_restore_ml/nmf/eval_wav_smpls/'
                piano_synthetic_sig = synthetic_sig[:orig_sig_len]
                eval_start = len(piano_synthetic_sig) // 4
                wavfile.write(eval_smpl_path + eval_name + '.wav', sig_sr, piano_synthetic_sig[eval_start: (eval_start + 500000)])

        # return synthetic_sig, noise_synethetic_sig
    else:

        if write_file:
            # Make synthetic WAV file - Important: signal elems to types of original signal (uint8 for brahms) or else MUCH LOUDER
            # wavfile.write(out_filepath, sig_sr, synthetic_sig.astype(orig_sig_type))
            wavfile.write(out_filepath, sig_sr, synthetic_sig)

    return synthetic_sig
