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
NUM_SCORE_NOTES = 61
SCORE_IGNORE_BOTTOM_NOTES = 15
SCORE_IGNORE_TOP_NOTES = 12
NUM_MARY_PIANO_NOTES = MARY_STOP_INDEX - MARY_START_INDEX
MAX_LEARN_ITER = 100

BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
ACTIVATION_RATIO = 8.0
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008


# Functions


# TEMP
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
                # TEMP
                plot_matrix(basis_vectors_fixed, "Fixed BV Before Learn", 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                plot_matrix(basis_vectors_learn, "Learned BV Before Learn", 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                plot_matrix(activations_for_fixed, "Activations of Fixed Before Learn", 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
                plot_matrix(activations_for_learn, "Activations of Learned Before Learn", 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
                # plot_matrix(basis_vectors_fixed, name="Fixed BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                # plot_matrix(basis_vectors_learn, name="Learned BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                # plot_matrix(activations_for_fixed, name="Activations of Fixed Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)
                # plot_matrix(activations_for_learn, name="Activations of Learned Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO)

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

                        # TEMP
                        plot_matrix(activations_for_fixed, 'Fixed Activations', 'time segments', 'k', ACTIVATION_RATIO, show=True)
                        # plot_matrix(activations_for_fixed, 'Fixed Activations', 'Components', ACTIVATION_RATIO)
                        # plot_matrix(activations_for_fixed[:11], 'Fixed Activations (Components 1-11)', 'Components', ACTIVATION_RATIO)
                        # plot_matrix(activations_for_fixed[:5], 'Fixed Activations (Components 1-5)', 'Components', ACTIVATION_RATIO)
                        # TEMP
                        plot_matrix(basis_vectors_fixed, 'Fixed Basis Vectors', 'k', 'frequency', BASIS_VECTOR_FULL_RATIO, show=True)
                        # plot_matrix(basis_vectors_fixed, 'Fixed Basis Vectors', 'Frequency (Hz)', BASIS_VECTOR_FULL_RATIO)

                        

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
                    # TEMP
                    plot_matrix(basis_vectors_fixed, "Fixed BV After Learn", 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                    plot_matrix(basis_vectors_learn, "Learned BV After Learn", 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                    plot_matrix(activations_for_fixed, "Activations of Fixed After Learn", 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
                    plot_matrix(activations_for_learn, "Activations of Learned After Learn", 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
                    # plot_matrix(basis_vectors_fixed, name="Fixed BV After Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                    # plot_matrix(basis_vectors_learn, name="Learned BV After Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO)
                    # plot_matrix(activations_for_fixed, name="Activations of Fixed After Learn", ylabel='Components', ratio=ACTIVATION_RATIO)
                    # plot_matrix(activations_for_learn, name="Activations of Learned After Learn", ylabel='Components', ratio=ACTIVATION_RATIO)

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
        # TEMP
        plot_matrix(activations, "Learned Activations", 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
        # plot_matrix(activations, name="Learned Activations", ylabel='Components', ratio=ACTIVATION_RATIO)
        print('In Learn - Shape of Learned Basis Vectors W:', basis_vectors.shape)
        # TEMP
        plot_matrix(basis_vectors, "Learned Basis Vectors", 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)

    return activations, basis_vectors









# Un-needed
# Semi-supervised NMF helper function
def partition_matrices(split_index, W, H, madeinit=False):
    if split_index > 0:     # Fixed part is left side (Wfix = noise)
        # So I don't make a memory mistake
        Wfixed = W[:, :split_index].copy()
        if madeinit:
            Wlearn = W[:, split_index:].copy()
        else:
            Wlearn = np.random.rand(W[:, split_index:].shape[0], 
                                                    W[:, split_index:].shape[1])
        Hfixed = H[:split_index, :].copy()
        Hlearn = H[split_index:, :].copy()
    
    else:                   # Fixed part is right side (Wfix = piano)
        # Modify learn index as a result of my failure to combine a flag w/ logic
        split_index *= -1
        
        Wfixed = W[:, split_index:].copy()
        if madeinit:
            Wlearn = W[:, :split_index].copy()
        else:
            Wlearn = np.random.rand(W[:, :split_index].shape[0], 
                                                 W[:, :split_index].shape[1])
        Hfixed = H[split_index:, :].copy()
        Hlearn = H[:split_index, :].copy()

        split_index *= -1
    
    return Wfixed, Wlearn, Hfixed, Hlearn

# # Learning optimization
# def make_row_sum_matrix(mtx, out_shape):
#     row_sums = mtx.sum(axis=1)
#     return np.repeat(row_sums, out_shape[1], axis=0)

# General case NMF algorithm
def nmf(input_matrix, k, basis_vectors=None, learn_index=0, madeinit=False, debug=False, incorrect=False, 
              learn_iter=MAX_LEARN_ITER, l1_penalty=0, mutual_use_update=True):
    
    # Orient the input matrix, for understandability in NMF (makes for columns (not rows) of basis vectors)
    input_matrix = input_matrix.T
    
    activations = np.random.rand(k, input_matrix.shape[1])
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
                plot_matrix(basis_vectors_fixed, name="Fixed BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                plot_matrix(basis_vectors_learn, name="Learned BV Before Learn", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                plot_matrix(activations_for_fixed, name="Activations of Fixed Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO, show=True)
                plot_matrix(activations_for_learn, name="Activations of Learned Before Learn", ylabel='Components', ratio=ACTIVATION_RATIO, show=True)

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

                        plot_matrix(activations_for_fixed, 'Fixed Activations', 'Components', ACTIVATION_RATIO, show=True)
                        # plot_matrix(activations_for_fixed[:11], 'Fixed Activations (Components 1-11)', 'Components', ACTIVATION_RATIO)
                        # plot_matrix(activations_for_fixed[:5], 'Fixed Activations (Components 1-5)', 'Components', ACTIVATION_RATIO)

                        plot_matrix(basis_vectors_fixed, 'Fixed Basis Vectors', 'Frequency', BASIS_VECTOR_FULL_RATIO, show=True)

                        

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
                    plot_matrix(basis_vectors_fixed, name="Fixed BV After Learn", ylabel='Frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                    plot_matrix(basis_vectors_learn, name="Learned BV After Learn", ylabel='Frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
                    plot_matrix(activations_for_fixed, name="Activations of Fixed After Learn", ylabel='Components', ratio=ACTIVATION_RATIO, show=True)
                    plot_matrix(activations_for_learn, name="Activations of Learned After Learn", ylabel='Components', ratio=ACTIVATION_RATIO, show=True)

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
        basis_vectors = np.random.rand(input_matrix.shape[0], k)
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
        plot_matrix(activations, name="Learned Activations", ylabel='Components', ratio=ACTIVATION_RATIO, show=True)
        print('In Learn - Shape of Learned Basis Vectors W:', basis_vectors.shape)
        plot_matrix(basis_vectors, name="Learned Basis Vectors", ylabel='Frequency (Hz)', ratio=BASIS_VECTOR_FULL_RATIO, show=True)

    # return activations, basis_vectors
    return basis_vectors, activations













# Useful functions

# NMF Learning step formulas:
    # H +1 = H * ((Wt dot (V / (W dot H))) / (Wt dot 1) )
    # W +1 = W * (((V / (W dot H)) dot Ht) / (1 dot Ht) )

# L1-Penalize fix - only do when basis vectors fixed, and (probably) only when Wfixed is piano cause piano H good

# TODO: Have a param to specify when to NOT learn voice in our basis vectors (we shouldn't) 
# For now, no param and we just shorten the brahms sig before this call

def extended_nmf(V, k, W=None, split_index=0, debug=False, incorrect=False, 
        learn_iter=MAX_LEARN_ITER, l1_pen=0, mutual_update=True, sslrn='None'):
    # Re-orient the input matrices for understandability in NMF
    V = V.T
    m, n = V.shape
    W = np.random.rand(m, k) if W is None else W.T
    H = np.random.rand(k, n)
    ones = np.ones(V.shape)
    if debug:
        print('IN NMF, V shape:', V.shape, 'W shape:', W.shape, 'H shape:', H.shape, 'ones shape:', ones.shape)
        print('Sum of input V:', np.sum(V))
        plot_matrix(W, 'W Before Learn', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(H, 'H Before Learn', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)

    if W is not None:
        if debug:
            print('Applying L1-Penalty of', l1_pen, 'to H')
            print('Supervised Learning') if (sslrn == 'None') else print('Semi-Supervised Learning', sslrn)

        if sslrn == 'None':
            # Supervised Learning
            for _ in range(learn_iter):
                H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
        elif sslrn == 'Piano':
            # Semi-supervised Learning Piano
            for _ in range(learn_iter):
                H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
                W[:, split_index:] *= (((V / (W[:, split_index:] @ H[split_index:, :])) @ H[split_index:, :].T) / 
                                        (ones @ H[split_index:, :].T))
        else:
            # Semi-supervised Learning Noise
            for _ in range(learn_iter):
                H *= ((W.T @ (V / (W @ H))) / ((W.T @ ones) + l1_pen))
                W[:, :split_index] *= (((V / (W[:, :split_index] @ H[:split_index, :])) @ H[:split_index, :].T) / 
                                        (ones @ H[:split_index, :].T))
    else:
        if debug:
            print('Unsupervised Learning')
        # Unsupervised Learning
        for _ in range(learn_iter):
            H *= ((W.T @ (V / (W @ H))) / (W.T @ ones))
            W *= (((V / (W @ H)) @ H.T) / (ones @ H.T))
    
    if debug:
        plot_matrix(W, 'W After Learn', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(H, 'H After Learn', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
    
    return W, H


def noise_split_matrices(activations, basis_vectors, num_noisebv, debug=False):
    piano_basis_vectors = basis_vectors[:, num_noisebv:].copy()
    piano_activations = activations[num_noisebv:].copy()
    noise_basis_vectors = basis_vectors[:, :num_noisebv].copy()
    noise_activations = activations[:num_noisebv].copy()
    if debug:
        print('In split')
        print('De-noised Piano Basis Vectors (first):', piano_basis_vectors[:][0])
        print('De-noised Piano Activations (first):', piano_activations[0])
        print('Sep. Noise Basis Vectors (first):', noise_basis_vectors[:][0])
        print('Sep. Noise Activations (first):', noise_activations[0])
        plot_matrix(piano_basis_vectors, 'De-noised Piano Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(piano_activations, 'De-noised Piano Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
        plot_matrix(noise_basis_vectors, 'Sep. Noise Basis Vectors', 'k', 'frequency', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        plot_matrix(noise_activations, 'Sep. Noise Activations', 'time segments', 'k', ratio=ACTIVATION_RATIO, show=True)
    return noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors


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


# - Don't change params, will mess up tests that could help narrow down extreme noise cutout NMF bug
# - Data Rules
#   - convert all sigs to float64 for calculations, then back to original type on return
# - Matrix Rules
#   - before NMF, in natural rows of vectors orientation
#   - in NMF, converted to orientation that makes sense for NMF W=(feats, k), H=(k, timesteps)
#   - after NMF, orient back to rows of vectors when necessary

def restore_with_nmf(sig, wdw_size, out_filepath, sig_sr, ova=True, marybv=False, noisebv=True, avgbv=True, semisuplearn='None', 
                  semisupmadeinit=False, write_file=True, debug=False, nohanbv=False, prec_noise=False, eqbv=False, incorrect_semisup=False,
                  learn_iter=MAX_LEARN_ITER, num_noisebv=10, noise_start=6, noise_stop=83, l1_penalty=0, write_noise_sig=False):
    orig_sig_type = sig.dtype

    print('\n--Making Piano & Noise Basis Vectors--\n')
    # Are in row orientation - natural
    basis_vectors = get_basis_vectors(wdw_size, ova=ova, mary=marybv, noise=noisebv, avg=avgbv, debug=debug, num_noise=num_noisebv, 
                                      precise_noise=prec_noise, eq=eqbv, noise_start=noise_start, noise_stop=noise_stop,
                                      randomize='None' if (semisupmadeinit and semisuplearn != 'None') else semisuplearn)
    # TEMP
    # Currently to avoid silly voices
    if 'brahms' in out_filepath:
        if debug:
            print('Sig before cut (bgn,end):', sig[:10], sig[-10:])
        # # New, keeps ending in
        # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]  
        # Take out voice from brahms sig for now, should take it from nmf from spectrogram later
        # sig = sig[WDW_NUM_AFTER_VOICE * wdw_size:]
        sig = sig[WDW_NUM_AFTER_VOICE * wdw_size: -(20 * wdw_size)]     
        # 0 values cause nan matrices, ?? TODO: Find optimal point to cut off sig
        if debug:
            print('Sig after cut (bgn,end):', sig[:10], sig[-10:])


    orig_sig_len = len(sig)
    print('\n--Making Brahms Spectrogram--\n')
    # TEMP
    # spectrogram, phases = make_spectrogram(sig, wdw_size, ova=ova, debug=debug)
    spectrogram, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=ova, debug=debug)

    if debug:
        print('Given Basis Vectors W (first):', basis_vectors.shape, basis_vectors[0])
        print('Given Brahms Spectrogram V (first timestep):', spectrogram.shape, spectrogram[0])
        # # TEMP
        # plot_matrix(spectrogram, 'Brahms Spectrogram', 'frequency', 'time segments', ratio=BASIS_VECTOR_FULL_RATIO, show=True)
        
    print('\nGoing into NMF--Learning Activations--\n') if semisuplearn == 'None' else print('\n--Going into NMF--Learning Activations & Basis Vectors--\n')
    k = NUM_MARY_PIANO_NOTES if marybv else NUM_PIANO_NOTES
    if noisebv:
        k += num_noisebv
    
    # TEMP
    # basis_vectors, activations = extended_nmf(spectrogram, k, W=basis_vectors, split_index=num_noisebv, 
    #                                             debug=debug, incorrect=incorrect_semisup, learn_iter=learn_iter,
    #                                             l1_pen=l1_penalty, sslrn=semisuplearn)
    spectrogram, basis_vectors = spectrogram.T, basis_vectors.T
    activations, _ = nmf_learn(spectrogram, k, basis_vectors=basis_vectors, debug=debug, learn_iter=learn_iter, 
                                l1_penalty=l1_penalty)

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
        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noisebv, debug=debug)
        print('\n--Making Synthetic Brahms Spectrogram (Source-Separating)--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        # TEMP
        # synthetic_spgm = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)
        # # Include noise within result to battle any normalizing wavfile.write might do
        synthetic_spgm = np.concatenate((synthetic_piano_spgm, synthetic_noise_spgm), axis=-1)
        if debug:
            print('Gotten De-noised Piano Basis Vectors W (first):', piano_basis_vectors.shape, piano_basis_vectors[:][0])
            print('Gotten De-noised Piano Activations H (first):', piano_activations.shape, piano_activations[0])
            print('Gotten De-pianoed Noise Basis Vectors W (first):', noise_basis_vectors.shape, noise_basis_vectors[:][0])
            print('Gotten De-pianoed Noise Activations H (first):', noise_activations.shape, noise_activations[0])
            print('Synthetic Signal Spectrogram V\' (first):', synthetic_spgm.shape, synthetic_spgm[:][0])
            plot_matrix(synthetic_piano_spgm, 'Synthetic Piano Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
            plot_matrix(synthetic_noise_spgm, 'Synthetic Noise Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
            plot_matrix(synthetic_spgm, 'Synthetic Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)
    else:
        # Unused
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
        noise_synthetic_sig = synthetic_sig[orig_sig_len:].copy()
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
