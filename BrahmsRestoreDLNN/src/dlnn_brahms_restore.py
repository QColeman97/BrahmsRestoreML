# Quinn Coleman - B.M.S. Thesis Brahms Restoration
# Advisor: Dr. Dennis Sun
# 8/31/20
# dlnn_brahms_restore - neural network to restore brahms recording
# Custom training loop version

# DATA RULES #
# - If writing a transformed signal, write it back using its original data type/range (wavfile lib)
# - Convert signals into float64 for processing (numpy default, no GPUs usit ed) (in make_spgm() do a check)
# - Convert data fed into NN into float32 (GPUs like it)
# - No functionality to train on 8-bit PCM signal (unsigned) b/c of rare case
##############

from scipy.io import wavfile
# import scipy.signal as sg
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Lambda, TimeDistributed, Layer, LSTM, Bidirectional, BatchNormalization, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# TF-NIGHTLY
# from tensorflow.keras import mixed_precision
import numpy as np
import datetime
import math
import random
import json
import os
import sys
import re
from copy import deepcopy


# TODO: See if both processes allow mem growth, is it faster or does GPU just work less hard?
#       if faster, keep both proc using mem growth
# For run-out-of-memory error
# gpus = tf.config.experimental.list_physical_devices('GPU')
# gpus = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(gpus))
# print("GPUs Available: ", gpus)

# mirrored_strategy = tf.distribute.MirroredStrategy()
# print("Num GPUs Available (according to mirrored strategy): ", mirrored_strategy.num_replicas_in_sync, "\n")

# Only use for narrowing down NaN bug to exploding gradient
# tf.debugging.enable_check_numerics()

# TEST - 2 GS's at same time? SUCCESS!!!
# BUT, set_memory_growth has perf disadvantages (slower) - give main GS full power
# GPU Mem as func of HP test
# for i in range(len(gpus)):
#     tf.config.experimental.set_memory_growth(gpus[i], True)

# # policy = None
# # MIXED PRECISION - only used on f35 (V100s)
# policy = mixed_precision.Policy('mixed_float16')
# Moved this to into logic, for non-PC run
# mixed_precision.set_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     # tf.config.experimental.set_virtual_device_configuration(
#     #     gpus[0],
#     #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#     # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
# np.set_printoptions(precision=3)    # Change if needed

# TEST FLOAT16 - FAILED
# tf.keras.backend.set_floatx('float16')

# CONSTANTS:
STD_SR_HZ = 44100
PIANO_WDW_SIZE = 4096
# Resolution (Windows per second) = STD_SR_HZ / PIANO_WDW_SIZE
SPGM_BRAHMS_RATIO = 0.08


# TEST - call() input for imp model
# global_phases1, global_phases2, global_phases3 = None, None, None

# DSP FUNCTIONS:
def plot_matrix(matrix, name, ylabel, ratio=0.08):
    matrix = matrix.T   # For this model
    num_wdws = matrix.shape[1]
    num_comp = matrix.shape[0]

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
    # plt.show()
    plt.savefig('../' + name + '.png')

# SIGNAL -> SPECTROGRAM
def signal_to_pos_fft(sgmt, wdw_size, ova=False, debug_flag=False):
    if len(sgmt) != wdw_size:
        deficit = wdw_size - len(sgmt)
        sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

    if debug_flag:
        print('Original segment (len =', len(sgmt), '):\n', sgmt[:5])

    if ova: # Perform lobing on ends of segment
        sgmt *= np.hanning(wdw_size)
    # pos_phases_fft = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1].copy()
    # pos_mag_fft = np.angle(np.fft.fft(sgmt))[: (wdw_size // 2) + 1].copy()
    
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


def make_spectrogram(signal, wdw_size, epsilon, ova=False, debug=False):
    # Pre-processing steps specific for Brahms (not training data)
    # If 8-bit PCM, convert to 16-bit PCM (signed to unsigned)
    if signal.dtype == 'uint8':
        signal = convert_sig_8bit_to_16bit(signal).astype('float64')
    if isinstance(signal[0], np.ndarray):   # Stereo signal = 2 channels
        # sig = np.array([((x[0] + x[1]) / 2) for x in signal.astype('float32')]) # float64
        signal = np.average(signal, axis=-1)
    # else:                                   # Mono signal = 1 channel    
    #     sig = np.array(signal).astype('float32')    # float64 - too big, lower performance

    # Data Granularity Check
    if signal.dtype != 'float64':
        signal = signal.astype('float64')
    num_spls = len(signal)
    # print('Len in makespgm:', num_spls)
    if debug:
        pass
    #    print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', signal[(wdw_size // 2): (wdw_size // 2) + 20]) if num_spls > 20 else print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', signal)

    # Hop size is half-length of window if OVA, else it's just window length (if length sufficient)
    hop_size = (wdw_size // 2) if (ova and num_spls >= (wdw_size + (wdw_size // 2))) else wdw_size
    # Number of segments depends on if OVA implemented
    num_sgmts = (math.ceil(num_spls / (wdw_size // 2)) - 1) if ova else math.ceil(num_spls / wdw_size)
    sgmt_len = (wdw_size // 2) + 1

    if debug:
        print('Num of Samples:', num_spls)
        print('Hop size:', hop_size)
        print('Num segments:', num_sgmts)
    
    # spectrogram, pos_phases = [], []
    spectrogram, pos_phases = np.empty((num_sgmts, sgmt_len)), np.empty((num_sgmts, sgmt_len))
    for i in range(num_sgmts):
        # Slicing a numpy array makes a view, so explicit copy
        sgmt = signal[i * hop_size: (i * hop_size) + wdw_size].copy()
        
        debug_flag = ((i == 0) or (i == 1)) if debug else False
        pos_mag_fft, pos_phases_fft = signal_to_pos_fft(sgmt, wdw_size, ova=ova, debug_flag=debug_flag)

        spectrogram[i] = pos_mag_fft
        pos_phases[i] = pos_phases_fft
        # spectrogram.append(pos_mag_fft)
        # pos_phases.append(pos_phases_fft)
    
    # Replace NaNs and 0s w/ epsilon
    spectrogram, pos_phases = np.nan_to_num(spectrogram), np.nan_to_num(pos_phases)
    spectrogram[spectrogram == 0], pos_phases[pos_phases == 0] = epsilon, epsilon

    # Safety measure to avoid overflow
    # TEST FLOAT16
    # MIXED PRECISION - hail mary try
    spectrogram = np.clip(spectrogram, np.finfo('float32').min, np.finfo('float32').max)
    # Spectrogram matrix w/ correct orientation (orig orient.)
    spectrogram = spectrogram.astype('float32')     # T Needed? (don't think so, only for plotting)
    #if debug:
        #plot_matrix(spectrogram, name='Built Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)

    return spectrogram, pos_phases

# SPECTROGRAM -> SIGNAL
# Returns real signal, given positive magnitude & phases of a DFT
def pos_fft_to_signal(pos_mag_fft, pos_phases_fft, wdw_size, ova=False, 
                      end_sig=None, debug_flag=False):
    # Append the mirrors of the synthetic magnitudes and phases to themselves
    neg_mag_fft = np.flip(pos_mag_fft[1: wdw_size // 2], 0)
    mag_fft = np.concatenate((pos_mag_fft, neg_mag_fft))

    # The mirror is negative b/c phases are flipped in sign (angle), but not for magnitudes
    neg_phases_fft = np.flip(pos_phases_fft[1: wdw_size // 2] * -1, 0)
    phases_fft = np.concatenate((pos_phases_fft, neg_phases_fft))

    # Multiply this magnitude fft w/ phases fft
    fft = mag_fft * np.exp(1j*phases_fft)
    # Do ifft on the fft -> waveform
    ifft = np.fft.ifft(fft)
    synthetic_sgmt = ifft.real

    if debug_flag:
        imaginaries = ifft.imag
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
            end_sig = np.zeros((wdw_size // 2))

        end_sum = ova_sgmt + end_sig    # Numpy element-wise addition of OVA parts
        synthetic_sgmt = np.concatenate((end_sum, end_sgmt))    # Concatenate OVA part with trailing end part

        if debug_flag:
            print('ova_sgmt (len =', len(ova_sgmt), '):\n', ova_sgmt[-10:], 
                  '\nend_sgmt (len =', len(end_sgmt), '):\n', end_sgmt[-10:], 
                  '\nend_sig (len =', len(end_sig), '):\n', end_sig[-10:], 
                  '\nend_sum (len =', len(end_sum), '):\n', end_sum[-10:])

    return synthetic_sgmt


# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_type, ova=False, debug=False):
    # Post-processing step from NN
    synthetic_spgm = synthetic_spgm.astype('float64')

    num_sgmts = synthetic_spgm.shape[0]#[1]
    # print('Num sgmts:', num_sgmts)
    # If both noise and piano in spgm, reuse phases in synthesis
    if num_sgmts != len(phases):   
        # phases += phases
        phases = np.concatenate((phases, phases))
    # synthetic_spgm = synthetic_spgm.T     # Get spectrogram back into orientation we did calculations on
    
    # synthetic_sig = []    # SLOW way - uses list
    synthetic_sig_len = int(((num_sgmts / 2) + 0.5) * wdw_size) if ova else num_sgmts * wdw_size
    # print('Synthetic Sig Len FULL (wdw_sizes):', synthetic_sig_len / wdw_size)
    # print('Synthetic Sig Len FULL:', synthetic_sig_len)
    # print('Putting', num_sgmts, 'sgmts into signal')
    synthetic_sig = np.empty((synthetic_sig_len))     # RAM too much use way
    # print('Synth sig mem location:', aid(synthetic_sig))
    # synthetic_sig = None
    for i in range(num_sgmts):
        ova_index = i * (wdw_size // 2)
        debug_flag = (i == 0 or i == 1) if debug else False

        # Do overlap-add operations if ova (but only if list already has >= 1 element)
        # if ova and len(synthetic_sig):
        if ova and (i > 0):
            # end_half_sgmt = synthetic_sig[-(wdw_size // 2):].copy()
            # end_half_sgmt = synthetic_sig[(i*wdw_size) - (wdw_size//2): i * wdw_size].copy()
            end_half_sgmt = synthetic_sig[ova_index: ova_index + (wdw_size//2)].copy()
            
            # print(synthetic_sig_len, '=?', i * wdw_size)
            # print('End Half Sgmt Len:', len(end_half_sgmt))
            # print('End Half Sgmt mem location:', aid(end_half_sgmt))
            
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                       end_sig=end_half_sgmt)
            # synthetic_sig = synthetic_sig[: -(wdw_size // 2)] + synthetic_sgmt
            
            
            # synthetic_sig[(i*wdw_size) - (wdw_size//2): ((i+1)*wdw_size) - (wdw_size//2)] = synthetic_sgmt
            
            
            # np.put(synthetic_sig, 
                #    range((i*wdw_size) - (wdw_size//2), ((i+1)*wdw_size) - (wdw_size//2)), 
                #    synthetic_sgmt)

            # synthetic_sgmt = np.concatenate((np.zeros((len(synthetic_sig) - (wdw_size//2))), synthetic_sgmt))
            # synthetic_sig = np.concatenate((synthetic_sig, np.zeros((wdw_size//2))))
            # synthetic_sig = synthetic_sig + synthetic_sgmt 
        else:
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                   wdw_size=wdw_size, ova=ova, debug_flag=debug_flag)
            # synthetic_sig += synthetic_sgmt
            # synthetic_sig[i * wdw_size: (i+1) * wdw_size] = synthetic_sgmt
        
        synthetic_sig[ova_index: ova_index + wdw_size] = synthetic_sgmt
            
            # np.put(synthetic_sig, range(i * wdw_size, (i+1) * wdw_size), synthetic_sgmt)

            # synthetic_sig = synthetic_sgmt if (i == 0) else np.concatenate((synthetic_sig, synthetic_sgmt))

        # print('Added sgmt', i+1)
        # print('Synth Sig Len (wdw_sizes):', ((i+1) / 2) + 0.5)
        # print('Synth sig mem location:', aid(synthetic_sig))


        if debug_flag:
            print('End of synth sig:', synthetic_sig[-20:])

    # synthetic_sig = np.array(synthetic_sig)

    if debug:
    # sig_copy = synthetic_sig.copy()
    # Adjust by factor if I want to compare clearly w/ orig sig (small wdw sizes)
        # print_synth_sig = np.around(synthetic_sig).astype('float32')
        (print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', synthetic_sig[(wdw_size // 2): (wdw_size // 2) + 20]) 
            if len(synthetic_sig) > 20 else 
                print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', synthetic_sig))

    if (orig_type == 'uint8'):  # Handle 8-bit PCM (unsigned)
        # Safety measure: prevent overflow
        synthetic_sig = np.clip(synthetic_sig, np.iinfo('int16').min, np.iinfo('int16').max)
        synthetic_sig = np.around(synthetic_sig).astype('int16')
        # Accuracy measure: round floats before converting to int    
        synthetic_sig = convert_sig_16bit_to_8bit(synthetic_sig)
    else:
        # Safety measure: prevent overflow
        synthetic_sig = (np.clip(synthetic_sig, np.finfo(orig_type).min, np.finfo(orig_type).max) 
                if orig_type == 'float32' else 
            np.clip(synthetic_sig, np.iinfo(orig_type).min, np.iinfo(orig_type).max))  
        # Accuracy measure: round floats before converting to original type
        synthetic_sig = np.around(synthetic_sig).astype(orig_type)

    return synthetic_sig


# SIGNAL CONVERSION FUNCTIONS

def convert_sig_8bit_to_16bit(sig):
    sig = sig.astype('int16')
    sig = sig - 128     # Bring to range [-128, 127]
    # sig = sig / 128     # Bring to range [-1.0, 0.99] ~ [-1.0, 1.0]
    # sig = sig * 32768   # Bring to range [-32768, 32512] ~ [-32768, 32767]
    sig = sig * 256     # Bring to range [-32768, 32512] ~ [-32768, 32767], no more info loss (no div, and trunc)
    return sig          # No need to round, since int preserved

# Badly named function, actually converts from type output by DSP (float64)
def convert_sig_16bit_to_8bit(sig):
    sig = sig / 256     # Bring to range [-128, 127]
    # sig = sig / 32768   # Bring to range [-1.0, 0.99] ~ [-1.0, 1.0]
    # sig = sig * 128     # Bring to range [-128, 127]
    # sig = sig.astype('int16')     # Prob not necessary, b/c uint8 will truncate it eventually
    sig = sig + 128     # Bring to range [0, 255]
    return sig.astype('uint8')



## NEURAL NETWORK DATA GENERATOR
# But not a dataset
# class RestoreDataSequence(Sequence):
#     def __init__(self, x_files, y1_files, y2_files, batch_size):
#         self.x_files, self.y1_files, self.y2_files = x_files, y1_files, y2_files
#         self.batch_size = batch_size

#     def __len__(self):
#         return math.ceil(len(self.x_files) / self.batch_size)

#     def __getitem__(self, index):
#         batch_x = self.x_files[index * self.batch_size: (index + 1) * self.batch_size]
#         batch_y1 = self.y1_files[index * self.batch_size: (index + 1) * self.batch_size]
#         batch_y2 = self.y2_files[index * self.batch_size: (index + 1) * self.batch_size]
#         return 

# class ArtificialDataset(tf.data.Dataset):
#     def _generator(self, num_samples):
#         # Opening the file
#         # time.sleep(0.03)

#         for sample_idx in range(num_samples):
#             # Reading data (line, record) from the file
#             # time.sleep(0.015)

#             yield (sample_idx,)

#     def __new__(cls, num_samples=3):
#         # return tf.data.Dataset.from_generator(
#         #     cls._generator,
#         #     output_types=tf.dtypes.int64,
#         #     output_shapes=(1,),
#         #     args=(num_samples,)
#         # )
#         return tf.data.Dataset.from_generator(
#         cls._generator, 
#         output_types=({'piano_noise_mixed': tf.float32, 'piano_true': tf.float32, 'noise_true': tf.float32}, 
#                       {'piano_pred': tf.float32, 'noise_pred': tf.float32})
#         )

# In order for this generator (unique output) to work w/ fit & cust training - batch it
def fixed_data_generator(x_files, y1_files, y2_files, num_samples, batch_size, num_seq, num_feat, pc_run, 
                         dmged_piano_artificial_noise=False, pad_len=-1, wdw_size=4096, epsilon=10 ** (-10)):
    while True: # Loop forever so the generator never terminates
        # for i in range(num_samples):
        for offset in range(0, num_samples, batch_size):
            x_batch_labels = x_files[offset:offset+batch_size]
            y1_batch_labels = y1_files[offset:offset+batch_size]
            y2_batch_labels = y2_files[offset:offset+batch_size]
            if (num_samples / batch_size == 0):
                # TEST FLOAT16
                # MIXED PRECISION - hail mary try
                actual_batch_size = batch_size
                x, y1, y2 = (np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'))
            else:
                actual_batch_size = len(x_batch_labels)
                x, y1, y2 = (np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'))
            
            for i in range(actual_batch_size):
                pn_filepath = x_batch_labels[i]
                pl_filepath = y1_batch_labels[i]
                nl_filepath = y2_batch_labels[i]
            # pn_filepath = x_files[i]
            # pl_filepath = y1_files[i]
            # nl_filepath = y2_files[i]
                
                # if dmged_piano_artificial_noise:
                #     # Get number from filename
                #     file_num_str = list(re.findall(r'\d+', pl_filepath))[-1]

                #     pn_sr, noise_piano_sig = wavfile.read(pn_filepath)
                #     noise_piano_sig = noise_piano_sig.astype('float64')
                #     pl_sr, piano_label_sig = wavfile.read(pl_filepath)
                #     nl_sr, noise_label_sig = wavfile.read(nl_filepath)
                #     pl_orig_type, nl_orig_type = piano_label_sig.dtype, noise_label_sig.dtype 
                #     piano_label_sig, noise_label_sig = piano_label_sig.astype('float64'), noise_label_sig.astype('float64')
                #     assert len(noise_piano_sig) == len(noise_label_sig) == len(piano_label_sig)   
                #     # assert len(noise_label_sig) == len(piano_label_sig)  
                #     # Stereo audio safety check
                #     if isinstance(noise_piano_sig[0], np.ndarray):   # Stereo signal = 2 channels
                #         # piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float32')]).astype(p_type)
                #         noise_piano_sig = np.average(noise_piano_sig, axis=-1)
                #     if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                #         # piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float32')]).astype(p_type)
                #         piano_label_sig = np.average(piano_label_sig, axis=-1)
                #     if isinstance(noise_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                #         # noise_label_sig = np.array([((x[0] + x[1]) / 2) for x in noise_label_sig.astype('float32')]).astype(n_type)
                #         noise_label_sig = np.average(noise_label_sig, axis=-1)

                #     # if i == 0:
                #     #     print('Filenames:', pl_filepath, nl_filepath)
                #     #     print('Piano Sig:', piano_label_sig[1000:1010], 'type:', piano_label_sig.dtype)
                #     #     print('Noise Sig:', noise_label_sig[1000:1010], 'type:', noise_label_sig.dtype)

                #     # NEW - create features w/ signal data augmentation
                #     # 1) Do amplitude variation aug on each source
                #     # DEBUG
                #     # piano_amp_factor = random.uniform(src_amp_low, src_amp_high)
                #     # noise_amp_factor = random.uniform(src_amp_low, src_amp_high)
                #     # piano_label_sig *= piano_amp_factor
                #     # Done for the debug wavfile.write()
                #     # piano_label_sig = np.clip(piano_label_sig, 
                #     #                           np.iinfo(pl_orig_type).min, 
                #     #                           np.iinfo(pl_orig_type).max)
                #     # piano_label_sig = np.around(piano_label_sig).astype(pl_orig_type)

                #     # noise_label_sig *= noise_amp_factor
                #     # noise_label_sig = np.clip(noise_label_sig, 
                #     #                           np.iinfo(nl_orig_type).min, 
                #     #                           np.iinfo(nl_orig_type).max)
                #     # noise_label_sig = np.around(noise_label_sig).astype(nl_orig_type)
                    
                #     # assert len(noise_label_sig) == len(piano_label_sig)  

                #     # if offset == 0 and i == 0:
                #     #     wavfile.write('/content/drive/My Drive/Quinn Coleman - Thesis/DLNN_Data/output/noise_beforepert.wav', 
                #     #                   nl_sr, noise_label_sig)

                #     # print('LEN:', len(noise_label_sig))
                #     # 2) TODO Do frequency perturbation aug on noise source
                #     # noise_label_sig = freq_perturb(noise_label_sig, wdw_size, epsilon)

                #     # assert len(noise_label_sig) == len(piano_label_sig)  

                #     # # if offset == 0 and i == 0:
                #     # #     wavfile.write('/content/drive/My Drive/Quinn Coleman - Thesis/DLNN_Data/output/noise_afterpert.wav', 
                #     # #                   nl_sr, noise_label_sig)
                #     # # 3) Mix - NOTE sigs must be same dtype
                #     # avg_src_sum = (np.sum(piano_label_sig) + np.sum(noise_label_sig)) / 2
                #     # src_percent_1 = random.randrange(int((src_amp_low*100) // 2), int((src_amp_high*100) // 2)) / 100
                #     # src_percent_2 = 1 - src_percent_1
                #     # piano_src_is_1 = bool(random.getrandbits(1))
                #     # if piano_src_is_1:
                #     #     piano_label_sig *= src_percent_1
                #     #     noise_label_sig *= src_percent_2
                #     # else:
                #     #     piano_label_sig *= src_percent_2
                #     #     noise_label_sig *= src_percent_1

                #     # noise_piano_sig = piano_label_sig + noise_label_sig
                #     # noise_piano_sig *= (avg_src_sum / np.sum(noise_piano_sig))             
                #     # # piano_label_sig = piano_label_sig.astype('float64')
                #     # # noise_label_sig = noise_label_sig.astype('float64')
                #     # # noise_piano_sig = noise_piano_sig.astype('float64')

                #     # Preprocessing - length pad, transform into spectrograms, add eps
                #     # if isinstance(noise_piano_sig[0], np.ndarray):   # Stereo signal = 2 channels
                #     #     noise_piano_sig = np.array([((x[0] + x[1]) / 2) for x in noise_piano_sig.astype('float64')])
                #     # if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                #     #     piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float64')])
                    
                #     deficit = pad_len - len(noise_piano_sig)
                #     noise_piano_sig = np.pad(noise_piano_sig, (0,deficit))
                #     piano_label_sig = np.pad(piano_label_sig, (0,deficit))
                #     noise_label_sig = np.pad(noise_label_sig, (0,deficit))
                #     # print('We\'re just sigs!')

                #     # noise_piano_spgm, np_phase
                #     noise_piano_spgm, _ = make_spectrogram(noise_piano_sig, wdw_size, epsilon, 
                #                                     ova=True, debug=False)#[0].astype('float32').T
                #     piano_label_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon,
                #                                     ova=True, debug=False)
                #     noise_label_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
                #                                     ova=True, debug=False)

                #     # Write to file for fixed data gen
                #     np.save('../dlnn_data/dmged_mix_numpy/mixed' + file_num_str, noise_piano_spgm)
                #     # np.save('../dlnn_data/piano_source_numpy/piano' + file_num_str, piano_label_spgm)
                #     np.save('../dlnn_data/dmged_noise_numpy/noise' + file_num_str, noise_label_spgm)
                # else:
                # MIXED PRECISION - hail mary try
                noise_piano_spgm = np.load(pn_filepath)# .astype('float32')
                piano_label_spgm = np.load(pl_filepath)# .astype('float32')
                noise_label_spgm = np.load(nl_filepath)# .astype('float32')

                x[i] = noise_piano_spgm
                y1[i] = piano_label_spgm
                y2[i] = noise_label_spgm
            
            # print('YIELDING SHAPE:', noise_piano_spgm.shape, piano_label_spgm.shape, noise_label_spgm.shape)
            # print('YIELDING TYPES:', noise_piano_spgm.dtype, piano_label_spgm.dtype, noise_label_spgm.dtype)

            # yield x, y1, y2
            # yield noise_piano_spgm, piano_label_spgm, noise_label_spgm
            # yield ({'piano_noise_mixed': noise_piano_spgm, 'piano_true': piano_label_spgm, 'noise_true': noise_label_spgm}, 
            #        piano_label_spgm, noise_label_spgm)
            # MAKE THIS WORK FOR FIT & CUSTOM TRAIN - docs return tuple of lists or tuple of dicts - use lists for max freedom
            # yield ([noise_piano_spgm, piano_label_spgm, noise_label_spgm], [piano_label_spgm, noise_label_spgm])
            # yield ({'piano_noise_mixed': noise_piano_spgm, 'piano_true': piano_label_spgm, 'noise_true': noise_label_spgm}, 
            #        {'piano_pred': piano_label_spgm, 'noise_pred': noise_label_spgm})
            # print('GEN YIELDING')
            # MIXED PRECISION
            # if policy is None:
            # if pc_run:
            #     yield ({'piano_noise_mixed': x, 'piano_true': y1, 'noise_true': y2}, 
            #            {'piano_pred': y1, 'noise_pred': y2})
            # else:
            #     yield ({'piano_noise_mixed': x, 'piano_true': y1, 'noise_true': y2}, 
            #            {'mp_piano_pred': y1, 'mp_noise_pred': y2})
            yield ([x, np.concatenate((y1, y2), axis=-1)])

# # Have a train dir, a val dir, and (a test dir?)
# # Generator that returns samples and two targets each (TF-matrices)
# # def my_generator(x_files, y1_files, y2_files, batch_size, train_seq, train_feat, 
# def my_generator(y1_files, y2_files, num_samples, batch_size, train_seq, train_feat, 
#                  wdw_size, epsilon, pad_len=3081621, src_amp_low=0.75, src_amp_high=1.15):
#     # print('DEBUG Batch Size in my_generator:', batch_size)
#     # paper amp rng is (0.25, 1.25) for generalizing more
#     while True: # Loop forever so the generator never terminates
#         # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
#         # print('In the TRAIN generator loop') if (num_samples == 45) else print('In the VAL generator loop')
#         for offset in range(0, num_samples, batch_size):
#             # print('OFFSET:', offset)
#             # print('Starting batch', (offset + batch_size) / batch_size, 'out of', num_samples / batch_size)
#             # Get the samples you'll use in this batch
#             # batch_samples = x_files[offset:offset+batch_size]
#             batch_labels1 = y1_files[offset:offset+batch_size]
#             batch_labels2 = y2_files[offset:offset+batch_size]
#             # Initialise x, y1 and y2 arrays for this batch (FLOAT 32 for DLNN)
#             if (num_samples / batch_size == 0):
#                 # TEST FLOAT16
#                 x, y1, y2 = (np.empty((batch_size, train_seq, train_feat)).astype('float32'),
#                              np.empty((batch_size, train_seq, train_feat)).astype('float32'),
#                              np.empty((batch_size, train_seq, train_feat)).astype('float32'))
#             else:
#                 actual_batch_size = len(batch_labels1)
#                 # x, y1, y2 = [], [], []
#                 x, y1, y2 = (np.empty((actual_batch_size, train_seq, train_feat)).astype('float32'),
#                              np.empty((actual_batch_size, train_seq, train_feat)).astype('float32'),
#                              np.empty((actual_batch_size, train_seq, train_feat)).astype('float32'))

#             # For each example
#             # for i, batch_sample in enumerate(batch_samples):
#             for i in range(len(batch_labels1)):
#                 # print('I:', i)
#                 # print('Making training sample', (i+1), 'out of', len(batch_labels1))
#                 # Load mixed source (x) and source labels (y1, y2)
#                 # pn_filepath = batch_sample
#                 pl_filepath = batch_labels1[i]
#                 nl_filepath = batch_labels2[i]

#                 # Get number from filename
#                 file_num_str = list(re.findall(r'\d+', pl_filepath))[-1]

#                 # pn_sr, noise_piano_sig = wavfile.read(pn_filepath)
#                 pl_sr, piano_label_sig = wavfile.read(pl_filepath)
#                 nl_sr, noise_label_sig = wavfile.read(nl_filepath)
#                 pl_orig_type, nl_orig_type = piano_label_sig.dtype, noise_label_sig.dtype 
#                 piano_label_sig, noise_label_sig = piano_label_sig.astype('float64'), noise_label_sig.astype('float64')
#                 # assert len(noise_piano_sig) == len(noise_label_sig) == len(piano_label_sig)   
#                 assert len(noise_label_sig) == len(piano_label_sig)  
#                 # Stereo audio safety check
#                 if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
#                     # piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float32')]).astype(p_type)
#                     piano_label_sig = np.average(piano_label_sig, axis=-1)
#                 if isinstance(noise_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
#                     # noise_label_sig = np.array([((x[0] + x[1]) / 2) for x in noise_label_sig.astype('float32')]).astype(n_type)
#                     noise_label_sig = np.average(noise_label_sig, axis=-1)

#                 # if i == 0:
#                 #     print('Filenames:', pl_filepath, nl_filepath)
#                 #     print('Piano Sig:', piano_label_sig[1000:1010], 'type:', piano_label_sig.dtype)
#                 #     print('Noise Sig:', noise_label_sig[1000:1010], 'type:', noise_label_sig.dtype)

#                 # NEW - create features w/ signal data augmentation
#                 # 1) Do amplitude variation aug on each source
#                 # DEBUG
#                 # piano_amp_factor = random.uniform(src_amp_low, src_amp_high)
#                 # noise_amp_factor = random.uniform(src_amp_low, src_amp_high)
#                 # piano_label_sig *= piano_amp_factor
#                 # Done for the debug wavfile.write()
#                 # piano_label_sig = np.clip(piano_label_sig, 
#                 #                           np.iinfo(pl_orig_type).min, 
#                 #                           np.iinfo(pl_orig_type).max)
#                 # piano_label_sig = np.around(piano_label_sig).astype(pl_orig_type)

#                 # noise_label_sig *= noise_amp_factor
#                 # noise_label_sig = np.clip(noise_label_sig, 
#                 #                           np.iinfo(nl_orig_type).min, 
#                 #                           np.iinfo(nl_orig_type).max)
#                 # noise_label_sig = np.around(noise_label_sig).astype(nl_orig_type)
                
#                 # assert len(noise_label_sig) == len(piano_label_sig)  

#                 # if offset == 0 and i == 0:
#                 #     wavfile.write('/content/drive/My Drive/Quinn Coleman - Thesis/DLNN_Data/output/noise_beforepert.wav', 
#                 #                   nl_sr, noise_label_sig)

#                 # print('LEN:', len(noise_label_sig))
#                 # 2) TODO Do frequency perturbation aug on noise source
#                 # noise_label_sig = freq_perturb(noise_label_sig, wdw_size, epsilon)

#                 # assert len(noise_label_sig) == len(piano_label_sig)  

#                 # if offset == 0 and i == 0:
#                 #     wavfile.write('/content/drive/My Drive/Quinn Coleman - Thesis/DLNN_Data/output/noise_afterpert.wav', 
#                 #                   nl_sr, noise_label_sig)
#                 # 3) Mix - NOTE sigs must be same dtype
#                 avg_src_sum = (np.sum(piano_label_sig) + np.sum(noise_label_sig)) / 2
#                 src_percent_1 = random.randrange(int((src_amp_low*100) // 2), int((src_amp_high*100) // 2)) / 100
#                 src_percent_2 = 1 - src_percent_1
#                 piano_src_is_1 = bool(random.getrandbits(1))
#                 if piano_src_is_1:
#                     piano_label_sig *= src_percent_1
#                     noise_label_sig *= src_percent_2
#                 else:
#                     piano_label_sig *= src_percent_2
#                     noise_label_sig *= src_percent_1

#                 noise_piano_sig = piano_label_sig + noise_label_sig
#                 noise_piano_sig *= (avg_src_sum / np.sum(noise_piano_sig))             
#                 # piano_label_sig = piano_label_sig.astype('float64')
#                 # noise_label_sig = noise_label_sig.astype('float64')
#                 # noise_piano_sig = noise_piano_sig.astype('float64')

#                 # Preprocessing - length pad, transform into spectrograms, add eps
#                 # if isinstance(noise_piano_sig[0], np.ndarray):   # Stereo signal = 2 channels
#                 #     noise_piano_sig = np.array([((x[0] + x[1]) / 2) for x in noise_piano_sig.astype('float64')])
#                 # if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
#                 #     piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float64')])
                
#                 deficit = pad_len - len(noise_piano_sig)
#                 noise_piano_sig = np.pad(noise_piano_sig, (0,deficit))
#                 piano_label_sig = np.pad(piano_label_sig, (0,deficit))
#                 noise_label_sig = np.pad(noise_label_sig, (0,deficit))
#                 # print('We\'re just sigs!')

#                 # noise_piano_spgm, np_phase
#                 noise_piano_spgm, _ = make_spectrogram(noise_piano_sig, wdw_size, epsilon, 
#                                                 ova=True, debug=False)#[0].astype('float32').T
#                 piano_label_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon,
#                                                 ova=True, debug=False)
#                 noise_label_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
#                                                 ova=True, debug=False)

#                 # Write to file for fixed data gen
#                 np.save('../dlnn_data/piano_noise_numpy/mixed' + file_num_str, noise_piano_spgm)
#                 np.save('../dlnn_data/piano_label_numpy/piano' + file_num_str, piano_label_spgm)
#                 np.save('../dlnn_data/noise_label_numpy/noise' + file_num_str, noise_label_spgm)

#                 # if offset == 0 and i == 0:
#                 #     global_phases1 = np_phase
#                 #     print('GLOBAL PHASES 1:', global_phases1.shape)
#                 #     np_phase.tofile('1')
#                 # elif offset == 0 and i == 1:
#                 #     global_phases2 = np_phase
#                 #     print('GLOBAL PHASES 2:', global_phases2.shape)
#                 #     np_phase.tofile('2')
#                 # elif offset == 0 and i == 2:
#                 #     global_phases3 = np_phase
#                 #     print('GLOBAL PHASES 3:', global_phases3.shape)
#                 #     np_phase.tofile('3')

#                 # if i == 0:
#                 #     print('Mixed Sig:', noise_piano_sig[1000:1010])
#                 #     print('Piano Spgm Sum:', np.sum(piano_label_spgm))
#                 #     print('Noise Spgm Sum:', np.sum(noise_label_spgm))
#                 #     print('Mixed Spgm Sum:', np.sum(noise_piano_spgm))
                
#                 # Add below into make_spgm()
#                 # noise_piano_spgm[noise_piano_spgm == 0] = epsilon
#                 # piano_label_spgm[piano_label_spgm == 0] = epsilon
#                 # noise_label_spgm[noise_label_spgm == 0] = epsilon
#                 # print('We\'re spectrograms now!')

#                 # print('NP Shape:', noise_piano_spgm.shape)
#                 # print('PL Shape:', piano_label_spgm.shape)
#                 # print('NL Shape:', noise_label_spgm.shape)

#                 # # DEBUG prints
#                 # print('OFFSET', offset, 'index', str(i) + ':')
#                 # print('Zero in np spgm:', 0 in noise_piano_spgm)
#                 # print('Zero in p spgm:', 0 in piano_label_spgm)
#                 # print('Zero in n spgm:', 0 in noise_label_spgm)
#                 # print('NaN in np spgm:', True in np.isnan(noise_piano_spgm))
#                 # print('NaN in p spgm:', True in np.isnan(piano_label_spgm))
#                 # print('NaN in n spgm:', True in np.isnan(noise_label_spgm))
#                 # print()

#                 # print('Finished training sample')
#                 # Add samples to arrays
#                 # if (num_samples / batch_size == 0):
#                 x[i] = noise_piano_spgm
#                 y1[i] = piano_label_spgm
#                 y2[i] = noise_label_spgm
#                 # else:
#                 #     x.append(noise_piano_spgm)
#                 #     y1.append(piano_label_spgm)
#                 #     y2.append(noise_label_spgm)

#             # if (num_samples / batch_size != 0):
#             # # Make sure they're numpy arrays (as opposed to lists)
#             #     x = np.array(x)
#             #     y1 = np.array(y1)
#             #     y2 = np.array(y2)

#             # print('\nBlowing out x,y1,y2:', x.shape, y1.shape, y2.shape)
#             # The generator-y part: yield the next training batch            
#             # yield [x_train, y1_train, y2_train], y1_train, y2_train
#             # yield {'piano_noise_mixed': x, 'piano_true': y1, 'noise_true': y2}
#             # IF DOESN'T WORK, TRY
#             # yield ({'piano_noise_mixed': x, 'piano_true': y1, 'noise_true': y2}, 
#             #        y1, y2)
#             # IF DOESN'T WORK, TRY
#             # print('IN GENERATOR YEILDING SHAPE:', (x.shape, y1.shape, y2.shape))

#             # print('GENERATOR YEILDING TYPES:', x.dtype, y1.dtype, y2.dtype)

#             yield (x, y1, y2)

#             # What fit expects
#             # {'piano_noise_mixed': X, 'piano_true': y1, 'noise_true': y2}
#             # {'piano_pred': y1, 'noise_pred': y2}


# NN DATA STATS FUNC - Only used when dataset changes
def get_stats(y1_filenames, y2_filenames, num_samples, train_seq, train_feat, 
              wdw_size, epsilon, pad_len, src_amp_low=0.75, src_amp_high=1.15):
    
    samples = np.empty((num_samples, train_seq, train_feat))
    # piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # noise_samples = np.empty((num_samples, train_seq, train_feat))
    # aug_piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # aug_noise_samples = np.empty((num_samples, train_seq, train_feat))
    for i in range(num_samples):
        _, piano_label_sig = wavfile.read(y1_filenames[i])
        _, noise_label_sig = wavfile.read(y2_filenames[i])

        # print('Piano Sig Type:')
        # print(piano_label_sig.dtype)
        # print(piano_label_sig)
        # print('Done')

        # pl_orig_type, nl_orig_type = piano_label_sig.dtype, noise_label_sig.dtype 
        piano_label_sig, noise_label_sig = piano_label_sig.astype('float64'), noise_label_sig.astype('float64')
        assert len(noise_label_sig) == len(piano_label_sig)  
        # Stereo audio safety check
        if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
            # piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float32')]).astype(p_type)
            piano_label_sig = np.average(piano_label_sig, axis=-1)
        if isinstance(noise_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
            # noise_label_sig = np.array([((x[0] + x[1]) / 2) for x in noise_label_sig.astype('float32')]).astype(n_type)
            noise_label_sig = np.average(noise_label_sig, axis=-1)


        avg_src_sum = (np.sum(piano_label_sig) + np.sum(noise_label_sig)) / 2
        src_percent_1 = random.randrange(int((src_amp_low*100) // 2), int((src_amp_high*100) // 2)) / 100
        src_percent_2 = 1 - src_percent_1
        piano_src_is_1 = bool(random.getrandbits(1))
        if piano_src_is_1:
            piano_label_sig *= src_percent_1
            noise_label_sig *= src_percent_2
        else:
            piano_label_sig *= src_percent_2
            noise_label_sig *= src_percent_1

        noise_piano_sig = piano_label_sig + noise_label_sig
        noise_piano_sig *= (avg_src_sum / np.sum(noise_piano_sig))     


        # # Pad up here now to support earlier tests
        # deficit = pad_len - len(piano_label_sig)
        # piano_label_sig = np.pad(piano_label_sig, (0,deficit))
        # # noise_label_sig = np.pad(noise_label_sig, (0,deficit))
        
        # piano_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon, 
        #                                         ova=True, debug=False)
        # # noise_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
        #                                         # ova=True, debug=False)

        # # VISUAL TEST
        # # print('PIANO SPGM #', i)
        # # for i in range(train_seq):
        # #     print(piano_spgm[i])
        # # print('DONE')

        # piano_samples[i] = piano_spgm
        # # noise_samples[i] = noise_spgm

        # piano_amp_factor = random.uniform(src_amp_low, src_amp_high)
        # noise_amp_factor = random.uniform(src_amp_low, src_amp_high)
        # piano_label_sig *= piano_amp_factor
        # noise_label_sig *= noise_amp_factor

        # aug_piano_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon, 
        #                                         ova=True, debug=False)
        # # aug_noise_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
        #                                         # ova=True, debug=False)
        # aug_piano_samples[i] = aug_piano_spgm
        # # aug_noise_samples[i] = aug_noise_spgm

        # noise_piano_sig = piano_label_sig + noise_label_sig
        deficit = pad_len - len(noise_piano_sig)
        noise_piano_sig = np.pad(noise_piano_sig, (0,deficit))

        noise_piano_spgm, np_phase = make_spectrogram(noise_piano_sig, wdw_size, epsilon, 
                                                ova=True, debug=False)
        samples[i] = noise_piano_spgm

    # # print('A different test: the average sum of piano and noise labels')
    # # # print('Avg sum of piano sources:', np.mean(np.sum(piano_samples, axis=-1)))
    # # print('Avg sum of noise sources:', np.mean(np.sum(noise_samples, axis=-1)))
    # # # print('Avg sum of aug piano sources:', np.mean(np.sum(aug_piano_samples, axis=-1)))
    # # print('Avg sum of aug noise sources:', np.mean(np.sum(aug_noise_samples, axis=-1)))
    # print('A different test: the average val of piano and noise labels')
    # print('Avg of piano sources:', np.mean(piano_samples))
    # # print('Avg of noise sources:', np.mean(noise_samples))
    # print('Avg of aug piano sources:', np.mean(aug_piano_samples))
    # # print('Avg of aug noise sources:', np.mean(aug_noise_samples))
    
    return np.mean(samples), np.std(samples)



# NEURAL NETWORK FUNCTIONS
# print('Tensorflow version:', tf.__version__)
# # Tells tf.function not to make graph, & run all ops eagerly (step debugging)
# # tf.config.run_functions_eagerly(True)             # For nightly release
# # tf.config.experimental_run_functions_eagerly(True)  # For TF 2.2 (non-nightly)
# print('Eager execution enabled? (default)', tf.executing_eagerly())

# Debugging settings
# Debugging key - Keras tensors are diff than Tensorflow tensors
# tf.enable_eager_execution() # Deprecated
# tf.debugging.set_log_device_placement(True) # For seeing which device being used
# tf.debugging.enable_check_numerics()

# Data Generator?
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class Standardize(Layer):
    def __init__(self, mean, std, **kwargs):
        super(Standardize, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call(self, input):
        input -= self.mean
        input /= self.std
        return input


class UnStandardize(Layer):
    def __init__(self, mean, std, **kwargs):
        super(UnStandardize, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def call(self, input):
        input *= self.std
        input += self.mean
        return input


# TF Masking layer has too compilcated operations for a lambda, and want to serialize model
class TimeFreqMasking(Layer):

    # Init is for input-independent variables
    # def __init__(self, piano_flag, **kwargs):
    def __init__(self, epsilon, **kwargs):
        # MAKE LAYER DEAL IN FLOAT16
        # TEST FLOAT16
        # kwargs['autocast'] = False
        # MIXED PRECISION - output layer needs to produce float32
        # kwargs['dtype'] = 'float32' # - or actually try in __init__ below
        # super(TimeFreqMasking, self).__init__(dtype='float32', **kwargs)
        super(TimeFreqMasking, self).__init__(**kwargs)
        # self.piano_flag = piano_flag
        self.epsilon = epsilon

    # No build method, b/c passing in multiple inputs to layer (no single shape)

    def call(self, inputs):
        # Try this alternative format if below doesn't work
        # self.total = tf.Variable(initial_value=y_hat_other, trainable=False)
        # self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        # return self.total

        # y_hat_self, y_hat_other, x_mixed = inputs[0], inputs[1], inputs[2]
        y_hat_self, y_hat_other, x_mixed = inputs

        # print('TYPES IN TF MASKING:', y_hat_self.dtype, y_hat_other.dtype, x_mixed.dtype)

        mask = tf.abs(y_hat_self) / (tf.abs(y_hat_self) + tf.abs(y_hat_other) + self.epsilon)
        # print('Mask Shape:', mask.shape)
        # ones = tf.convert_to_tensor(np.ones(mask.shape).astype('float32'))
        # print('Ones Shape:', ones.shape)
        # y_tilde_self = mask * x_mixed if (self.piano_flag) else (ones - mask) * x_mixed
        y_tilde_self = mask * x_mixed

        # print('Y Tilde Shape:', y_tilde_self.shape)
        return y_tilde_self
    
    # config only contains things in __init__
    def get_config(self):
        config = super(TimeFreqMasking, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
    
    def from_config(cls, config):
        return cls(**config)


# In TF 2.0, if loss has extra arg, need add_loss() - Chollet
# Custom Loss Class Example from - https://github.com/tensorflow/tensorflow/issues/32142
# For multi-output -> "call" on just one output (piano) even though messy... (model.add_loss())
# https://stackoverflow.com/questions/54069363/output-multiple-losses-added-by-add-loss-in-keras
# ELSE MAYBE BEST solution = custom output layers, each with own loss (2 loss funcs), and pass in placeholder loss to model (layer.add_loss())
# https://github.com/keras-team/keras/blob/e8484633473c340defbe03a092be2d4856d56302/examples/variational_autoencoder.py
# Try add_loss, but with only a function - not GREAT design, b/c recalc last_dim each call
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss

# https://stackoverflow.com/questions/55149026/tensorflow-2-0-do-you-need-a-tf-function-decorator-on-top-of-each-function
# tf.function is only to decorate highest-level computations (training loops)
# tf.print is only for inside low-level functions then
# @tf.function
# def l2_norm_squared(a, b, last_dim):
#     return tf.math.reduce_sum(tf.reshape(a - b, shape=(-1, last_dim)) ** 2, axis=-1)

# Prefer native TF API over Keras backend API whenever possible, mostly
# https://stackoverflow.com/questions/59361689/redundancies-in-tf-keras-backend-and-tensorflow-libraries
# AND for TF simplicity, dont put loss calc in a function
# last_dim = noise_pred.shape[1] * noise_pred.shape[2]
# disc_loss = (
#     tf.math.reduce_sum(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
#     (loss_const * tf.math.reduce_sum(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
#     tf.math.reduce_sum(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) -
#     (loss_const * tf.math.reduce_sum(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1))
# )

# TRY - B/C tf.function is symbolic tensors - no error there
# Assign loss to each output -> keras should average/sum it
# FIX - only return one value = custom loss needs to return a tf scalar?
# @tf.function
# def custom_loss(self_true, self_pred, other_true, other_pred, loss_const):
#     # @tf.function
#     def closure(self_true, self_pred):
#         last_dim = other_pred.shape[1] * other_pred.shape[2]
#         return (
#             tf.math.reduce_mean(tf.reshape(self_pred - self_true, shape=(-1, last_dim)) ** 2) - 
#             (loss_const * tf.math.reduce_mean(tf.reshape(self_pred - other_true, shape=(-1, last_dim)) ** 2)) +
#             tf.math.reduce_mean(tf.reshape(other_pred - other_true, shape=(-1, last_dim)) ** 2) -
#             (loss_const * tf.math.reduce_mean(tf.reshape(other_pred - self_true, shape=(-1, last_dim)) ** 2))
#         )
#     return closure(self_true, self_pred)

# Loss function for subclassed model
def discriminative_loss(piano_true, noise_true, piano_pred, noise_pred, loss_const):
    # print('TYPES:', piano_true.dtype, noise_true.dtype, piano_pred.dtype, noise_pred.dtype)
    last_dim = piano_pred.shape[1] * piano_pred.shape[2]
    return (
        tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
        (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
        tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
        (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
    )
# # FIX - only return one value = less memory taken?
# def discriminative_loss(piano_true, noise_true, piano_pred, noise_pred, loss_const):
#     last_dim = piano_pred.shape[1] * piano_pred.shape[2]
#     return (
#         tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2) - 
#         (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2)) +
#         tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2) -
#         (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2))
#     )

# output = Concatenate() ([piano_true, noise_true])
# y_preds = Concatenate() ([piano_pred, noise_pred, loss_const_tensor])
def discrim_loss(y_true, y_pred):
    piano_true, noise_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    loss_const = y_pred[-1, :, :][0][0]
    piano_pred, noise_pred = tf.split(y_pred[:-1, :, :], num_or_size_splits=2, axis=0)

    last_dim = piano_pred.shape[1] * piano_pred.shape[2]
    return (
        tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
        (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
        tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
        (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
    )

def make_model(features, sequences, name='Model', epsilon=10 ** (-10),
                    loss_const=0.05, config=None, t_mean=None, t_std=None, 
                    optimizer=tf.keras.optimizers.RMSprop(),
                    pre_trained_wgts=None,
                    # GPU mem as func of HP TEST
                    # test=16, 
                    test=0, 
                    pc_run=False,
                    keras_fit=False):
    # TEST FLOAT16
    # MIXED PRECISION
    input_layer = Input(shape=(sequences, features), name='piano_noise_mixed')
    # input_layer = Input(shape=(sequences, features), dtype='float16', 
    #                     name='piano_noise_mixed')

    if config is not None:
        num_layers = len(config['layers'])
        prev_layer_type = None  # Works b/c all RNN stacks are size > 1
        for i in range(num_layers):
            layer_config = config['layers'][i]
            curr_layer_type = layer_config['type']

            # Standardize option
            if config['scale'] and i == 0:
                x = Standardize(t_mean, t_std) (input_layer)

            # Add skip connection if necessary
            if (config['rnn_res_cntn'] and prev_layer_type is not None and
                  prev_layer_type != 'Dense' and curr_layer_type == 'Dense'):
                x = Concatenate() ([x, input_layer])
    
            if curr_layer_type == 'RNN':
                if config['bidir']:
                    x = Bidirectional(SimpleRNN(features // layer_config['nrn_div'], 
                            activation=layer_config['act'], 
                            use_bias=config['bias_rnn'],
                            dropout=config['rnn_dropout'][0],
                            recurrent_dropout=config['rnn_dropout'][1],
                            return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
                else:
                    x = SimpleRNN(features // layer_config['nrn_div'], 
                            activation=layer_config['act'], 
                            use_bias=config['bias_rnn'],
                            dropout=config['rnn_dropout'][0],
                            recurrent_dropout=config['rnn_dropout'][1],
                            return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)

            elif curr_layer_type == 'LSTM':
                if config['bidir']:
                    x = Bidirectional(LSTM(features // layer_config['nrn_div'], 
                            activation=layer_config['act'], 
                            use_bias=config['bias_rnn'],
                            dropout=config['rnn_dropout'][0],
                            recurrent_dropout=config['rnn_dropout'][1],
                            return_sequences=True)) (input_layer if (i == 0 and not config['scale']) else x)
                else:
                    x = LSTM(features // layer_config['nrn_div'], 
                            activation=layer_config['act'], 
                            use_bias=config['bias_rnn'],
                            dropout=config['rnn_dropout'][0],
                            recurrent_dropout=config['rnn_dropout'][1],
                            return_sequences=True) (input_layer if (i == 0 and not config['scale']) else x)
            elif curr_layer_type == 'Dense':
                if i == (num_layers - 1):   # Last layer is fork layer
                    # Reverse standardization at end of model if appropriate
                    if config['scale']:
                        piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                        activation=layer_config['act'], 
                                                        use_bias=config['bias_dense']), 
                                                    name='piano_hat'
                                                   ) (x)
                        noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                        activation=layer_config['act'], 
                                                        use_bias=config['bias_dense']), 
                                                    name='noise_hat'
                                                   ) (x)
                        if config['bn']:
                            piano_hat = BatchNormalization() (piano_hat)
                            noise_hat = BatchNormalization() (noise_hat)
                        
                        piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
                        noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
                
                    else:
                        piano_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                        activation=layer_config['act'], 
                                                        use_bias=config['bias_dense']), 
                                                    name='piano_hat'
                                                   ) (x)
                        noise_hat = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                                        activation=layer_config['act'], 
                                                        use_bias=config['bias_dense']),
                                                    name='noise_hat'
                                                   ) (x)
                        if config['bn']:
                            piano_hat = BatchNormalization() (piano_hat)
                            noise_hat = BatchNormalization() (noise_hat)

                else:
                    x = TimeDistributed(Dense(features // layer_config['nrn_div'],
                                            activation=layer_config['act'], 
                                            use_bias=config['bias_dense']), 
                                       ) (input_layer if (i == 0 and not config['scale']) else x)
                    if config['bn']:
                        x = BatchNormalization() (x)

            prev_layer_type = curr_layer_type
    
    elif test > 0:
        if test == 1:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        # The difference in mem use between test 1 @ 2 is much an long a rnn takes
        if test == 2:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 3:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 4:
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x)
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        # The difference in mem use between test 2 @ 4 is how much more an lstm takes
        if test == 5:
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 6:
            x = LSTM(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer)
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 7:
            x = Standardize(t_mean, t_std) (input_layer)
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
        # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
        if test == 8:
            x = Standardize(t_mean, t_std) (input_layer)
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
        if test == 9:
            x = Standardize(t_mean, t_std) (input_layer)
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            piano_hat = UnStandardize(t_mean, t_std) (piano_hat)
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
            noise_hat = UnStandardize(t_mean, t_std) (noise_hat)
        if test == 10:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x)
            x = Concatenate() ([x, input_layer])
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        # The difference in mem use between test 1 @ 2 is much an long a rnn takes
        if test == 11:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = Concatenate() ([x, input_layer])
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 12:
            x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = Concatenate() ([x, input_layer])
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

        if test == 13:
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (input_layer) 
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (x) 
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        # The difference in mem use between test 1 @ 2 is much an long a rnn takes
        if test == 14:
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (input_layer) 
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 15:
            x = Bidirectional(SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True)) (input_layer) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

        if test == 16:
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (x) 
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        # The difference in mem use between test 6 & 2 is what doubling dim red does to mem usage
        if test == 17:
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (x) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
        if test == 18:
            x = SimpleRNN(features - 1, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
            piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
            noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch

    # Use pre-configurations (default)
    else:
        x = SimpleRNN(features // 2, 
                      activation='relu', 
                      return_sequences=True) (input_layer) 
        x = SimpleRNN(features // 2, 
                  activation='relu',
                  return_sequences=True) (x)
        piano_hat = TimeDistributed(Dense(features), name='piano_hat') (x)  # source 1 branch
        noise_hat = TimeDistributed(Dense(features), name='noise_hat') (x)  # source 2 branch
    piano_pred = TimeFreqMasking(epsilon=epsilon, 
                                 name='piano_pred') ((piano_hat, noise_hat, input_layer))
    noise_pred = TimeFreqMasking(epsilon=epsilon, 
                                 name='noise_pred') ((noise_hat, piano_hat, input_layer))

    # model = Model(inputs=input_layer, outputs=[piano_pred, noise_pred])

    # # Keras debug block
    # debug_piano_model = Model(
    #     inputs=model.inputs,
    #     # inputs=model.layers[3].output,
    #     # outputs=[model.layers[0].output] + model.outputs,
    #     outputs=[model.layers[2].output, model.layers[3].output, model.layers[5].output],
    #     name='Debug Piano Model (rnn2 out -> piano_hat out -> piano_pred out)'
    # )
    # debug_noise_model = Model(
    #     inputs=model.inputs,
    #     outputs=[model.layers[2].output, model.layers[4].output, model.layers[6].output],
    #     name='Debug Noise Model (rnn2 out -> noise_hat out -> noise_pred out)'
    # )
    # xs = tf.random.normal((3, sequences, features))
    # # print('DEBUG Piano Model Summary:')
    # # print(debug_piano_model.summary())
    # print('DEBUG Piano Model Run:')
    # # print(debug_piano_model(xs, training=True))

    # debug_piano_model_outputs = debug_piano_model(xs, training=True)
    # rnn_o, dense_o, mask_o = debug_piano_model_outputs[0].numpy(), debug_piano_model_outputs[1].numpy(), debug_piano_model_outputs[2].numpy()
    # print('Shape rnn out:', rnn_o.shape)
    # print('Shape dense out:', dense_o.shape)
    # print('Shape mask out:', mask_o.shape)
    # # print('Inf in rnn out:', True in np.isinf(rnn_o))
    # # print('Inf in dense out:', True in np.isinf(dense_o))
    # # print('Inf in mask out:', True in np.isinf(mask_o))
    # # print('NaN in rnn out:', True in np.isnan(rnn_o))
    # # print('NaN in dense out:', True in np.isnan(dense_o))
    # # print('NaN in mask out:', True in np.isnan(mask_o))
    # print()

    # # print('DEBUG Noise Model Summary:')
    # # print(debug_noise_model.summary())
    # print('DEBUG Noise Model Run:')
    # # print(debug_noise_model(xs, training=True))
    # debug_noise_model_outputs = debug_noise_model(xs, training=True)
    # rnn_o, dense_o, mask_o = debug_noise_model_outputs[0].numpy(), debug_noise_model_outputs[1].numpy(), debug_noise_model_outputs[2].numpy()
    # print('Shape rnn out:', rnn_o.shape)
    # print('Shape dense out:', dense_o.shape)
    # print('Shape mask out:', mask_o.shape)
    # # print('Inf in rnn out:', True in np.isinf(rnn_o))
    # # print('Inf in dense out:', True in np.isinf(dense_o))
    # # print('Inf in mask out:', True in np.isinf(mask_o))
    # # print('NaN in rnn out:', True in np.isnan(rnn_o))
    # # print('NaN in dense out:', True in np.isnan(dense_o))
    # # print('NaN in mask out:', True in np.isnan(mask_o))
    # print()
    # # print('Model Layers:')
    # # print([layer.name for layer in model.layers])
    # # ['piano_noise_mixed', 'simple_rnn', 'simple_rnn_1', 'piano_hat', 'noise_hat', 'piano_pred', 'noise_pred']    
    
    # disc_loss = None
    if pre_trained_wgts is not None:
            # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
            preds_and_lc = Concatenate(axis=0) ([piano_pred, 
                                                 noise_pred, 
                                                #  loss_const_tensor
                                                 tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                                ])
            # MIXED PRECISION - not sure if test case is necessary
            # if not pc_run:
            #     preds_and_lc = Activation('linear', name='mp_output', dtype='float32') (preds_and_lc)

            model = Model(inputs=input_layer, outputs=preds_and_lc)

            print('Only loading pre-trained weights for prediction')
            model.set_weights(pre_trained_wgts)
    elif keras_fit:
        # # print('MODEL OUTPUT TYPES:', piano_pred.dtype, noise_pred.dtype)
        # # print('MODEL TARGETS:', piano_pred.dtype, noise_pred.dtype)
        # # TEST FLOAT16
        # piano_true = Input(shape=(sequences, features), name='piano_true')
        # noise_true = Input(shape=(sequences, features), name='noise_true')
        # # piano_true = Input(shape=(sequences, features), dtype='float32', 
        # #                 name='piano_true')
        # # noise_true = Input(shape=(sequences, features), dtype='float32', 
        # #                 name='noise_true')
        # model = Model(inputs=[input_layer, piano_true, noise_true],
        #         outputs=[piano_pred, noise_pred])

        # loss_const = tf.constant(loss_const) # For performance/less mem
        # # FLOAT16 TEST
        # # loss_const = tf.dtypes.cast(loss_const, tf.float16)
        # # print('MODEL LOSS_CONST:', loss_const.dtype)
        # # 1 val instead of 1 val/batch makes for less mem used
        # last_dim = noise_pred.shape[1] * noise_pred.shape[2]
        # disc_loss = (
        #     tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2) - 
        #     (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2)) +
        #     tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2) -
        #     (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2))
        # )
        # # # FLOAT16 TEST
        # # disc_loss = (
        # #     tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16) - 
        # #     (loss_const * tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16)) +
        # #     tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16) -
        # #     (loss_const * tf.dtypes.cast(tf.math.reduce_mean(tf.dtypes.cast(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)), tf.float16) ** 2, axis=-1), tf.float16))
        # # )
        # # OOM BUG TEST - change one factor - loss
        # # model.add_loss(disc_loss)
        # # model.compile(optimizer=optimizer, loss={'piano_pred': 'mse', 'noise_pred': 'mse'})

        # loss_const_tensor = tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
        preds_and_lc = Concatenate(axis=0) ([piano_pred, 
                                             noise_pred, 
                                            #  loss_const_tensor
                                             tf.broadcast_to(tf.constant(loss_const), [1, sequences, features])
                                             ])
        # MIXED PRECISION
        # if not pc_run:
        #     preds_and_lc = Activation('linear', name='mp_output', dtype='float32') (preds_and_lc)

        model = Model(inputs=input_layer, outputs=preds_and_lc)
        # model = Model(inputs=input_layer, outputs=output)

        # Combine piano_pred, noise_pred & loss_const into models output!
        # loss_const_tensor = tf.reshape(tf.constant(loss_const), [None, sequences, 1])
        # output = Concatenate() ([piano_pred, noise_pred, loss_const_tensor])

        model.compile(optimizer=optimizer, loss=discrim_loss)
        # TRY - B/C tf.function is symbolic tensors - no error there
        # Assign loss to each output -> keras should average/sum it
        # # Problematic, returns a func or eager func, not a tensor
        # @tf.function
        # def piano_loss(noise_true, noise_pred, loss_const):
        #     def closure(piano_true, piano_pred):
        #         last_dim = noise_pred.shape[1] * noise_pred.shape[2]
        #         return (
        #             tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
        #             (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
        #             tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) -
        #             (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1))
        #         )
        #     return closure

        # @tf.function
        # def noise_loss(piano_true, piano_pred, loss_const):
        #     def closure(noise_true, noise_pred):
        #         last_dim = piano_pred.shape[1] * piano_pred.shape[2]
        #         return (
        #             tf.math.reduce_mean(tf.reshape(noise_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1) - 
        #             (loss_const * tf.math.reduce_mean(tf.reshape(noise_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1)) +
        #             tf.math.reduce_mean(tf.reshape(piano_pred - piano_true, shape=(-1, last_dim)) ** 2, axis=-1) -
        #             (loss_const * tf.math.reduce_mean(tf.reshape(piano_pred - noise_true, shape=(-1, last_dim)) ** 2, axis=-1))
        #         )
        #     return closure

        # model.compile(optimizer=optimizer,
        #               loss={
        #                     'piano_pred': custom_loss(piano_true, piano_pred, noise_true, noise_pred, loss_const),
        #                     'noise_pred': custom_loss(noise_true, noise_pred, piano_true, piano_pred, loss_const)
        #                    })
        #                 #   loss={
        #                 #       'piano_pred': piano_loss(noise_true, noise_pred, loss_const),
        #                 #       'noise_pred': noise_loss(piano_true, piano_pred, loss_const)
        #                 #        })

    return model    #, disc_loss


# CUSTOM TRAINING LOOP
# class TrainStep():
#     def __init__(self):
#         self.logits1, self.logits2 = None, None
#         self.loss, self.grads = None, None

#     @tf.function
#     def __call__(self, x, y1, y2, model, loss_const, optimizer):
#         with tf.GradientTape() as tape:
#             logits1, logits2 = model(x, training=True)
#             loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
#         grads = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         return loss


# # def get_train_step_func():
# @tf.function
# def train_step(x, y1, y2, model, loss_const, optimizer):
#     with tf.GradientTape() as tape:
#         logits1, logits2 = model(x, training=True)
#         loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
#     grads = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     return loss
#     # return train_step

# # def get_test_step_func():
# @tf.function
# def test_step(x, y1, y2, model, loss_const):
#     val_logits1, val_logits2 = model(x, training=False)
#     loss = discriminative_loss(y1, y2, val_logits1, val_logits2, loss_const)
#     return loss
#     # return test_step

# # def train_step_for_dist(x, y1, y2, model, loss_const, optimizer):
# def train_step_for_dist(inputs, model, loss_const, optimizer, dist_bs):
#     x, y1, y2 = inputs
#     with tf.GradientTape() as tape:
#         logits1, logits2 = model(x, training=True)
#         per_example_loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
#         loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=dist_bs)
#     grads = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     return loss

# # def test_step_for_dist(x, y1, y2, model, loss_const):
# def test_step_for_dist(inputs, model, loss_const, dist_bs):
#     x, y1, y2 = inputs
#     val_logits1, val_logits2 = model(x, training=False)
#     per_example_loss = discriminative_loss(y1, y2, val_logits1, val_logits2, loss_const)
#     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=dist_bs)

# # @tf.function
# # def distributed_train_step(x, y1, y2, model, loss_const, optimizer):
# @tf.function
# def distributed_train_step(dist_inputs, model, loss_const, optimizer, dist_bs):
#     per_replica_losses = mirrored_strategy.run(train_step_for_dist, 
#                                                args=(dist_inputs, model, loss_const, optimizer, dist_bs))
#     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, 
#                                     axis=None)

# # @tf.function
# # def distributed_test_step(x, y1, y2, model, loss_const):
# @tf.function
# def distributed_test_step(dist_inputs, model, loss_const, dist_bs):
#     per_replica_losses = mirrored_strategy.run(test_step_for_dist, 
#                                                args=(dist_inputs, model, loss_const, dist_bs))
#     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, 
#                                     axis=None)

def make_gen_callable(_gen):
    def gen():
        for x,y in _gen:
            yield x,y
    return gen

def custom_fit(model, train_dataset, val_dataset,
                # train_step_func, test_step_func,
               num_train, num_val, n_feat, n_seq, batch_size, 
               loss_const, epochs=20, 
               optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
               patience=100, epsilon=10 ** (-10), config=None, recent_model_path=None, pc_run=False,
               t_mean=None, t_std=None, grid_search_iter=None, gs_path=None, combos=None, gs_id=''):
    @tf.function
    def mixed_prec_train_step(x, y1, y2):
        with tf.GradientTape() as tape:
            logits1, logits2 = model(x, training=True)
            loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
            # MIXED PRECISION
            scaled_loss = optimizer.get_scaled_loss(loss)
        # grads = tape.gradient(loss, model.trainable_weights)
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def train_step(x, y1, y2):
        with tf.GradientTape() as tape:
            logits1, logits2 = model(x, training=True)
            loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    # def get_test_step_func():
    @tf.function
    def test_step(x, y1, y2):
        val_logits1, val_logits2 = model(x, training=False)
        loss = discriminative_loss(y1, y2, val_logits1, val_logits2, loss_const)
        return loss

    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        print('EPOCH:', epoch + 1)

        train_steps_per_epoch=math.ceil(num_train / batch_size)
        val_steps_per_epoch=math.ceil(num_val / batch_size)

        train_iter = iter(train_dataset)
        val_iter = iter(val_dataset)
        # if pc_run:
        # TRAIN LOOP
        # train_step_func, test_step_func = get_train_step_func(), get_test_step_func()
        total_loss, num_batches = 0.0, 0
        # for step, (x_batch_train, y1_batch_train, y2_batch_train) in enumerate(train_generator):
        for step in range(train_steps_per_epoch):
            # profile no more than 10 steps @ a time - save memory
            # avoid profiling first few batches for accuracy
            # if step % 2 == 0 and step > 1:
            #     with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            #         x_batch_train, y1_batch_train, y2_batch_train = next(train_iter)

            # # if epoch == 0 and step == 0:
            #     # tf.summary.trace_on(graph=True, profiler=True)
            #         loss_tensor = train_step(x_batch_train, y1_batch_train, y2_batch_train)
            #     # with train_summary_writer.as_default():
            #     #     tf.summary.trace_export(
            #     #         name='train_trace',
            #     #         step=step,
            #     #         profiler_outdir=train_log_dir)
            # else:
            x_batch_train, y1_batch_train, y2_batch_train = next(train_iter)
            if pc_run:
                loss_tensor = train_step(x_batch_train, y1_batch_train, y2_batch_train)
            else:
                loss_tensor = mixed_prec_train_step(x_batch_train, y1_batch_train, y2_batch_train)
            # loss_tensor = train_step_func(x_batch_train, y1_batch_train, y2_batch_train,
            #                               model, loss_const, optimizer)
            # loss_tensor = train_step(x_batch_train, y1_batch_train, y2_batch_train,
            #                             model, loss_const, optimizer)
            # loss_value = tf.math.reduce_mean(loss_tensor).numpy()
            total_loss += tf.math.reduce_mean(loss_tensor).numpy()
            num_batches += 1

            readable_step = step + 1
            # Log every batch
            if step == 0:
                print('Training execution (steps):', end = " ")
            print('(' + str(readable_step) + ')', end="")

            if readable_step == train_steps_per_epoch:
                break
            # else:
            #     x_batch_train, y1_batch_train, y2_batch_train = next(train_iter)

            #     loss_tensor = train_step(x_batch_train, y1_batch_train, y2_batch_train)

            #     # loss_tensor = train_step_func(x_batch_train, y1_batch_train, y2_batch_train,
            #     #                               model, loss_const, optimizer)
            #     # loss_tensor = train_step(x_batch_train, y1_batch_train, y2_batch_train,
            #     #                             model, loss_const, optimizer)
            #     # loss_value = tf.math.reduce_mean(loss_tensor).numpy()
            #     total_loss += tf.math.reduce_mean(loss_tensor).numpy()
            #     num_batches += 1

            #     readable_step = step + 1
            #     # Log every batch
            #     if step == 0:
            #         print('Training execution (steps):', end = " ")
            #     print('(' + str(readable_step) + ')', end="")

            #     if readable_step == train_steps_per_epoch:
            #         break

        avg_train_loss = total_loss / num_batches
        print(' - epoch loss:', avg_train_loss)
        history['loss'].append(avg_train_loss)

        # Tensorboard
        # with train_summary_writer.as_default():
        #     tf.summary.scalar("Loss", avg_train_loss, step=epoch)

        # VALIDATION LOOP
        total_loss, num_batches = 0.0, 0
        # for step, (x_batch_val, y1_batch_val, y2_batch_val) in enumerate(validation_generator):
        for step in range(val_steps_per_epoch):
            # if step % 2 == 0 and step > 1:
            #     with tf.profiler.experimental.Trace('val', step_num=step, _r=1):
            #         x_batch_val, y1_batch_val, y2_batch_val = next(val_iter)

            # # tf.summary.trace_on(graph=True, profiler=True)
            #         loss_tensor = test_step(x_batch_val, y1_batch_val, y2_batch_val)
            # # with test_summary_writer.as_default():
            # #     tf.summary.trace_export(
            # #         name='val_trace',
            # #         step=step,
            # #         profiler_outdir=test_log_dir)

            # else:
            x_batch_val, y1_batch_val, y2_batch_val = next(val_iter)
            loss_tensor = test_step(x_batch_val, y1_batch_val, y2_batch_val)
            # loss_tensor = test_step_func(x_batch_val, y1_batch_val, y2_batch_val,
            #                              model, loss_const)
            # loss_tensor = test_step(x_batch_val, y1_batch_val, y2_batch_val,
            #                         model, loss_const)
            total_loss += tf.math.reduce_mean(loss_tensor).numpy()
            num_batches += 1

            readable_step = step + 1
            if step == 0:
                print('Validate execution (steps):', end = " ")
            print('(' + str(readable_step) + ')', end="")
            if readable_step == val_steps_per_epoch:
                break

        avg_val_loss = total_loss / num_batches
        print(' - epoch val. loss:', avg_val_loss)        
        history['val_loss'].append(avg_val_loss)

        # Tensorboard
        # with test_summary_writer.as_default():
        #     tf.summary.scalar("Loss", avg_val_loss, step=epoch)
    
        # # else:
        #     # From docs: batch size must be equal to global batch size (refactor earlier if needed)
        #     # Assume better to give worker less batches than too many? - give even num
        #     # batch_size_per_replica = batch_size // 2
        #     # global_batch_size = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync
            
        #     with mirrored_strategy.scope():
        #         def compute_loss(y1, y2, logits1, logits2, loss_const):
        #             per_example_loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
        #             return tf.nn.compute_average_loss(per_example_loss, 
        #                                               global_batch_size=batch_size)
        #                                               # global_batch_size=global_batch_size)

        #     # Put functions inside scope
        #     def train_step_for_dist(inputs):
        #         x, y1, y2 = inputs
        #         with tf.GradientTape() as tape:
        #             logits1, logits2 = model(x, training=True)
        #             # per_example_loss = discriminative_loss(y1, y2, logits1, logits2, loss_const)
        #             # loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        #             loss = compute_loss(y1, y2, logits1, logits2, loss_const)
        #         grads = tape.gradient(loss, model.trainable_weights)
        #         optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #         return loss

        #     def test_step_for_dist(inputs):
        #         x, y1, y2 = inputs
        #         val_logits1, val_logits2 = model(x, training=False)
        #         # per_example_loss = discriminative_loss(y1, y2, val_logits1, val_logits2, loss_const)
        #         # return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        #         return compute_loss(y1, y2, val_logits1, val_logits2, loss_const)

        #     @tf.function
        #     def distributed_train_step(dist_inputs):
        #         per_replica_losses = mirrored_strategy.run(train_step_for_dist, 
        #                                                    args=(dist_inputs,))
        #         return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, 
        #                                         axis=None)

        #     @tf.function
        #     def distributed_test_step(dist_inputs):
        #         per_replica_losses = mirrored_strategy.run(test_step_for_dist, 
        #                                                    args=(dist_inputs,))
        #         return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, 
        #                                         axis=None)

        #     # TRAIN DATASET FROM GENERATOR
        #     train_dataset = tf.data.Dataset.from_generator(
        #         make_gen_callable(train_generator), output_types=(tf.float32), 
        #         output_shapes=tf.TensorShape([3, None, n_seq, n_feat])
        #     )
        #     # TRAIN LOOP
        #     total_loss, num_batches = 0.0, 0
        #     # Cross fingers for this line
        #     # train_dataset = train_dataset.batch(global_batch_size)
        #     dist_train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)
        #     train_iter = iter(dist_train_dataset)
        #     for step in range(train_steps_per_epoch):
        #         # x_batch_train, y1_batch_train, y2_batch_train = next(iterator)
        #         # loss_value = distributed_train_step(x_batch_train, y1_batch_train, y2_batch_train,
        #         #                                     model, loss_const, optimizer)
        #         # loss_value = distributed_train_step(next(iterator), model, loss_const, optimizer, global_batch_size)

        #         # debug = next(iterator)
        #         # print('next(iterator):', debug._values[0].shape, 'TYPE:', type(debug._values[0]), 
        #         # 'next thing:', debug._values[1].shape, 'TYPE:', type(debug._values[1]), 'LEN:', len(debug._values))
                
        #         # loss_value = distributed_train_step(next(iterator))
        #         total_loss += distributed_train_step(next(train_iter))
        #         num_batches += 1

        #         readable_step = step + 1
        #         # Log every batch
        #         if step == 0:
        #             print('Training execution (steps):', end = " ")
        #         print('(' + str(readable_step) + ')', end="")

        #         if readable_step == train_steps_per_epoch:
        #             break
            
        #     avg_train_loss = total_loss / num_batches

        #     print(' - epoch loss:', avg_train_loss)
        #     history['loss'].append(avg_train_loss)

        #     # VALIDATION DATASET FROM GENERATOR
        #     val_dataset = tf.data.Dataset.from_generator(
        #         make_gen_callable(validation_generator), output_types=(tf.float32), 
        #         output_shapes=tf.TensorShape([3, None, n_seq, n_feat])
        #     )
        #     # VALIDATION LOOP
        #     total_loss, num_batches = 0.0, 0
        #     # Cross fingers for this line
        #     # val_dataset = val_dataset.batch(global_batch_size)
        #     dist_val_dataset = mirrored_strategy.experimental_distribute_dataset(val_dataset)
        #     val_iter = iter(dist_val_dataset)
        #     for step in range(train_steps_per_epoch):
        #         # x_batch_val, y1_batch_val, y2_batch_val = next(iterator)
        #         # loss_value = distributed_test_step(x_batch_val, y1_batch_val, y2_batch_val,
        #         #                                    model, loss_const)

        #         # loss_value = distributed_test_step(next(iterator), model, loss_const, global_batch_size)
        #         # loss_value = distributed_test_step(next(iterator))
        #         total_loss += distributed_test_step(next(val_iter))
        #         num_batches += 1

        #         readable_step = step + 1
        #         if step == 0:
        #             print('Validate execution (steps):', end = " ")
        #         print('(' + str(readable_step) + ')', end="")
        #         if readable_step == val_steps_per_epoch:
        #             break

        #     avg_val_loss = total_loss / num_batches

        #     print(' - epoch val. loss:', avg_val_loss)        
        #     history['val_loss'].append(avg_val_loss)

        # Early stopping on no improvement in last (x-1) epochs - stop after x epochs
        if len(history['val_loss']) >= patience:
            # Track improvement over last (x-1) epochs
            improvement, prev_val = [], None
            for loss_val in history['val_loss'][-1 * patience:]:
                if prev_val is not None:
                    improvement.append((loss_val <= prev_val) or (not np.isnan(loss_val) and np.isnan(prev_val)))
                prev_val = loss_val
            # Stop training, no improvement in last x epochs
            if not any(improvement):
                break

    # tf.profiler.experimental.stop()
    # tf.profiler.experimental.client.trace('grpc://localhost:6009',
    #                                   'gs://logdir', 2000)
    return model, history


# MODEL TRAIN & EVAL FUNCTION - Training Loop From Scratch
def evaluate_source_sep(# train_dataset, val_dataset,
                        train_generator, validation_generator,
                        # train_step_func, test_step_func,
                        num_train, num_val, n_feat, n_seq, batch_size, 
                        loss_const, epochs=20, 
                        optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                        patience=100, epsilon=10 ** (-10), config=None, recent_model_path=None, pc_run=False,
                        t_mean=None, t_std=None, grid_search_iter=None, gs_path=None, combos=None, gs_id='',
                        keras_fit=False):
    # Generator returns tuple of lists ~ ([3 len], [2 len])
    # TODO maybe initialize the generator in here too? Would need to make generator callable first
    # TEST FLOAT16
    # MIXED PRECISION - hail mary try
    # if policy is None:
    # if pc_run:
    #     train_dataset = tf.data.Dataset.from_generator(
    #         make_gen_callable(train_generator), 
    #         output_types=({'piano_noise_mixed': tf.float32, 'piano_true': tf.float32, 'noise_true': tf.float32}, 
    #                     {'piano_pred': tf.float32, 'noise_pred': tf.float32}),
    #         # output_types=([tf.float32, tf.float32, tf.float32], [tf.float32, tf.float32]),
    #         # output_types=(tf.float32),
    #         output_shapes=({'piano_noise_mixed': tf.TensorShape([None, n_seq, n_feat]), 'piano_true': tf.TensorShape([None, n_seq, n_feat]), 'noise_true': tf.TensorShape([None, n_seq, n_feat])}, 
    #                     {'piano_pred': tf.TensorShape([None, n_seq, n_feat]), 'noise_pred': tf.TensorShape([None, n_seq, n_feat])}),      # For my gen & keras functional API & custom training
    #         # output_shapes=({'piano_noise_mixed': tf.TensorShape([n_seq, n_feat]), 'piano_true': tf.TensorShape([n_seq, n_feat]), 'noise_true': tf.TensorShape([n_seq, n_feat])}, 
    #         #                {'piano_pred': tf.TensorShape([n_seq, n_feat]), 'noise_pred': tf.TensorShape([n_seq, n_feat])}),      # For keras functional API & custom training
    #         # output_shapes=([tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)], 
    #         #                [tf.TensorShape(None), tf.TensorShape(None)])    # For keras functional API & custom training
    #         # output_shapes=([(n_seq, n_feat), (n_seq, n_feat), (n_seq, n_feat)], 
    #         #                [(n_seq, n_feat), (n_seq, n_feat)])    # For keras functional API & custom training
    #         # output_shapes=([tf.TensorShape([n_seq, n_feat]), tf.TensorShape([n_seq, n_feat]), tf.TensorShape([n_seq, n_feat])], 
    #         #                [tf.TensorShape([n_seq, n_feat]), tf.TensorShape([n_seq, n_feat])])    # For keras functional API & custom training
    #         # output_shapes=tf.TensorShape([3, n_seq, n_feat])    # No batch, for model.fit()
    #         # output_shapes=tf.TensorShape([3, None, n_seq, n_feat])
    #     )
    #     val_dataset = tf.data.Dataset.from_generator(
    #         make_gen_callable(validation_generator), 
    #         output_types=({'piano_noise_mixed': tf.float32, 'piano_true': tf.float32, 'noise_true': tf.float32}, 
    #                     {'piano_pred': tf.float32, 'noise_pred': tf.float32}),
    #         output_shapes=({'piano_noise_mixed': tf.TensorShape([None, n_seq, n_feat]), 'piano_true': tf.TensorShape([None, n_seq, n_feat]), 'noise_true': tf.TensorShape([None, n_seq, n_feat])}, 
    #                     {'piano_pred': tf.TensorShape([None, n_seq, n_feat]), 'noise_pred': tf.TensorShape([None, n_seq, n_feat])}),      # For my gen & keras functional API & custom training
    #         # output_shapes=([tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None)], 
    #         #                [tf.TensorShape(None), tf.TensorShape(None)])    # For keras functional API & custom training
    #         # output_shapes=tf.TensorShape([3, n_seq, n_feat])    # No batch, for model.fit()
    #         # output_shapes=tf.TensorShape([3, None, n_seq, n_feat])
    #     )
    # else:
    #     train_dataset = tf.data.Dataset.from_generator(
    #         make_gen_callable(train_generator), 
    #         output_types=({'piano_noise_mixed': tf.float32, 'piano_true': tf.float32, 'noise_true': tf.float32}, 
    #                     {'mp_piano_pred': tf.float32, 'mp_noise_pred': tf.float32}),
    #         output_shapes=({'piano_noise_mixed': tf.TensorShape([None, n_seq, n_feat]), 'piano_true': tf.TensorShape([None, n_seq, n_feat]), 'noise_true': tf.TensorShape([None, n_seq, n_feat])}, 
    #                     {'mp_piano_pred': tf.TensorShape([None, n_seq, n_feat]), 'mp_noise_pred': tf.TensorShape([None, n_seq, n_feat])}),      # For my gen & keras functional API & custom training
    #     )
    #     val_dataset = tf.data.Dataset.from_generator(
    #         make_gen_callable(validation_generator), 
    #         output_types=({'piano_noise_mixed': tf.float32, 'piano_true': tf.float32, 'noise_true': tf.float32}, 
    #                     {'mp_piano_pred': tf.float32, 'mp_noise_pred': tf.float32}),
    #         output_shapes=({'piano_noise_mixed': tf.TensorShape([None, n_seq, n_feat]), 'piano_true': tf.TensorShape([None, n_seq, n_feat]), 'noise_true': tf.TensorShape([None, n_seq, n_feat])}, 
    #                     {'mp_piano_pred': tf.TensorShape([None, n_seq, n_feat]), 'mp_noise_pred': tf.TensorShape([None, n_seq, n_feat])}),      # For my gen & keras functional API & custom training
    #     )

    train_dataset = tf.data.Dataset.from_generator(
        make_gen_callable(train_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )
    val_dataset = tf.data.Dataset.from_generator(
        make_gen_callable(validation_generator), 
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, n_seq, n_feat), (None, n_seq, n_feat*2)),
    )

    # print('TRAIN DATASET ELEMENTS:', train_dataset.element_spec)
    # print('VALID DATASET ELEMENTS:', val_dataset.element_spec)
    # print('TRAIN DATASET TYPE:', train_dataset)
    # print('VALID DATASET TYPE:', val_dataset)
    # Input pipeline optimizations
    # TODO - parallelize pre-processing -> move preprocessing to tf first
    # Vectorize pre-processing, by batching before & transform whole batch of data
    # If doing this ^, do it before call to cache()
    # BUT if transformed data (sig->spgm) to big for cache, call cache() after
    # Batch dataset method does not work with MY generator dataset was made with
    train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    # train_dataset.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    # val_dataset.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
    # print('-after changes- TRAIN DATASET TYPE:', train_dataset)
    # print('-after changes- VALID DATASET TYPE:', val_dataset)
    
    # print('X shape:', X.shape, 'y1 shape:', y1.shape, 'y2 shape:', y2.shape)
    # print('X shape:', X.shape)
    # tf.profiler.experimental.server.start(6009)
    # tf.profiler.experimental.start('logdir')
    print('Making model...')
    # if pc_run:
    model = make_model(n_feat, n_seq, name='Training Model', epsilon=epsilon, loss_const=loss_const,
                            config=config, t_mean=t_mean, t_std=t_std, optimizer=optimizer,
                            pc_run=pc_run, keras_fit=keras_fit)
    # print('KERAS LOSS TENSOR:', keras_fit_loss)
    
        # optimizer = optimizer
    # else:
    #     with mirrored_strategy.scope():
    #         model = make_model(n_feat, n_seq, name='Training Model', epsilon=epsilon, 
    #                                 config=config, t_mean=t_mean, t_std=t_std)
    #         optimizer = optimizer
    print(model.summary())

    # MIXED PRECISION
    # if not pc_run:
        # optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = '../logs/gradient_tape/' + current_time + '/train'
    # test_log_dir = '../logs/gradient_tape/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print('Going into training now...')
    if keras_fit:
        # log_dir = '../logs/keras_fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        hist = model.fit(train_dataset,
                     steps_per_epoch=math.ceil(num_train / batch_size),
                     epochs=epochs,
                     validation_data=val_dataset,
                     validation_steps=math.ceil(num_val / batch_size),
                     callbacks=[EarlyStopping('val_loss', patience=patience, mode='min')])#,
                                # Done memory profiling
                                # TensorBoard(log_dir=log_dir, profile_batch='2, 4')])   # 10' # by default, profiles 2nd batch
        history = hist.history
    else:
        model, history = custom_fit(model, train_dataset, val_dataset,
                                    num_train, num_val, n_feat, n_seq, batch_size, 
                                    loss_const, epochs=20, 
                                    optimizer=tf.keras.optimizers.RMSprop(clipvalue=0.75),
                                    patience=100, epsilon=10 ** (-10), config=None, 
                                    recent_model_path=None, pc_run=False, t_mean=None, t_std=None, 
                                    grid_search_iter=None, gs_path=None, combos=None, gs_id='')
    # Need to install additional unnecessary libs
    # if not pc_run and grid_search_iter is None:
    #     tf.keras.utils.plot_model(model, 
    #                               (gs_path + 'model' + str(grid_search_iter) + 'of' + str(combos) + '.png'
    #                               if grid_search_iter is not None else
    #                               'last_trained_model.png'), 
    #                               show_shapes=True)
 
    pc_run_str = '' if pc_run else '_noPC'
    if pc_run and grid_search_iter is None:
        #  Can't for imperative models
        model.save(recent_model_path)

        # print('History Dictionary Keys:', hist.history.keys())
        # 'val_loss', 'val_piano_pred_loss', 'val_noise_pred_loss',
        # 'loss', 'piano_pred_loss', 'noise_pred_loss'

        # print('Val Loss:\n', hist.history['val_loss'])
        # print('Loss:\n', hist.history['loss'])
        # CUSTOM TRAIN LOOP CHANGES FOR BOTH BLOCKS HERE
        print('Val Loss:\n', history['val_loss'])
        print('Loss:\n', history['loss'])

        epoch_r = range(1, len(history['loss'])+1)
        plt.plot(epoch_r, history['val_loss'], 'b', label = 'Validation Loss')
        plt.plot(epoch_r, history['loss'], 'bo', label = 'Training Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig('../train_val_loss_chart' + pc_run_str + '.png')
    # Consider if too much storage use, when model runs faster w/ OOM fix
    # else:
    #     epoch_r = range(1, len(history['loss'])+1)
    #     plt.plot(epoch_r, history['val_loss'], 'b', label = 'Validation Loss')
    #     plt.plot(epoch_r, history['loss'], 'bo', label = 'Training Loss')
    #     plt.title('Training & Validation Loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     # plt.show()
    #     if len(gs_id) > 0:
    #         gs_id += '_'
    #     plt.savefig(gs_path + gs_id + 'train_val_loss_chart_' + 
    #                 str(grid_search_iter) + '_of_' + str(combos) + pc_run_str + '.png')

    return model, history['loss'], history['val_loss']


# GRID SEARCH FUNCTION

# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasRegressor

# TODO - do my own Grid Search - call evaluate_source_sep for each hyperparam combo
#   b/c not worth complication for something so simple
# TODO - if not good, do Keras Tuner Random Search w/ help from this(?): https://github.com/keras-team/keras-tuner/issues/122
# class MyKerasRegressor(KerasRegressor):

#     def fit(self, X, y, **kwargs):

#         train_batch_size, loss_const, epochs = 5, 0.05, 10

#         train_generator = my_generator(x_train_files, y1_train_files, y2_train_files, 
#                                     batch_size=train_batch_size, wdw_size=wdw_size, 
#                                     epsilon=epsilon, pad_len=MAX_SIG_LEN)
#         validation_generator = my_generator(x_val_files, y1_val_files, y2_val_files, 
#                                     batch_size=train_batch_size, wdw_size=wdw_size, 
#                                     epsilon=epsilon, pad_len=MAX_SIG_LEN)

#         return self.__history

def get_hp_configs(bare_config_path, pc_run=False):
    # IMPORTANT: 1st GS - GO FOR WIDE RANGE OF OPTIONS & LESS OPTIONS PER HP
    # TEST FLOAT16 - double batch size
    # MIXED PRECISION   - double batch size (can't on PC still b/c OOM), for V100: multiple of 8
    # batch_size_optns = [3] if pc_run else [8] # [16, 24] OOM on f35 w/ old addloss model
    # OOM BOUND TEST
    # batch_size_optns = [3] if pc_run else [12, 18]  
    batch_size_optns = [3] if pc_run else [8, 16]  
    # epochs total options 10, 50, 100, but keep low b/c can go more if neccesary later (early stop pattern = 5)
    epochs_optns = [10]
    # loss_const total options 0 - 0.3 by steps of 0.05
    loss_const_optns = [0.05, 0.2]
    # Optimizers ... test out Adaptive Learning Rate Optimizers (RMSprop & Adam) Adam ~ RMSprop w/ momentum
    # Balance between gradient clipping and lr for exploding gradient
    # If time permits, later grid searches explore learning rate & momentum to fine tune
    
    # OLD
    # WORKED!!!! (very low lr - 2 orders of mag lower than default) at 9:45 am checked output
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=10, learning_rate=0.00001) # Random HP
    # Try next?
    # ALMOST worked, bcame NaN at end, so bad result
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=1, learning_rate=0.0001) # Random HP
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=0.5)
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=1)
    # ALMOST
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # Random HP
    # Find optimal balance betwwen clipval & lr for random HPs
    # ALMOST worked, NaN at end
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=10, learning_rate=0.0001) # Random HP
    # (very low clipvalue - 2/3 orders of mag higher than default ~1/0.1)
    # Failed
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue=100, learning_rate=0.0001) # Random HP
    # Is learning rate only thing that matters? YES or does clip val help no
    # ALMOST, but became NaN earlier - lr is more effective
    #   When clip calue is too high w/ lr -> bad, else almost works
    #       does work when learning rate = 0.00001
    # train_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001) # Random HP
    
    # Optimizers, should non-PC have more options b/c LSTM could help w/ expl gradi
    #   TODO Test - on PC w/ LSTM to find out after this

    # Trials for good results
    # Does clipvalue do anything by itslef (cv=100)? - no  10? - yes, 19 million
    # What's the ghighest learning rate by itself that makes it work? 
    #   0.0001 (R & A)->0.0005 (not R & not A) -> 0.0008 no
    #   11 million                                            
    # If we gradclip with a a little higher learning rate, still work? 
    #   cv=10 & lr=0.0005 yes, cv=10 & lr=0.001 no, cv=5 & lr=0.0008 yes, cv=5 & lr=0.001 no, cv=10 & lr=0.0008 no
    #   5 million (RMSprop)                         14 million val loss                         
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam')
    #                   ]
    # FYI - Best seen: clipvalue=10, lr=0.0005 - keep in mind for 2nd honed-in GS
    # # FLOAT16
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-4), -1, 0.0001, 'Adam'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10, epsilon=1e-4), 10, 0.001, 'Adam')
    #                   ]
    # MIXED PRECISION - doesn't support gradient clipping or specifically clipvalue
    # if pc_run:
    optimizer_optns = [
                    (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
                    (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
                    (tf.keras.optimizers.Adam(learning_rate=0.0001), -1, 0.0001, 'Adam'),
                    (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam')
                    ]
    # else:
    #     optimizer_optns = [
    #                     (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
    #                     (tf.keras.optimizers.Adam(learning_rate=0.0001), -1, 0.0001, 'Adam'),
    #                     ]
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(learning_rate=0.0001), -1, 0.0001, 'RMSprop'),
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(learning_rate=0.0001), -1, 0.0001, 'Adam'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam')
    #                   ]

    train_configs = {'batch_size': batch_size_optns, 'epochs': epochs_optns,
                     'loss_const': loss_const_optns, 'optimizer': optimizer_optns}
    
    # # REPL TEST - arch config, all config, optiizer config
    # dropout_optns = [(0.0,0.0)]    # For RNN only    IF NEEDED CAN GO DOWN TO 2 (conservative value)
    # scale_optns = [False]
    # rnn_skip_optns = [False]
    # bias_rnn_optns = [True]     # False
    # bias_dense_optns = [True]   # False
    # bidir_optns = [True]
    # bn_optns = [False]                    # For Dense only
    # # TEST - failed - OOM on PC
    # # rnn_optns = ['LSTM'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model              
    # rnn_optns = ['RNN'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model
    # if pc_run:
    #     # TEST PC
    #     # with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #     with open(bare_config_path + 'hp_arch_config_final.json') as hp_file:
    #         bare_config_optns = [json.load(hp_file)['archs'][3]]
    # else:
    #     # with open(bare_config_path + 'hp_arch_config_largedim.json') as hp_file:
    #     with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #         bare_config_optns = json.load(hp_file)['archs']

    # IMPORTANT: Order arch config options by mem-intensive HPs first,
    #           to protect against OOM on grid search
    # dropout_optns = [(0.0,0.0), (0.2,0.2), (0.2,0.5), (0.5,0.2), (0.5,0.5)]   # For RNN only
    dropout_optns = [(0.0,0.0), (0.25,0.25)]    # For RNN only    IF NEEDED CAN GO DOWN TO 2 (conservative value)
    scale_optns = [False, True]
    rnn_skip_optns = [False, True]
    bias_rnn_optns = [True]     # False
    bias_dense_optns = [True]   # False
    bidir_optns = [False, True]
    bn_optns = [False, True]                    # For Dense only
    rnn_optns = ['RNN'] if pc_run else ['RNN', 'LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model  
    # MIXED PRECISION combat NaNs            
    # rnn_optns = ['RNN'] if pc_run else ['LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model
    # rnn_optns = ['RNN'] # F35 OOM w/ mixed precision BUT batch size too high?
    if pc_run:
        # TEST PC
        # with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
        with open(bare_config_path + 'hp_arch_config_final.json') as hp_file:
            bare_config_optns = json.load(hp_file)['archs']
    else:
        # with open(bare_config_path + 'hp_arch_config_largedim.json') as hp_file:
        with open(bare_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
            bare_config_optns = json.load(hp_file)['archs']
    
    # iter_hp = 0 # test
    arch_config_optns = []
    for config in bare_config_optns:  
        for rnn_optn in rnn_optns:
            for bidir_optn in bidir_optns:
                for scale_optn in scale_optns:
                    for bn_optn in bn_optns:  
                        for rnn_skip_optn in rnn_skip_optns:
                            for bias_rnn_optn in bias_rnn_optns:
                                for bias_dense_optn in bias_dense_optns:
                                    for dropout_optn in dropout_optns:   
                                        # Make a unique copy for each factor combo
                                        curr_config = deepcopy(config)  
                                        curr_config['scale'] = scale_optn
                                        curr_config['rnn_res_cntn'] = rnn_skip_optn
                                        curr_config['bias_rnn'] = bias_rnn_optn
                                        curr_config['bias_dense'] = bias_dense_optn
                                        curr_config['bidir'] = bidir_optn
                                        curr_config['rnn_dropout'] = dropout_optn
                                        curr_config['bn'] = bn_optn
                                        if rnn_optn == 'LSTM':
                                            for i, layer in enumerate(config['layers']):
                                                if layer['type'] == 'RNN':
                                                    curr_config['layers'][i]['type'] = rnn_optn
                                        # if iter_hp < 5:
                                        #     print('Iter even number should have RNN:', iter_hp, rnn_optn, curr_config)
                                        # Append updated config
                                        arch_config_optns.append(curr_config) 
                                        # if iter_hp < 3:
                                        #     print('arch_config_optns:', arch_config_optns)
                                        # iter_hp += 1
    # print('About to return first index:', arch_config_optns[0])
    return train_configs, arch_config_optns


# def grid_search(y1_train_files, y2_train_files, y1_val_files, y2_val_files,
def grid_search(x_train_files, y1_train_files, y2_train_files, 
                x_val_files, y1_val_files, y2_val_files,
                # train_step_func, test_step_func,
                n_feat, n_seq, 
                # wdw_size, 
                epsilon, 
                # max_sig_len, 
                t_mean, t_std,
                train_configs, arch_config_optns,
                # arch_config_path, 
                gsres_path, early_stop_pat=3, pc_run=False, 
                gs_id='', restart=False, keras_fit=False):
    # model = MyKerasRegressor(build_fn=make_model, 
    #                          features=n_feat, sequences=n_seq)
    # param_grid = {batch_size: batch_size, epochs: epochs}#, 
    #               #lost_const: loss_const}
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)#, cv=3)

    # IMPORTANT to take advantage of what's known in test data to minimize factors
    # Factors: batchsize, epochs, loss_const, optimizers, gradient clipping,
    # learning rate, lstm/rnn, layer type/arch, tanh activation, num neurons in layer,
    # bidiric layer, amp var aug range, batch norm, skip connection over lstms,
    # standardize input & un-standardize output  
    # Maybe factor? Dmged/non-dmged piano input 
    print('\nPC RUN:', pc_run, '\n\nGRID SEARCH ID:', gs_id if len(gs_id) > 0 else 'N/A', '\n')

    num_train, num_val = len(y1_train_files), len(y1_val_files)

    # # All factors version below:
    # # IMPORTANT FOR BATCH SIZE: Factors of total samples * (1 - val_split) for performance
    # #   - total options: 1,3,5,9,15
    # # IMPORTANT: 1st GS - GO FOR WIDE RANGE OF OPTIONS & LESS OPTIONS PER HP
    # # batch_size_optns = [3] if pc_run else [3, 5, 15]

    # # Being careful about batch size effect on mem -> start low
    # # batch_size_optns = [1] if pc_run else [3, 9]    # Lowering batch size 3 -> 1 b/c OOM Error on GS iter 15
    # # FOR PC: Runs longer, so 1 less batch size option is good for ~2 weeks runtime
    # batch_size_optns = [3] if pc_run else [4, 10]    # Lowering batch size 3 -> 1 b/c OOM Error on GS iter 15
    # # epochs total options 10, 50, 100, but keep low b/c can go more if neccesary later (early stop pattern = 5)
    # epochs_optns = [10]
    # # loss_const total options 0 - 0.3 by steps of 0.05
    # # Paper - joint training causes sensitivity to gamma, keep in low range of (0.05 - 0.2)
    #     # FAILED - IMPORTANT LATER: Possible exception - can split up one HP, if we run 2 on PC (must be HP easy on mem) ALL OTHER HPs SAME
    #     # CHANGE - DO NOT COMMIT TIL CHANGE IMPL
    #     # batch_size_optns = [3]
    #     # rnn_optns = ['RNN']
    #     # with open(arch_config_path + 'hp_arch_config.json') as hp_file:
    #     #     bare_config_optns = json.load(hp_file)['archs']
    #     # loss_const_optns = [0.02, 0.1] if pc_run else [0.2, 0.3]
    # loss_const_optns = [0.05, 0.2]
    # # Remove no clipval? - 1st GS
    # optimizer_optns = [
    #                   (tf.keras.optimizers.RMSprop(clipvalue=10), 10, 0.001, 'RMSprop'),
    #                   (tf.keras.optimizers.Adam(clipvalue=10), 10, 0.001, 'Adam')
    #                   ]
    # # optimizer_optns = [(tf.keras.optimizers.RMSprop(), 0, 0.001, 'RMSprop'), 
    # #                   (tf.keras.optimizers.RMSprop(clipvalue=0.25), 0.25, 0.001, 'RMSprop'), 
    # #                   (tf.keras.optimizers.RMSprop(clipvalue=0.5), 0.5, 0.001, 'RMSprop'), 
    # #                   (tf.keras.optimizers.RMSprop(clipvalue=0.75), 0.75, 0.001, 'RMSprop'),
    # #                   (tf.keras.optimizers.Adam(), 0, 0.001, 'Adam'), 
    # #                   (tf.keras.optimizers.Adam(clipvalue=0.25), 0.25, 0.001, 'Adam'), 
    # #                   (tf.keras.optimizers.Adam(clipvalue=0.5), 0.5, 0.001, 'Adam'), 
    # #                   (tf.keras.optimizers.Adam(clipvalue=0.75), 0.75, 0.001, 'Adam')#,
    # #                   ]
    # # Optimizers ... test out Adaptive Learning Rate Optimizers (RMSprop & Adam) Adam ~ RMSprop w/ momentum
    # # If time permits, later grid searches explore learning rate & momentum to fine tune
    # # dropout_optns = [(0.0,0.0), (0.2,0.2), (0.2,0.5), (0.5,0.2), (0.5,0.5)]   # For RNN only
    # dropout_optns = [(0.0,0.0), (0.25,0.25)]    # For RNN only    IF NEEDED CAN GO DOWN TO 2 (conservative value)
    # scale_optns = [True, False]
    # rnn_skip_optns = [True, False]
    # bias_rnn_optns = [True]     # False
    # bias_dense_optns = [True]   # False
    # bidir_optns = [True, False]
    # bn_optns = [True, False]            # For Dense only
    # # TEST - dont believe this should matter (got to iter 16 last w/ & 2 batchsize)
    # # rnn_optns = ['RNN', 'LSTM']
    # rnn_optns = ['RNN'] if pc_run else ['LSTM']  # F35 sesh crashed doing dropouts on LSTM - old model

    # # Optional - for future when I'm not hitting SNR correctly
    # # amp_var_rng_optns = [(0.5, 1.25), (0.75, 1.15), (0.9, 1.1)]

    # # TEST - dont beleive this should matter (got to iter 16 last w/ & 2 batchsize)
    # # with open(arch_config_path + 'hp_arch_config_nodimreduc.json') as hp_file:
    # #     bare_config_optns = json.load(hp_file)['archs']
    # if pc_run:
    #     # TEST PC
    #     # with open(arch_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #     with open(arch_config_path + 'hp_arch_config_final.json') as hp_file:
    #         bare_config_optns = json.load(hp_file)['archs']
    # else:
    #     # with open(arch_config_path + 'hp_arch_config_largedim.json') as hp_file:
    #     with open(arch_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
    #         bare_config_optns = json.load(hp_file)['archs']

    # # Comment-out block below for all-factors version
    # arch_config_optns = []   # Add variations of each bare config to official
    # for config in bare_config_optns:    
    #     for scale_optn in scale_optns:
    #         for rnn_skip_optn in rnn_skip_optns:
    #             for bias_rnn_optn in bias_rnn_optns:
    #                 for bias_dense_optn in bias_dense_optns:
    #                     for bidir_optn in bidir_optns:
    #                         for dropout_optn in dropout_optns:  
    #                             for bn_optn in bn_optns:   
    #                                 for rnn_optn in rnn_optns:
    #                                     # Make a unique copy for each factor combo
    #                                     curr_config = config.copy()
    #                                     curr_config['scale'] = scale_optn
    #                                     curr_config['rnn_res_cntn'] = rnn_skip_optn
    #                                     curr_config['bias_rnn'] = bias_rnn_optn
    #                                     curr_config['bias_dense'] = bias_dense_optn
    #                                     curr_config['bidir'] = bidir_optn
    #                                     curr_config['rnn_dropout'] = dropout_optn
    #                                     curr_config['bn'] = bn_optn
    #                                     if rnn_optn == 'LSTM':
    #                                         for i, layer in enumerate(config['layers']):
    #                                             if layer['type'] == 'RNN':
    #                                                 curr_config['layers'][i]['type'] = rnn_optn
    #                                     # Append updated config
    #                                     arch_config_optns.append(curr_config)       
    
    # # # TEST - some factors version below
    # # # batch_size_optns = [5]    # IMPORTANT: Factors of total samples * (1 - val_split) for performance
    # # # CUDA_OUT_OF_MEMORY Error Debug
    # # batch_size_optns = [3]    # IMPORTANT: Factors of total samples * (1 - val_split) for performance
    # # epochs_optns = [10]
    # # loss_const_optns = [0.1]
    # # optimizer_optns = [(tf.keras.optimizers.RMSprop(), 0, 0.001, 'RMSprop')]     
    # # dropout_optns = [(0.0,0.0)]
    # # arch_config_optns = []   # Add variations of each bare config to official
    # # for config in bare_config_optns[:1]:
    # #     # TEMP - 2 tests @ once: 1 for f35 test, 2 for diff scalinng methods
    # #     for scale_optn in [False]:    # TODO: Be skeptical of scaling, b/c must scale at end before activation
    # #         for rnn_skip_optn in [True]:
    # #             for bias_rnn_optn in [False]:
    # #                 for bias_dense_optn in [False]:
    # #                     for dropout_optn in dropout_optns:      # For RNN only
    # #                         for bidir_optn in [True]:
    # #                             for bn_optn in [True]:   # For Dense only
    # #                                 for rnn_optn in ['LSTM']:
    # #                                     # Make a unique copy for each factor combo
    # #                                     curr_config = config.copy()
    # #                                     curr_config['scale'] = scale_optn
    # #                                     # if scale_optn = True: # Unneseccary block
    # #                                     #     curr_config['layers'].insert(0, {'type': 'Scale'})
    # #                                     #     curr_config['layers'].append({'type': 'Un-Scale'})
    # #                                     curr_config['rnn_res_cntn'] = rnn_skip_optn
    # #                                     curr_config['bias_rnn'] = bias_rnn_optn
    # #                                     curr_config['bias_dense'] = bias_dense_optn
    # #                                     curr_config['rnn_dropout'] = dropout_optn
    # #                                     curr_config['bidir'] = bidir_optn
    # #                                     curr_config['bn'] = bn_optn
    # #                                     if rnn_optn == 'LSTM':
    # #                                         for i, layer in enumerate(config['layers']):
    # #                                             if layer['type'] == 'RNN':
    # #                                                 curr_config['layers'][i]['type'] = rnn_optn
    # #                                     # Append updated config
    # #                                     arch_config_optns.append(curr_config) 

    batch_size_optns = train_configs['batch_size']
    epochs_optns = train_configs['epochs']
    loss_const_optns = train_configs['loss_const']
    optimizer_optns = train_configs['optimizer']

    combos = (len(batch_size_optns) * len(epochs_optns) * len(loss_const_optns) *
              len(optimizer_optns) * len(arch_config_optns))
    print('\nGS COMBOS:', combos, '\n')

    # Start where last left off, if applicable:
    if not restart:
        gs_iters_so_far = []
        # Search through grid search output directory
        base_dir = os.getcwd()
        os.chdir(gsres_path)
        if len(gs_id) > 0:
            gs_result_files = [f_name for f_name in os.listdir(os.getcwd()) if 
                               (f_name.endswith('txt') and f_name[0].isdigit() and f_name[0] == gs_id)]

            for f_name in gs_result_files:
                gs_iter = [int(token) for token in f_name.split('_') if token.isdigit()][1]  
                gs_iters_so_far.append(gs_iter)

        else:
            gs_result_files = [f_name for f_name in os.listdir(os.getcwd()) if 
                               (f_name.endswith('txt') and f_name[0] == 'r')]

            for f_name in gs_result_files:
                gs_iter = [int(token) for token in f_name.split('_') if token.isdigit()][0]  
                gs_iters_so_far.append(gs_iter)

        os.chdir(base_dir)
        # Know the last done job, if any jobs were done
        gs_iters_so_far.sort(reverse=True)
        last_done = gs_iters_so_far[0] if (len(gs_iters_so_far) > 0) else 0

        # Unsafe to user - do no grid search at all instead
        # if last_done == combos:
            # restart = True
        
        if last_done > 0:
            print('RESUMING GRID SEARCH AT ITERATION', last_done + 1, '\n')

    # Format grid search ID for filenames:
    if len(gs_id) > 0:
        gs_id += '_'

    # IMPORTANT: Grab HPs in order of mem-instensiveness, reduces chances OOM on grid search
    # Full grid search loop
    grid_results_val, grid_results, gs_iter = {}, {}, 1
    for batch_size in batch_size_optns:     # Batch size is tested first -> fast OOM-handling iterations
        for arch_config in arch_config_optns:
            for epochs in epochs_optns:
                for loss_const in loss_const_optns:
                    for opt, clip_val, lr, opt_name in optimizer_optns:

                        if restart or (gs_iter > last_done):

                            # CUSTOM TRAINING
                            # if not pc_run:
                            #     og_batch_size = batch_size
                            #     batch_size_per_replica = batch_size // 2
                            #     batch_size = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync

                            # print('DEBUG Batch Size in Grid Search:', batch_size)
                            # train_generator = my_generator(y1_train_files, y2_train_files, 
                            #         num_train,
                            #         batch_size=batch_size, train_seq=n_seq,
                            #         train_feat=n_feat, wdw_size=wdw_size, 
                            #         epsilon=epsilon, pad_len=max_sig_len)
                            # validation_generator = my_generator(y1_val_files, y2_val_files, 
                            #         num_val,
                            #         batch_size=batch_size, train_seq=n_seq,
                            #         train_feat=n_feat, wdw_size=wdw_size, 
                            #         epsilon=epsilon, pad_len=max_sig_len)
                            train_generator = fixed_data_generator(
                                    x_train_files, y1_train_files, y2_train_files, num_train,
                                    batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run)
                            validation_generator = fixed_data_generator(
                                    x_val_files, y1_val_files, y2_val_files, num_val,
                                    batch_size=batch_size, num_seq=n_seq, num_feat=n_feat, pc_run=pc_run)
                            # train_generator = SpgmGenerator(
                            #         x_train_files, y1_train_files, y2_train_files, num_train,
                            #         batch_size=batch_size, num_seq=n_seq, num_feat=n_feat)
                            # validation_generator = SpgmGenerator(
                            #         x_val_files, y1_val_files, y2_val_files, num_val,
                            #         batch_size=batch_size, num_seq=n_seq, num_feat=n_feat)

                            _, losses, val_losses = evaluate_source_sep(train_generator,
                                                                    validation_generator,
                                                                    # train_step_func, test_step_func,
                                                                    num_train, num_val,
                                                                    n_feat, n_seq, 
                                                                    batch_size, loss_const,
                                                                    epochs, opt, 
                                                                    patience=early_stop_pat,
                                                                    epsilon=epsilon,
                                                                    config=arch_config, pc_run=pc_run,
                                                                    t_mean=t_mean, t_std=t_std,
                                                                    grid_search_iter=gs_iter,
                                                                    gs_path=gsres_path,
                                                                    combos=combos,
                                                                    keras_fit=keras_fit)
                            
                            # CUSTOM TRAINING
                            # if not pc_run:
                            #     batch_size = og_batch_size

                            # Do multiple runs of eval_src_sep to avg over randomness?
                            curr_basic_loss = {'batch_size': batch_size, 
                                                'epochs': epochs, 'gamma': loss_const,
                                                'optimizer': opt_name, 'clip value': clip_val,
                                                'learning rate': lr, 'all_loss': losses}
                            curr_basic_val_loss = {'batch_size': batch_size, 
                                                'epochs': epochs, 'gamma': loss_const,
                                                'optimizer': opt_name, 'clip value': clip_val,
                                                'learning rate': lr, 'all_loss': val_losses}

                            # Not supported yet Python 3.4 (127x machines)
                            grid_results[losses[-1]] = {**arch_config, **curr_basic_loss}
                            grid_results_val[val_losses[-1]] = {**arch_config, **curr_basic_val_loss}

                            #grid_results[losses[-1]] = merge_two_dicts(arch_config, curr_basic_loss)
                            #grid_results_val[val_losses[-1]] = merge_two_dicts(arch_config, curr_basic_val_loss)

                            # WRITE these results to file too
                            pc_run_str = '' if pc_run else '_noPC'
                            with open(gsres_path + gs_id + 'result_' + str(gs_iter) + '_of_' + str(combos) + pc_run_str + '.txt', 'w') as w_fp:
                                w_fp.write(str(val_losses[-1]) + '\n')
                                w_fp.write('VAL LOSS ^\n')
                                w_fp.write(str(losses[-1]) + '\n')
                                w_fp.write('LOSS ^\n')
                                w_fp.write(json.dumps({**arch_config, **curr_basic_val_loss}) + '\n')
                                #w_fp.write(json.dumps(merge_two_dicts(arch_config, curr_basic_val_loss)) + '\n')
                                w_fp.write('VAL LOSS FACTORS ^\n')
                                w_fp.write(json.dumps({**arch_config, **curr_basic_loss}) + '\n')
                                #w_fp.write(json.dumps(merge_two_dicts(arch_config, curr_basic_loss)) + '\n')
                                w_fp.write('LOSS FACTORS^\n')


                            # grid_results[losses[-1]] = {'batch_size': batch_size, 
                            #                     'epochs': epochs, 'gamma': loss_const,
                            #                     'optimizer': opt, 'clip value': clip_val,
                            #                     'learning rate': lr, 'all_loss': losses}
                            # grid_results_val[val_losses[-1]] = {'batch_size': batch_size, 
                            #                     'epochs': epochs, 'gamma': loss_const,
                            #                     'optimizer': opt, 'clip value': clip_val,
                            #                     'learning rate': lr, 'all_loss': val_losses}

                            print('DONE W/ GRID-SEARCH ITER', (str(gs_iter) + '/' + str(combos) + ':\n'), 
                                'batch_size:', batch_size, 'epochs:', epochs, 'loss_const:', loss_const,
                                'optimizer:', opt, 'clipvalue:', clip_val, 'learn_rate:', lr, 
                                '\narch_config', arch_config, '\n')

                        gs_iter += 1

    return grid_results, grid_results_val


def analyze_grid_search_results(grid_res, grid_res_val):
    if not grid_res and not grid_res_val:
        print('No grid search results from current run -- haven\'t you finished this already? (look in files)')
    else:
        print('Grid Search Statistics of Current Run (full stats in file):')
        grid_res_items = [(key, val) if (key != np.nan) else (np.inf, val) 
                        for (key, val) in grid_res.items()]
        grid_res_val_items = [(key, val) if (key != np.nan) else (np.inf, val) 
                            for (key, val) in grid_res_val.items()]

        sorted_grid_results = sorted(grid_res_items)
        sorted_grid_val_results = sorted(grid_res_val_items)
        # summarize results
        print("Best Loss: %f using %s" % (sorted_grid_results[0][0], sorted_grid_results[0][1]))
        print("Best Val Loss: %f using %s" % (sorted_grid_val_results[0][0], sorted_grid_val_results[0][1]))
        print('--Losses--')
        print(sorted_grid_results)
        print('--Validation Losses--')
        print(sorted_grid_val_results)
        print('\n')


# MODEL INFERENCE FUNCTION
def infer(x, phases, wdw_size, model, loss_const, optimizer, seq_len, 
          n_feat, batch_size, epsilon, output_path, sr, orig_sig_type,
          config=None, t_mean=None, t_std=None, pc_run=False):
    # Must make new model, b/c Brahms spgm has different num timesteps
    x = np.expand_dims(x, axis=0)   # Give a samples dimension (1 sample)
    print('x shape to be predicted on:', x.shape)
    print('Inference Model:')
    model = make_model(n_feat, seq_len, loss_const=loss_const, optimizer=optimizer,
                       pre_trained_wgts=model.get_weights(), name='Inference Model',
                       epsilon=epsilon, config=config, t_mean=t_mean, t_std=t_std,
                       pc_run=pc_run)
    print(model.summary())

    # For small amts of input that fit in one batch: __call__ > predict - didn't work :/
    # clear_spgm, noise_spgm = model([x, x, x], batch_size=batch_size, training=False)
    result_spgms = model.predict(x, batch_size=batch_size)
    clear_spgm, noise_spgm = tf.split(result_spgms[:-1, :, :], num_or_size_splits=2, axis=0)
    # clear_spgm, noise_spgm = model.predict([x, x, x], batch_size=batch_size)
    # print('RAW PREDICTIONS -- Clear Spgm Shape:', clear_spgm.shape, 
    #       '\nNoise Spgm Shape:', noise_spgm.shape)
    clear_spgm = clear_spgm.numpy().reshape(-1, n_feat)
    noise_spgm = noise_spgm.numpy().reshape(-1, n_feat)
    # print('Clear Spgm Shape:', clear_spgm.shape)
    # print('Noise Spgm Shape:', noise_spgm.shape)
    # print('NaN in clear spgm?', True in np.isnan(clear_spgm))
    # print('NaN in noise spgm?', True in np.isnan(noise_spgm))
    # print('Clear spgm contents (timestep 1000):\n', clear_spgm[1000])

    if pc_run:
        plot_matrix(clear_spgm, name='clear_output_spgm', ylabel='Frequency (Hz)', 
                ratio=SPGM_BRAHMS_RATIO)
        plot_matrix(noise_spgm, name='noise_output_spgm', ylabel='Frequency (Hz)', 
                ratio=SPGM_BRAHMS_RATIO)

    synthetic_sig = make_synthetic_signal(clear_spgm, phases, wdw_size, 
                                          orig_sig_type, ova=True, debug=False)
    wavfile.write(output_path + 'restore.wav', sr, synthetic_sig)

    synthetic_sig = make_synthetic_signal(noise_spgm, phases, wdw_size, 
                                          orig_sig_type, ova=True, debug=False)
    wavfile.write(output_path + 'noise.wav', sr, synthetic_sig)


# BRAHMS RESTORATION FUNCTION (USES INFERENCE)
def restore_audio_file(output_path, model, wdw_size, epsilon, loss_const, optimizer,
                       test_filepath=None, test_sig=None, test_sr=None, 
                       config=None, t_mean=None, t_std=None, pc_run=False):
    if test_filepath:
        # Load in testing data - only use sr of test
        print('Restoring audio of file:', test_filepath)
        test_sr, test_sig = wavfile.read(test_filepath)
    test_sig_type = test_sig.dtype

    # Spectrogram creation - test. Only use phases of test
    test_spgm, test_phases = make_spectrogram(test_sig, wdw_size, epsilon, ova=True, debug=False)
    # test_spgm = test_spgm.astype('float32').T
    # test_spgm[test_spgm == 0] = epsilon
    # print('Test Phases:', test_phases)
    # new_test_phases = []
    # for phases in test_phases:
    #     phases[phases == 0] = epsilon
    #     new_test_phases.append(phases)
    # test_phases = new_test_phases
    # print('Numpy Epsilon:', np.finfo(np.float32).eps)
    # print('Our epsilon:', epsilon)
    # print('Test phases len:', len(test_phases))

    # NaN TEST
    print('Zero in test spgm:', 0 in test_spgm)
    print('Zero in test phases:', 0 in test_phases)
    print('NaN in test spgm:', True in np.isnan(test_spgm))
    print('NaN in test phases:', True in np.isnan(test_phases))
    # zero_in_ph, nan_in_ph = [], []
    # for phases in test_phases:
    #     zero_in_ph.append(0 in phases)
    #     nan_in_ph.append(True in np.isnan(phases))
    # # print('Zero in test phases:', True in zero_in_ph)
    # print('NaN in test phases:', True in nan_in_ph)

    print('Test (Brahms) Spgm Shape:', test_spgm.shape)    # (1272, 2049)
    test_feat = test_spgm.shape[1]
    test_seq = test_spgm.shape[0]
    test_batch_size = 1
    print('Inference Input Stats:')
    print('N Feat:', test_feat, 'Seq Len:', test_seq, 'Batch Size(1):', test_batch_size)

    print('Test Spgm contents (timestep 1000):\n', test_spgm[1000])

    infer(test_spgm, test_phases, wdw_size, model, loss_const=loss_const,
        optimizer=optimizer, seq_len=test_seq, 
        n_feat=test_feat, batch_size=test_batch_size, epsilon=epsilon,
        output_path=output_path, sr=test_sr, orig_sig_type=test_sig_type,
        config=config, t_mean=t_mean, t_std=t_std, pc_run=pc_run)

# https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
# Only want to predict one batch = 1 song, solutions:
#   - Online learning (train one batch/song at a time/epoch) - possible instability in training
#   - Predict all batches at once (impossible - only one orig Brahms)
#   - CHOSE Copy weights from fit network to a newly created network


# MAIN FUNCTION
def main():
    # PROGRAM ARGUMENTS #
    if len(sys.argv) < 3:
        print('\nUsage: dlnn_brahms_restore.py <mode> <PC> [-f] [-k] [-d] [gs_id]')
        print('Parameter Options:')
        print('Mode     t               - Train model, then restore brahms with model')
        print('         g               - Perform grid search (default: starts where last left off)')
        print('         r               - Restore brahms with last-trained model')
        print('PC       true            - Uses HPs for lower GPU-memory consumption (< 4GB)')
        print('         false           - Uses HPs for higher GPU-memory limit (PC HPs + nonPC HPs = total for now)')
        print('-f                       - (Optional) Force restart grid search (grid search mode) OR force random HPs (train mode)')
        print('-k                       - (Optional) Train with keras.fit() - don\'t know if it\'s supported')
        print('-d                       - (Optional) Distribute training (over 2 GPUs on 1 machine - 1 GPU default unless -k)')
        print('gs_id    <single digit>  - (Optional) grid search unique ID for running concurrently')
        print('\nTIP: Keep IDs different for PC/non-PC runs on same machine')
        sys.exit(1)


    mode = sys.argv[1] 
    pc_run = True if (sys.argv[2].lower() == 'true') else False
    dmged_piano_artificial_noise_mix = False
    test_on_synthetic = False
    wdw_size = PIANO_WDW_SIZE
    data_path = '../dlnn_data/'
    arch_config_path = '../config/'
    gs_output_path = '../output_grid_search/'
    recent_model_path = '../recent_model'
    infer_output_path = '../output_restore/'
    brahms_path = '../brahms.wav'
    
    keras_fit, dist_training = False, False
    # EMPERICALLY DERIVED HPs
    # Note: FROM PO-SEN PAPER - about loss_const
    #   Empirically, the value γ is in the range of 0.05∼0.2 in order
    #   to achieve SIR improvements and maintain SAR and SDR.
    # Orig batch size 5, orig loss const 0.05, orig clipval 0.9 - Colab
    # HP TEST
    # train_batch_size = 6 if pc_run else 4   # Batchsize is even to dist on 2 GPUs?
    train_batch_size = 3 if pc_run else 4   # Batchsize is even to dist on 2 GPUs?
    train_loss_const = 0.05
    train_epochs = 10
    # CURR FIX - Exploding gradient
    train_optimizer = tf.keras.optimizers.RMSprop(clipvalue=0.9, learning_rate=0.0001)
    training_arch_config = None

    epsilon, patience, val_split = 10 ** (-10), train_epochs, 0.25 #(1/3)

    # TRAINING DATA SPECIFIC CONSTANTS (Change when data changes) #
    MAX_SIG_LEN, TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = 3784581, 1847, 2049
    # TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = 1847, 2049
    TRAIN_MEAN, TRAIN_STD = 1728.2116672701493, 6450.4985228518635
    TOTAL_SMPLS = 61 # 60 # Performance: Make divisible by batch_size (actual total = 61) ... questionable

    # System-dependant changes
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    print("GPUs Available: ", gpus)
    if not pc_run:
        print("Setting memory growth on GPUs")
        for i in range(len(gpus)):
            tf.config.experimental.set_memory_growth(gpus[i], True)
    
    # BLOCK MUST BE MADE AFTER SET MEM GROWTH ABOVE
    # policy = None
    # MIXED PRECISION - only used on f35 (V100s)
    # policy = mixed_precision.Policy('mixed_float16')
    print('Tensorflow version:', tf.__version__)
    # Tells tf.function not to make graph, & run all ops eagerly (step debugging)
    # tf.config.run_functions_eagerly(True)             # For nightly release
    # tf.config.experimental_run_functions_eagerly(True)  # For TF 2.2 (non-nightly)
    print('Eager execution enabled? (default)', tf.executing_eagerly())

    # INFER ONLY
    if mode == 'r':
        try:
            model = tf.keras.models.load_model(recent_model_path)
        except:
            print('ERROR: No SaveModel to load from')
        else:
            print(model.summary())
            restore_audio_file(infer_output_path, model, wdw_size, epsilon,
                               train_loss_const, train_optimizer, 
                               brahms_path,
                               t_mean=TRAIN_MEAN, t_std=TRAIN_STD, pc_run=pc_run)
    else:
        # CUSTOM TRAINING HACK - FAILED
        # if not pc_run:
        #     try:
        #         tf.config.experimental.set_memory_growth(gpus[0], True)
        #         print('Set mem growth for GPU 1')
        #     except:
        #         print('ERROR: Couldn\'t set memory growth for GPU 1')

        #     try:
        #         tf.config.experimental.set_memory_growth(gpus[1], True)
        #         print('Set mem growth for GPU 2')
        #     except:
        #         print('ERROR: Couldn\'t set memory growth for GPU 2')

        # train_step_func, test_step_func = get_train_step_func(), get_test_step_func()
        # Mixed precision - f35
        # if policy is not None:
        # if not pc_run:
        #     mixed_precision.set_policy(policy)
        #     print('Compute dtype: %s' % policy.compute_dtype)
        #     print('Variable dtype: %s' % policy.variable_dtype)

        train_configs, arch_config_optns = get_hp_configs(arch_config_path, pc_run=pc_run)
        # print('First arch config optn after return:', arch_config_optns[0])

        # Load in train/validation data
        noise_piano_filepath_prefix = ((data_path + 'dmged_mix_numpy/mixed')
            if dmged_piano_artificial_noise_mix else (data_path + 'piano_noise_numpy/mixed'))
        piano_label_filepath_prefix = ((data_path + 'piano_source_numpy/piano')
            if dmged_piano_artificial_noise_mix else (data_path + 'piano_source_numpy/piano'))
        noise_label_filepath_prefix = ((data_path + 'dmged_noise_numpy/noise')
            if dmged_piano_artificial_noise_mix else (data_path + 'noise_source_numpy/noise'))

        # TRAIN & INFER
        if mode == 't':
            random_hps = False
            for arg_i in range(3, 6):
                if arg_i < len(sys.argv):
                    if sys.argv[arg_i] == '-f':
                        random_hps = True
                        print('\nTRAINING TO USE RANDOM (NON-EMPIRICALLY-OPTIMAL) HP\'S\n')
                    elif sys.argv[arg_i] == '-k':
                        keras_fit = True
                        print('\nTRAINING WITH KERAS FIT\n')
                    elif sys.argv[arg_i] == '-d':
                        dist_training = True
                        print('\nDISTRIBUTING TRAINING OVER 2 GPU\'S\n')

            # Define which files to grab for training. Shuffle regardless.
            # (Currently sample is to test on 1 synthetic sample (not Brahms))
            sample = test_on_synthetic
            # sample = False   # If taking less than total samples
            if sample:  # Used now for testing on synthetic data
                TOTAL_SMPLS += 1
                actual_samples = TOTAL_SMPLS - 1  # How many to leave out (1)
                sample_indices = list(range(TOTAL_SMPLS))
                random.shuffle(sample_indices)
                
                test_index = sample_indices[actual_samples]
                sample_indices = sample_indices[:actual_samples]
                test_piano = piano_label_filepath_prefix + str(test_index) + '.wav'
                test_noise = noise_label_filepath_prefix + str(test_index) + '.wav'
                test_sr, test_piano_sig = wavfile.read(test_piano)
                _, test_noise_sig = wavfile.read(test_noise)
                test_sig = test_piano_sig + test_noise_sig
            else:
                actual_samples = TOTAL_SMPLS
                sample_indices = list(range(TOTAL_SMPLS))
                # DEBUG
                random.shuffle(sample_indices)
            
            # if dmged_piano_artificial_noise_mix:
            #     x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            #     y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            #     y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            # else:
            x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            
            # # Temp - do to calc max len for padding - it's 3081621 (for youtube src data)
            # # it's 3784581 (for Spotify/Youtube Final Data)
            # # MAX_SIG_LEN = None
            # # for x_file in x_files:
            # #     _, sig = wavfile.read(x_file)
            # #     if MAX_SIG_LEN is None or len(sig) > MAX_SIG_LEN:
            # #         MAX_SIG_LEN = len(sig)
            # # print('MAX SIG LEN:', MAX_SIG_LEN)
            max_sig_len = MAX_SIG_LEN

            # Validation & Training Split
            indices = list(range(actual_samples))
            val_indices = indices[:math.ceil(actual_samples * val_split)]
            x_train_files = np.delete(x_files, val_indices)
            y1_train_files = np.delete(y1_files, val_indices)
            y2_train_files = np.delete(y2_files, val_indices)
            x_val_files = x_files[val_indices]
            y1_val_files = y1_files[val_indices]
            y2_val_files = y2_files[val_indices]
            num_train, num_val = len(y1_train_files), len(y1_val_files)

            # DEBUG PRINT
            # print('y1_train_files:', y1_train_files[:10])
            # print('y1_val_files:', y1_val_files[:10])
            # print('y2_train_files:', y2_train_files[:10])
            # print('y2_val_files:', y2_val_files[:10])

            # Temp - get training data dim (from dummy) (for model & data making)
            # max_len_sig = np.ones((MAX_SIG_LEN))
            # dummy_train_spgm = make_spectrogram(max_len_sig, wdw_size, 
            #                                     ova=True, debug=False)[0].astype('float32').T
            # TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = dummy_train_spgm.shape
            train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN

            # CUSTOM TRAINING Dist training needs a "global_batch_size"
            # if not pc_run:
            #     batch_size_per_replica = train_batch_size // 2
            #     train_batch_size = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync

            print('Train Input Stats:')
            print('N Feat:', train_feat, 'Seq Len:', train_seq, 'Batch Size:', train_batch_size)

            # Create data generators, evaluate model with them, and infer
            # train_generator = my_generator(y1_train_files, y2_train_files, num_train,
            #                     batch_size=train_batch_size, train_seq=train_seq,
            #                     train_feat=train_feat, wdw_size=wdw_size, 
            #                     epsilon=epsilon, pad_len=max_sig_len)
            # validation_generator = my_generator(y1_val_files, y2_val_files, num_val,
            #                     batch_size=train_batch_size, train_seq=train_seq,
            #                     train_feat=train_feat, wdw_size=wdw_size, 
            #                     epsilon=epsilon, pad_len=max_sig_len)
            train_generator = fixed_data_generator(x_train_files, y1_train_files, y2_train_files, num_train,
                                batch_size=train_batch_size, num_seq=train_seq, num_feat=train_feat, pc_run=pc_run,
                                dmged_piano_artificial_noise=dmged_piano_artificial_noise_mix,
                                pad_len=max_sig_len)
            validation_generator = fixed_data_generator(x_val_files, y1_val_files, y2_val_files, num_val,
                                batch_size=train_batch_size, num_seq=train_seq, num_feat=train_feat, pc_run=pc_run,
                                dmged_piano_artificial_noise=dmged_piano_artificial_noise_mix,
                                pad_len=max_sig_len)
            # train_generator = SpgmGenerator(x_train_files, y1_train_files, y2_train_files, num_train,
            #                     batch_size=train_batch_size, num_seq=train_seq, num_feat=train_feat)
            # validation_generator = SpgmGenerator(x_val_files, y1_val_files, y2_val_files, num_val,
            #                     batch_size=train_batch_size, num_seq=train_seq, num_feat=train_feat)

            # if pc_run:
            #     # TEST PC
            #     # with open(arch_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
            #     with open(arch_config_path + 'hp_arch_config_final.json') as hp_file:
            #         bare_config_optns = json.load(hp_file)['archs']
            # else:
            #     # with open(arch_config_path + 'hp_arch_config_largedim.json') as hp_file:
            #     with open(arch_config_path + 'hp_arch_config_final_no_pc.json') as hp_file:
            #         bare_config_optns = json.load(hp_file)['archs']

            # rnn_optns = ['RNN'] if pc_run else ['LSTM']
            # # TEST PC
            # # rnn_optns = ['LSTM'] if pc_run else ['LSTM'

            # dropout_optns = [(0.0,0.0)]
            # arch_config_optns = []   # Add variations of each bare config to official
            # for config in bare_config_optns[3:4]:  #[3:4]:    # rand base = #71 last
            #     for scale_optn in [True]:  
            #         for rnn_skip_optn in [True]:    # false last
            #             for bias_rnn_optn in [True]:
            #                 for bias_dense_optn in [True]:
            #                     for dropout_optn in dropout_optns:      # For RNN only
            #                         for bidir_optn in [False]:
            #                             for bn_optn in [True]:   # For Dense only # true last
            #                                 for rnn_optn in rnn_optns:
            #                                     # Important: skip bad output cases
            #                                     if bias_rnn_optn == False and bias_dense_optn == False:
            #                                         continue

            #                                     # Make a unique copy for each factor combo
            #                                     curr_config = config.copy()
            #                                     curr_config['scale'] = scale_optn
            #                                     curr_config['rnn_res_cntn'] = rnn_skip_optn
            #                                     curr_config['bias_rnn'] = bias_rnn_optn
            #                                     curr_config['bias_dense'] = bias_dense_optn
            #                                     curr_config['rnn_dropout'] = dropout_optn
            #                                     curr_config['bidir'] = bidir_optn
            #                                     curr_config['bn'] = bn_optn
            #                                     if rnn_optn == 'LSTM':
            #                                         for i, layer in enumerate(config['layers']):
            #                                             if layer['type'] == 'RNN':
            #                                                 curr_config['layers'][i]['type'] = rnn_optn
            #                                     # Append updated config
            #                                     arch_config_optns.append(curr_config) 

            # # DEBUG - normal HPs, RNN -> LSTM
            # # rnn_optns = ['LSTM']

            # # dropout_optns = [(0.0,0.0)]
            # # arch_config_optns = []   # Add variations of each bare config to official
            # # for config in bare_config_optns[0:1]:  #[3:4]:    # rand base = #71 last
            # #     for scale_optn in [False]:  
            # #         for rnn_skip_optn in [False]:    # false last
            # #             for bias_rnn_optn in [True]:
            # #                 for bias_dense_optn in [True]:
            # #                     for dropout_optn in dropout_optns:      # For RNN only
            # #                         for bidir_optn in [False]:
            # #                             for bn_optn in [False]:   # For Dense only # true last
            # #                                 for rnn_optn in rnn_optns:
            # #                                     # Important: skip bad output cases
            # #                                     if bias_rnn_optn == False and bias_dense_optn == False:
            # #                                         continue

            # #                                     # Make a unique copy for each factor combo
            # #                                     curr_config = config.copy()
            # #                                     curr_config['scale'] = scale_optn
            # #                                     curr_config['rnn_res_cntn'] = rnn_skip_optn
            # #                                     curr_config['bias_rnn'] = bias_rnn_optn
            # #                                     curr_config['bias_dense'] = bias_dense_optn
            # #                                     curr_config['rnn_dropout'] = dropout_optn
            # #                                     curr_config['bidir'] = bidir_optn
            # #                                     curr_config['bn'] = bn_optn
            # #                                     if rnn_optn == 'LSTM':
            # #                                         for i, layer in enumerate(config['layers']):
            # #                                             if layer['type'] == 'RNN':
            # #                                                 curr_config['layers'][i]['type'] = rnn_optn
            # #                                     # Append updated config
            # #                                     arch_config_optns.append(curr_config) 

            # REPL TEST - arch config, all config, optiizer config
            if random_hps:
                # Index into random arch config, and other random HPs
                arch_rand_index = random.randint(0, len(arch_config_optns)-1)
                # arch_rand_index = 0
                # print('ARCH RAND INDEX:', arch_rand_index)
                training_arch_config = arch_config_optns[arch_rand_index]
                # print('ARCH CONFIGS AT PREV & NEXT INDICES:\n', arch_config_optns[arch_rand_index-1], 
                #       '---\n', arch_config_optns[arch_rand_index+1])
                # print('In random HPs section, rand_index:', arch_rand_index)
                # print('FIRST ARCH CONFIG OPTION SHOULD HAVE RNN:\n', arch_config_optns[0])
                for hp, optns in train_configs.items():
                    # print('HP:', hp, 'OPTNS:', optns)
                    hp_rand_index = random.randint(0, len(optns)-1)
                    # hp_rand_index = 0
                    if hp == 'batch_size':
                        # print('BATCH SIZE RAND INDEX:', hp_rand_index)
                        train_batch_size = optns[hp_rand_index]
                    elif hp == 'epochs':
                        # print('EPOCHS RAND INDEX:', hp_rand_index)
                        train_epochs = optns[hp_rand_index]
                    elif hp == 'loss_const':
                        # print('LOSS CONST RAND INDEX:', hp_rand_index)
                        train_loss_const = optns[hp_rand_index]
                    elif hp == 'optimizer':
                        # hp_rand_index = 2
                        # print('OPT RAND INDEX:', hp_rand_index)
                        train_optimizer, clip_val, lr, opt_name = (
                            optns[hp_rand_index]
                        )

                # Early stop for random HPs
                # TIME TEST
                # patience = 4
                # training_arch_config = arch_config_optns[0]
                print('RANDOM TRAIN ARCH FOR USE:')
                print(training_arch_config)
                print('RANDOM TRAIN HPs FOR USE:')
                print('Batch size:', train_batch_size, 'Epochs:', train_epochs,
                      'Loss constant:', train_loss_const, 'Optimizer:', opt_name, 
                      'Clip value:', clip_val, 'Learning rate:', lr)
            # else:
            #     print('CONFIG:', training_arch_config)

            # TEMP - update for each unique dataset
            # train_mean, train_std = get_stats(y1_train_files, y2_train_files, num_train,
            #                                   train_seq=train_seq, train_feat=train_feat, 
            #                                   wdw_size=wdw_size, epsilon=epsilon, 
            #                                   pad_len=max_sig_len)
            # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
            # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20
            train_mean, train_std = TRAIN_MEAN, TRAIN_STD

            model, _, _ = evaluate_source_sep(train_generator, validation_generator, 
                                    # train_step_func, test_step_func, 
                                    num_train, num_val,
                                    n_feat=train_feat, n_seq=train_seq, 
                                    batch_size=train_batch_size, 
                                    loss_const=train_loss_const, epochs=train_epochs,
                                    optimizer=train_optimizer, patience=patience, epsilon=epsilon,
                                    recent_model_path=recent_model_path, pc_run=pc_run,
                                    config=training_arch_config, t_mean=train_mean, t_std=train_std,
                                    keras_fit=keras_fit)
            
            if sample:
                restore_audio_file(infer_output_path, model, wdw_size, epsilon, 
                                train_loss_const, train_optimizer, 
                                test_filepath=None, 
                                test_sig=test_sig, test_sr=test_sr,
                                config=training_arch_config, t_mean=train_mean, t_std=train_std, pc_run=pc_run)
            else:
                restore_audio_file(infer_output_path, model, wdw_size, epsilon,
                                train_loss_const, train_optimizer, 
                                test_filepath=brahms_path,
                                config=training_arch_config, t_mean=train_mean, t_std=train_std, pc_run=pc_run)

        # GRID SEARCH
        elif mode == 'g':
            # Dennis - think of good metrics (my loss is obvious first start)
            # - To use SKLearn GridSearchCV, make class inheriting from kerasregressor that
            #       1) uses a generator in fit
            #       2) uses my custom loss as the score - no need

            restart, gs_id = False, ''
            for arg_i in range(3, 7):
                if arg_i < len(sys.argv):
                    if sys.argv[arg_i] == '-f':
                        restart = True
                        print('\nGRID SEARCH TO FORCE RESTART\n')
                    elif sys.argv[arg_i] == '-k':
                        keras_fit = True
                        print('\nTRAINING WITH KERAS FIT\n')
                    elif sys.argv[arg_i] == '-d':
                        dist_training = True
                        print('\nDISTRIBUTING TRAINING OVER 2 GPU\'S\n')
                    elif sys.argv[arg_i].isdigit() and len(sys.argv[arg_i]) == 1:
                        gs_id = sys.argv[arg_i]
                        print('GRID SEARCH ID:', gs_id, '\n')

            early_stop_pat = 5
            # Define which files to grab for training. Shuffle regardless.
            actual_samples = TOTAL_SMPLS
            sample_indices = list(range(TOTAL_SMPLS))
            random.shuffle(sample_indices)

            x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.wav')
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.wav')
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.wav')
                        for i in sample_indices])
            # max_sig_len = MAX_SIG_LEN

            # Validation & Training Split
            indices = list(range(actual_samples))
            val_indices = indices[:math.ceil(actual_samples * val_split)]
            x_train_files = np.delete(x_files, val_indices)
            y1_train_files = np.delete(y1_files, val_indices)
            y2_train_files = np.delete(y2_files, val_indices)
            x_val_files = x_files[val_indices]
            y1_val_files = y1_files[val_indices]
            y2_val_files = y2_files[val_indices]

            # Temp - get training data dim (from dummy) (for model & data making)
            # max_len_sig = np.ones((MAX_SIG_LEN))
            # dummy_train_spgm = make_spectrogram(max_len_sig, wdw_size, 
            #                                     ova=True, debug=False)[0].astype('float32').T
            # TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = dummy_train_spgm.shape
            train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN
            print('Grid Search Input Stats:')
            print('N Feat:', train_feat, 'Seq Len:', train_seq)

            # TEMP - update for each unique dataset
            # num_train, num_val = len(y1_train_files), len(y1_val_files)
            # train_mean, train_std = get_stats(y1_train_files, y2_train_files, num_train,
            #                                   train_seq=train_seq, train_feat=train_feat, 
            #                                   wdw_size=wdw_size, epsilon=epsilon, 
            #                                   pad_len=max_sig_len)
            # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
            # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20
            train_mean, train_std = TRAIN_MEAN, TRAIN_STD

            grid_res, grid_res_val = grid_search(x_train_files, y1_train_files, y2_train_files,
                                        x_val_files, y1_val_files, y2_val_files,
                                        # train_step_func, test_step_func,
                                        n_feat=train_feat, n_seq=train_seq,
                                        # wdw_size=wdw_size, 
                                        epsilon=epsilon,
                                        # max_sig_len=max_sig_len, 
                                        t_mean=train_mean, t_std=train_std,
                                        train_configs=train_configs,
                                        arch_config_optns=arch_config_optns,
                                        # arch_config_path=arch_config_path, 
                                        gsres_path=gs_output_path,
                                        early_stop_pat=early_stop_pat, 
                                        pc_run=pc_run, gs_id=gs_id, 
                                        restart=restart,
                                        keras_fit=keras_fit)
            
            analyze_grid_search_results(grid_res, grid_res_val)


if __name__ == '__main__':
    main()

