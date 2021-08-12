# Audio Data Pre/Post-Processing (DSP) Functions for ML Models
# & some supplementary functions using these

from unittest import signals
from scipy.io import wavfile
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# DSP CONSTANTS:
STD_SR_HZ = 44100
PIANO_WDW_SIZE = 4096
# Resolution (Windows per second) = STD_SR_HZ / PIANO_WDW_SIZE
SPGM_BRAHMS_RATIO = 0.08    # for pyplot
EPSILON = 10 ** (-10)
BRAHMS_SILENCE_WDWS = 15

def plot_signal(sig, orig_type, name, plot_path=None, show=False):
    # Make plottable mono signal
    if isinstance(sig[0], np.ndarray):
        sig = np.average(sig, axis=-1).astype(orig_type)

    t = np.arange(0, len(sig))
    fig, ax = plt.subplots()
    ax.plot(t, sig)

    name_suffix = ''
    if orig_type == 'int16':
        name_suffix += ' (16-bit PCM)'
    ax.set(xlabel='time (samples)', ylabel='amplitude',
           title=name + name_suffix)
    if plot_path is not None:
        fig.savefig(plot_path)
    if show:
        plt.show()
    return ax.get_aspect()

# Supports matrices & arrays
def plot_matrix(matrix, name, xlabel, ylabel, ratio=0.08, show=False, true_dim=False, plot_path=None):    
    def frequency_in_hz(x, pos):
        return '%.2f Hz' % ((x * STD_SR_HZ)/PIANO_WDW_SIZE)
    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines
    formatter = FuncFormatter(frequency_in_hz)

    fig, ax = plt.subplots()
    if xlabel == 'frequency':
        ax.xaxis.set_major_formatter(formatter)
    elif ylabel == 'frequency':
        ax.yaxis.set_major_formatter(formatter)
    ax.title.set_text(name)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if len(matrix.shape) == 1:
        n_rows, n_cols = matrix.shape[0], 1
    else:
        n_rows, n_cols = matrix.shape
    if true_dim:
        _ = ax.imshow(np.log(matrix), aspect=0.5, origin='lower', interpolation='none')
    else:
        ratio = n_cols / n_rows
        print('RATIO FOR', n_rows, 'x', str(n_cols)+':', ratio)
        extent = [-0.5, n_cols-0.5, -0.5, n_rows-0.5]
        # Make frequency show up to ~4,300 Hz ~ C8 fund. freq.
        if xlabel == 'frequency':
            extent = [-0.5, (n_cols//5)-0.5, -0.5, n_rows-0.5]
            ratio /= 5
        elif ylabel == 'frequency':
            extent = [-0.5, n_cols-0.5, -0.5, (n_rows//5)-0.5]
            ratio *= 5
        try:
            _ = ax.imshow(np.log(matrix), 
                aspect=ratio,
                origin='lower',
                extent=extent,
                # vmin= 0,      # Remove for CONSTANT COLORS
                # vmax = 10,    # Remove for CONSTANT COLORS
                interpolation='none')
        except TypeError as e:
            print('CAUGHT ERROR IN PLOT_MATRIX', e)

    if show:
        plt.show()
    else:
        if plot_path is None:
            plt.savefig(name + '.png')
        else:
            plt.savefig(plot_path)


# SIGNAL -> SPECTROGRAM
def sig_length_to_spgm_shape(n_smpls, wdw_size=PIANO_WDW_SIZE, hop_size_divisor=2, ova=True):
    hop_size = ((wdw_size // hop_size_divisor) 
                if (ova and n_smpls >= (wdw_size + (wdw_size // hop_size_divisor))) 
                else wdw_size) 
    # If OVA is active, only works for hop_size = wdw_size // 2
    num_sgmts = (math.ceil(n_smpls / hop_size) - 1) if ova else math.ceil(n_smpls / wdw_size)
    num_feats = (wdw_size // 2) + 1     # Nothing to do w/ hop size - result of DSP
    return (num_sgmts, num_feats)

# Returns pos. magnitude & phases of a DFT, given a signal segment
def signal_to_pos_fft(sgmt, wdw_size, ova=False, debug_flag=False):
    if len(sgmt) != wdw_size:
        deficit = wdw_size - len(sgmt)
        sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

    if debug_flag:
        print('Original segment (len =', str(len(sgmt))+'):\n', sgmt[:5])

    if ova: # Perform lobing on ends of segment
        sgmt *= np.hanning(wdw_size)
    fft = np.fft.fft(sgmt)
    phases_fft = np.angle(fft)
    mag_fft = np.abs(fft)
    pos_phases_fft = phases_fft[: (wdw_size // 2) + 1]
    pos_mag_fft = mag_fft[: (wdw_size // 2) + 1]
    # # From docs - for an even number of input points, A[n/2] represents both positive and negative Nyquist frequency (summed?)

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


# Construct stft (not actually spectrogram b/c viewing isn't use case)
def make_spectrogram(signal, wdw_size, epsilon, ova=False, debug=False, hop_size_divisor=2):
    # Pre-processing steps
    # If 8-bit PCM, convert to 16-bit PCM (signed to unsigned) - specific for Brahms (not training data)
    if signal.dtype == 'uint8' or signal.dtype == 'float32':
        signal = convert_wav_format_up(signal, print_sum=True)
    # Data Granularity Check
    if signal.dtype != 'float64':
        signal = signal.astype('float64')
    # Convert stereo signal to mono signal, 2 channels -> 1 channel
    if isinstance(signal[0], np.ndarray):
        signal = np.average(signal, axis=-1)

    num_spls = len(signal)
    if debug:
       print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', signal[(wdw_size // 2): (wdw_size // 2) + 20]) if num_spls > 20 else print('ORIGINAL SIG (FLOAT64) BEFORE SPGM:\n', signal)

    # Hop size is half-length of window if OVA (if length sufficient), else it's just window length
    hop_size = ((wdw_size // hop_size_divisor) 
                if (ova and num_spls >= (wdw_size + (wdw_size // hop_size_divisor))) 
                else wdw_size)
    num_sgmts, sgmt_len = sig_length_to_spgm_shape(num_spls, wdw_size=wdw_size, 
                                        hop_size_divisor=hop_size_divisor, ova=ova)
    # TEMP - old
    # # Number of segments depends on if OVA implemented    
    # # Works for hop_size = wdw_size // 2
    # num_sgmts = (math.ceil(num_spls / hop_size) - 1) if ova else math.ceil(num_spls / wdw_size)
    # sgmt_len = (wdw_size // 2) + 1 # Nothing to do w/ hop size - result of DSP
    if debug:
        print('Num of Samples:', num_spls)
        print('Hop size:', hop_size)
        print('Num segments:', num_sgmts)
    
    spectrogram, pos_phases = np.empty((num_sgmts, sgmt_len)), np.empty((num_sgmts, sgmt_len))
    for i in range(num_sgmts):
        # Make a copy of slice, so we don't go into un-allocated mem when padding
        sgmt = signal[i * hop_size: (i * hop_size) + wdw_size].copy()
        
        debug_flag = ((i == 0) or (i == 1)) if debug else False
        pos_mag_fft, pos_phases_fft = signal_to_pos_fft(sgmt, wdw_size, ova=ova, debug_flag=debug_flag)
        spectrogram[i] = pos_mag_fft
        pos_phases[i] = pos_phases_fft
    # Replace NaNs and 0s w/ epsilon
    spectrogram, pos_phases = np.nan_to_num(spectrogram), np.nan_to_num(pos_phases)
    spectrogram[spectrogram == 0], pos_phases[pos_phases == 0] = epsilon, epsilon
    if debug:
        # plot_matrix(spectrogram, 'Built Spectrogram', 'frequency', 'time segments', ratio=SPGM_BRAHMS_RATIO, show=True)
        # diff - make pretty looking
        plot_matrix(spectrogram.T[:, BRAHMS_SILENCE_WDWS:-BRAHMS_SILENCE_WDWS], 'Built Spectrogram', 'frequency', 'time segments', ratio=SPGM_BRAHMS_RATIO, show=True)

    return spectrogram, pos_phases


# SPECTROGRAM -> SIGNAL
# Returns real signal, given positive magnitude & phases of a DFT
def pos_fft_to_signal(pos_mag_fft, pos_phases_fft, wdw_size, ova=False, 
                      end_sig=None, debug_flag=False, hop_size_divisor=2):
    # Append the mirrors of the synthetic magnitudes and phases to themselves
    neg_mag_fft = np.flip(pos_mag_fft[1: wdw_size // 2], axis=0)
    mag_fft = np.concatenate((pos_mag_fft, neg_mag_fft))

    # The mirror is negative b/c phases are flipped in sign (angle), but not for magnitudes
    neg_phases_fft = np.flip(pos_phases_fft[1: wdw_size // 2] * -1, axis=0)
    phases_fft = np.concatenate((pos_phases_fft, neg_phases_fft))

    # Multiply this magnitude fft w/ phases fft
    fft = mag_fft * np.exp(1j*phases_fft)
    # Do ifft on the fft -> waveform
    ifft = np.fft.ifft(fft)
    # This is safe, b/c ifft imag values are 0 for this data
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
        ova_sgmt, end_sgmt = sgmt_halves[0], sgmt_halves[1]
        # # for hop size != wdw_size//2 support...      unfinished
        # sgmt_portions = np.split(synthetic_sgmt, hop_size_divisor)
        # if hop_size_divisor == 2:
        #     ova_sgmt, end_sgmt = sgmt_portions[0], sgmt_portions[1] # First, then second half
        # elif hop_size_divisor == 4:
        #     # print('sgmt portion 1:', sgmt_portions[0])
        #     ova_sgmt, end_sgmt = sgmt_portions[0], np.concatenate((sgmt_portions[1], sgmt_portions[2], sgmt_portions[3]))

        if end_sig is None:
            # end_sig = np.zeros((wdw_size // 2))
            end_sig = np.zeros((wdw_size // hop_size_divisor))

        end_sum = ova_sgmt + end_sig    # Numpy element-wise addition of OVA parts
        synthetic_sgmt = np.concatenate((end_sum, end_sgmt))    # Concatenate OVA part with trailing end part

        if debug_flag:
            print('ova_sgmt (len =', len(ova_sgmt), '):\n', ova_sgmt[-10:], 
                  '\nend_sgmt (len =', len(end_sgmt), '):\n', end_sgmt[-10:], 
                  '\nend_sig (len =', len(end_sig), '):\n', end_sig[-10:], 
                  '\nend_sum (len =', len(end_sum), '):\n', end_sum[-10:])
    return synthetic_sgmt


# Construct synthetic waveform (actually takes in stft)
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_type, ova=False, debug=False, hop_size_divisor=2):
    synthetic_spgm = synthetic_spgm.astype('float64')   # Post-processing step from NN

    num_sgmts = synthetic_spgm.shape[0]
    # If both noise and piano in spgm, reuse phases in synthesis
    if num_sgmts != len(phases):   
        phases = np.concatenate((phases, phases))
    
    hop_size = wdw_size // hop_size_divisor
    # Support for different hop sizes
    synthetic_sig_len = int(((num_sgmts/hop_size_divisor) + (1-(1/hop_size_divisor))) * wdw_size) if ova else num_sgmts * wdw_size
    # print('Synthetic Sig Len FULL:', synthetic_sig_len)
    synthetic_sig = np.empty((synthetic_sig_len))
    for i in range(num_sgmts):
        ova_index = (i * (wdw_size // hop_size_divisor)) if ova else (i * wdw_size)
        debug_flag = (i == 0 or i == 1) if debug else False

        if ova and (i > 0):
            end_half_sgmt = synthetic_sig[ova_index: ova_index + hop_size]
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                       end_sig=end_half_sgmt, hop_size_divisor=hop_size_divisor)
        else:
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                   wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                   hop_size_divisor=hop_size_divisor)
 
        synthetic_sig[ova_index: ova_index + wdw_size] = synthetic_sgmt
            
        if debug_flag:
            print('End of synth sig:', synthetic_sig[-20:])

    if debug:
        (print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', synthetic_sig[(wdw_size // 2): (wdw_size // 2) + 20]) 
        # (print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', synthetic_sig[50:100]) 
            if len(synthetic_sig) > 20 else 
                print('SYNTHETIC SIG (FLOAT64) AFTER SPGM:\n', synthetic_sig))

    if (orig_type == 'uint8'):  # Handle 8-bit PCM (unsigned)
        synthetic_sig = convert_wav_format_down(synthetic_sig, print_sum=True)
    elif (orig_type == 'float32'):
        synthetic_sig = convert_wav_format_down(synthetic_sig, to_bit_depth='float32', print_sum=True)
    else:
        # Safety measure: prevent overflow by clipping, assume other WAV format types are int (int16)
        if np.amax(np.abs(synthetic_sig)) > np.iinfo(orig_type).max:    # amax doing max of flattened array
            print('Warning: signal values greater than original signal\'s capacity. Losing data.')
        synthetic_sig = np.clip(synthetic_sig, np.iinfo(orig_type).min, np.iinfo(orig_type).max) 
        # Accuracy measure: round floats before converting to original type
        synthetic_sig = np.around(synthetic_sig).astype(orig_type)

    return synthetic_sig


# WAV FORMAT CONVERSION FUNCTIONS
# Convert bit-depth & range to that of highest numeric range WAV format fed into system (16-bit int PCM)
# Only supports conversion from 'uint8' and 'float32'
# Only supports conversion to highest range yet - 16-bit int PCM ('int16')
def convert_wav_format_up(sig, to_bit_depth='int16', print_sum=False):
    if to_bit_depth == 'int16':
        if sig.dtype == 'uint8':
            save_sig = sig
            sig = sig.astype('int16')
            sig -= 128
            sig *= 256
            if print_sum:
                print('SUM OF BRAHMS CONVERTED TO 16-BIT INT PCM:', np.sum(sig), 'BEFORE CONV:', np.sum(save_sig))
        elif sig.dtype == 'float32':
            sig = sig * 32767    # 32768 is full negative range but not positive, stay safe
            sig = np.around(sig).astype('int16')
        else:
            if print_sum:
                print('SUM OF SIG (16-BIT INT PCM):', np.sum(sig))
    return sig

def convert_wav_format_down(sig, to_bit_depth='uint8', safe=True, print_sum=False):
    if to_bit_depth == 'uint8':
        if sig.dtype != 'int16' and safe:
            # Safety measure: prevent overflow by clipping
            if np.amax(np.abs(sig)) > np.iinfo('int16').max:    # amax doing max of flattened array
                print('Warning: signal values greater than 16-bit PCM WAV capacity. Losing data.')
            sig = np.clip(sig, np.iinfo('int16').min, np.iinfo('int16').max)
            sig = np.around(sig).astype('int16')
        sig = sig / 256
        sig = sig.astype('int16')
        sig += 128
    elif to_bit_depth == 'float32':
        sig = sig / 32767    # clip after scaling to preserve accuracy
        if safe:
            # Safety measure: prevent overflow by clipping
            # if np.amax(np.abs(sig)) > np.finfo('float32').max:    # amax doing max of flattened array
            #     print('Warning: signal values greater than float32 capacity. Losing data.')
            # sig = np.clip(sig, np.finfo('float32').min, np.finfo('float32').max)
            if np.amax(np.abs(sig)) > 1.0:    # amax doing max of flattened array
                print('Warning: signal values greater than 32-bit floating-point WAV capacity. Losing data.')
            sig = np.clip(sig, -1.0, 1.0)
    sig = sig.astype(to_bit_depth) 
    if print_sum and to_bit_depth == 'uint8':
        print('SUM OF BRAHMS CONVERTED BACK TO 8-BIT INT PCM:', np.sum(sig))
    return sig


# To get all noise part in brahms, rule of thumb = 25 windows
def write_partial_sig(sig, wdw_size, start_index, end_index, out_filepath, sig_sr):
    sig = sig[(start_index * wdw_size): (end_index * wdw_size)]
    wavfile.write(out_filepath, sig_sr, sig)


def reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=False, segment=False, write_file=False, debug=False):
    orig_sig_type = sig.dtype
    debug_plot_path = os.getcwd() + '/brahms_restore_ml/nmf/plot_pics/'
    plot_signal(sig, orig_sig_type, name='Piano Recording Signal', plot_path=(debug_plot_path + 'piano_waveform.png'), show=True)

    print('\n--Making Signal Spectrogram--\n')
    if segment:
        # ARG FOR TESTING - SO WE CAN FIND NO VOICE SEGMENT
        # no_voice_sig = sig[(77 * wdw_size):]# + (wdw_size // 2):] 
        # out_filepath = 'novoice_' + out_filepath
        spectrogram, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=ova, debug=debug)
    else:
        print('\n--Making Signal Spectrogram--\n')
        spectrogram, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=ova, debug=debug)

    # plot_matrix(spectrogram[15:-15].T, name='Original Recording', 
    #             xlabel='time (4096-sample windows)', ylabel='frequency', plot_path=(debug_plot_path + 'piano_spectrogram.png'), show=False)
    plot_matrix(spectrogram.T, name='Piano Recording Spectrogram', 
                xlabel='time (4096-sample windows)', ylabel='frequency', plot_path=(debug_plot_path + 'piano_spectrogram.png'), show=False)
    
    print('\n--Making Synthetic Signal--\n')
    synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, orig_sig_type, ova=ova, debug=debug)
    if write_file:
        eval_smpl_path = os.getcwd() + '/brahms_restore_ml/nmf/eval_wav_smpls/'
        eval_start = len(synthetic_sig) // 4
        wavfile.write(eval_smpl_path + 'orig_smpl.wav', sig_sr, synthetic_sig[eval_start: (eval_start + 500000)])

        wavfile.write(out_filepath, sig_sr, synthetic_sig)

    return synthetic_sig