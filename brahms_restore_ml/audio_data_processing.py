# Audio Data Pre/Post-Processing (DSP) Functions for ML Models

from scipy.io import wavfile
# import scipy.signal as sg
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import math
# from copy import deepcopy

# DSP CONSTANTS:
STD_SR_HZ = 44100
PIANO_WDW_SIZE = 4096
# Resolution (Windows per second) = STD_SR_HZ / PIANO_WDW_SIZE
SPGM_BRAHMS_RATIO = 0.08
EPSILON = 10 ** (-10)

def plot_matrix(matrix, name, xlabel, ylabel, ratio=0.08, show=False, true_dim=False):
    n_rows, n_cols = matrix.shape
    
    def frequency_in_hz(x, pos):
        # return '%.1f Hz' % x
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

    n_rows, n_cols = matrix.shape
    if true_dim:
        _ = ax.imshow(np.log(matrix), aspect=0.5, origin='lower')
    else:
        # ratio = 1
        # if xlabel == 'frequency' or xlabel == 'time segments':
        #     ratio = 10
        #     if ylabel == 'k':
        #         ratio *= 5
        # else:
        #     ratio = 0.1
        #     if xlabel == 'k':
        #         ratio /= 2
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

        _ = ax.imshow(np.log(matrix), 
            aspect=ratio,
            origin='lower',
            extent=extent)
    # else:
    #     ax.title.set_text(name)
    #     ax.set_ylabel(ylabel)
    #     _ = ax.imshow(matrix, extent=[0, n_cols, n_rows, 0])
    #     fig.tight_layout()
    #     # bottom, top = plt.ylim()
    #     # print('Bottom:', bottom, 'Top:', top)
    #     plt.ylim(n_rows, 0.0)   # Crop an axis (to ~double the piano frequency max)
    #     ax.set_aspect(ratio)    # Set a visually nice ratio
    plt.show() if show else plt.savefig(name + '.png')


# SIGNAL -> SPECTROGRAM
# Returns pos. magnitude & phases of a DFT, given a signal segment
def signal_to_pos_fft(sgmt, wdw_size, ova=False, debug_flag=False):
    if len(sgmt) != wdw_size:
        deficit = wdw_size - len(sgmt)
        sgmt = np.pad(sgmt, (0,deficit))  # pads on right side (good b/c end of signal), (deficit, 0) pads on left side # , mode='constant')

    if debug_flag:
        print('Original segment (len =', str(len(sgmt))+'):\n', sgmt[:5])

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


# (actually returns only a STFT)
def make_spectrogram(signal, wdw_size, epsilon, ova=False, debug=False, hop_size_divisor=2):
    # Pre-processing steps
    # If 8-bit PCM, convert to 16-bit PCM (signed to unsigned) - specific for Brahms (not training data)
    if signal.dtype == 'uint8':
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

    # Hop size is half-length of window if OVA, else it's just window length (if length sufficient)
    hop_size = ((wdw_size // hop_size_divisor) 
                if (ova and num_spls >= (wdw_size + (wdw_size // hop_size_divisor))) 
                else wdw_size)
    # Number of segments depends on if OVA implemented    
    # Works for hop_size = wdw_size // 2
    num_sgmts = (math.ceil(num_spls / hop_size) - 1) if ova else math.ceil(num_spls / wdw_size)
    sgmt_len = (wdw_size // 2) + 1 # Nothing to do w/ hop size - result of DSP

    if debug:
        print('Num of Samples:', num_spls)
        print('Hop size:', hop_size)
        print('Num segments:', num_sgmts)
    
    spectrogram, pos_phases = np.empty((num_sgmts, sgmt_len)), np.empty((num_sgmts, sgmt_len))
    for i in range(num_sgmts):
        # Slicing a numpy array makes a view, so explicit copy
        sgmt = signal[i * hop_size: (i * hop_size) + wdw_size].copy()
        
        debug_flag = ((i == 0) or (i == 1)) if debug else False
        pos_mag_fft, pos_phases_fft = signal_to_pos_fft(sgmt, wdw_size, ova=ova, debug_flag=debug_flag)
        spectrogram[i] = pos_mag_fft
        pos_phases[i] = pos_phases_fft

    # Replace NaNs and 0s w/ epsilon
    spectrogram, pos_phases = np.nan_to_num(spectrogram), np.nan_to_num(pos_phases)
    spectrogram[spectrogram == 0], pos_phases[pos_phases == 0] = epsilon, epsilon

    if debug:
        plot_matrix(spectrogram, 'Built Spectrogram', 'time segments', 'frequency', ratio=SPGM_BRAHMS_RATIO, show=True)

    return spectrogram, pos_phases


# SPECTROGRAM -> SIGNAL
# Returns real signal, given positive magnitude & phases of a DFT
def pos_fft_to_signal(pos_mag_fft, pos_phases_fft, wdw_size, ova=False, 
                      end_sig=None, debug_flag=False, hop_size_divisor=2):
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
        # sgmt_halves = np.split(synthetic_sgmt, 2)
        # ova_sgmt, end_sgmt = sgmt_halves[0], sgmt_halves[1] # First, then second half
        sgmt_portions = np.split(synthetic_sgmt, hop_size_divisor)
        if hop_size_divisor == 2:
            ova_sgmt, end_sgmt = sgmt_portions[0], sgmt_portions[1] # First, then second half
        elif hop_size_divisor == 4:
            # print('sgmt portion 1:', sgmt_portions[0])
            ova_sgmt, end_sgmt = sgmt_portions[0], np.concatenate((sgmt_portions[1], sgmt_portions[2], sgmt_portions[3]))

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


# Construct synthetic waveform
def make_synthetic_signal(synthetic_spgm, phases, wdw_size, orig_type, ova=False, debug=False, hop_size_divisor=2):
    # Post-processing step from NN
    synthetic_spgm = synthetic_spgm.astype('float64')

    num_sgmts = synthetic_spgm.shape[0]#[1]
    # print('Num sgmts:', num_sgmts)
    # If both noise and piano in spgm, reuse phases in synthesis
    if num_sgmts != len(phases):   
        # phases += phases
        phases = np.concatenate((phases, phases))
    
    hop_size = wdw_size // hop_size_divisor

    # Support for different hop sizes
    # synthetic_sig_len = int(((num_sgmts / 2) + 0.5) * wdw_size) if ova else num_sgmts * wdw_size
    synthetic_sig_len = int(((num_sgmts / hop_size_divisor) + (1-(1/hop_size_divisor))) * wdw_size) if ova else num_sgmts * wdw_size
    # print('Synthetic Sig Len FULL (wdw_sizes):', synthetic_sig_len / wdw_size)
    print('Synthetic Sig Len FULL:', synthetic_sig_len)
    # print('Putting', num_sgmts, 'sgmts into signal')
    synthetic_sig = np.empty((synthetic_sig_len))     # RAM too much use way
    # print('Synth sig mem location:', aid(synthetic_sig))
    # synthetic_sig = None
    for i in range(num_sgmts):
        ova_index = i * (wdw_size // hop_size_divisor)
        debug_flag = (i == 0 or i == 1) if debug else False

        # Do overlap-add operations if ova (but only if list already has >= 1 element)
        # if ova and len(synthetic_sig):
        if ova and (i > 0):
            # end_half_sgmt = synthetic_sig[-(wdw_size // 2):].copy()
            # end_half_sgmt = synthetic_sig[(i*wdw_size) - (wdw_size//2): i * wdw_size].copy()
            # end_half_sgmt = synthetic_sig[ova_index: ova_index + (wdw_size//2)].copy()
            end_half_sgmt = synthetic_sig[ova_index: ova_index + hop_size].copy()
            
            # print(synthetic_sig_len, '=?', i * wdw_size)
            # print('End Half Sgmt Len:', len(end_half_sgmt))
            # print('End Half Sgmt mem location:', aid(end_half_sgmt))
            
            synthetic_sgmt = pos_fft_to_signal(pos_mag_fft=synthetic_spgm[i], pos_phases_fft=phases[i], 
                                                       wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                       end_sig=end_half_sgmt, hop_size_divisor=hop_size_divisor)
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
                                                   wdw_size=wdw_size, ova=ova, debug_flag=debug_flag,
                                                   hop_size_divisor=hop_size_divisor)
            # synthetic_sig += synthetic_sgmt
            # synthetic_sig[i * wdw_size: (i+1) * wdw_size] = synthetic_sgmt
        # print('Synth sig len:', len(synthetic_sig), 'OVA index:', ova_index)
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
        # # Safety measure: prevent overflow
        # synthetic_sig = np.clip(synthetic_sig, np.iinfo('int16').min, np.iinfo('int16').max)
        # synthetic_sig = np.around(synthetic_sig).astype('int16')
        # # Accuracy measure: round floats before converting to int    
        # synthetic_sig = convert_sig_16bit_to_8bit(synthetic_sig)
        synthetic_sig = convert_wav_format_down(synthetic_sig, print_sum=True)
    else:
        # Safety measure: prevent overflow
        synthetic_sig = (np.clip(synthetic_sig, np.finfo(orig_type).min, np.finfo(orig_type).max) 
                if orig_type == 'float32' else 
            np.clip(synthetic_sig, np.iinfo(orig_type).min, np.iinfo(orig_type).max))  
        # Accuracy measure: round floats before converting to original type
        synthetic_sig = np.around(synthetic_sig).astype(orig_type)

    return synthetic_sig


# WAV FORMAT CONVERSION FUNCTIONS
# Convert bit-depth & range to that of highest quality WAV format fed into system (16-bit int PCM)
# Can only handle max of 16-bit int PCM
def convert_wav_format_up(sig, to_bit_depth='int16', print_sum=False):
    if to_bit_depth == 'int16':
        if sig.dtype == 'uint8':
            save_sig = sig
            sig = sig.astype('int16')
            sig -= 128
            sig *= 256
            if print_sum:
                print('SUM OF BRAHMS CONVERTED TO 16-BIT INT PCM:', np.sum(sig), 'BEFORE CONV:', np.sum(save_sig))
        elif print_sum:
            print('SUM OF SIG (16-BIT INT PCM):', np.sum(sig))
    return sig

# Can only handle max of 16-bit int PCM
def convert_wav_format_down(sig, to_bit_depth='uint8', safe=True, print_sum=False):
    if to_bit_depth == 'uint8':
        if sig.dtype != 'int16' and safe:
            if np.amax(np.abs(sig)) > 32768:    # amax doing max of flattened array
                print('WARNING: signal values greater than int16 capacity. Losing data.')
            sig = np.clip(sig, -32768, 32767)
            sig = np.around(sig).astype('int16')
        sig = sig / 256
        sig = sig.astype('int16')
        sig += 128
        sig = sig.astype('uint8')
    if print_sum:
        print('SUM OF BRAHMS CONVERTED BACK TO 8-BIT INT PCM:', np.sum(sig))
    return sig


# To get all noise part in brahms, rule of thumb = 25 windows
def write_partial_sig(sig, wdw_size, start_index, end_index, out_filepath, sig_sr):
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