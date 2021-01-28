import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
from scipy.linalg import svd


STD_SR_HZ = 44100

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
    plt.savefig('../kl_euc_div_comp/' + name + '.png')

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
    spectrogram = np.clip(spectrogram, np.finfo('float32').min, np.finfo('float32').max)
    # Spectrogram matrix w/ correct orientation (orig orient.)
    spectrogram = spectrogram.astype('float32')     # T Needed? (don't think so, only for plotting)
    #if debug:
        #plot_matrix(spectrogram, name='Built Spectrogram', ylabel='Frequency (Hz)', ratio=SPGM_BRAHMS_RATIO)

    return spectrogram, pos_phases


def nmf(input_matrix, k, learn_iter=100):
    W = np.random.rand(input_matrix.shape[0], k)
    H = np.random.rand(k, input_matrix.shape[1])
    ones = np.ones(input_matrix.shape) # so dimensions match W transpose dot w/ V

    for learn_i in range(learn_iter):
                
        H *= ((W.T @ (input_matrix / (W @ H))) / (W.T @ ones))
                        
        W *= (((input_matrix / (W @ H)) @ H.T) / (ones @ H.T))

    return W, H


def make_basis_vector(waveform, wdw_size, epsilon, ova=False):
    spectrogram, _ = make_spectrogram(waveform, wdw_size, epsilon, ova=ova)
    # Spectrogram in features = time orientation
    spectrogram = spectrogram.T

    W, H = nmf(spectrogram, 1)
    print('W shape:', W.shape, 'H shape:', H.shape)

    basis_vector = W[:, 0]
    print('--- Basis vector shape (NMF):', basis_vector.shape)

    return basis_vector


# Idea - rank-1 approx = take avg of the pos. mag. spectrogram NOT the signal
def make_rank_1_approx(waveform, wdw_size, epsilon, ova=False, debug=False):

    spectrogram, _ = make_spectrogram(waveform, wdw_size, epsilon, ova=ova)
    # Spectrogram in features = time orientation
    spectrogram = spectrogram.T

    # Equivalent to W of rank-1 NMF
    basis_vector = np.mean(spectrogram, axis=1)
    print('--- Rank-1 approx. shape (Avged Freq Spectrum):', basis_vector.shape)

    return basis_vector


def make_left_singular_vector(waveform, wdw_size, epsilon, ova=False):

    spectrogram, _ = make_spectrogram(waveform, wdw_size, epsilon, ova=ova)
    # Spectrogram in features = time orientation
    spectrogram = spectrogram.T

    # Perform rank-1 SVD (LSA = SVD for dimensionality reduction)
    U, s, VT = svd(spectrogram)
    # Make sigma into diag matrix
    S = np.zeros((spectrogram.shape[0], spectrogram.shape[1]))
    diagonals = np.diag(s)
    S[:spectrogram.shape[1], :spectrogram.shape[1]] = np.diag(s)

    print('U shape:', U.shape, 'S shape:', S.shape, 'V^T.shape:', VT.shape)

    left_singular_vector = U[:,0]   # First column of U is the most important eigen-frequency-spectrum
    print('--- Left singular vector shape (SVD):', left_singular_vector.shape)

    return left_singular_vector


def main():

    wdw_size, ova, avg = 4096, True, True
    epsilon = 10 ** (-10)
    
    audio_file = '../all_notes_ff_wav/Piano.ff.C4.wav'
    note_sr, stereo_sig = wavfile.read(audio_file)
    orig_note_sig_type = stereo_sig.dtype
    # Convert to mono signal (avg left & right channels) 
    sig = np.average(stereo_sig.astype('float64'), axis=-1)

    # Trim out the silence on both ends of sig
    amp_thresh = max(sig) * 0.01
    while sig[0] < amp_thresh:
        sig = sig[1:]
    while sig[-1] < amp_thresh:
        sig = sig[:-1]

    rank_1_approx = make_rank_1_approx(sig, wdw_size, epsilon, ova=ova)
    plot_matrix(np.expand_dims(rank_1_approx, axis=0), 'rank_1_approx', 'frequency (Hz)', ratio=0.001)

    left_singular_vector = make_left_singular_vector(sig, wdw_size, epsilon, ova=ova)
    plot_matrix(np.expand_dims(left_singular_vector, axis=0), 'left_singular_vector', 'frequency (Hz)', ratio=0.001)

    basis_vector = make_basis_vector(sig, wdw_size, epsilon, ova=ova)
    plot_matrix(np.expand_dims(basis_vector, axis=0), 'basis_vector', 'frequency (Hz)', ratio=0.001)
    
    bv_to_r1_diff = basis_vector - rank_1_approx
    bv_to_r1_mse = np.mean(np.square(basis_vector - rank_1_approx))
    bv_to_r1_mae = np.mean(np.abs(basis_vector - rank_1_approx))
    print('\nBasis Vector to Rank-1 Approx. Diff:\n', bv_to_r1_diff, '\nMSE:', bv_to_r1_mse, '\nMAE:', bv_to_r1_mae)

    lsv_to_r1_diff = left_singular_vector - rank_1_approx
    lsv_to_r1_mse = np.mean(np.square(left_singular_vector - rank_1_approx))
    lsv_to_r1_mae = np.mean(np.abs(left_singular_vector - rank_1_approx))
    print('\nLeft Singular Vector to Rank-1 Approx. Diff:\n', lsv_to_r1_diff, '\nMSE:', lsv_to_r1_mse, '\nMAE:', lsv_to_r1_mae)

    # B/c we can
    bv_to_lsv_diff = basis_vector - left_singular_vector
    bv_to_lsv_mse = np.mean(np.square(basis_vector - left_singular_vector))
    bv_to_lsv_mae = np.mean(np.abs(basis_vector - left_singular_vector))
    print('\nBasis Vector to Left Singular Vector Diff:\n', bv_to_lsv_diff, '\nMSE:', bv_to_lsv_mse, '\nMAE:', bv_to_lsv_mae)


if __name__ == '__main__':
    main()