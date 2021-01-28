import os
from scipy.io import wavfile
# from sklearn.preprocessing import normalize
from . import nmf
from ..audio_data_processing import *

# Idea - rank-1 approx = take avg of the pos. mag. spectrogram NOT the signal
def make_basis_vector(waveform,  wf_type, wf_sr, num, wdw_size, ova=False, avg=False, debug=False):
    # TEMP
    # waveform = waveform.astype('float64')
    # # Convert to mono signal (avg left & right channels) if needed
    # sig = np.average(waveform, axis=-1) if (len(waveform.shape) > 1) else waveform

    # New
    # # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
    # amp_thresh = max(abs(sig)) * 0.01 # New
    # # amp_thresh = max(sig) * 0.01
    # while abs(sig[0]) < amp_thresh: # New
    # # while sig[0] < amp_thresh:
    #     sig = sig[1:]
    # while abs(sig[-1]) < amp_thresh: # New
    # # while sig[-1] < amp_thresh:
    #     sig = sig[:-1]
    
    # Write trimmed piano note signals to WAV - check if trim is good
    # if i == 40 or i == 43 or i == 46 or i < 10:
    #     wavfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/trimmed_notes/trimmed_note_' + str(i) + '.wav', 
    #                   note_sr, sig.astype(orig_note_sig_type))

    # if debug:
    #     print('In make bv')
    # TEMP
    spectrogram, phases = make_spectrogram(waveform, wdw_size, ova=ova)
    # spectrogram, _ = make_spectrogram(waveform, wdw_size, EPSILON, ova=ova)
    # spectrogram, _ = make_spectrogram(sig, wdw_size, EPSILON, ova=ova)
    if debug:
        print('In make piano bv - V of piano note @ 1st timestep:', spectrogram[0][:10])
    if avg:
        # # OLD WAY - Averaged the signal, not a spectrogram
        # num_sgmts = math.floor(len(sig) / wdw_size) # Including incomplete windows throws off averaging
        # all_sgmts = np.array([sig[i * wdw_size: (i + 1) * wdw_size] for i in range(num_sgmts)])
        # sgmt = np.mean(all_sgmts, axis=0)
        # if ova:
        #     sgmt *= np.hanning(wdw_size)
        # # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
        # return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]
        basis_vector = np.mean(spectrogram, axis=0) # Actually the bv that makes best rank-1 approx. of V (piano note spectrogram) - the avg
        # TEMP
        # norm_bv = basis_vector / np.linalg.norm(basis_vector)
        # # norm_bv = normalize(basis_vector[:,np.newaxis], axis=0).ravel()
        # basis_vector = norm_bv * 1000000
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

        basis_vector = spectrogram[nmf.BEST_PIANO_BV_SGMT, :].copy()

    if debug:
        # WRITE BASIS VECTOR REPEATED OUT TO A WAV FILE
        if wf_sr > 0 and avg and ova:
            avg_spgm = np.array([basis_vector for _ in range(spectrogram.shape[1])]).T
            avg_sig = make_synthetic_signal(avg_spgm, phases, wdw_size, wf_type, ova=ova, debug=False)
            wavfile.write('brahms_restore_ml/nmf/avged_ova_notes/avged_ova_note_' + str(num) + '.wav', 
                          wf_sr, avg_sig.astype(wf_type))
        print('In make piano bv - BV of piano note:', basis_vector[:10])
        # print('Shape of note spectrogram:', spectrogram.shape)
        # print('Shape of basis vector made from this:', basis_vector.shape, '\n')

    return basis_vector


# def make_basis_vector_old(waveform, wdw_size, ova=False, avg=False):
#     if avg:
#         num_sgmts = math.floor(len(waveform) / wdw_size) # Including incomplete windows throws off averaging
#         all_sgmts = np.array([waveform[i * wdw_size: (i + 1) * wdw_size] for i in range(num_sgmts)])
#         sgmt = np.mean(all_sgmts, axis=0)
    
#     else:
#         sgmt = waveform[(nmf.BEST_PIANO_BV_SGMT - 1) * wdw_size: nmf.BEST_PIANO_BV_SGMT * wdw_size].copy()  # BEST_PIANO_BV_SGMT is naturally-indexed
#         # print("Type of elem in piano note sig:", type(sgmt[0]))
#         if len(sgmt) != wdw_size:
#                 deficit = wdw_size - len(sgmt)
#                 sgmt = np.pad(sgmt, (deficit, 0), mode='constant')
        
#     if ova:
#         sgmt *= np.hanning(wdw_size)
#     # Positive frequencies in ascending order, including 0Hz and the middle frequency (pos & neg)
#     return np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1].copy()


# Time/#segments is irrelevant to # of basis vectors made (so maximize)
def make_noise_basis_vectors(num, wdw_size, ova=False, eq=False, debug=False, precise_noise=False, eq_thresh=800000,
                             start=0, stop=25):
    # TEMP
    # # Old - now source noise from accuracte audacity picking
    _, brahms_sig = wavfile.read('brahms.wav')

    # # Convert to mono signal (avg left & right channels) 
    # brahms_sig = np.array([((x[0] + x[1]) / 2) for x in brahms_sig.astype('float64')])

    # # Precise noise is pointless -> b/c we want to mximize what we draw noise from
    # noise_sig_len = 2 if ova else 1 # The 1 is an educated guess, 2 is empircally derived
    # # Second 2 hits solid noise - based on Audacity waveform (22nd wdw if sr=44100, wdw_size=4096)
    # noise_sgmt_num = math.ceil((STD_SR_HZ * 2.2) / wdw_size)    # 2.2 seconds (24rd window to (not including) 26th window)
    # if precise_noise:
    #     noise_sig = brahms_sig[(noise_sgmt_num - 1) * wdw_size: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
    # else:
    # # All noise from beginning of clip
    #     noise_sig = brahms_sig[: (noise_sgmt_num + noise_sig_len - 1) * wdw_size] 
    # newer
    noise_sig = brahms_sig[(start * wdw_size): (stop * wdw_size)].copy()

    # Equalize noise bv's? - no doesnt make sense to
    # if eq:  # Make it louder
    #     while np.max(np.abs(sig)) < sig_thresh:
    #         sig *= 1.1

    # # New
    # real_currdir = os.path.dirname(os.path.realpath(__file__))
    # _, noise_sig = wavfile.read(real_currdir + '/../../brahms_noise_izotope_rx.wav')

    print('\n----Making Noise Spectrogram--\n')
    # TEMP
    spectrogram, _ = make_spectrogram(noise_sig, wdw_size, ova=ova, debug=debug)
    # spectrogram, _ = make_spectrogram(noise_sig, wdw_size, EPSILON, ova=ova, debug=debug)
    print('\n----Learning Noise Basis Vectors--\n')
    # TEMP
    spectrogram = spectrogram.T
    _, noise_basis_vectors = nmf.nmf_learn(spectrogram, num, debug=debug)
    # noise_basis_vectors, _ = nmf.extended_nmf(spectrogram, num, debug=debug)
    noise_basis_vectors = noise_basis_vectors.T     # Get out of NMF-context orientation

    # TEMP
    # # NEW - normalize noise basis vectors
    # for i in range(noise_basis_vectors.shape[0]):
    #     # basis_vector = noise_basis_vectors[i].copy()
    #     # norm_bv = basis_vector / np.linalg.norm(basis_vector)
    #     # # norm_bv = normalize(basis_vector[:,np.newaxis], axis=0).ravel()
    #     # basis_vector = norm_bv * 1000000
    #     noise_basis_vectors[i] /= np.linalg.norm(noise_basis_vectors[i])
    #     noise_basis_vectors[i] *= 1000000

    if debug:
        print('Making noise basis vectors. Noise signal:', noise_sig[:10])
        print('Noise Spectogram V (& Sum):', spectrogram.shape, np.sum(spectrogram), spectrogram.T[0][:10])
        print('First learned Noise Basis Vector of W:', noise_basis_vectors.shape, noise_basis_vectors[0][:10])
        plot_matrix(noise_basis_vectors, 'Learned Noise Basis Vectors', 'frequency', 'k', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)

    # if False:  # Make louder # if eq:
    #     new_bvs = []
    #     for bv in noise_basis_vectors:
    #         while np.max(bv[1:]) < bv_thresh:
    #             bv *= 1.1
    #         new_bvs.append(bv)
    #     noise_basis_vectors = np.array(new_bvs)

    # TEMP
    return list(noise_basis_vectors)
    # return noise_basis_vectors

def make_basis_vectors(wdw_size, filepath, ova=False, avg=False, mary_flag=False, eq=False, eq_thresh=800000, debug=False):
    # TEMP
    # basis_vectors, sorted_notes = np.empty((nmf.NUM_PIANO_NOTES, (wdw_size//2)+1)), []
    basis_vectors, sorted_notes = [], []
    # Read in ordered piano notes, IMPORTANT: if I use fund freqs, they may be imprecise
    real_currdir = os.path.dirname(os.path.realpath(__file__))
    with open(real_currdir + '/piano_notes_and_fund_freqs.csv', 'r') as notes_f:
        for i, line in enumerate(notes_f.readlines()):
            sorted_notes.append(line.split(',')[0])

    # TEMP - all under indented
    with open(filepath, 'w') as bv_f:
        # all_notes_ff_wav yielding a list of filename strings, need to sort it by note
        base_dir = os.getcwd()
        os.chdir(real_currdir + '/all_notes_ff_wav')
        unsorted_audio_files = [x for x in os.listdir(os.getcwd()) if x.endswith('wav')]
        sorted_file_names = ['Piano.ff.' + x + '.wav' for x in sorted_notes]
        audio_files = sorted(unsorted_audio_files, key=lambda x: sorted_file_names.index(x))

        if debug:
            print('Piano notes:', sorted_notes[:10], sorted_notes[-10:])
            print('Audio files:', audio_files[:2], audio_files[-2:])

        if mary_flag:
            start, stop = nmf.MARY_START_INDEX, nmf.MARY_STOP_INDEX
        else:
            start, stop = 0, len(audio_files)
        
        for i in range(start, stop):
            audio_file = audio_files[i]
            note_sr, note_sig = wavfile.read(audio_file) 
            note_sig_type = note_sig.dtype  
            # TEMP
            note_sig = note_sig.astype('float64')
            # Convert to mono signal (avg left & right channels) if needed
            note_sig = np.average(note_sig, axis=-1) if (len(note_sig.shape) > 1) else note_sig

            # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
            amp_thresh = max(note_sig) * 0.01
            while note_sig[0] < amp_thresh:
                note_sig = note_sig[1:]
            while note_sig[-1] < amp_thresh:
                note_sig = note_sig[:-1]
            basis_vector = make_basis_vector(note_sig, note_sig_type, note_sr, i, wdw_size, ova=ova, avg=avg, debug=debug)
            # TEMP
            # basis_vectors[i] = basis_vector
            basis_vectors.append(basis_vector)
            bv_f.write(','.join([str(x) for x in basis_vector]) + '\n')

    os.chdir(base_dir)
    # TEMP
    # np.save(filepath, basis_vectors)

    if debug:
        print('Done making bvs. First basis vector:', basis_vectors[0][:10])

    return basis_vectors


# We don't save bvs w/ noise anymnore, 
# we just calc noise and pop it on top of restored-from-file piano bvs

# W LOGIC
# Basis vectors in essence are the "best" dft of a sound w/ constant pitch (distinct freq signature)
def get_basis_vectors(wdw_size, ova=False, mary=False, noise=False, avg=False, debug=False, precise_noise=False, eq=False, 
                      num_noise=0, noise_start=6, noise_stop=25, randomize='None'):
    if randomize == 'Piano':
        # Piano basis vectors are random for semisupervised learn piano
        basis_vectors = np.random.rand(nmf.NUM_PIANO_NOTES, (wdw_size//2) + 1)
    else:
        # Save/load basis vectors (w/o noise) to/from numpy files
        real_currdir = os.path.dirname(os.path.realpath(__file__))
        filepath = real_currdir + '/np_saves_bv/basis_vectors'
        if mary:
            filepath += '_mary'
        if ova:
            filepath += '_ova'
        if avg:
            filepath += '_avg'
        if eq:
            filepath += '_eqsig' # '_eqmeansig' '_eqmediansig'
        # TEMP
        filepath += '.csv'
        # filepath += '.npy'

        try:
            # TEMP
            # basis_vectors = np.load(filepath)
            # print('FILE FOUND - READ IN BASIS VECTORS:', filepath)
            with open(filepath, 'r') as bv_f:
                print('FILE FOUND - READING IN BASIS VECTORS:', filepath)
                basis_vectors = [[float(sub) for sub in string.split(',')] for string in bv_f.readlines()]
        except:
            print('FILE NOT FOUND - MAKING BASIS VECTORS:', filepath)
            basis_vectors = make_basis_vectors(wdw_size, filepath, ova=ova, avg=avg, mary_flag=mary, eq=eq, eq_thresh=800000, debug=debug)

        if debug:
            # TEMP
            # print('Basis Vectors Shape:', basis_vectors.shape)
            print('Basis Vectors Length:', len(basis_vectors))
            print('GOTTEN 1st Basis Vector:', basis_vectors[0][:10])

    # Make and add noise bv's
    if noise:
        if randomize == 'Noise':
            noise_basis_vectors = np.random.rand(num_noise, (wdw_size//2) + 1)
        else:
            noise_basis_vectors = make_noise_basis_vectors(num_noise, wdw_size, ova=ova, eq=eq, debug=debug, 
                                                        precise_noise=precise_noise, eq_thresh=800000,
                                                        start=noise_start, stop=noise_stop)
        # TEMP
        # # Noise first, piano notes second
        # basis_vectors = np.concatenate((noise_basis_vectors, basis_vectors))
        # if debug:
        #     print('Noise Basis Vectors Shape:', noise_basis_vectors.shape)
        #     print('Basis Vectors Shape After putting together:', basis_vectors.shape)
        basis_vectors = (noise_basis_vectors + basis_vectors)
        if debug:
            print('Noise Basis Vectors Length:', len(noise_basis_vectors))
            print('Basis Vectors Length After putting together:', len(basis_vectors))
    # TEMP
    basis_vectors = np.array(basis_vectors)

    # # Basis vectors in column orientation for NMF readability
    # basis_vectors = basis_vectors.T
    if debug:
        print('Shape of built basis vectors:', basis_vectors.shape)
        plot_matrix(basis_vectors, 'Made Basis Vectors', 'frequency', 'k', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)

    return basis_vectors
