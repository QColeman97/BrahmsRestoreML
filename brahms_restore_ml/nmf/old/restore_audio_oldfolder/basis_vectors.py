# import sys
# import os
# User friendly below
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append('/Users/quinnmc/Desktop/BMSThesisContent/BMSThesis/BrahmsRestoreDLNN/audio_data_processing')
from audio_data_processing import *

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
            wavfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/avged_ova_notes/avged_ova_note_' + str(num) + '.wav', 
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
    _, brahms_sig = wavfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms.wav')
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
    with open('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/piano_notes_and_fund_freqs.csv', 'r') as notes_f:
        for line in notes_f.readlines():
            sorted_notes.append(line.split(',')[0])
    
    with open(filepath, 'w') as bv_f:
        base_dir = os.getcwd()
        os.chdir('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/all_notes_ff_wav')
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
                # TODO: Aggressive manual override for fussy basis vectors (85, 86, 88)
                if i == 85 or i == 87:
                    sig *= 1.5
                elif i == 84:   # A7 note that is distinct to ear, noisiest?
                    sig *= 2.0

            # Write trimmed piano note signals to WAV - check if trim is good
            # if i == 40 or i == 43 or i == 46 or i < 10:
            #     wavfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/trimmed_notes/trimmed_note_' + str(i) + '.wav', 
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
    filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/csv_saves_bv/basis_vectors'
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
