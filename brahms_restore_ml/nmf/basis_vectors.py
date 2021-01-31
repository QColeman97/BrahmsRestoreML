import os
from scipy.io import wavfile
# from sklearn.preprocessing import normalize
from . import nmf
from ..audio_data_processing import *

# Idea - rank-1 approx = take avg of the pos. mag. spectrogram NOT the signal
def make_basis_vector(waveform, wf_type, wf_sr, num, wdw_size, ova=False, avg=False, debug=False, write_path=None):
    waveform = waveform.astype('float64')
    # Convert to mono signal (avg left & right channels) if needed
    sig = np.average(waveform, axis=-1) if (len(waveform.shape) > 1) else waveform
    # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
    amp_thresh = np.amax(np.abs(sig)) * 0.01
    while np.abs(sig[0]) < amp_thresh:
        sig = sig[1:]
    while np.abs(sig[-1]) < amp_thresh:
        sig = sig[:-1]
    if write_path is not None:  # before sig ruined by make_spectrogram
        # optional - write trimmed piano note signals to WAV - check if trim is good
        wavfile.write(write_path + 'trimmed_note_' + str(num) + '.wav', wf_sr, sig.astype(wf_type))

    spectrogram, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=ova)
    if debug:
        print('Making piano bv', num, '- V of piano note @ 1st timestep:', spectrogram[0][:10])
    if avg:
        # Actually the bv that makes best rank-1 approx. of V (piano note spectrogram) - the avg
        basis_vector = np.mean(spectrogram, axis=0)
        # TEMP
        # norm_bv = basis_vector / np.linalg.norm(basis_vector)
        # # norm_bv = normalize(basis_vector[:,np.newaxis], axis=0).ravel()
        # basis_vector = norm_bv * 1000000
    else: # unused branch
        basis_vector = spectrogram[nmf.BEST_PIANO_BV_SGMT]
    if debug:
        print('Making piano bv', num, '- BV of piano note:', basis_vector[:10])
        if num == 30:   # random choice
            plot_matrix(basis_vector, '30th basis vector', 'null', 'frequencies', show=True)
    if write_path is not None:   
        # write basis vector magnitude-repeated out to wav file
        bv_spgm = np.array([basis_vector for _ in range(spectrogram.shape[0])])
        bv_sig = make_synthetic_signal(bv_spgm, phases, wdw_size, wf_type, ova=ova, debug=False)
        wavfile.write(write_path + 'bv_' + str(num) + '.wav', wf_sr, bv_sig.astype(wf_type))

    return basis_vector


# Time/#segments is irrelevant to # of basis vectors made (so maximize)
def make_noise_basis_vectors(num, wdw_size, ova=False, eq=False, debug=False, precise_noise=False, eq_thresh=800000,
                             start=0, stop=25):
    real_currdir = os.path.dirname(os.path.realpath(__file__))
    _, noise_sig = wavfile.read(real_currdir + '/../../brahms_noise_izotope_rx.wav')
    if debug:   # before make_spectrogram ruins sig
        print('Making noise basis vectors. Noise signal:', noise_sig[:10])

    print('\n----Making Noise Spectrogram--\n')
    spectrogram, _ = make_spectrogram(noise_sig, wdw_size, EPSILON, ova=ova, debug=debug)
    print('\n----Learning Noise Basis Vectors--\n')
    # Transpose V from natural orientation to NMF-liking orientation
    spectrogram = spectrogram.T
    # TEMP
    # _, noise_basis_vectors = nmf.nmf_learn(spectrogram, num, debug=debug)
    noise_basis_vectors, _ = nmf.extended_nmf(spectrogram, num, debug=debug)
    noise_basis_vectors = noise_basis_vectors.T     # Get out of NMF-context orientation

    # # NEW - normalize noise basis vectors
    # for i in range(noise_basis_vectors.shape[0]):
    #     noise_basis_vectors[i] /= np.linalg.norm(noise_basis_vectors[i])
    #     noise_basis_vectors[i] *= 1000000
    if debug:
        print('Noise Spectogram V (& Sum):', spectrogram.shape, np.sum(spectrogram), spectrogram.T[0][:10])
        print('First learned Noise Basis Vector of W:', noise_basis_vectors.shape, noise_basis_vectors[0][:10])
        plot_matrix(noise_basis_vectors, 'Learned Noise Basis Vectors', 'frequency', 'k', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)
    
    return noise_basis_vectors

def make_basis_vectors(wdw_size, filepath, ova=False, avg=False, mary_flag=False, eq=False, eq_thresh=800000, debug=False):
    basis_vectors = np.empty((nmf.NUM_MARY_PIANO_NOTES if mary_flag else nmf.NUM_PIANO_NOTES, (wdw_size//2)+1))
    sorted_notes = []
    # Read in ordered piano notes, don't use fund. freqs
    real_currdir = os.path.dirname(os.path.realpath(__file__))
    with open(real_currdir + '/piano_notes_and_fund_freqs.csv', 'r') as notes_f:
        for i, line in enumerate(notes_f.readlines()):
            sorted_notes.append(line.split(',')[0])

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
    for i, note_index in enumerate(range(start, stop)):
        audio_file = audio_files[note_index]
        note_sr, note_sig = wavfile.read(audio_file) 
        note_sig_type = note_sig.dtype  
        basis_vector = make_basis_vector(note_sig, note_sig_type, note_sr, note_index, wdw_size, ova=ova, avg=avg, debug=debug)
        basis_vectors[i] = basis_vector

    os.chdir(base_dir)
    np.save(filepath, basis_vectors)
    if debug:
        print('Done making bvs.', basis_vectors.shape, 'First basis vector:', basis_vectors[0][:10])

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
        filepath += '.npy'

        try:
            basis_vectors = np.load(filepath)
            print('FILE FOUND - READ IN BASIS VECTORS:', filepath)
        except:
            print('FILE NOT FOUND - MAKING BASIS VECTORS:', filepath)
            basis_vectors = make_basis_vectors(wdw_size, filepath, ova=ova, avg=avg, mary_flag=mary, eq=eq, eq_thresh=800000, debug=debug)

        if debug:
            print('Basis Vectors Shape:', basis_vectors.shape)  
            print('GOTTEN 1st Basis Vector:', basis_vectors[0][:10])

    # Make and add noise bv's
    if noise:
        if randomize == 'Noise':
            noise_basis_vectors = np.random.rand(num_noise, (wdw_size//2) + 1)
        else:
            noise_basis_vectors = make_noise_basis_vectors(num_noise, wdw_size, ova=ova, eq=eq, debug=debug, 
                                                        precise_noise=precise_noise, eq_thresh=800000,
                                                        start=noise_start, stop=noise_stop)
        # Noise first, piano notes second
        basis_vectors = np.concatenate((noise_basis_vectors, basis_vectors))
        if debug:
            print('Noise Basis Vectors Shape:', noise_basis_vectors.shape)
            print('Basis Vectors Shape After putting together:', basis_vectors.shape)
    if debug:
        print('Shape of built basis vectors:', basis_vectors.shape)
        plot_matrix(basis_vectors, 'Made Basis Vectors', 'frequency', 'k', ratio=nmf.BASIS_VECTOR_FULL_RATIO, show=True)

    return basis_vectors
