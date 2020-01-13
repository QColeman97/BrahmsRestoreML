# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for restore_audio.py.

# Things to test:
# - sig reconstruction
# - basis vectors correct
# - 

import unittest
from restore_audio import *

STD_SR_HZ = 44100
MARY_SR_HZ = 16000
PIANO_WDW_SIZE = 4096 # 32768 # 16384 # 8192 # 4096 # 2048
DEBUG_WDW_SIZE = 4
RES = STD_SR_HZ / PIANO_WDW_SIZE
BEST_WDW_NUM = 5
# Activation Matrix (H) Learning Part
MAX_LEARN_ITER = 100
BASIS_VECTOR_FULL_RATIO = 0.01
BASIS_VECTOR_MARY_RATIO = 0.001
SPGM_BRAHMS_RATIO = 0.08
SPGM_MARY_RATIO = 0.008

# Spectrogram (V) Part
brahms_filepath = 'brahms.wav'
synthetic_brahms_filepath = 'synthetic_brahms.wav'
synthetic_ova_brahms_filepath = 'synthetic_brahms_ova.wav'
debug_filepath = 'brahms_debug.wav'
debug_ova_filepath = 'brahms_debug_ova.wav'

class RestoreAudioTests(unittest.TestCase):

    def test_sig_reconst_by_diff(self):
        # What Dennis had me test before break
        rand_sig = np.random.rand(44100)    # Length requires DEBUG_WDW_SIZE
        spectrogram, phases = make_spectrogram(rand_sig, DEBUG_WDW_SIZE, ova=True)
        synthetic_rand_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, ova=True)

        synthetic_rand_sig *= (4/3) # Amplitude ratio made by ova

        # Difference between total signal values (excluding the end window halves)
        sig_diff = np.sum(np.abs(rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)] - synthetic_rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)]))
        
        # plt.plot(rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
        # plt.plot(synthetic_rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
        # plt.show()
        # print('Diff:', sig_diff)
        
        self.assertAlmostEqual(sig_diff, 0)

    def test_sig_reconst_by_diff_2(self):
        rand_sig = np.random.rand(44100)        
        spectrogram, phases = make_spectrogram(rand_sig, DEBUG_WDW_SIZE, ova=True)
        synthetic_rand_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, ova=True)

        ratios = rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)] / synthetic_rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)]

        self.assertAlmostEqual(ratios[0], 4/3)

    def test_reconst_beginning(self):
        # sig = [5]*12
        sig = np.ones(12)
        wdw_size = 6
        spectrogram, phases = make_spectrogram(sig, wdw_size)
        synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size)

        # print('Synthetic non-OVA sig:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])

        # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
        match = np.ones(wdw_size)[:wdw_size // 2]

        np.testing.assert_array_equal(synthetic_sig[:wdw_size // 2], match)

    def test_overlap_add_beginning(self):
        # sig = [5]*12
        sig = np.ones(12)
        wdw_size = 6
        spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True, debug=False)
        synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True, debug=False)

        print('Synthetic OVA sig beginning:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])
        # [0.        0.3454915 0.9045085]

        # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
        match = np.hanning(wdw_size)[:wdw_size // 2]

        np.testing.assert_array_almost_equal(synthetic_sig[:wdw_size // 2], match)

    # def test_overlap_add_middle(self):
    #     # sig = [5]*12
    #     sig = np.ones(12)
    #     wdw_size = 6
    #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

    #     print('Synthetic OVA sig middle:\n', synthetic_sig, '\n', synthetic_sig[wdw_size // 2: -(wdw_size // 2)])

    #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2: -(wdw_size // 2)]
    #     match = np.hanning(wdw_size)[wdw_size // 2: -(wdw_size // 2)]
    #     print('Match:\n', match)
    #     print()

    #     np.testing.assert_array_almost_equal(synthetic_sig[wdw_size // 2: -(wdw_size // 2)], match)

    def test_overlap_add_end(self):
        # sig = [5]*12
        sig = np.ones(12)
        wdw_size = 6
        spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
        synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

        # print('Synthetic OVA sig end:\n', synthetic_sig, '\n', synthetic_sig[-(wdw_size // 2):])

        # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2:]
        match = np.hanning(wdw_size)[-(wdw_size // 2):]
        # print('Match:\n', match)
        # print()

        np.testing.assert_array_almost_equal(synthetic_sig[-(wdw_size // 2):], match)

    # # Useless test
    # def test_mary_activations_mary_bv(self):
    #     mary_activations = make_mary_bv_test_activations()
    #     mary_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True)

    #     synthetic_spgm = mary_basis_vectors @ mary_activations
                
    #     self.assertEqual(synthetic_spgm.shape, (mary_basis_vectors.shape[0], mary_activations.shape[1]))

    # # Can't do b/c don't make activations for full bv yet
    # # def test_mary_activations_full_bv(self):
    # #     mary_activations = make_mary_bv_test_activations()
    # #     norm_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)

    # #     synthetic_spgm = norm_basis_vectors @ mary_activations

    # #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, wdw_size, ova=ova_flag)

    # #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=ova_flag)

    # def test_another_2(self):
    #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
    #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
    #     # plot_matrix(basis_vectors, name="Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
    #     # plot_matrix(spectrogram, name="Original Spectrogram", ratio=SPGM_BRAHMS_RATIO)

    #     # print('Shape of Spectrogram V:', spectrogram.shape)
    #     # print('Shape of Basis Vectors W:', basis_vectors.shape)
    #     # print('Learning Activations...')
    #     # activations = make_activations(spectrogram, basis_vectors)
    #     # print('Shape of Activations H:', activations.shape)

    #     activations = make_mary_bv_test_activations()
    #     print('Shape of Hand-made Activations H:', activations.shape)

    #     with open('activations_my_mary.csv', 'w') as a_f:
    #     # with open('activations_learned_brahms_trunc.csv', 'w') as a_f:
    #     # with open('activations_learned_mary_trunc.csv', 'w') as a_f:
    #         for component in activations:
    #             a_f.write(','.join([('%.4f' % x) for x in component]) + '\n')

    #     synthetic_spgm = basis_vectors @ activations
    #     # plot_matrix(synthetic_spgm, name="Synthetic Spectrogram", ratio=SPGM_BRAHMS_RATIO)

    #     # print('---SYNTHETIC SPGM TRANSITION----')
    #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
    #     # print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
    #     # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

    #     synthetic_sig /= (4/3)

    #     out_filepath = synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
    #     # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
    #     wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
    #     # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)


    # def test_my_mary_activations(self):
    #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
    #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
      
    #     activations = make_activations(spectrogram, basis_vectors)
    

    #     synthetic_spgm = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
    #     # print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
    #     # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

    #     synthetic_sig /= (4/3)

    #     # out_filepath = synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
    #     # # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
    #     # wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
    #     # # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)

    # MAKE_BASIS_VECTOR
    def test_make_best_basis_vector(self):
        wdw_size, sgmt_num = 3, 2
        signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
        basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

        sgmt = np.array([5,0,1])
        match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

        np.testing.assert_array_equal(basis_vector, match)

    def test_make_best_basis_vector_short(self):
        wdw_size, sgmt_num = 3, 2
        signal = np.array([1,3,4,5,0])
        basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

        sgmt = np.array([5,0,0])
        match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

        np.testing.assert_array_equal(basis_vector, match)

    def test_make_avg_basis_vector(self):
        wdw_size, sgmt_num = 3, 2
        signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
        basis_vector = make_basis_vector(signal, sgmt_num, wdw_size, avg=True)

        sgmt = np.array([2.75,2.5,2.25])
        match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

        np.testing.assert_array_equal(basis_vector, match)

    # GET_BASIS_VECTORS
    # def test_make_basis_vectors(self):
    #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=True, debug=True)
        
    #     _, c4_sig = wavfile.read('Piano.ff.C4.wav')
    #     _, db4_sig = wavfile.read('Piano.ff.Db4.wav')
    #     _, d4_sig = wavfile.read('Piano.ff.D4.wav')
    #     _, eb4_sig = wavfile.read('Piano.ff.Eb4.wav')
    #     _, e4_sig = wavfile.read('Piano.ff.E4.wav')
    #     sigs = [c4_sig, db4_sig, d4_sig, eb4_sig, e4_sig]
    #     sigs = [np.array([((x[0] + x[1]) / 2) for x in sig]) for sig in sigs]
    #     # amp_thresh = max(sig) * 0.01  # ?????

    #     match = np.array([make_basis_vector(sig, BEST_WDW_NUM, PIANO_WDW_SIZE, avg=True) for sig in sigs])

    # def test_make_basis_vectors_2(self):
        # get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=False, debug=True)

    # # Mary.wav?
    # def test_make_spectrogram(self):
    #     pass

if __name__ == '__main__':
    unittest.main()


# Just saving stuff from past version of restore_audio (main):

#  # _, example_sig = wavfile.read("piano.wav")
#     # _, brahms_sig = wavfile.read(brahms_filepath)
#     # mary_sig, _ = librosa.load("Mary.wav", sr=STD_SR_HZ)   # Upsample Mary to 44.1kHz
#     # debug_sig = np.array([0,1,1,0])
#     # debug_sig = np.array([0,1,1,0,1,0])
#     # debug_sig_2 = np.random.rand(44100)

#     # DEBUG BLOCK - True for debug
#     if mode == 'DEBUG':
#         print('\n\n')
#         spectrogram, phases = make_spectrogram(debug_sig_2, DEBUG_WDW_SIZE, ova=ova_flag)
#         print('\n---SYNTHETIC SPGM TRANSITION----\n')
#         synthetic_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, ova=ova_flag)
#         print('Debug Synthetic Sig:\n', synthetic_sig[:20])

#         # Also try actual sig
#         print('\n\n')
#         spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
#         print('\n---SYNTHETIC SPGM TRANSITION----\n')
#         synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=ova_flag)
#         print('Actual Synthetic Sig:\n', np.array(synthetic_sig).astype('uint8')[:20])
#         # Make synthetic WAV file
#         out_filepath = debug_ova_filepath if ova_flag else debug_filepath
#         wavfile.write(out_filepath, STD_SR_HZ, np.array(synthetic_sig).astype('uint8'))

#     else:
#         basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
#         spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
#         plot_matrix(basis_vectors, name="Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
#         plot_matrix(spectrogram, name="Original Spectrogram", ratio=SPGM_BRAHMS_RATIO)

#         print('Shape of Spectrogram V:', spectrogram.shape)
#         print('Shape of Basis Vectors W:', basis_vectors.shape)
#         print('Learning Activations...')
#         activations = make_activations(spectrogram, basis_vectors)
#         print('Shape of Activations H:', activations.shape)

#         # activations = make_mary_bv_test_activations()
#         # print('Shape of Hand-made Activations H:', activations.shape)

#         # with open('learned_brahms_activations_trunc.csv', 'w') as a_f:
#         # # with open('learned_mary_activations_trunc.csv', 'w') as a_f:
#         #     for component in activations:
#         #         a_f.write(','.join([('%.4f' % x) for x in component]) + '\n')

#         synthetic_spgm = basis_vectors @ activations
#         plot_matrix(synthetic_spgm, name="Synthetic Spectrogram", ratio=SPGM_BRAHMS_RATIO)

#         print('---SYNTHETIC SPGM TRANSITION----')
#         synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
#         print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
#         # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

#         synthetic_sig /= (4/3)

#         out_filepath = synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
#         # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
#         wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
#         # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)