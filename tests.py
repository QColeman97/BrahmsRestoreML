# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for restore_audio.py.

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

    def test_row_col_winner_1(self):
        # board = (0, 0, 0, 1, 1, 1, 0, 0, 0)
        # width = int(len(board) ** 0.5)
        # winner = row_col_winner(board, width, True)
        # self.assertEqual(winner, 1)

        # What Dennis had me test before break

        spectrogram, phases = make_spectrogram(debug_sig_2, DEBUG_WDW_SIZE, ova=ova_flag)
        print('\n---SYNTHETIC SPGM TRANSITION----\n')
        synthetic_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, ova=ova_flag)
        print('Debug Synthetic Sig:\n', synthetic_sig[:20])

        diff = np.sum(np.abs(debug_sig_2[(wdw_size // 2):-(wdw_size // 2)] - synthetic_sig[(wdw_size // 2):-(wdw_size // 2)]))
        print()       
        print('Diff:', diff)
        print()
        
        plt.plot(debug_sig_2[wdw_size*2:(wdw_size*2) + 100])
    
        plt.plot(synthetic_sig[wdw_size*2:(wdw_size*2) + 100])
        plt.show()

        ratios = debug_sig_2[(wdw_size // 2):-(wdw_size // 2)] / synthetic_sig[(wdw_size // 2):-(wdw_size // 2)]
        print('Ratios:\n', ratios)

    def test_another(self):
        print('\n\n')
        spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
        print('\n---SYNTHETIC SPGM TRANSITION----\n')
        synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=ova_flag)
        print('Actual Synthetic Sig:\n', np.array(synthetic_sig).astype('uint8')[:20])
        # Make synthetic WAV file
        out_filepath = debug_ova_filepath if ova_flag else debug_filepath
        wavfile.write(out_filepath, STD_SR_HZ, np.array(synthetic_sig).astype('uint8'))

    def test_another_2(self):
        basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
        spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
        plot_matrix(basis_vectors, name="Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
        plot_matrix(spectrogram, name="Original Spectrogram", ratio=SPGM_BRAHMS_RATIO)

        print('Shape of Spectrogram V:', spectrogram.shape)
        print('Shape of Basis Vectors W:', basis_vectors.shape)
        print('Learning Activations...')
        activations = make_activations(spectrogram, basis_vectors)
        print('Shape of Activations H:', activations.shape)

        # activations = make_mary_bv_test_activations()
        # print('Shape of Hand-made Activations H:', activations.shape)

        # with open('learned_brahms_activations_trunc.csv', 'w') as a_f:
        # # with open('learned_mary_activations_trunc.csv', 'w') as a_f:
        #     for component in activations:
        #         a_f.write(','.join([('%.4f' % x) for x in component]) + '\n')

        synthetic_spgm = basis_vectors @ activations
        plot_matrix(synthetic_spgm, name="Synthetic Spectrogram", ratio=SPGM_BRAHMS_RATIO)

        print('---SYNTHETIC SPGM TRANSITION----')
        synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
        print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
        # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

        synthetic_sig /= (4/3)

        out_filepath = synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
        # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
        wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
        # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)



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