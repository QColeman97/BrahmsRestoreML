# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for restore_audio.py.

# Testing on Mary.wav for general purposes

# Things to test:
# - sig reconstruction
# - basis vectors correct
# -

import unittest
from restore_audio import *

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv = 5

brahms_filepath = 'brahms.wav'
mary_filepath = 'Mary.wav'
test_path = 'output_test/'


class RestoreAudioTests(unittest.TestCase):

    # UNCOMMENT FOR GOOD TESTS

    # # As signal get's longer, the average ratio between synthetic and original gets smaller
    # def test_ova_reconst_by_sig_diff(self):
    #     if write_flag:
    #         out_filepath = test_path + 'reconst_Mary_ova.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, write_file=write_flag)
    #     # synthetic_sig *= (4/3) # Amplitude ratio made by ova

    #     # u_bound = len(sig)
    #     comp_synthetic_sig = synthetic_sig[: len(sig)]

    #     # Difference between total signal values (excluding the end window halves)
    #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] - comp_synthetic_sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)]))

    #     ratios = sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] / comp_synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]

    #     print('Ratio:', sum(ratios) / len(ratios))

    #     # plt.plot(sig[PIANO_WDW_SIZE*2:(PIANO_WDW_SIZE*2) + 100])
    #     # plt.plot(comp_synthetic_sig[PIANO_WDW_SIZE*2:(PIANO_WDW_SIZE*2) + 100])

    #     # Uncomment for the plot
    #     # plt.plot(sig[(len(sig) - 100):])
    #     # plt.plot(comp_synthetic_sig[(len(sig) - 100):])
    #     # plt.show()
    #     # print('Diff:', sig_diff)

    #     # self.assertEqual(len(sig), len(synthetic_sig[: u_bound]))
    #     self.assertAlmostEqual(sig_diff, 0)

    # def test_reconst_by_sig_diff(self):
    #     if write_flag:
    #         out_filepath = test_path + 'reconst_Mary.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, write_file=write_flag)

    #     # Difference between total signal values (excluding the end window halves)
    #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] - synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]))

    #     # plt.plot(rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    #     # plt.plot(synthetic_rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    #     # plt.show()
    #     # print('Diff:', sig_diff)

    #     self.assertAlmostEqual(sig_diff, 0)

    # def test_ova_reconst_by_sig_diff_2(self):
    #     rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    #     synthetic_rand_sig = reconstruct_audio(rand_sig, DEBUG_WDW_SIZE, '', 0, ova=True)
    #     synthetic_rand_sig *= (4/3) # Amplitude ratio made by ova

    #     # Difference between total signal values (excluding the end window halves)
    #     sig_diff = np.sum(np.abs(rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)] - synthetic_rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)]))

    #     # plt.plot(rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    #     # plt.plot(synthetic_rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    #     # plt.show()
    #     # print('Diff:', sig_diff)

    #     self.assertAlmostEqual(sig_diff, 0)

    # def test_ova_reconst_by_sig_ratio(self):
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE

    #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, '', sr, ova=True)

    #     # Difference between total signal values (excluding the end window halves)
    #     ratios = sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] / synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]

    #     self.assertAlmostEqual(ratios[0], 4/3)

    # def test_ova_reconst_beginning(self):
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz

    #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, '', sr, ova=True)

    #     match = (sig[: PIANO_WDW_SIZE] * np.hanning(PIANO_WDW_SIZE))[: (PIANO_WDW_SIZE // 2)]

    #     np.testing.assert_array_equal(synthetic_sig[: (PIANO_WDW_SIZE // 2)], match)

    # def test_reconst_beginning(self):
    #     # sig = [5]*12
    #     sig = np.ones(12)
    #     wdw_size = 6
    #     spectrogram, phases = make_spectrogram(sig, wdw_size)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size)

    #     # print('Synthetic non-OVA sig:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])

    #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
    #     match = np.ones(wdw_size)[:wdw_size // 2]

    #     np.testing.assert_array_equal(synthetic_sig[:wdw_size // 2], match)

    # def test_overlap_add_beginning(self):
    #     # sig = [5]*12
    #     sig = np.ones(12)
    #     wdw_size = 6
    #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True, debug=debug_flag)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True, debug=debug_flag)

    #     print('Synthetic OVA sig beginning:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])
    #     # [0.        0.3454915 0.9045085]

    #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
    #     match = np.hanning(wdw_size)[:wdw_size // 2]

    #     np.testing.assert_array_almost_equal(synthetic_sig[:wdw_size // 2], match)

    # # def test_overlap_add_middle(self):
    # #     # sig = [5]*12
    # #     sig = np.ones(12)
    # #     wdw_size = 6
    # #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

    # #     print('Synthetic OVA sig middle:\n', synthetic_sig, '\n', synthetic_sig[wdw_size // 2: -(wdw_size // 2)])

    # #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2: -(wdw_size // 2)]
    # #     match = np.hanning(wdw_size)[wdw_size // 2: -(wdw_size // 2)]
    # #     print('Match:\n', match)
    # #     print()

    # #     np.testing.assert_array_almost_equal(synthetic_sig[wdw_size // 2: -(wdw_size // 2)], match)

    # def test_overlap_add_end(self):
    #     # sig = [5]*12
    #     sig = np.ones(12)
    #     wdw_size = 6
    #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

    #     # print('Synthetic OVA sig end:\n', synthetic_sig, '\n', synthetic_sig[-(wdw_size // 2):])

    #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2:]
    #     match = np.hanning(wdw_size)[-(wdw_size // 2):]
    #     # print('Match:\n', match)
    #     # print()

    #     np.testing.assert_array_almost_equal(synthetic_sig[-(wdw_size // 2):], match)

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

    #     out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
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

    #     # out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
    #     # # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
    #     # wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
    #     # # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)

    # UNCOMMENT FOR GOOD TESTS

    # # MAKE_BASIS_VECTOR
    # def test_make_best_basis_vector(self):
    #     wdw_size, sgmt_num = 3, 2
    #     signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
    #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

    #     sgmt = np.array([5,0,1])
    #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    #     np.testing.assert_array_equal(basis_vector, match)

    # def test_make_best_basis_vector_short(self):
    #     wdw_size, sgmt_num = 3, 2
    #     signal = np.array([1,3,4,5,0])
    #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

    #     sgmt = np.array([5,0,0])
    #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    #     np.testing.assert_array_equal(basis_vector, match)

    # def test_make_avg_basis_vector(self):
    #     wdw_size, sgmt_num = 3, 2
    #     signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
    #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size, avg=True)

    #     sgmt = np.array([2.75,2.5,2.25])
    #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    #     np.testing.assert_array_equal(basis_vector, match)

    # def test_unsupervised_nmf(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_Mary_unsupnmf_ova.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, num_components=88, debug=debug_flag)
    #     synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #                              synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_marybv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_Mary_marybv_ova.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=True, noisebv=True,
    #                                   avgbv=True, write_file=write_flag, debug=debug_flag)

    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_fullbv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_Mary_fullbv_ova.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    #                                   avgbv=True, write_file=write_flag, debug=debug_flag)

    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # # Output file not noticably worse b/c no noise in Mary.wav
    # def test_restore_no_hanningbv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_Mary_nohanbv_ova.wav'
    #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    #                                   avgbv=True, write_file=write_flag, debug=debug_flag, nohanbv=True)

    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # # Brahms for these tests (bad audio)
    # def test_restore_brahms_bare(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_brahms_ova(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True,
    #                                   write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_brahms_avgbv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_avgbv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, avgbv=True,
    #                                   write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_brahms_noisebv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_noisebv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, noisebv=True,
    #                                   write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # # Two factors
    # def test_restore_brahms_ova_avgbv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova_avgbv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   ova=True, avgbv=True, write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_brahms_ova_noisebv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova_noisebv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   ova=True, noisebv=True, write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_brahms_avgbv_noisebv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_avgbv_noisebv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   avgbv=True, noisebv=True, write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # # Three factors
    # def test_restore_brahms_ova_noisebv_avgbv(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   ova=True, noisebv=True, avgbv=True, write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_unsupnmf(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_unsupnmf.wav'
        sr, sig = wavfile.read(brahms_filepath)
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, 88, debug=debug_flag)
        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

    # # Output file noticably worse - crackles and a constant high frequency
    # def test_restore_no_hanningbv_brahms(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_nohanbv_ova_noisebv_avgbv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    #                                   avgbv=True, write_file=write_flag, debug=debug_flag, nohanbv=True)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_precise_noise_brahms(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova_precnoisebv_avgbv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    #                                   avgbv=True, write_file=write_flag, debug=debug_flag, prec_noise=True)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)

    # def test_restore_eqbv_brahms(self):
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv_eqpianobv.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
    #                                   ova=True, noisebv=True, avgbv=True, eqbv=True, write_file=write_flag, debug=debug_flag)

    #     # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
    #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    #     # self.assertEqual(sig_diff, 0)


    def test_parition_mtx_madeinit_lfix(self):
        w = np.arange(12).reshape((4,3))
        h = np.arange(18).reshape((3,6))
        # If learn index > 0, fixed part is left side
        w_f, w_l, h_f, h_l = partition_matrices(2, w, h, madeinit=True)

        # If index = 2, w_f is left side of w
        np.testing.assert_array_equal(w, np.concatenate((w_f, w_l), axis=1))
        np.testing.assert_array_equal(h, np.concatenate((h_f, h_l), axis=0))

    def test_parition_mtx_randinit_lfix(self):
        w = np.arange(12).reshape((4,3))
        h = np.arange(18).reshape((3,6))
        # If learn index > 0, fixed part is left side
        w_f, w_l, h_f, h_l = partition_matrices(2, w, h)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 w, np.concatenate((w_f, w_l), axis=1))
        np.testing.assert_array_equal(h, np.concatenate((h_f, h_l), axis=0))

    def test_parition_mtx_madeinit_rfix(self):
        w = np.arange(12).reshape((4,3))
        h = np.arange(18).reshape((3,6))
        # If learn index < 0, fixed part is right side
        w_f, w_l, h_f, h_l = partition_matrices(-2, w, h, madeinit=True)

        # If index = -2, w_f is right side of w
        np.testing.assert_array_equal(w, np.concatenate((w_l, w_f), axis=1))
        np.testing.assert_array_equal(h, np.concatenate((h_l, h_f), axis=0))

    def test_parition_mtx_randinit_rfix(self):
        w = np.arange(12).reshape((4,3))
        h = np.arange(18).reshape((3,6))
        # If learn index < 0, fixed part is right side
        w_f, w_l, h_f, h_l = partition_matrices(-2, w, h)

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 w, np.concatenate((w_l, w_f), axis=1))
        np.testing.assert_array_equal(h, np.concatenate((h_l, h_f), axis=0))


    def test_restore_brahms_ssln_piano_madeinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=True, debug=debug_flag, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv].copy()
        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv], noise_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv].shape, noise_vectors.shape)

    # No difference in sound
    def test_restore_brahms_ssln_piano_madeinit_removemorenoise(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_morenoiseremoved.wav'
        sr, sig = wavfile.read(brahms_filepath)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=True, debug=True, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv].copy()
        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=True, num_noisebv=num_noise_bv + 3) # HERE we take more noise out (learned in piano part)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv], noise_vectors)


    def test_restore_brahms_ssln_piano_randinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_randinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=False, debug=True, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=True)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv], noise_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv].shape, noise_vectors.shape)

    def test_restore_brahms_ssln_noise_madeinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit.wav'

        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv), madeinit=True, debug=debug_flag, incorrect=False)

        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(
            given_basis_vectors[:, num_noise_bv:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv:].shape, basis_vectors[:, :].shape)

    def test_restore_brahms_ssln_noise_randinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_randinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv), madeinit=False, debug=True, incorrect=False)

        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=True)

        # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(
            given_basis_vectors[:, num_noise_bv:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv:].shape, basis_vectors[:, :].shape)

    def test_restore_brahms_ssln_piano_madeinit_incorrect(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_incorrect.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, incorrect_semisup=True)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=True, debug=debug_flag, incorrect=True)

        noise_vectors = basis_vectors[:, :num_noise_bv].copy()
        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, :num_noise_bv], noise_vectors)

    def test_restore_brahms_ssln_piano_randinit_incorrect(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_randinit_incorrect.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, incorrect_semisup=True)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=False, debug=debug_flag, incorrect=True)

        noise_vectors = basis_vectors[:, :num_noise_bv].copy()
        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, :num_noise_bv], noise_vectors)

    def test_restore_brahms_ssln_noise_madeinit_incorrect(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit_incorrect.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, incorrect_semisup=True)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv), madeinit=True, debug=debug_flag, incorrect=True)

        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, num_noise_bv:], basis_vectors[:, :])

    def test_restore_brahms_ssln_noise_randinit_incorrect(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_randinit_incorrect.wav'
        sr, sig = wavfile.read(brahms_filepath)
        # synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True,
        #                               semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, incorrect_semisup=True)

        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv), madeinit=False, debug=debug_flag, incorrect=True)

        activations, basis_vectors = remove_noise_vectors(
            activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, num_noise_bv:], basis_vectors[:, :])


    # Mess w/ params of this one test
    def test_restore_brahms_iter(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_iter25.wav'
        sr, sig = wavfile.read(brahms_filepath)
        
        given_basis_vectors = get_basis_vectors(
            BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv, madeinit=True, debug=debug_flag, incorrect=False, learn_iter=25)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        mean_abs_error = np.mean(np.abs(spectrogram - synthetic_spectrogram))
        print('MAE @ 25 iter:', mean_abs_error)

    
    def test_diff_noisebv_num_look(self):
        num_noise = 10000
        make_noise_basis_vectors(PIANO_WDW_SIZE, ova=True, debug=True, num=num_noise)
        make_noise_basis_vectors(PIANO_WDW_SIZE, ova=True, debug=True, num=num_noise)
        make_noise_basis_vectors(PIANO_WDW_SIZE, ova=True, debug=True, num=num_noise)
        make_noise_basis_vectors(PIANO_WDW_SIZE, ova=True, debug=True, num=num_noise)
        make_noise_basis_vectors(PIANO_WDW_SIZE, ova=True, debug=True, num=num_noise)


    # Mess w/ params of this one test - LOOK at noise bv plot
    def test_restore_brahms_noisebv_num(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_noisebv10000.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=True, 
                                      num_noisebv=10000, l1_penalty=0)


    # L1-PENALTY TESTS
    
    # Use L1-PENALTY constant to change these test cases (l1-pen value wise)
    # Want to test all combos of Brahms activation penalties (piano and noise) - edit to only do l1-pen on H for fixed W
    # Best product so far (semi-sup learn made piano) allows these 3 combos:
    #   - activations for (learned made) piano                          - REMOVED
    #   - activations for (fixed) noise
    #   - activations for (fixed) piano
    #   - activations for (fixed) piano and noise
    #   - activations for both (learned) piano and (fixed) noise        - REMOVED
    
    # Fixed Piano W - Penalized Piano Activations
    def test_restore_brahms_l1pen_pianoh(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(L1_PENALTY_TEST) + '_piano_h.wav'
        sr, sig = wavfile.read(brahms_filepath)

        given_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                               madeinit=True, debug=True, l1_penalty=L1_PENALTY_TEST, pen='Piano')
        print('\nL1-Penalty to No L1-Penalty Transition\n')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                                               madeinit=True, debug=True)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)

    def test_restore_brahms_l1pen_noiseh(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(L1_PENALTY_TEST) + '_noise_h.wav'
        sr, sig = wavfile.read(brahms_filepath)

        given_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                               madeinit=True, debug=True, l1_penalty=L1_PENALTY_TEST, pen='Noise')
        print('\nL1-Penalty to No L1-Penalty Transition\n')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                                               madeinit=True, debug=True)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)

    def test_restore_brahms_l1pen_allh(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(L1_PENALTY_TEST) + '_all_h.wav'
        sr, sig = wavfile.read(brahms_filepath)

        given_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag)
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                               madeinit=True, debug=True, l1_penalty=L1_PENALTY_TEST, pen='Both')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (88 + num_noise_bv), basis_vectors=given_basis_vectors, learn_index=num_noise_bv, 
                                           madeinit=True, debug=debug_flag)
        
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)


    # GET_BASIS_VECTORS
    # def test_make_basis_vectors(self):
    #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=True, debug=debug_flag)
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
        # get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=False, debug=debug_flag)
    # # Mary.wav?
    # def test_make_spectrogram(self):
    #     pass
    
if __name__ == '__main__':
    unittest.main()


# Just saving stuff from past version of restore_audio (main):

#     # _, example_sig = wavfile.read("piano.wav")
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
#         out_filepath += debug_ova_filepath if ova_flag else debug_filepath
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

#         out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
#         # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
#         wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
#         # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)
