import unittest
import sys
sys.path.append('/Users/quinnmc/Desktop/AudioRestore/restore_audio')
from restore_audio import *
import soundfile

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10
l1_penalty_test = 4096
learn_iter_test = 100

# brahms_filepath = '/Users/quinnmc/Desktop/AudioRestore/brahms_16bitPCM.wav'
brahms_filepath = '/Users/quinnmc/Desktop/AudioRestore/brahms.wav'
mary_filepath = '/Users/quinnmc/Desktop/AudioRestore/Mary.wav'
test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv/'
# test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv_from16bitBrahms/'
# test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv_convto16bitBrahms/'
noise_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_noise/'
noise_learnnoise_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_noise_learnnoise/'
penalty_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_penalty/'

class RestoreTests(unittest.TestCase):

    # TEST UNSUP LEARN

    # def test_restore_brahms_unsupnmf(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     # sig, sr = soundfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype

    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_unsupnmf.wav'

    #     # TEMP - don't let silence into sig (so not last part)
    #     # This makes matrix of nan's go away
    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]    # For nice comparing
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, NUM_PIANO_NOTES, debug=debug_flag)
    #     synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
    #                                           orig_sig_type, ova=True, debug=debug_flag)

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)
    #         # wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))
    #         # soundfile.write(out_filepath, synthetic_sig, sr, subtype='PCM_S8')


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




    # SEMISUP LEARN TEST #

    # def test_parition_mtx_madeinit_lfix(self):
    #     w = np.arange(12).reshape((4,3))
    #     h = np.arange(18).reshape((3,6))
    #     # If learn index > 0, fixed part is left side
    #     w_f, w_l, h_f, h_l = partition_matrices(2, w, h, madeinit=True)

    #     # If index = 2, w_f is left side of w
    #     np.testing.assert_array_equal(w, np.concatenate((w_f, w_l), axis=1))
    #     np.testing.assert_array_equal(h, np.concatenate((h_f, h_l), axis=0))

    # def test_parition_mtx_randinit_lfix(self):
    #     w = np.arange(12).reshape((4,3))
    #     h = np.arange(18).reshape((3,6))
    #     # If learn index > 0, fixed part is left side
    #     w_f, w_l, h_f, h_l = partition_matrices(2, w, h)

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              w, np.concatenate((w_f, w_l), axis=1))
    #     np.testing.assert_array_equal(h, np.concatenate((h_f, h_l), axis=0))

    # def test_parition_mtx_madeinit_rfix(self):
    #     w = np.arange(12).reshape((4,3))
    #     h = np.arange(18).reshape((3,6))
    #     # If learn index < 0, fixed part is right side
    #     w_f, w_l, h_f, h_l = partition_matrices(-2, w, h, madeinit=True)

    #     # If index = -2, w_f is right side of w
    #     np.testing.assert_array_equal(w, np.concatenate((w_l, w_f), axis=1))
    #     np.testing.assert_array_equal(h, np.concatenate((h_l, h_f), axis=0))

    # def test_parition_mtx_randinit_rfix(self):
    #     w = np.arange(12).reshape((4,3))
    #     h = np.arange(18).reshape((3,6))
    #     # If learn index < 0, fixed part is right side
    #     w_f, w_l, h_f, h_l = partition_matrices(-2, w, h)

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              w, np.concatenate((w_l, w_f), axis=1))
    #     np.testing.assert_array_equal(h, np.concatenate((h_l, h_f), axis=0))


    # def test_restore_brahms_ssln_piano_madeinit(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_' + str(num_noise_bv_test) + 'nbv.wav'
    
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
    #                                           orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_array_equal(given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)
    #     # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    # # No difference in sound
    # def test_restore_brahms_ssln_piano_madeinit_removemorenoise(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_morenoiseremoved_' + str(num_noise_bv_test) + 'nbv.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test + 3, 
    #     #                                                   debug=debug_flag) # HERE we take more noise out (learned in piano part)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
    #                                           orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_array_equal(
    #         given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)


    # def test_restore_brahms_ssln_piano_randinit(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_piano_randinit_' + str(num_noise_bv_test) + 'nbv.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=False, debug=debug_flag, incorrect=False)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_array_equal(
    #         given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)
    #     # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    # def test_restore_brahms_ssln_noise_madeinit(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit_' + str(num_noise_bv_test) + 'nbv.wav'
        
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, incorrect=False)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

    #     np.testing.assert_array_equal(given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
    #     # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)

    # def test_restore_brahms_ssln_noise_randinit(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_noise_randinit_' + str(num_noise_bv_test) + 'nbv.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=(-1 * num_noise_bv_test), madeinit=False, debug=debug_flag, incorrect=False)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

    #     np.testing.assert_array_equal(
    #         given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
    #     # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)

    # def test_restore_brahms_ssln_piano_madeinit_incorrect(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_incorrect_nbv.wav'
      
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=True)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)

    # def test_restore_brahms_ssln_piano_randinit_incorrect(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_piano_randinit_incorrect.wav'
      
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=False, debug=debug_flag, incorrect=True)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)

    # def test_restore_brahms_ssln_noise_madeinit_incorrect(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit_incorrect.wav'
        
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, incorrect=True)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])

    # def test_restore_brahms_ssln_noise_randinit_incorrect(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_ssln_noise_randinit_incorrect.wav'
        
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(
    #         sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=(-1 * num_noise_bv_test), madeinit=False, debug=debug_flag, incorrect=True)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
    #                              given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])


    # # Mess w/ params of this one test
    # def test_restore_brahms_iter(self):
    #     learn_iter = 25
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = test_path + 'restored_brahms_iter25.wav'
        
    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
    #                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False, 
    #                                            learn_iter=learn_iter)
        
    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     print('\n--Making Synthetic Spectrogram--\n')
    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # synthetic_spectrogram = basis_vectors @ activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig)

    #     mean_abs_error = np.mean(np.abs(spectrogram - synthetic_spectrogram[:, :spectrogram.shape[1]]))
    #     print('MAE @ 25 iter:', mean_abs_error)

    
    # def test_diff_noisebv_num_look(self):
    #     num_noise = 50
    #     make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)




    # NUM NOISE BV TESTS #

    # # Sanity Checkmark - noise spectrogram looks like a rank-1 approx.
    # def test_restore_brahms_noisebvnum_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # # Step up window count of noise bv's

    # def test_restore_brahms_noisebvnum_noise77wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws30(self):
    #     num_noise = 30
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws35(self):
    #     num_noise = 35
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws40(self):
    #     num_noise = 40
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws45(self):
    #     num_noise = 45
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws50(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws55(self):
    #     num_noise = 55
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws60(self):
    #     num_noise = 60
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws65(self):
    #     num_noise = 65
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws70(self):
    #     num_noise = 70
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws75(self):
    #     num_noise = 75
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws77(self):
    #     num_noise = 77
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)
    



    # NUM NOISE BV TESTS - FOR LEARNING THE NOISE (NOT LEARNING PIANO)

    # # Sanity Checkmark - noise spectrogram looks like a rank-1 approx.
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                        
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws19(self):
    #     num_noise = 19
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

                                      
    # Step up window count of noise bv's

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    
    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)


    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws30(self):
    #     num_noise = 30
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws35(self):
    #     num_noise = 35
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws40(self):
    #     num_noise = 40
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws45(self):
    #     num_noise = 45
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws50(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws55(self):
    #     num_noise = 55
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws60(self):
    #     num_noise = 60
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws65(self):
    #     num_noise = 65
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws70(self):
    #     num_noise = 70
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws75(self):
    #     num_noise = 75
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws77(self):
    #     num_noise = 77
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)


    # L1-PENALTY TESTS
    
    # # Use L1-PENALTY constant to change these test cases (l1-pen value wise)
    # # Want to test all combos of Brahms activation penalties (piano and noise) - edit to only do l1-pen on H for fixed W
    # # Best product so far (semi-sup learn made piano) allows these 3 combos:
    # #   - activations for (learned made) piano                          - REMOVED
    # #   - activations for (fixed) noise
    # #   - activations for (fixed) piano
    # #   - activations for (fixed) piano and noise
    # #   - activations for both (learned) piano and (fixed) noise        - REMOVED
    
    # # Fixed Piano W - Penalized Piano Activations
    # def test_restore_brahms_l1pen_pianoh(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     # if write_flag:
    #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_piano_h_sslrnmade_noise_' + str(num_noise_bv_test) + 'nbv.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    #                                             noise_start=6, noise_stop=25)
        
    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
    #     # Pos. learn index = learn piano
    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                            learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, l1_penalty=l1_penalty_test)
    #     # Compare to no l1-Penalty
    #     non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                                            learn_index=(-1* num_noise_bv_test), madeinit=True, debug=debug_flag)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = noise_split_matrices(
    #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     # synthetic_spectrogram = basis_vectors @ activations
    #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]
    #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    #     # if write_flag:
    #     #     wavfile.write(out_filepath, sr, synthetic_sig)
        
    #     print('Penalize Piano Activations - Semi-Supervised Learn Made Noise')
    #     pen_h_sum = np.sum(piano_activations)
    #     nonpen_h_sum = np.sum(non_pen_piano_activations)
    #     print('Penalized Hpiano Sum:', pen_h_sum, 'Non-Penalized Hpiano Sum:', nonpen_h_sum)
    #     print('(No diff) Penalized Wpiano Sum:', np.sum(piano_basis_vectors), 'Non-Penalized Wpiano Sum:', np.sum(non_pen_piano_basis_vectors))
        
    #     print('(Shouldn\'t be) Penalized V\'noise Sum:', np.sum(synthetic_noise_spgm), 'Non-Penalized V\'noise Sum:', np.sum(non_pen_synthetic_noise_spgm))
    #     print('Penalized V\'piano Sum:', np.sum(synthetic_piano_spgm), 'Non-Penalized V\'piano Sum:', np.sum(non_pen_synthetic_piano_spgm))

    #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    #     self.assertGreater(nonpen_h_sum, pen_h_sum)

    # def test_restore_brahms_l1pen_noiseh(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     # if write_flag:
    #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_noise_h_sslrnmade_piano.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    #                                             noise_start=6, noise_stop=25)
        
    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
    #     # Pos. learn index = learn piano
    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, l1_penalty=l1_penalty_test)
    #     # Compare to no l1-Penalty
    #     non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                                            learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = noise_split_matrices(
    #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     # synthetic_spectrogram = basis_vectors @ activations
    #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]
    #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    #     # if write_flag:
    #     #     wavfile.write(out_filepath, sr, synthetic_sig)
        
    #     print('Penalize Noise Activations - Semi-Supervised Learn Made Piano')
    #     pen_h_sum = np.sum(noise_activations)
    #     nonpen_h_sum = np.sum(non_pen_noise_activations)
    #     print('Penalized Hnoise Sum:', pen_h_sum, 'Non-Penalized Hnoise Sum:', nonpen_h_sum)
    #     print('(No diff) Penalized Wnoise Sum:', np.sum(noise_basis_vectors), 'Non-Penalized Wnoise Sum:', np.sum(non_pen_noise_basis_vectors))
        
    #     print('Penalized V\'noise Sum:', np.sum(synthetic_noise_spgm), 'Non-Penalized V\'noise Sum:', np.sum(non_pen_synthetic_noise_spgm))
    #     print('(Shouldn\'t be) Penalized V\'piano Sum:', np.sum(synthetic_piano_spgm), 'Non-Penalized V\'piano Sum:', np.sum(non_pen_synthetic_piano_spgm))

    #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    #     self.assertGreater(nonpen_h_sum, pen_h_sum)

    # def test_restore_brahms_l1pen_allh(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     # if write_flag:
    #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_all_h.wav'

    #     given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    #                                             noise_start=6, noise_stop=25)
        
    #     sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    #     orig_sig_len = len(sig)
    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
    #     # Pos. learn index = learn piano
    #     activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                            debug=debug_flag, l1_penalty=l1_penalty_test)
    #     # Compare to no l1-Penalty
    #     non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
    #                                                            debug=debug_flag)

    #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = noise_split_matrices(
    #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)
        
    #     pen_h_sum = np.sum(activations)
    #     nonpen_h_sum = np.sum(non_pen_activations)

    #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    #     # synthetic_spectrogram = basis_vectors @ activations
    #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    #     synthetic_sig = synthetic_sig[orig_sig_len:]
    #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    #     # if write_flag:
    #     #     wavfile.write(out_filepath, sr, synthetic_sig)

    #     # After noise removed, non-penalized H actually has less sum, prob because all the penalty was focused on noise?
    #     # pen_h_sum = np.sum(activations)
    #     # nonpen_h_sum = np.sum(non_pen_activations)
    #     print('Penalize All Activations - Supervised')
    #     print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
    #     print('(No diff) Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
    #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    #     self.assertGreater(nonpen_h_sum, pen_h_sum)



    # Varying L1-Penalty Tests #
    # Penalize the Piano Activations, Noise Activations, or All Activations #
    # Piano Activations 
    def test_restore_brahms_pianoh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_pianoh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_pianoh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # Noise Activations 
    def test_restore_brahms_noiseh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'
        
        print('Noise H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_noiseh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_noiseh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # # All Activations 
    def test_restore_brahms_allh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_allh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_allh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # def test_restore_brahms_allh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # def test_restore_brahms_allh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

if __name__ == '__main__':
    unittest.main()
