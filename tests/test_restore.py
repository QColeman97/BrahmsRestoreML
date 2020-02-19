import unittest
import sys
sys.path.append('/Users/quinnmc/Desktop/AudioRestore/restore_audio')
from restore_audio import *

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 5
l1_penalty_test = 100
learn_iter_test = 25

brahms_filepath = '/Users/quinnmc/Desktop/AudioRestore/brahms.wav'
mary_filepath = '/Users/quinnmc/Desktop/AudioRestore/Mary.wav'
test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv/'

class RestoreTests(unittest.TestCase):

    def test_restore_brahms_unsupnmf(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_unsupnmf.wav'
            orig_sig_type = sig.dtype   # CRUCIAL INCLUSION?

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]    # For nice comparing
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, NUM_PIANO_NOTES, debug=debug_flag)
        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

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
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit.wav'
            orig_sig_type = sig.dtype
    
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv_test], noise_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    # No difference in sound
    def test_restore_brahms_ssln_piano_madeinit_removemorenoise(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_morenoiseremoved.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test + 3, 
                                                          debug=debug_flag) # HERE we take more noise out (learned in piano part)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv_test], noise_vectors)


    def test_restore_brahms_ssln_piano_randinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_randinit.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=False, debug=debug_flag, incorrect=False)

        noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv_test], noise_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    def test_restore_brahms_ssln_noise_madeinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit.wav'
            orig_sig_type = sig.dtype
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, incorrect=False)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(
            given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)

    def test_restore_brahms_ssln_noise_randinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_randinit.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=False, debug=debug_flag, incorrect=False)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(
            given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)

    def test_restore_brahms_ssln_piano_madeinit_incorrect(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_incorrect.wav'
            orig_sig_type = sig.dtype
      
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=True)

        noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, :num_noise_bv_test], noise_vectors)

    def test_restore_brahms_ssln_piano_randinit_incorrect(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_randinit_incorrect.wav'
            orig_sig_type = sig.dtype
      
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=False, debug=debug_flag, incorrect=True)

        noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, :num_noise_bv_test], noise_vectors)

    def test_restore_brahms_ssln_noise_madeinit_incorrect(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit_incorrect.wav'
            orig_sig_type = sig.dtype
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, incorrect=True)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])

    def test_restore_brahms_ssln_noise_randinit_incorrect(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_randinit_incorrect.wav'
            orig_sig_type = sig.dtype
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=False, debug=debug_flag, incorrect=True)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                                 given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])


    # Mess w/ params of this one test
    def test_restore_brahms_iter(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_iter25.wav'
            orig_sig_type = sig.dtype
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False, 
                                               learn_iter=learn_iter_test)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(
            synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        mean_abs_error = np.mean(np.abs(spectrogram - synthetic_spectrogram))
        print('MAE @ 25 iter:', mean_abs_error)

    
    def test_diff_noisebv_num_look(self):
        num_noise = 50
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)


    # Mess w/ params of this one test - LOOK at noise bv plot
    def test_restore_brahms_noisebv_num(self):
        num_noise = 20
        if write_flag:
            out_filepath = test_path + 'restored_brahms_noisebv' + str(num_noise) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise)


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
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_piano_h_sslrnmade_noise.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)
        
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                               learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, l1_penalty=l1_penalty_test, pen='Piano')
        print('\nL1-Penalty to No L1-Penalty Transition\n')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                                               learn_index=(-1* num_noise_bv_test), madeinit=True, debug=debug_flag)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)

    def test_restore_brahms_l1pen_noiseh(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_noise_h_sslrnmade_piano.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)
        
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, l1_penalty=l1_penalty_test, pen='Noise')
        print('\nL1-Penalty to No L1-Penalty Transition\n')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)

    def test_restore_brahms_l1pen_allh(self):
        sr, sig = wavfile.read(brahms_filepath)
        if write_flag:
            out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_all_h.wav'
            orig_sig_type = sig.dtype

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)
        
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE:]
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        
        # Pos. learn index = learn piano
        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                               debug=debug_flag, l1_penalty=l1_penalty_test, pen='Both')
        print('\nL1-Penalty to No L1-Penalty Transition\n')
        # Compare to no l1-Penalty
        non_pen_activations, non_pen_basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors, 
                                                               debug=debug_flag)
        
        pen_h_sum = np.sum(activations)
        nonpen_h_sum = np.sum(non_pen_activations)

        activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

        synthetic_spectrogram = basis_vectors @ activations
        non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        non_pen_synthetic_sig = make_synthetic_signal(non_pen_synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

        # After noise removed, non-penalized H actually has less sum, prob because all the penalty was focused on noise?
        # pen_h_sum = np.sum(activations)
        # nonpen_h_sum = np.sum(non_pen_activations)
        print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
        print('Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
        print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
        print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
        self.assertGreater(nonpen_h_sum, pen_h_sum)


if __name__ == '__main__':
    unittest.main()
