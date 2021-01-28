from brahms_restore_ml.nmf import nmf
from brahms_restore_ml.nmf import basis_vectors as bv
from brahms_restore_ml import audio_data_processing as dsp
import unittest
from scipy.io import wavfile
import soundfile
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10
l1_penalty_test = 4096
learn_iter_test = 100

brahms_filepath = 'brahms.wav'
mary_filepath = 'brahms_restore_ml/nmf/Mary.wav'
test_path = 'brahms_restore_ml/nmf/output/output_test/output_test_sslrn/'

class SemiSupLearnTests(unittest.TestCase):
    # SEMISUP LEARN TEST #
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
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_' + str(num_noise_bv_test) + 'nbv.wav'
    
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
                                              orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        np.testing.assert_array_equal(given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    # No difference in sound
    def test_restore_brahms_ssln_piano_madeinit_removemorenoise(self):
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_madeinit_morenoiseremoved_' + str(num_noise_bv_test) + 'nbv.wav'

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False)

        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test + 3, 
        #                                                   debug=debug_flag) # HERE we take more noise out (learned in piano part)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
                                              orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)


    def test_restore_brahms_ssln_piano_randinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_piano_randinit_' + str(num_noise_bv_test) + 'nbv.wav'

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=False, debug=debug_flag, incorrect=False)

        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # noise_vectors = basis_vectors[:, :num_noise_bv_test].copy()
        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        np.testing.assert_array_equal(
            given_basis_vectors[:, :num_noise_bv_test], noise_basis_vectors)
        # self.assertEqual(given_basis_vectors[:, :num_noise_bv_test].shape, noise_vectors.shape)

    def test_restore_brahms_ssln_noise_madeinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_madeinit_' + str(num_noise_bv_test) + 'nbv.wav'
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=True, debug=debug_flag, incorrect=False)

        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        # # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)

    def test_restore_brahms_ssln_noise_randinit(self):
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ssln_noise_randinit_' + str(num_noise_bv_test) + 'nbv.wav'

        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(
            sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=(-1 * num_noise_bv_test), madeinit=False, debug=debug_flag, incorrect=False)

        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        # # print('Basis Vectors Shape after de-noise:', basis_vectors.shape)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        # print('Given Basis Vectors Shape:', given_basis_vectors.shape, 'Basis Vectors Shape:', basis_vectors.shape)

        np.testing.assert_array_equal(
            given_basis_vectors[:, num_noise_bv_test:], basis_vectors[:, :])
        # self.assertEqual(given_basis_vectors[:, num_noise_bv_test:].shape, basis_vectors[:, :].shape)


if __name__ == '__main__':
    unittest.main()
