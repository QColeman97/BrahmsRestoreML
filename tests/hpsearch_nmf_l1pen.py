from brahms_restore_ml.nmf import nmf
from brahms_restore_ml.nmf import basis_vectors as bv
from brahms_restore_ml import audio_data_processing as dsp
import unittest
import soundfile
from scipy.io import wavfile
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 2
# l1_penalty_test = 4096  # 100
# learn_iter_test = 100   # 25

brahms_filepath = 'brahms.wav'
mary_filepath = 'brahms_restore_ml/nmf/Mary.wav'
# penalty_test_path = 'brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_penalty/'
penalty_test_path = 'brahms_restore_ml/nmf/output/output_hpsearch_nomask/penalty/'
penalty_dmgp_test_path = 'brahms_restore_ml/nmf/output/output_hpsearch_nomask/dmged_piano_penalty/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class RestorePenaltyTests(unittest.TestCase):    
    # L1-PENALTY TESTS
    
    # Use L1-PENALTY constant to change these test cases (l1-pen value wise)
    # Want to test all combos of Brahms activation penalties (piano and noise) - edit to only do l1-pen on H for fixed W
    # Best product so far (semi-sup learn made piano) allows these 3 combos:
    #   - activations for (learned made) piano                          - REMOVED
    #   - activations for (fixed) noise
    #   - activations for (fixed) piano
    #   - activations for (fixed) piano and noise
    #   - activations for both (learned) piano and (fixed) noise        - REMOVED
    
    # Usage makes sense for only H matrix
    # Usage only works when corres W is fixed, or corrs W is learned but normalized after each MU?
    #                                           no, b/c W will try to make up for H normalized or not

    # # # Fixed Piano W - Penalized Piano Activations
    # # def test_restore_brahms_l1pen_pianoh(self):
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     # if write_flag:
    # #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_piano_h_sslrnmade_noise_' + str(num_noise_bv_test) + 'nbv.wav'

    # #     given_basis_vectors = bv.get_basis_vectors(nmf.PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    # #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    # #                                             noise_start=6, noise_stop=25)
        
    # #     sig = sig[nmf.WDW_NUM_AFTER_VOICE * nmf.PIANO_WDW_SIZE: -(20 * nmf.PIANO_WDW_SIZE)]
    # #     orig_sig_len = len(sig)
    # #     spectrogram, phases = dsp.make_spectrogram(sig, nmf.PIANO_WDW_SIZE, dsp.EPSILON, ova=True, debug=debug_flag)
        
    # #     # Pos. learn index = learn piano
    # #     activations, basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                            split_index=(-1 * num_noise_bv_test), made_init=True, debug=debug_flag, l1_pen=l1_penalty_test)
    # #     # Compare to no l1-Penalty
    # #     non_pen_activations, non_pen_basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                                            split_index=(-1* num_noise_bv_test), made_init=True, debug=debug_flag)

    # #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = nmf.noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = nmf.noise_split_matrices(
    # #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    # #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    # #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    # #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    # #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    # #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)

    # #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     # synthetic_spectrogram = basis_vectors @ activations
    # #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    # #     synthetic_sig = dsp.make_synthetic_signal(synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     non_pen_synthetic_sig = dsp.make_synthetic_signal(non_pen_synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     synthetic_sig = synthetic_sig[orig_sig_len:]
    # #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    # #     # if write_flag:
    # #     #     wavfile.write(out_filepath, sr, synthetic_sig)
        
    # #     print('Penalize Piano Activations - Semi-Supervised Learn Made Noise')
    # #     pen_h_sum = np.sum(piano_activations)
    # #     nonpen_h_sum = np.sum(non_pen_piano_activations)
    # #     print('Penalized Hpiano Sum:', pen_h_sum, 'Non-Penalized Hpiano Sum:', nonpen_h_sum)
    # #     print('(No diff) Penalized Wpiano Sum:', np.sum(piano_basis_vectors), 'Non-Penalized Wpiano Sum:', np.sum(non_pen_piano_basis_vectors))
        
    # #     print('(Shouldn\'t be) Penalized V\'noise Sum:', np.sum(synthetic_noise_spgm), 'Non-Penalized V\'noise Sum:', np.sum(non_pen_synthetic_noise_spgm))
    # #     print('Penalized V\'piano Sum:', np.sum(synthetic_piano_spgm), 'Non-Penalized V\'piano Sum:', np.sum(non_pen_synthetic_piano_spgm))

    # #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    # #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    # #     self.assertGreater(nonpen_h_sum, pen_h_sum)

    # # def test_restore_brahms_l1pen_noiseh(self):
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     # if write_flag:
    # #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_noise_h_sslrnmade_piano.wav'

    # #     given_basis_vectors = bv.get_basis_vectors(nmf.PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    # #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    # #                                             noise_start=6, noise_stop=25)
        
    # #     sig = sig[nmf.WDW_NUM_AFTER_VOICE * nmf.PIANO_WDW_SIZE: -(20 * nmf.PIANO_WDW_SIZE)]
    # #     orig_sig_len = len(sig)
    # #     spectrogram, phases = dsp.make_spectrogram(sig, nmf.PIANO_WDW_SIZE, dsp.EPSILON, ova=True, debug=debug_flag)
        
    # #     # Pos. learn index = learn piano
    # #     activations, basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                            split_index=num_noise_bv_test, made_init=True, debug=debug_flag, l1_pen=l1_penalty_test)
    # #     # Compare to no l1-Penalty
    # #     non_pen_activations, non_pen_basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                                            split_index=num_noise_bv_test, made_init=True, debug=debug_flag)

    # #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = nmf.noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = nmf.noise_split_matrices(
    # #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    # #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    # #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    # #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    # #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    # #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)

    # #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     # synthetic_spectrogram = basis_vectors @ activations
    # #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    # #     synthetic_sig = dsp.make_synthetic_signal(synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     non_pen_synthetic_sig = dsp.make_synthetic_signal(non_pen_synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     synthetic_sig = synthetic_sig[orig_sig_len:]
    # #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    # #     # if write_flag:
    # #     #     wavfile.write(out_filepath, sr, synthetic_sig)
        
    # #     print('Penalize Noise Activations - Semi-Supervised Learn Made Piano')
    # #     pen_h_sum = np.sum(noise_activations)
    # #     nonpen_h_sum = np.sum(non_pen_noise_activations)
    # #     print('Penalized Hnoise Sum:', pen_h_sum, 'Non-Penalized Hnoise Sum:', nonpen_h_sum)
    # #     print('(No diff) Penalized Wnoise Sum:', np.sum(noise_basis_vectors), 'Non-Penalized Wnoise Sum:', np.sum(non_pen_noise_basis_vectors))
        
    # #     print('Penalized V\'noise Sum:', np.sum(synthetic_noise_spgm), 'Non-Penalized V\'noise Sum:', np.sum(non_pen_synthetic_noise_spgm))
    # #     print('(Shouldn\'t be) Penalized V\'piano Sum:', np.sum(synthetic_piano_spgm), 'Non-Penalized V\'piano Sum:', np.sum(non_pen_synthetic_piano_spgm))

    # #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    # #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    # #     self.assertGreater(nonpen_h_sum, pen_h_sum)

    # # def test_restore_brahms_l1pen_allh(self):
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     # if write_flag:
    # #     #     out_filepath = test_path + 'restored_brahms_l1pen' + str(l1_penalty_test) + '_all_h.wav'

    # #     given_basis_vectors = bv.get_basis_vectors(nmf.PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
    # #                                             debug=debug_flag, num_noise=num_noise_bv_test,
    # #                                             noise_start=6, noise_stop=25)
        
    # #     sig = sig[nmf.WDW_NUM_AFTER_VOICE * nmf.PIANO_WDW_SIZE: -(20 * nmf.PIANO_WDW_SIZE)]
    # #     orig_sig_len = len(sig)
    # #     spectrogram, phases = dsp.make_spectrogram(sig, nmf.PIANO_WDW_SIZE, dsp.EPSILON, ova=True, debug=debug_flag)
        
    # #     # Pos. learn index = learn piano
    # #     activations, basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                            debug=debug_flag, l1_pen=l1_penalty_test)
    # #     # Compare to no l1-Penalty
    # #     non_pen_activations, non_pen_basis_vectors = nmf.extended_nmf(spectrogram, (nmf.NUM_PIANO_NOTES + num_noise_bv_test), W=given_basis_vectors, 
    # #                                                            debug=debug_flag)

    # #     noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = nmf.noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     non_pen_noise_activations, non_pen_noise_basis_vectors, non_pen_piano_activations, non_pen_piano_basis_vectors = nmf.noise_split_matrices(
    # #         non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     synthetic_piano_spgm = piano_basis_vectors @ piano_activations
    # #     synthetic_noise_spgm = noise_basis_vectors @ noise_activations
    # #     synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

    # #     non_pen_synthetic_piano_spgm = non_pen_piano_basis_vectors @ non_pen_piano_activations
    # #     non_pen_synthetic_noise_spgm = non_pen_noise_basis_vectors @ non_pen_noise_activations
    # #     non_pen_synthetic_spectrogram = np.concatenate((non_pen_synthetic_noise_spgm, non_pen_synthetic_piano_spgm), axis=1)
        
    # #     pen_h_sum = np.sum(activations)
    # #     nonpen_h_sum = np.sum(non_pen_activations)

    # #     # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
    # #     # non_pen_activations, non_pen_basis_vectors = remove_noise_vectors(non_pen_activations, non_pen_basis_vectors, num_noise_bv_test, debug=debug_flag)

    # #     # synthetic_spectrogram = basis_vectors @ activations
    # #     # non_pen_synthetic_spectrogram = non_pen_basis_vectors @ non_pen_activations

    # #     synthetic_sig = dsp.make_synthetic_signal(synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     non_pen_synthetic_sig = dsp.make_synthetic_signal(non_pen_synthetic_spectrogram, phases, nmf.PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
    # #     synthetic_sig = synthetic_sig[orig_sig_len:]
    # #     non_pen_synthetic_sig = non_pen_synthetic_sig[orig_sig_len:]

    # #     # if write_flag:
    # #     #     wavfile.write(out_filepath, sr, synthetic_sig)

    # #     # After noise removed, non-penalized H actually has less sum, prob because all the penalty was focused on noise?
    # #     # pen_h_sum = np.sum(activations)
    # #     # nonpen_h_sum = np.sum(non_pen_activations)
    # #     print('Penalize All Activations - Supervised')
    # #     print('Penalized H Sum:', pen_h_sum, 'Non-Penalized H Sum:', nonpen_h_sum)
    # #     print('(No diff) Penalized W Sum:', np.sum(basis_vectors), 'Non-Penalized W Sum:', np.sum(non_pen_basis_vectors))
    # #     print('Penalized V\' Sum:', np.sum(synthetic_spectrogram), 'Non-Penalized V\' Sum:', np.sum(non_pen_synthetic_spectrogram))
    # #     print('Penalized Sig\' Sum:', np.sum(synthetic_sig), 'Non-Penalized Sig\' Sum:', np.sum(non_pen_synthetic_sig))
    # #     self.assertGreater(nonpen_h_sum, pen_h_sum)


    # Varying L1-Penalty Tests #
    # Penalize the Piano Activations, Noise Activations, or All Activations #
    # Piano Activations 
    # def test_restore_brahms_pianoh_l1pen1(self):
    #     pen = 1
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph_no_l1pen.wav'

    #     print('Piano H Non-Penalized:')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test)
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # # def test_restore_brahms_pianoh_l1pen2(self):
    # #     pen = 2
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen4(self):
    # #     pen = 4
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen8(self):
    # #     pen = 8
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen16(self):
    # #     pen = 16
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen32(self):
    # #     pen = 32
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen64(self):
    # #     pen = 64
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen128(self):
    # #     pen = 128
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen256(self):
    # #     pen = 256
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen1024(self):
    # #     pen = 1024
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen2048(self):
    # #     pen = 2048
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen4096(self):
    # #     pen = 4096
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen8192(self):
    # #     pen = 8192
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_pianoh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # # OPTIMUM POINT HERE? NO
    # def test_restore_brahms_pianoh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen8388608(self):
    # #     pen = 8388608
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen16777216(self):
    # #     pen = 16777216
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen33554432(self):
    #     pen = 33554432
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # UPPER BOUND FOUND HERE - 67108864
    # def test_restore_brahms_pianoh_l1pen67108864(self):
    #     pen = 67108864
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen134217728(self):
    # #     pen = 134217728
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen268435456(self):
    # #     pen = 268435456
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_pianoh_l1pen1000000000(self):
    # #     pen = 1000000000
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    # #     print('Piano H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # Noise Activations 
    # def test_restore_brahms_noiseh_l1pen1(self):
    #     pen = 1
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh_no_l1pen.wav'
        
    #     print('Noise H Non-Penalized:')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test)
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # # def test_restore_brahms_noiseh_l1pen2(self):
    # #     pen = 2
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen4(self):
    # #     pen = 4
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen8(self):
    # #     pen = 8
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen16(self):
    # #     pen = 16
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen32(self):
    # #     pen = 32
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen64(self):
    # #     pen = 64
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen128(self):
    # #     pen = 128
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen256(self):
    # #     pen = 256
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen512(self):
    # #     pen = 512
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen1024(self):
    # #     pen = 1024
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen2048(self):
    # #     pen = 2048
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen4096(self):
    # #     pen = 4096
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen8192(self):
    # #     pen = 8192
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen16384(self):
    # #     pen = 16384
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen32678(self):
    # #     pen = 32768
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen65536(self):
    # #     pen = 65536
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen131072(self):
    # #     pen = 131072
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # # def test_restore_brahms_noiseh_l1pen262144(self):
    # #     pen = 262144
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen524288(self):
    # #     pen = 524288
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen1048576(self):
    # #     pen = 1048576
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen2097152(self):
    # #     pen = 2097152
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen4194304(self):
    # #     pen = 4194304
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen8388608(self):
    # #     pen = 8388608
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen16777216(self):
    # #     pen = 16777216
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)





    # # def test_restore_brahms_noiseh_l1pen33554432(self):
    # #     pen = 33554432
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen67108864(self):
    # #     pen = 67108864
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen134217728(self):
    # #     pen = 134217728
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_noiseh_l1pen268435456(self):
    # #     pen = 268435456
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    # #     print('Noise H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # # DMGED PIANO PARAMS
    # def test_restore_brahms_dmgp_pianoh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_dmgp_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen, dmged_pianobv=True)

    # def test_restore_brahms_dmgp_pianoh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_dmgp_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen, dmged_pianobv=True)
    
    # def test_restore_brahms_dmgp_pianoh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_dmgp_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen, dmged_pianobv=True)

    # def test_restore_brahms_dmgp_pianoh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_dmgp_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen, dmged_pianobv=True)

    def test_restore_brahms_dmgp_pianoh_l1pen524288(self):
        pen = 524288
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_dmgp_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen, dmged_pianobv=True)


    # # # All Activations 
    # def test_restore_brahms_allh_l1pen1(self):
    #     pen = 1
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah_no_l1pen.wav'

    #     print('All H Non-Penalized:')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test)
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # # def test_restore_brahms_allh_l1pen2(self):
    # #     pen = 2
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen4(self):
    # #     pen = 4
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen8(self):
    # #     pen = 8
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen16(self):
    # #     pen = 16
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen32(self):
    # #     pen = 32
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen64(self):
    # #     pen = 64
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen128(self):
    # #     pen = 128
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen256(self):
    # #     pen = 256
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen512(self):
    # #     pen = 512
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen1024(self):
    # #     pen = 1024
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen2048(self):
    # #     pen = 2048
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen4096(self):
    # #     pen = 4096
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen8192(self):
    # #     pen = 8192
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen16384(self):
    # #     pen = 16384
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen32678(self):
    # #     pen = 32768
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen65536(self):
    # #     pen = 65536
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # # def test_restore_brahms_allh_l1pen131072(self):
    # #     pen = 131072
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen262144(self):
    # #     pen = 262144
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen524288(self):
    # #     pen = 524288
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen1048576(self):
    # #     pen = 1048576
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen2097152(self):
    # #     pen = 2097152
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # # def test_restore_brahms_allh_l1pen4194304(self):
    # #     pen = 4194304
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # # def test_restore_brahms_allh_l1pen8388608(self):
    # #     pen = 8388608
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen16777216(self):
    # #     pen = 16777216
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen33554432(self):
    #     pen = 33554432
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen67108864(self):
    # #     pen = 67108864
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen134217728(self):
    # #     pen = 134217728
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # def test_restore_brahms_allh_l1pen268435456(self):
    # #     pen = 268435456
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     orig_sig_type = sig.dtype
    # #     if write_flag:
    # #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    # #     print('All H Penalty ' + str(pen) + ':')
    # #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # # UPPER BOUND FOUND HERE
    # def test_restore_brahms_allh_l1pen1000000000(self):
    #     pen = 1000000000
    #     # pen = 0
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'
    #         # out_filepath = penalty_test_path + 'ZERONOTPASSED_restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen) #,
    #                                 #   write_noise_sig=True)

    # # Find optimum l1-pen for sup 
    # def test_restore_brahms_allh_l1pen10000000(self):
    #     pen = 10000000 # started working
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

if __name__ == '__main__':
    unittest.main()
