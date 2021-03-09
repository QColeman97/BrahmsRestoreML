from brahms_restore_ml.nmf.nmf import *
# from brahms_restore_ml.nmf import basis_vectors as bv
# from brahms_restore_ml import audio_data_processing as dsp
import unittest
from scipy.io import wavfile
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False

brahms_filepath = os.getcwd() + '/brahms.wav'
# noise_sup_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_noise_fixed/'
# noise_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_noise_learnpiano/'
# noise_learnnoise_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_noise_learnnoise/'
noise_sup_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch_nomask/num_noise_fixed/'
noise_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch_nomask/num_noise_learn_piano/'
noise_learnnoise_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch_nomask/num_noise_learn_noise/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class RestoreNoiseTests(unittest.TestCase):

    # NUM NOISE BV TESTS - SEMI-SUP FOR LEARNING THE PIANO (NOT LEARNING NOISE)
    # # RANDINIT
    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                                       
    # # def test_restore_brahms_sslrn_randinit_noisebvnum_noise30_izotope(self):
    # #     num_noise = 30
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise50_izotope(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_noise100_izotope(self):
    #     num_noise = 100
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # # MADEINIT
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws4(self):
    # #     num_noise = 4
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                         
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws6(self):
    # #     num_noise = 6
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws7(self):
    # #     num_noise = 7
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws8(self):
    # #     num_noise = 8
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws9(self):
    # #     num_noise = 9
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws11(self):
    # #     num_noise = 11
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws12(self):
    # #     num_noise = 12
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws13(self):
    # #     num_noise = 13
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws14(self):
    # #     num_noise = 14
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                                       
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise30_izotope(self):
    # #     num_noise = 30
    # #     if write_flag:
    # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise50_izotope(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_noise100_izotope(self):
    #     num_noise = 100
    #     if write_flag:
    #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # # # Step up window count of noise bv's

    # # # 77 windows includes noise - takes out too much piano
    # # # def test_restore_brahms_noisebvnum_noise77wdws1(self):
    # # #     num_noise = 1
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws2(self):
    # # #     num_noise = 2
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws3(self):
    # # #     num_noise = 3
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws4(self):
    # # #     num_noise = 4
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws5(self):
    # # #     num_noise = 5
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws6(self):
    # # #     num_noise = 6
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws7(self):
    # # #     num_noise = 7
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws8(self):
    # # #     num_noise = 8
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws9(self):
    # # #     num_noise = 9
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws10(self):
    # # #     num_noise = 10
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws11(self):
    # # #     num_noise = 11
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws12(self):
    # # #     num_noise = 12
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws13(self):
    # # #     num_noise = 13
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws14(self):
    # # #     num_noise = 14
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws15(self):
    # # #     num_noise = 15
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws20(self):
    # # #     num_noise = 20
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws25(self):
    # # #     num_noise = 25
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws30(self):
    # # #     num_noise = 30
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws35(self):
    # # #     num_noise = 35
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws40(self):
    # # #     num_noise = 40
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws45(self):
    # # #     num_noise = 45
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws50(self):
    # # #     num_noise = 50
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws55(self):
    # # #     num_noise = 55
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws60(self):
    # # #     num_noise = 60
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws65(self):
    # # #     num_noise = 65
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws70(self):
    # # #     num_noise = 70
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws75(self):
    # # #     num_noise = 75
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_noise77wdws77(self):
    # # #     num_noise = 77
    # # #     if write_flag:
    # # #         out_filepath = noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)


















    # NUM NOISE BV TESTS - SEMI-SUP FOR LEARNING THE NOISE (NOT LEARNING PIANO)
    # # RANDINIT
    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
         
    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise50_izotope(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise100_izotope(self):
    #     num_noise = 100
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # MADEINIT
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws2(self):
        num_noise = 2
        if write_flag:
            out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws4(self):
    # #     num_noise = 4
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws6(self):
    # #     num_noise = 6
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws7(self):
    # #     num_noise = 7
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws8(self):
    # #     num_noise = 8
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws9(self):
    # #     num_noise = 9
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                        
    # # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws11(self):
    # # #     num_noise = 11
    # # #     if write_flag:
    # # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws12(self):
    # # #     num_noise = 12
    # # #     if write_flag:
    # # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws13(self):
    # # #     num_noise = 13
    # # #     if write_flag:
    # # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws14(self):
    # # #     num_noise = 14
    # # #     if write_flag:
    # # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws15(self):
    # #     num_noise = 15
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                              
    # # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise30_izotope(self):
    # #     num_noise = 30
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise50_izotope(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise100_izotope(self):
    #     num_noise = 100
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # # Step up window count of noise bv's

    # # 77 windows includes noise - takes out too much piano
    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws1(self):
    # #     num_noise = 1
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws2(self):
    # #     num_noise = 2
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws3(self):
    # #     num_noise = 3
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws4(self):
    # #     num_noise = 4
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws5(self):
    # #     num_noise = 5
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws6(self):
    # #     num_noise = 6
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws7(self):
    # #     num_noise = 7
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws8(self):
    # #     num_noise = 8
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws9(self):
    # #     num_noise = 9
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws10(self):
    # #     num_noise = 10
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    
    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws11(self):
    # #     num_noise = 11
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws12(self):
    # #     num_noise = 12
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws13(self):
    # #     num_noise = 13
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws14(self):
    # #     num_noise = 14
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)


    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws15(self):
    # #     num_noise = 15
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws20(self):
    # #     num_noise = 20
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws25(self):
    # #     num_noise = 25
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws30(self):
    # #     num_noise = 30
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws35(self):
    # #     num_noise = 35
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws40(self):
    # #     num_noise = 40
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws45(self):
    # #     num_noise = 45
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws50(self):
    # #     num_noise = 50
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws55(self):
    # #     num_noise = 55
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws60(self):
    # #     num_noise = 60
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws65(self):
    # #     num_noise = 65
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws70(self):
    # #     num_noise = 70
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws75(self):
    # #     num_noise = 75
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # def test_restore_brahms_noisebvnum_lnoise_noise77wdws77(self):
    # #     num_noise = 77
    # #     if write_flag:
    # #         out_filepath = noise_learnnoise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)












    # # NUM NOISE BV TESTS - FOR SUPERVISED
    # def test_restore_brahms_sup_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sup_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sup_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sup_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sup_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)
                                      
    # def test_restore_brahms_sup_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, prec_noise=True)

    # def test_restore_brahms_sup_noise50_izotope(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # def test_restore_brahms_sup_noise100_izotope(self):
    #     num_noise = 100
    #     if write_flag:
    #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_izotope.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, write_noise_sig=True)

    # # # Step up window count of noise bv's

    # # # 77 windows includes voice - takes out too much piano - disregard from here
    # # def test_restore_brahms_noisebvnum_sup_noise77wdws1(self):
    # #     num_noise = 1
    # #     if write_flag:
    # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # #     sr, sig = wavfile.read(brahms_filepath)
    # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # #                                   write_file=write_flag, debug=debug_flag, 
    # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws2(self):
    # # #     num_noise = 2
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws3(self):
    # # #     num_noise = 3
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws4(self):
    # # # #     num_noise = 4
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws5(self):
    # # #     num_noise = 5
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws6(self):
    # # # #     num_noise = 6
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws7(self):
    # # # #     num_noise = 7
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws8(self):
    # # # #     num_noise = 8
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws9(self):
    # # # #     num_noise = 9
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws10(self):
    # # #     num_noise = 10
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    
    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws11(self):
    # # # #     num_noise = 11
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws12(self):
    # # # #     num_noise = 12
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws13(self):
    # # # #     num_noise = 13
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws14(self):
    # # # #     num_noise = 14
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)


    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws15(self):
    # # #     num_noise = 15
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws20(self):
    # # # #     num_noise = 20
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws25(self):
    # # # #     num_noise = 25
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws30(self):
    # # # #     num_noise = 30
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws35(self):
    # # # #     num_noise = 35
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws40(self):
    # # # #     num_noise = 40
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws45(self):
    # # # #     num_noise = 45
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # def test_restore_brahms_noisebvnum_sup_noise77wdws50(self):
    # # #     num_noise = 50
    # # #     if write_flag:
    # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # #     sr, sig = wavfile.read(brahms_filepath)
    # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # #                                   write_file=write_flag, debug=debug_flag, 
    # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws55(self):
    # # # #     num_noise = 55
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws60(self):
    # # # #     num_noise = 60
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws65(self):
    # # # #     num_noise = 65
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws70(self):
    # # # #     num_noise = 70
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws75(self):
    # # # #     num_noise = 75
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)

    # # # # def test_restore_brahms_noisebvnum_sup_noise77wdws77(self):
    # # # #     num_noise = 77
    # # # #     if write_flag:
    # # # #         out_filepath = noise_sup_test_path + 'sup_noisebv' + str(num_noise) + '_77wdws.wav'
    # # # #     sr, sig = wavfile.read(brahms_filepath)
    # # # #     synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    # # # #                                   write_file=write_flag, debug=debug_flag, 
    # # # #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True, prec_noise=True)


if __name__ == '__main__':
    unittest.main()