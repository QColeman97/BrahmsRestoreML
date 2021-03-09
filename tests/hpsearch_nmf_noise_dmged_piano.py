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
sup_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/dmged_piano_num_noise_fixed/'
learn_noise_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/dmged_piano_num_noise_learn_noise/'
learn_piano_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/dmged_piano_num_noise_learn_piano/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class RestoreNoiseTests(unittest.TestCase):

    # DMGED PIANO - NUM NOISE BV TESTS - SEMI-SUP FOR LEARNING THE NOISE (NOT LEARNING PIANO)
    # RANDINIT
    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws1(self):
        num_noise = 1
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws2(self):
        num_noise = 2
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws3(self):
        num_noise = 3
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
         
    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws5(self):
        num_noise = 5
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise19wdws10(self):
        num_noise = 10
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise50_izotope(self):
        num_noise = 50
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_randinit_noisebvnum_lnoise_noise100_izotope(self):
        num_noise = 100
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_randinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    # MADEINIT
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws1(self):
        num_noise = 1
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws2(self):
        num_noise = 2
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws3(self):
        num_noise = 3
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws5(self):
        num_noise = 5
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws10(self):
        num_noise = 10
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise50_izotope(self):
        num_noise = 50
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise100_izotope(self):
        num_noise = 100
        if write_flag:
            out_filepath = learn_noise_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)

    # SEMI-SUP LEARN PIANO - MADEINIT AS DMGED
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws1(self):
        num_noise = 1
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws2(self):
        num_noise = 2
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws3(self):
        num_noise = 3
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws5(self):
        num_noise = 5
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise19wdws10(self):
        num_noise = 10
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise50_izotope(self):
        num_noise = 50
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)

    def test_restore_brahms_sslrn_madeinit_noisebvnum_lnoise_noise100_izotope(self):
        num_noise = 100
        if write_flag:
            out_filepath = learn_piano_test_path + 'sslrn_madeinit_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)


    # NUM NOISE BV TESTS - FOR SUPERVISED
    def test_restore_brahms_sup_noise19wdws1(self):
        num_noise = 1
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sup_noise19wdws2(self):
        num_noise = 2
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)

    def test_restore_brahms_sup_noise19wdws3(self):
        num_noise = 3
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sup_noise19wdws5(self):
        num_noise = 5
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
                                      
    def test_restore_brahms_sup_noise19wdws10(self):
        num_noise = 10
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_19wdws.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, noise_stop=25, write_noise_sig=True, dmged_pianobv=True, prec_noise=True)
    
    def test_restore_brahms_sup_noise50_izotope(self):
        num_noise = 50
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)

    def test_restore_brahms_sup_noise100_izotope(self):
        num_noise = 100
        if write_flag:
            out_filepath = sup_test_path + 'sup_noisebv' + str(num_noise) + '_izotope.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise, write_noise_sig=True, dmged_pianobv=True)


if __name__ == '__main__':
    unittest.main()
