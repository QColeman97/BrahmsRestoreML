from brahms_restore_ml.nmf.nmf import *
import unittest
from scipy.io import wavfile
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 1 # 2
l1_penalty_test = 4096
learn_iter_test = 100

brahms_filepath = os.getcwd() + '/brahms.wav'
# mary_filepath = 'brahms_restore_ml/nmf/Mary.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_sslrn/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class SemiSupLearnTests(unittest.TestCase):
    # SEMISUP LEARN TEST #
    def test_restore_brahms_ssln_piano_madeinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_sslrn_piano_madeinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True)


    def test_restore_brahms_ssln_piano_randinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_sslrn_piano_randinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True)


    def test_restore_brahms_ssln_noise_madeinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_sslrn_noise_madeinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True)


    def test_restore_brahms_ssln_noise_randinit(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_sslrn_noise_randinit.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True)


if __name__ == '__main__':
    unittest.main()
