from brahms_restore_ml.nmf.nmf import *
import unittest
from scipy.io import wavfile

# Testing global vars
write_flag = True
debug_flag = False
# semi-sup results don't depend on num noisebv's, except when not - needing ~50
num_noise_bv_test = 50 # 5 # 50 # 2

brahms_filepath = os.getcwd() + '/brahms.wav'
# mary_filepath = 'brahms_restore_ml/nmf/Mary.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_sslrn/'
limits_test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_sslrn_limits/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class SemiSupLearnTests(unittest.TestCase):
    # SEMISUP LEARN PARAMS #
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

    # SEMISUP LEARN - RAND-INIT BASIS VECTORS RATIO PARAMS #
    def test_restore_brahms_ssln_piano_randinit_100(self):
        num_pianobvs = 100
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_piano_randinit_' + str(num_pianobvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True, num_pbv_unlocked=num_pianobvs)
    
    def test_restore_brahms_ssln_piano_randinit_500(self):
        num_pianobvs = 500
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_piano_randinit_' + str(num_pianobvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True, num_pbv_unlocked=num_pianobvs)

    def test_restore_brahms_ssln_piano_randinit_1000(self):
        num_pianobvs = 1000
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_piano_randinit_' + str(num_pianobvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, write_noise_sig=True, num_pbv_unlocked=num_pianobvs)
    
    def test_restore_brahms_ssln_piano_randinit_equalratio(self):
        num_pianobvs = 100
        num_noisebvs = 100
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_piano_randinit_equalratio.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noisebvs, write_noise_sig=True, num_pbv_unlocked=num_pianobvs)

    def test_restore_brahms_ssln_noise_randinit_100(self):
        num_noisebvs = 100
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_noise_randinit_' + str(num_noisebvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noisebvs, write_noise_sig=True)

    def test_restore_brahms_ssln_noise_randinit_500(self):
        num_noisebvs = 500
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_noise_randinit_' + str(num_noisebvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noisebvs, write_noise_sig=True)

    def test_restore_brahms_ssln_noise_randinit_1000(self):
        num_noisebvs = 1000
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_noise_randinit_' + str(num_noisebvs) + '.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noisebvs, write_noise_sig=True)
    
    def test_restore_brahms_ssln_noise_randinit_equalratio(self):
        num_noisebvs = NUM_SCORE_NOTES
        num_pianobvs = NUM_SCORE_NOTES
        if write_flag:
            out_filepath = limits_test_path + 'restored_brahms_sslrn_noise_randinit_equalratio.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_with_nmf(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=False, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noisebvs, write_noise_sig=True, num_pbv_unlocked=num_pianobvs)


if __name__ == '__main__':
    unittest.main()
