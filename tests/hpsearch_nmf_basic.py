from brahms_restore_ml.nmf import nmf
import unittest
from scipy.io import wavfile
import os

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 1

brahms_filepath = os.getcwd() + '/brahms.wav'
mary_441kHz_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary_44100Hz_32bitfp_librosa.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_hpsearch/output_hpsearch_basic/'
# Hp-search
# This script & output path is for testing & comparing the best results using each respective feature

class BasicRestoreTests(unittest.TestCase):
    # Brahms for these tests (hp-tuning grid search)
    # 0 factors
    def test_restore_brahms_bare(self):
        out_filepath = test_path + 'restored_brahms_bare.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=False, noisebv=False, avgbv=False,
                                      write_file=write_flag, debug=debug_flag)
    
    # 1 factor
    def test_restore_brahms_ova(self):
        print('Testing restore brahms: ova')

        out_filepath = test_path + 'restored_brahms_ova.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        noisebv=False, avgbv=False,
                                        write_file=write_flag, debug=debug_flag)

    def test_restore_brahms_avgbv(self):
        print('Testing restore brahms: avg')

        out_filepath = test_path + 'restored_brahms_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        ova=False, noisebv=False,
                                        write_file=write_flag, debug=debug_flag)

    def test_restore_brahms_noisebv(self):
        print('Testing restore brahms: noise')

        out_filepath = test_path + 'restored_brahms_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        ova=False, avgbv=False,
                                        num_noisebv=num_noise_bv_test, write_file=write_flag, debug=debug_flag)

    # Two factors
    def test_restore_brahms_ova_avgbv(self):
        print('Testing restore brahms: ova, avg')

        out_filepath = test_path + 'restored_brahms_ova_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        noisebv=False,
                                        write_file=write_flag, debug=debug_flag)

    def test_restore_brahms_ova_noisebv(self):
        print('Testing restore brahms: ova, noise')

        out_filepath = test_path + 'restored_brahms_ova_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      avgbv=False, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

    def test_restore_brahms_avgbv_noisebv(self):
        print('Testing restore brahms: avg, noise')

        out_filepath = test_path + 'restored_brahms_avgbv_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=False, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

    # Three factors
    def test_restore_brahms_ova_noisebv_avgbv(self):
        print('Testing restore brahms: ova, avg, noise')

        out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_with_nmf(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      num_noisebv=num_noise_bv_test,
                                      write_file=write_flag, debug=debug_flag)

    # Mary had a little lamb tests
    def test_mary_bvs_supervised_nmf(self):
        mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
        write_path = test_path + 'mary_bvs_nmf.wav'
        nmf.restore_with_nmf(mary_sig, nmf.PIANO_WDW_SIZE, write_path, mary_sr, marybv=True, 
                        noisebv=False)

    def test_mary_all_bvs_supervised_nmf(self):
        mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
        write_path = test_path + 'mary_allbvs_nmf.wav'
        nmf.restore_with_nmf(mary_sig, nmf.PIANO_WDW_SIZE, write_path, mary_sr, marybv=False, 
                        noisebv=False)

    def test_mary_bvs_supervised_nmf_wnoisebvs(self):       # experimental
        mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
        write_path = test_path + 'mary_bvs_nmf_noisebvs.wav'
        nmf.restore_with_nmf(mary_sig, nmf.PIANO_WDW_SIZE, write_path, mary_sr, marybv=True, 
                        noisebv=True)

    def test_mary_all_bvs_supervised_nmf_wnoisebvs(self):   # experimental
        mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
        write_path = test_path + 'mary_allbvs_nmf_noisebvs.wav'
        nmf.restore_with_nmf(mary_sig, nmf.PIANO_WDW_SIZE, write_path, mary_sr, marybv=False, 
                        noisebv=True)


if __name__ == '__main__':
    unittest.main()
