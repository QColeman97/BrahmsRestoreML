from brahms_restore_ml.nmf import nmf
import unittest
from scipy.io import wavfile
import os

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 5

brahms_filepath = os.getcwd() + '/brahms.wav'
mary_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_basic/'

# This script & output path is for testing & comparing the best results using each respective feature

class BasicRestoreTests(unittest.TestCase):
    # Brahms for these tests (bad audio)
    def test_restore_brahms_bare(self):
        # print('Testing restore brahms: bare')
        # print('Real dir:', os.path.dirname(os.path.realpath(__file__)))
        # print('Brahms path?', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'conf.json'))
        # print('Curr?', os.getcwd())

        out_filepath = test_path + 'restored_brahms_bare.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=False, noisebv=False, avgbv=False,
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_ova(self):
        print('Testing restore brahms: ova')

        out_filepath = test_path + 'restored_brahms_ova.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        noisebv=False, avgbv=False,
                                        write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_avgbv(self):
        print('Testing restore brahms: avg')

        out_filepath = test_path + 'restored_brahms_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        ova=False, noisebv=False,
                                        write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_noisebv(self):
        print('Testing restore brahms: noise')

        out_filepath = test_path + 'restored_brahms_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        ova=False, avgbv=False,
                                        num_noisebv=num_noise_bv_test, write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    # Two factors
    def test_restore_brahms_ova_avgbv(self):
        print('Testing restore brahms: ova, avg')

        out_filepath = test_path + 'restored_brahms_ova_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                        noisebv=False,
                                        write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_ova_noisebv(self):
        print('Testing restore brahms: ova, noise')

        out_filepath = test_path + 'restored_brahms_ova_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      avgbv=False, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_avgbv_noisebv(self):
        print('Testing restore brahms: avg, noise')

        out_filepath = test_path + 'restored_brahms_avgbv_noisebv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=False, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    # Three factors
    def test_restore_brahms_ova_noisebv_avgbv(self):
        print('Testing restore brahms: ova, avg, noise')

        out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv.wav' if write_flag else None
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr,
                                      num_noisebv=num_noise_bv_test,
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)


if __name__ == '__main__':
    unittest.main()
