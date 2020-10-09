import unittest
import sys
sys.path.append('/Users/quinnmc/Desktop/AudioRestore/restore_audio')
from restore_audio import *

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 5

brahms_filepath = '/Users/quinnmc/Desktop/AudioRestore/brahms.wav'
# brahms_filepath = '/Users/quinnmc/Desktop/AudioRestore/brahms_16bitPCM.wav'
mary_filepath = '/Users/quinnmc/Desktop/AudioRestore/Mary.wav'
test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv/'
# test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_writefix_newbv_from16bitBrahms/'

class BasicRestoreTests(unittest.TestCase):
    # Brahms for these tests (bad audio)
    def test_restore_brahms_bare(self):
        print('Testing restore brahms: bare')

        if write_flag:
            out_filepath = test_path + 'restored_brahms.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_ova(self):
        print('Testing restore brahms: ova')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True,
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_avgbv(self):
        print('Testing restore brahms: avg')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_avgbv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, avgbv=True,
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_noisebv(self):
        print('Testing restore brahms: noise')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_noisebv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, noisebv=True, 
                                      num_noisebv=num_noise_bv_test, write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    # Two factors
    def test_restore_brahms_ova_avgbv(self):
        print('Testing restore brahms: ova, avg')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova_avgbv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=True, avgbv=True, write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_ova_noisebv(self):
        print('Testing restore brahms: ova, noise')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova_noisebv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=True, noisebv=True, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_brahms_avgbv_noisebv(self):
        print('Testing restore brahms: avg, noise')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_avgbv_noisebv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      avgbv=True, noisebv=True, num_noisebv=num_noise_bv_test, 
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    # Three factors
    def test_restore_brahms_ova_noisebv_avgbv(self):
        print('Testing restore brahms: ova, avg, noise')

        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=True, noisebv=True, num_noisebv=num_noise_bv_test, avgbv=True, 
                                      write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)


if __name__ == '__main__':
    unittest.main()
