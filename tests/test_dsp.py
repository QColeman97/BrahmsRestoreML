# test_dsp.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for dsp functions.

# Run with $ python -m unittest tests.test_dsp

# Test on Mary.wav for general purposes
# Test on brahms.wav for listening

from brahms_restore_ml.audio_data_processing import *
import unittest
import numpy as np

mary_441kHz_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary_44100Hz_32bitfp_librosa.wav'
piano_filepath = os.getcwd() + '/brahms_restore_ml/nmf/piano.wav'
brahms_filepath = os.getcwd() + '/brahms.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_dsp/'

# Testing global vars
write_flag = True
debug_flag = False

class DSPTests(unittest.TestCase):
    # wavfile tests
    def test_convert_wav_format_up(self):
        sig = np.array([0, 128, 255], dtype='uint8')
        conv_sig = convert_wav_format_up(sig)
        correct = np.array([-32768, 0, 32512], dtype='int16')
        np.testing.assert_array_equal(conv_sig, correct)
    
    def test_convert_wav_format_down(self):
        sig = np.array([-32768, 0, 32767], dtype='int16')
        conv_sig = convert_wav_format_down(sig)
        correct = np.array([0, 128, 255], dtype='uint8')
        np.testing.assert_array_equal(conv_sig, correct)

    def test_convert_wav_format_down_safe_uint8(self):
        sig = np.array([-40000, -32768, 0, 32767, 40000], dtype='float64')
        conv_sig = convert_wav_format_down(sig)
        correct = np.array([0, 0, 128, 255, 255], dtype='uint8')
        np.testing.assert_array_equal(conv_sig, correct)

    def test_convert_wav_format_down_unsafe_uint8(self):
        sig = np.array([-40000, -32768, 0, 32767, 40000], dtype='float64')
        conv_sig = convert_wav_format_down(sig, safe=False)
        correct = np.array([0, 0, 128, 255, 255], dtype='uint8')
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, conv_sig, correct)

    def test_convert_wav_format_down_safe_float32(self):
        sig = np.array([-40000, -32768, 0, 32767, 40000], dtype='float64')
        conv_sig = convert_wav_format_down(sig, to_bit_depth='float32')
        correct = np.array([-1., -1., 0., 1., 1.], dtype='float32')
        np.testing.assert_array_equal(conv_sig, correct)

    def test_convert_wav_format_down_unsafe_float32(self):
        sig = np.array([-40000, -32768, 0, 32767, 40000], dtype='float64')
        conv_sig = convert_wav_format_down(sig, to_bit_depth='float32', safe=False)
        correct = np.array([-1., -1., 0., 1., 1.], dtype='float32')
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, conv_sig, correct)

    def test_sig_to_pos_mag_fft(self):
        # Test this returns of the mags: zero freq, pos freqs & nyquist freq
        sig = np.array([1.,0.,-1.,0.])
        mag, _ = signal_to_pos_fft(sig, len(sig), ova=True)
        self.assertEqual(mag.shape, ((len(sig)//2)+1,))

    def test_pos_mag_fft_to_sig(self):
        pmfft, pmphases = np.array([1.,2.,3.,4.]), np.array([1.,2.,3.,4.])
        sig = pos_fft_to_signal(pmfft, pmphases, 6, ova=False)
        self.assertEqual(sig.shape, (6,))   
    
    def test_pos_mag_fft_to_sig_ova1(self):
        pmfft, pmphases = np.array([1.,2.,3.,4.]), np.array([1.,2.,3.,4.])
        end_of_bigger_sig = np.array([1.,1.,1.])
        sig = pos_fft_to_signal(pmfft, pmphases, 6, ova=True, end_sig=end_of_bigger_sig)
        begin_sig = pos_fft_to_signal(pmfft, pmphases, 6, ova=True)
        np.testing.assert_array_equal(sig[3:], begin_sig[3:])

    def test_pos_mag_fft_to_sig_ova2(self):
        pmfft, pmphases = np.array([1.,2.,3.,4.]), np.array([1.,2.,3.,4.])
        end_of_bigger_sig = np.array([1.,1.,1.])
        sig = pos_fft_to_signal(pmfft, pmphases, 6, ova=True, end_sig=end_of_bigger_sig)
        begin_sig = pos_fft_to_signal(pmfft, pmphases, 6, ova=True)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, sig[:3], begin_sig[:3])
    
    def test_sig_to_pmfft_to_sig(self):
        # Test correct time-domain - freq-domain conversion
        sig = np.array([1.,0.,-1.,0.])
        wdw_size = len(sig)
        pmfft, pmphases = signal_to_pos_fft(sig, wdw_size, ova=True)
        synth_sig = pos_fft_to_signal(pmfft, pmphases, wdw_size, ova=True)
        np.testing.assert_array_equal(sig, synth_sig)
    
    def test_stft(self):
        # case: fits with windows
        sig = np.array([1.,0.,-1.,0., 1, 0])
        spgm, _ = make_spectrogram(sig, 4, EPSILON, ova=True)
        t_sgmts, n_feat = 2, 3
        self.assertEqual(spgm.shape, (t_sgmts, n_feat))

    def test_stft_nofit(self):
        # case: doesn't fit with windows
        sig = np.array([1.,0.,-1.,0.,1,0,-1.,0.,1.,0.,-1.,0.,1.,0.])
        spgm, _ = make_spectrogram(sig, 8, EPSILON, ova=True)
        t_sgmts, n_feat = 3, 5
        self.assertEqual(spgm.shape, (t_sgmts, n_feat))

    def test_istft(self):
        spgm = np.array([[0.,1.,0.], [0.,0.5,0.]])
        phases = np.array([[0.5,0.5,0.5], [2.,1.,0.]])
        synthetic_sig = make_synthetic_signal(spgm, phases, 4, 'uint8', ova=True)
        self.assertEqual(synthetic_sig.shape, (6,))

    def test_sig_to_stft_to_sig_uint8(self):
        # Test correct time-amp-domain - time-freq-amp-domain conversion
        sig = np.array([0,128,255,128,0,128,255], dtype='uint8')
        wdw_size = 4
        spgm, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=True)
        synth_sig = make_synthetic_signal(spgm, phases, wdw_size, 'int16', ova=True)
        self.assertTrue(synth_sig[1] < synth_sig[2] and
                        synth_sig[2] > synth_sig[3] and synth_sig[3] > synth_sig[4] and
                        synth_sig[4] < synth_sig[5])

    def test_sig_to_stft_to_sig_float32(self):
        # Test correct time-amp-domain - time-freq-amp-domain conversion
        sig = np.array([-1.,0.,1.,0.,-1.,0.,1.], dtype='float32')
        wdw_size = 4
        spgm, phases = make_spectrogram(sig, wdw_size, EPSILON, ova=True)
        synth_sig = make_synthetic_signal(spgm, phases, wdw_size, 'float32', ova=True)
        print('synth sig:', synth_sig)
        self.assertTrue(synth_sig[1] < synth_sig[2] and
                        synth_sig[2] > synth_sig[3] and synth_sig[3] > synth_sig[4] and
                        synth_sig[4] < synth_sig[5])

    def test_reconstruct_mary(self):
        sig_sr, sig = wavfile.read(mary_441kHz_filepath)
        write_path = test_path + 'mary_reconstructed.wav'
        reconstruct_audio(sig, PIANO_WDW_SIZE, write_path, sig_sr, ova=True, 
                          write_file=True, debug=False)

    def test_reconstruct_piano_sample(self):
        sig_sr, sig = wavfile.read(piano_filepath)
        write_path = test_path + 'piano_reconstructed.wav'
        reconstruct_audio(sig, PIANO_WDW_SIZE, write_path, sig_sr, ova=True, 
                          write_file=True, debug=False)

    def test_reconstruct_brahms(self):
        sig_sr, sig = wavfile.read(brahms_filepath)
        write_path = test_path + 'brahms_reconstructed.wav'
        reconstruct_audio(sig, PIANO_WDW_SIZE, write_path, sig_sr, ova=True, 
                          write_file=True, debug=False)

    # TODO? test if sum reconstructed sig = sum input sig
        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(nmf.PIANO_WDW_SIZE // 2): -(nmf.PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(nmf.PIANO_WDW_SIZE // 2): (len(sig) - (nmf.PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

if __name__ == '__main__':
    unittest.main()
