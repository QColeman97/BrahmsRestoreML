import unittest
import sys
sys.path.append('/Users/quinnmc/Desktop/AudioRestore/restore_audio')
from restore_audio import *
import soundfile

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10
l1_penalty_test = 4096
learn_iter_test = 100

noise_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_noise/'
noise_learnnoise_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_noise_learnnoise/'

class RestoreNoiseTests(unittest.TestCase):

    # NUM NOISE BV TESTS #

    # # Sanity Checkmark - noise spectrogram looks like a rank-1 approx.
    # def test_restore_brahms_noisebvnum_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_noise19wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # # Step up window count of noise bv's

    # def test_restore_brahms_noisebvnum_noise77wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws30(self):
    #     num_noise = 30
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws35(self):
    #     num_noise = 35
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws40(self):
    #     num_noise = 40
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws45(self):
    #     num_noise = 45
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws50(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws55(self):
    #     num_noise = 55
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws60(self):
    #     num_noise = 60
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws65(self):
    #     num_noise = 65
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws70(self):
    #     num_noise = 70
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws75(self):
    #     num_noise = 75
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_noise77wdws77(self):
    #     num_noise = 77
    #     if write_flag:
    #         out_filepath = noise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)


    # NUM NOISE BV TESTS - FOR LEARNING THE NOISE (NOT LEARNING PIANO)

    # # Sanity Checkmark - noise spectrogram looks like a rank-1 approx.
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                        
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)
                                      
    # def test_restore_brahms_noisebvnum_lnoise_noise19wdws19(self):
    #     num_noise = 19
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_19wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=25, write_noise_sig=True)

                                      
    # Step up window count of noise bv's

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws1(self):
    #     num_noise = 1
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws2(self):
    #     num_noise = 2
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws3(self):
    #     num_noise = 3
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws4(self):
    #     num_noise = 4
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws5(self):
    #     num_noise = 5
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws6(self):
    #     num_noise = 6
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws7(self):
    #     num_noise = 7
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws8(self):
    #     num_noise = 8
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws9(self):
    #     num_noise = 9
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws10(self):
    #     num_noise = 10
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    
    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws11(self):
    #     num_noise = 11
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws12(self):
    #     num_noise = 12
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws13(self):
    #     num_noise = 13
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws14(self):
    #     num_noise = 14
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)


    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws15(self):
    #     num_noise = 15
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws20(self):
    #     num_noise = 20
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws25(self):
    #     num_noise = 25
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws30(self):
    #     num_noise = 30
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws35(self):
    #     num_noise = 35
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws40(self):
    #     num_noise = 40
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws45(self):
    #     num_noise = 45
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws50(self):
    #     num_noise = 50
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws55(self):
    #     num_noise = 55
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws60(self):
    #     num_noise = 60
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws65(self):
    #     num_noise = 65
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws70(self):
    #     num_noise = 70
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws75(self):
    #     num_noise = 75
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

    # def test_restore_brahms_noisebvnum_lnoise_noise77wdws77(self):
    #     num_noise = 77
    #     if write_flag:
    #         out_filepath = noise_learnnoise_test_path + 'restored_brahms_noisebv' + str(num_noise) + '_77wdws.wav'
    #     sr, sig = wavfile.read(brahms_filepath)
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise, noise_stop=83, write_noise_sig=True)

if __name__ == '__main__':
    unittest.main()
