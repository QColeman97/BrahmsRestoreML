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

penalty_test_path = '/Users/quinnmc/Desktop/AudioRestore/output_test_penalty/'

class RestorePenaltyTests(unittest.TestCase):

    # Varying L1-Penalty Tests #
    # Penalize the Piano Activations, Noise Activations, or All Activations #
    # Piano Activations 
    def test_restore_brahms_pianoh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_pianoh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_pianoh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_pianoh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

    #     print('Piano H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_pianoh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ph' + str(pen) + 'l1pen.wav'

        print('Piano H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # Noise Activations 
    def test_restore_brahms_noiseh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'
        
        print('Noise H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_noiseh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_noiseh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_noiseh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

    #     print('Noise H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_noiseh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_nh' + str(pen) + 'l1pen.wav'

        print('Noise H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Piano', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # # All Activations 
    def test_restore_brahms_allh_l1pen1(self):
        pen = 1
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Non-Penalized:')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test)

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_allh_l1pen2(self):
    #     pen = 2
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen4(self):
    #     pen = 4
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen8(self):
    #     pen = 8
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16(self):
    #     pen = 16
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen32(self):
    #     pen = 32
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen64(self):
    #     pen = 64
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen128(self):
    #     pen = 128
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen256(self):
    #     pen = 256
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen512(self):
    #     pen = 512
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen1024(self):
    #     pen = 1024
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen2048(self):
    #     pen = 2048
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen4096(self):
    #     pen = 4096
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen8192(self):
    #     pen = 8192
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16384(self):
    #     pen = 16384
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen32678(self):
    #     pen = 32768
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen65536(self):
    #     pen = 65536
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)
    
    # def test_restore_brahms_allh_l1pen131072(self):
    #     pen = 131072
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen262144(self):
    #     pen = 262144
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen524288(self):
    #     pen = 524288
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen1048576(self):
    #     pen = 1048576
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen2097152(self):
    #     pen = 2097152
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # def test_restore_brahms_allh_l1pen4194304(self):
    #     pen = 4194304
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)


    # def test_restore_brahms_allh_l1pen8388608(self):
    #     pen = 8388608
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    # def test_restore_brahms_allh_l1pen16777216(self):
    #     pen = 16777216
    #     sr, sig = wavfile.read(brahms_filepath)
    #     orig_sig_type = sig.dtype
    #     if write_flag:
    #         out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

    #     print('All H Penalty ' + str(pen) + ':')
    #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
    #                                   semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
    #                                   num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen33554432(self):
        pen = 33554432
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen67108864(self):
        pen = 67108864
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen134217728(self):
        pen = 134217728
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

    def test_restore_brahms_allh_l1pen268435456(self):
        pen = 268435456
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = penalty_test_path + 'restored_brahms_ah' + str(pen) + 'l1pen.wav'

        print('All H Penalty ' + str(pen) + ':')
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=write_flag, debug=debug_flag, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen)

if __name__ == '__main__':
    unittest.main()
