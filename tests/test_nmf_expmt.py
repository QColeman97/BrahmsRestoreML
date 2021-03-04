# from brahms_restore_ml.nmf import nmf
from brahms_restore_ml.nmf.nmf import WDW_NUM_AFTER_VOICE, restore_audio
from brahms_restore_ml.nmf import basis_vectors as bv
from brahms_restore_ml import audio_data_processing as dsp
import unittest
from scipy.io import wavfile
import soundfile
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10
l1_penalty_test = 4096
learn_iter_test = 100

brahms_filepath = 'brahms.wav'
# mary_filepath = 'brahms_restore_ml/nmf/Mary.wav'
test_path = 'brahms_restore_ml/nmf/output/output_test/output_test_other/'

class ExperimentalTests(unittest.TestCase):

    # SNR - convert to 16bit signals for better granularity when calc SNR
    def test_brahms_snr(self):
        sr, sig = wavfile.read(brahms_filepath)
        signal = convert_sig_8bit_to_16bit(sig)
        brahms_sig = np.array([((x[0] + x[1]) / 2) for x in signal.astype('float64')])
        # Keep it the length NMF deals with and outputs
        brahms_sig = brahms_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)].copy()  

        # Clean sigs - for DLNN
        x, clean_sig1 = wavfile.read(brahms_clean_filepath + '1intempo.wav')
        # print('CLEAN SIG 1 DATATYPE:', clean_sig1.dtype, '\n')
        clean_datatype = clean_sig1.dtype
        clean_sig1 = np.array([((x[0] + x[1]) / 2) for x in clean_sig1.astype('float64')])
        x, clean_sig2 = wavfile.read(brahms_clean_filepath + '2intempo_24SNR.wav')
        clean_sig2 = np.array([((x[0] + x[1]) / 2) for x in clean_sig2.astype('float64')])
        x, clean_sig3 = wavfile.read(brahms_clean_filepath + '3intempo.wav')
        clean_sig3 = np.array([((x[0] + x[1]) / 2) for x in clean_sig3.astype('float64')])
        x, clean_sig4 = wavfile.read(brahms_clean_filepath + '4intempo.wav')
        clean_sig4 = np.array([((x[0] + x[1]) / 2) for x in clean_sig4.astype('float64')])
        x, clean_sig5 = wavfile.read(brahms_clean_filepath + '5.wav')
        clean_sig5 = np.array([((x[0] + x[1]) / 2) for x in clean_sig5.astype('float64')])
        x, clean_sig6 = wavfile.read(brahms_clean_filepath + '6.wav')
        clean_sig6 = np.array([((x[0] + x[1]) / 2) for x in clean_sig6.astype('float64')])
        x, clean_sig7 = wavfile.read(brahms_clean_filepath + '7_5SNR.wav')
        clean_sig7 = np.array([((x[0] + x[1]) / 2) for x in clean_sig7.astype('float64')])
        x, clean_sig8 = wavfile.read(brahms_clean_filepath + '8.wav')
        clean_sig8 = np.array([((x[0] + x[1]) / 2) for x in clean_sig8.astype('float64')])

        out_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms_noise_19noisewdws_fixednoise.wav'
        # Noise
        output_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='None', semisupmadeinit=True, write_file=False, debug=False, 
                                      num_noisebv=num_noise_bv_test, noise_stop=25, return_16bit=True)
       
        brahms_noise_sig = output_sig.astype('float64')[:len(brahms_sig)].copy()
        # noise_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms_justnoise_19noisewdws_fixednoise.wav'
        # _, brahms_noise_sig = wavfile.read(noise_filepath)
        # brahms_noise_sig = brahms_noise_sig.astype('float64')

        # print('Len of brahms:', len(brahms_noise_sig), len(brahms_sig), '2x len:', len(output_sig))

        # power = sum of abs squares of samples / length
        # SNR = ratio of Psig to Pnoise
        brahms_noise_sig = brahms_noise_sig * (4/3)

        sig_val = np.mean(brahms_sig ** 2)
        noise_val = np.mean(brahms_noise_sig ** 2)
        # target_val = sig_val - noise_val
        snr = sig_val / noise_val
        print('SNR kinda:', snr)

        # print('Clean sig 1:', clean_sig1.dtype, clean_sig1[1000000:1000050])
        clean_sig_val1 = np.mean(clean_sig1 ** 2)
        clean_snr1 = clean_sig_val1 / noise_val
        print('Clean SNR 1:', clean_snr1)
        clean_sig1_1 = clean_sig1 / 2
        clean_sig_val1_1 = np.mean(clean_sig1_1 ** 2)
        clean_snr1_1 = clean_sig_val1_1 / noise_val
        print('Clean SNR 1 Updated:', clean_snr1_1)
        clean_sig1_2 = clean_sig1 / 4
        clean_sig_val1_2 = np.mean(clean_sig1_2 ** 2)
        clean_snr1_2 = clean_sig_val1_2 / noise_val
        print('Clean SNR 1 Updated Again:', clean_snr1_2)
        wavfile.write(brahms_clean_filepath + '1_27SNR.wav', x, clean_sig1_2.astype(clean_datatype))
        
        # print('Clean sig 2:', clean_sig2.dtype, clean_sig2[1000000:1000050])
        clean_sig_val2 = np.mean(clean_sig2 ** 2)
        clean_snr2 = clean_sig_val2 / noise_val
        print('Clean SNR 2:', clean_snr2)
        
        # print('Clean sig 3:', clean_sig3.dtype, clean_sig3[1000000:1000050])
        clean_sig_val3 = np.mean(clean_sig3 ** 2)
        clean_snr3 = clean_sig_val3 / noise_val
        print('Clean SNR 3:', clean_snr3)
        clean_sig3_1 = clean_sig3 / 2
        clean_sig_val3_1 = np.mean(clean_sig3_1 ** 2)
        clean_snr3_1 = clean_sig_val3_1 / noise_val
        print('Clean SNR 3 Updated:', clean_snr3_1)
        wavfile.write(brahms_clean_filepath + '3_36SNR.wav', x, clean_sig3_1.astype(clean_datatype))

        # print('Clean sig 4:', clean_sig4.dtype, clean_sig4[1000000:1000050])
        clean_sig_val4 = np.mean(clean_sig4 ** 2)
        clean_snr4 = clean_sig_val4 / noise_val
        print('Clean SNR 4:', clean_snr4)
        clean_sig4_1 = clean_sig4 / 2
        clean_sig_val4_1 = np.mean(clean_sig4_1 ** 2)
        clean_snr4_1 = clean_sig_val4_1 / noise_val
        print('Clean SNR 4 Updated:', clean_snr4_1)
        wavfile.write(brahms_clean_filepath + '4_13SNR.wav', x, clean_sig4_1.astype(clean_datatype))
        
        # print('Clean sig 5:', clean_sig5.dtype, clean_sig5[1000000:1000050])
        clean_sig_val5 = np.mean(clean_sig5 ** 2)
        clean_snr5 = clean_sig_val5 / noise_val
        print('Clean SNR 5:', clean_snr5)
        clean_sig5_1 = clean_sig5 / 2
        clean_sig_val5_1 = np.mean(clean_sig5_1 ** 2)
        clean_snr5_1 = clean_sig_val5_1 / noise_val
        print('Clean SNR 5 Updated:', clean_snr5_1)
        wavfile.write(brahms_clean_filepath + '5_16SNR.wav', x, clean_sig5_1.astype(clean_datatype))
        
        # print('Clean sig 6:', clean_sig6.dtype, clean_sig6[1000000:1000050])
        clean_sig_val6 = np.mean(clean_sig6 ** 2)
        clean_snr6 = clean_sig_val6 / noise_val
        print('Clean SNR 6:', clean_snr6)
        clean_sig6_1 = clean_sig6 / 2
        clean_sig_val6_1 = np.mean(clean_sig6_1 ** 2)
        clean_snr6_1 = clean_sig_val6_1 / noise_val
        print('Clean SNR 6 Updated:', clean_snr6_1)
        wavfile.write(brahms_clean_filepath + '6_18SNR.wav', x, clean_sig6_1.astype(clean_datatype))
        
        # print('Clean sig 7:', clean_sig7.dtype, clean_sig7[1000000:1000050])
        clean_sig_val7 = np.mean(clean_sig7 ** 2)
        clean_snr7 = clean_sig_val7 / noise_val
        print('Clean SNR 7:', clean_snr7)
        
        # print('Clean sig 8:', clean_sig8.dtype, clean_sig8[1000000:1000050])
        clean_sig_val8 = np.mean(clean_sig8 ** 2)
        clean_snr8 = clean_sig_val8 / noise_val
        print('Clean SNR 8:', clean_snr8)
        clean_sig8_1 = clean_sig8 / 2
        clean_sig_val8_1 = np.mean(clean_sig8_1 ** 2)
        clean_snr8_1 = clean_sig_val8_1 / noise_val
        print('Clean SNR 8 Updated:', clean_snr8_1)
        wavfile.write(brahms_clean_filepath + '8_9SNR.wav', x, clean_sig8_1.astype(clean_datatype))

        # print('Brahms sig dtype:', brahms_sig.dtype, '\n', brahms_sig[1000000:1000050])
        # print('Brahms noise sig dtype:', brahms_noise_sig.dtype, '\n', brahms_noise_sig[1000000:1000050])

        # brahms_sig = brahms_sig[1000000:1000050].copy()
        # brahms_noise_sig = brahms_noise_sig[1000000:1000050].copy()

        # plt.plot(brahms_sig)
        # plt.plot(brahms_noise_sig)
        # plt.title('Brahms & Noise Signals')
        # plt.ylabel('Amplitude')
        # plt.show()

        # sig_val = np.mean(brahms_sig ** 2)
        # noise_val = np.mean(brahms_noise_sig ** 2)
        # snr = sig_val / noise_val
        # print('sgmt SNR:', snr)

        plt.plot(brahms_sig)
        plt.plot(brahms_noise_sig)
        plt.title('Brahms & Noise Signals')
        plt.ylabel('Amplitude')
        plt.show()


    # TEST UNSUP LEARN
    def test_restore_brahms_unsupnmf(self):
        sr, sig = wavfile.read(brahms_filepath)
        # sig, sr = soundfile.read(brahms_filepath)
        orig_sig_type = sig.dtype

        if write_flag:
            out_filepath = test_path + 'restored_brahms_unsupnmf.wav'

        # TEMP - don't let silence into sig (so not last part)
        # This makes matrix of nan's go away
        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]    # For nice comparing
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, NUM_PIANO_NOTES, debug=debug_flag)
        synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, 
                                              orig_sig_type, ova=True, debug=debug_flag)

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)
            # wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))
            # soundfile.write(out_filepath, synthetic_sig, sr, subtype='PCM_S8')


    # Output file noticably worse - crackles and a constant high frequency
    def test_restore_no_hanningbv_brahms(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_nohanbv_ova_noisebv_avgbv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
                                      avgbv=True, write_file=write_flag, debug=debug_flag, nohanbv=True)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_precise_noise_brahms(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova_precnoisebv_avgbv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
                                      avgbv=True, write_file=write_flag, debug=debug_flag, prec_noise=True)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)

    def test_restore_eqbv_brahms(self):
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ova_noisebv_avgbv_eqpianobv.wav'
        sr, sig = wavfile.read(brahms_filepath)
        synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr,
                                      ova=True, noisebv=True, avgbv=True, eqbv=True, write_file=write_flag, debug=debug_flag)

        # sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')])
        # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
        #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

        # self.assertEqual(sig_diff, 0)


    # Mess w/ params of this one test
    def test_restore_brahms_iter(self):
        learn_iter = 25
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_iter25.wav'
        
        given_basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, debug=debug_flag, num_noise=num_noise_bv_test)

        sig = sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
        orig_sig_len = len(sig)
        spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

        activations, basis_vectors = nmf_learn(spectrogram, (NUM_PIANO_NOTES + num_noise_bv_test), basis_vectors=given_basis_vectors,
                                               learn_index=num_noise_bv_test, madeinit=True, debug=debug_flag, incorrect=False, 
                                               learn_iter=learn_iter)
        
        noise_activations, noise_basis_vectors, piano_activations, piano_basis_vectors = noise_split_matrices(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        print('\n--Making Synthetic Spectrogram--\n')
        synthetic_piano_spgm = piano_basis_vectors @ piano_activations
        synthetic_noise_spgm = noise_basis_vectors @ noise_activations
        synthetic_spectrogram = np.concatenate((synthetic_noise_spgm, synthetic_piano_spgm), axis=1)

        # activations, basis_vectors = remove_noise_vectors(activations, basis_vectors, num_noise_bv_test, debug=debug_flag)
        # synthetic_spectrogram = basis_vectors @ activations

        synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, orig_sig_type, ova=True, debug=debug_flag)
        synthetic_sig = synthetic_sig[orig_sig_len:]

        if write_flag:
            wavfile.write(out_filepath, sr, synthetic_sig)

        mean_abs_error = np.mean(np.abs(spectrogram - synthetic_spectrogram[:, :spectrogram.shape[1]]))
        print('MAE @ 25 iter:', mean_abs_error)

    
    def test_diff_noisebv_num_look(self):
        num_noise = 50
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
        make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag)



    # Probably a bug in NMF
    # FIXED PIANO BV CONSTANT FREQUENCY - TESTS

    # Observation: The higher the penalty, the higher the const freq
    # High freq - most easily noticable in fixed piano semisup-learning
    def test_fixpianobv_highfreq_1(self):
        pen = 0
        sr, sig = wavfile.read(brahms_filepath)
        orig_sig_type = sig.dtype
        if write_flag:
            out_filepath = test_path + 'restored_brahms_ph' + str(pen) + 'l1pen_EQMEANMOD_2_A7bvadjust.wav'
        else:
            out_filepath = None

        synthetic_sig = nmf.restore_audio(sig, nmf.PIANO_WDW_SIZE, out_filepath, sr, ova=True, noisebv=True, avgbv=True, 
                                      semisuplearn='Noise', semisupmadeinit=True, write_file=True, debug=True, 
                                      num_noisebv=num_noise_bv_test, l1_penalty=pen, eqbv=True)


if __name__ == '__main__':
    unittest.main()
