# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for restore_audio.py.

# Testing on Mary.wav for general purposes

# Things to test:
# - sig reconstruction
# - basis vectors correct
# -

import unittest
import sys
# sys.path.append('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/restore_audio')
# from restore_audio import *
import soundfile
from scipy import stats
# from sklearn.metrics import mean_absolute_error

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10
l1_penalty_test = 100
learn_iter_test = 25

brahms_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms.wav'
brahms_clean_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreDLNN/HungarianDanceNo1Rec/(youtube)wav/BrahmsHungDance1_'
mary_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/Mary.wav'
# mary32_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/Mary_44100Hz.wav'
piano_echo_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/piano-echo.wav'
test_path = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/output_test_other/'
# noise_test_path = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/output_test_noise/'

class RestOfTests(unittest.TestCase):

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

    
    # # Partial Brahms Sig Playground
    # def test_partial_brahms(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     start, stop = 0, 25
    #     out_filepath = ('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms_partial_wav/sgmt' 
    #                     + str(start) + '-' + str(stop) + '.wav')

    #     write_partial_sig(sig, PIANO_WDW_SIZE, start, stop, out_filepath, sr)
    
    # # Pointless test? Begin noise basis vector number tests
    # # def test_bv_sound(self):
    # #     num_noise = 25
    # #     noise_bvs = np.array(make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, 
    # #                                                   ova=True, debug=debug_flag)).T
    # #     out_filepath = noise_test_path + 'NoiseBasisVectors' + str(num_noise) + '.wav'
    # #     wavfile.write(out_filepath, 44100, noise_bvs)


    # # Begin signal conversion tests
    # def test_no_data_loss(self):
    #     # The multiplicative factor needed in OVA is dependant on window size (wdw_size = 4, factor = 4/3)
    #     # Try to find a relationship
    #     wdw_size = 4096 # 4
    #     # 8-BIT SIGNAL
    #     # arr = np.random.rand(500, 2) * 255
    #     arr = np.random.rand(20480) * 255
    #     # arr = np.random.rand(500) * 255
    #     arr = arr.astype('uint8')
    #     # 16-BIT SIGNAL
    #     # arr = np.random.rand(500) * 16384
    #     # arr = arr - 32768
    #     # arr = arr.astype('int16')
    #     orig_arr = arr
    #     # If not a mono signal
    #     # orig_arr = np.array([((x[0] + x[1]) / 2) for x in arr.astype('float64')]).astype('uint8')

    #     # comp_orig_arr = convert_sig_8bit_to_16bit(orig_arr)
    #     # # sig = np.array(signal).astype('float64')
    #     # comp_orig_arr = comp_orig_arr.astype('float64')

    #     # CONVERSION BACK AND FORTH
    #     # arr = convert_sig_8bit_to_16bit(arr)
    #     # arr = convert_sig_16bit_to_8bit(arr)
    #     # -- substitute
    #     spectrogram, phases = make_spectrogram(arr, wdw_size, ova=True, debug=debug_flag)
    #     arr = make_synthetic_signal(spectrogram, phases, wdw_size, orig_arr.dtype, ova=True, debug=debug_flag)

    #     # print('ABOUT to compare these:')
    #     # print(arr[(wdw_size // 2): (wdw_size // 2) + 20])
    #     # print(orig_arr[(wdw_size // 2): (wdw_size // 2) + 20])

    #     # print('MAE of arr and orig_arr:', mean_absolute_error(orig_arr, arr))
    #     # print('Orig arr:\n', orig_arr[:10])
    #     # print('New arr:\n', arr[:10])

    #     # ratios = comp_orig_arr[(wdw_size // 2): -(wdw_size // 2)].copy() / comp_arr[(wdw_size // 2): -(wdw_size // 2)].copy()
    #     # print('Ratios:\n', ratios)
    #     # # print('Sum of ratios:', np.sum(ratios))
    #     # # ratio = np.sum(ratios) / ratios.shape[0]
    #     # ratio = np.nanmean(ratios)
    #     # print('RATIO:', ratio)  # Correct ratio is 4/3 ~ 1.333

    #     # print('Plotting arrs before adjust by factor')
    #     # plt.plot(orig_arr)
    #     # plt.plot(arr)
    #     # plt.show()

    #     # # LOGIC FOR COMPARISON IN UNIT TEST
    #     # # Displace signal to center around 0, then do our ratio change
    #     # arr = arr.astype('int16')
    #     # arr = arr - 128

    #     # # Get rid of OVA artifacts for comparison
    #     # # orig_arr = orig_arr.astype('float64')
    #     # arr = arr.astype('float64')
    #     # arr *= (4/3) # ratio # Amplitude ratio made by this ova

    #     # # Bring signal to original position
    #     # # TRY - prob better than below - closer to actual factor change
    #     # arr = np.around(arr).astype('int16')
    #     # # arr = arr.astype('int16')
    #     # arr = arr + 128
    #     # arr = arr.astype('uint8')

    #     # print('Orig arr:\n', orig_arr)
    #     # print('Arr:\n', arr)
    #     # print('Plotting arrs after adjust by factor')
    #     # plt.plot(orig_arr)
    #     # plt.plot(arr)
    #     # plt.show()

    #     # # Swap these around to test each (first doesn't let second go)
    #     print('Compare converted arr (8 to 16 to 8 bit), to orig arr')
    #     np.testing.assert_array_equal(arr[(wdw_size // 2): -(wdw_size // 2)], orig_arr[(wdw_size // 2): -(wdw_size // 2)])


    # # THIS TEST NOT FINISHED
    # # Begin signal conversion tests
    # def test_no_data_loss_16bitversion(self):
    #     wdw_size = 4
    #     # arr = np.random.rand(500, 2) * 255
    #     arr = np.random.rand(200) * 255
    #     arr = arr.astype('uint8')
    #     orig_arr = arr
    #     # If not a mono signal
    #     # orig_arr = np.array([((x[0] + x[1]) / 2) for x in arr.astype('float64')]).astype('uint8')

    #     # FOR COMPARING OUR 16-bit CONVERSION, TO SOUNDFILE CONVERSION
    #     # wavfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_8bitPCM.wav', 44100, arr)
    #     # conv_8bit_sig, conv_8bit_sr = soundfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_8bitPCM.wav')
    #     # soundfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_16bitPCM.wav', conv_8bit_sig, conv_8bit_sr, subtype='PCM_16')
        
    #     sr_16bit, arr_16bit_file = wavfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_16bitPCM.wav')
    #     orig_16bit_arr = arr_16bit_file
    #     spectrogram, phases = make_spectrogram(arr_16bit_file, wdw_size, ova=True, debug=debug_flag)
    #     arr_16bit = make_synthetic_signal(spectrogram, phases, wdw_size, orig_16bit_arr.dtype, ova=True, debug=debug_flag)

    #     # CONVERSION BACK AND FORTH
    #     # arr = convert_sig_8bit_to_16bit(arr)
    #     # arr = convert_sig_16bit_to_8bit(arr)
    #     # -- substitute
    #     spectrogram, phases = make_spectrogram(arr, wdw_size, ova=True, debug=debug_flag)
    #     arr = make_synthetic_signal(spectrogram, phases, wdw_size, orig_arr.dtype, ova=True, debug=debug_flag)

    #     # LOGIC FOR COMPARISON IN UNIT TEST
    #     # Displace both signals to center around 0, then do our ratio change
    #     # orig_arr = orig_arr.astype('int16')
    #     # arr = arr.astype('int16')
    #     # orig_arr = orig_arr - 128
    #     # arr = arr - 128

    #     # So we can compare w/ 16bit Brahms
    #     # # orig_arr = orig_arr / 128
    #     # # arr = arr / 128
    #     # # orig_arr = orig_arr * 32768
    #     # # arr = arr * 32768
    #     # arr_16bit = arr_16bit / 32768
    #     # arr_16bit = arr_16bit * 128

    #     # # ratio = list(stats.mode(arr[(wdw_size // 2): -(wdw_size // 2)] /
    #     # #                         orig_arr[(wdw_size // 2): -(wdw_size // 2)])[0])[0]
    #     # ratios = orig_arr[(wdw_size // 2): -(wdw_size // 2)] / arr[(wdw_size // 2): -(wdw_size // 2)]
    #     # print('Ratios:\n', ratios)
    #     # ratio = sum(ratios) / len(ratios)
    #     # print('RATIO:', ratio)  # Correct ratio is 4/3 ~ 1.333

    #     # print('Plotting before')
    #     # plt.plot(orig_arr)
    #     # plt.plot(arr)
    #     # plt.plot(arr_16bit)
    #     # plt.show()

    #     # orig_arr = orig_arr.astype('float64')
    #     # arr = arr.astype('float64')
    #     # arr_16bit = arr_16bit.astype('float64')
    #     # # Get rid of OVA artifacts for comparison
    #     # # arr = arr[(wdw_size // 2): -(wdw_size // 2)]
    #     # # arr = arr.astype('float64')
    #     # arr *= (4/3) # ratio # Amplitude ratio made by this ova
    #     # # arr = arr.astype('uint8')
    #     # arr_16bit *= (4/3)

    #     # print('Orig as float:\n', orig_arr)
    #     # print('New as float:\n', arr)

    #     # orig_arr = np.around(orig_arr).astype('int16')
    #     # arr = np.around(arr).astype('int16')
    #     # arr_16bit = np.around(arr_16bit).astype('int16')

    #     # print('Orig as int:\n', orig_arr)
    #     # print('New as int:\n', arr)

    #     # print('Plotting after')
    #     # plt.plot(orig_arr)
    #     plt.plot(arr)
    #     plt.plot(arr_16bit)
    #     plt.show()

    #     if debug_flag:
    #         print('Amp ratio:', ratio)
    #         print('Orig arr:\n', orig_arr)
    #         print('Arr:\n', arr)

    #     # Standalone, compare our conv arr to 16bit orignally arr
    #     print('Compare converted arr (8 to 16 bit), to orig 16-bit arr')
    #     np.testing.assert_array_equal(arr[(wdw_size // 2): -(wdw_size // 2)], arr_16bit[(wdw_size // 2): -(wdw_size // 2)])
    #     # # Swap these around to test each (first doesn't let second go)
    #     # print('Compare converted arr (8 to 16 to 8 bit), to orig arr')
    #     # np.testing.assert_array_equal(arr[(wdw_size // 2): -(wdw_size // 2)], orig_arr[(wdw_size // 2): -(wdw_size // 2)])
    #     # print('Compare 16 bit arr (downscaled to 8 bit-ish), to orig arr')
    #     # np.testing.assert_array_equal(arr_16bit[(wdw_size // 2): -(wdw_size // 2)], orig_arr[(wdw_size // 2): -(wdw_size // 2)])

    # # Function to compare my conv (lossy) to soundfile & librosa, test their performance
    # def test_lossless_conv(self):
    #     signal = np.array([[120, 10], [251, 220], [0, 255]]).astype('uint8')

    #     _, sf_16bit = wavfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_16bitPCM.wav')

    #     my_16bit = signal.astype('int16')
    #     # Signed to unsigned
    #     my_16bit = my_16bit - 128
    #     # Bring to range [-1, 1]
    #     my_16bit = my_16bit / 128       # This line loses precision
    #     # Bring to range [-32768, 32767]
    #     my_16bit = my_16bit * 32768
    #     my_16bit = my_16bit.astype('int16')

    #     print('My 16bit (lossy):\n', my_16bit)
    #     print('Sound file 16bit:\n', sf_16bit)

    # def test_write_lossless_part1(self):
    #     signal = np.array([[120, 10], [251, 220], [0, 255]]).astype('uint8')
    #     wavfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_8bitPCM.wav', 44100, signal)

    # def test_write_lossless_part2(self):
    #     test_8bit_sig, sr = soundfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_8bitPCM.wav')
    #     soundfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/test_16bitPCM.wav', test_8bit_sig, 44100, subtype='PCM_16')

    # def test_convert_wav_8_to_16(self):
    #     sig, sr = soundfile.read(brahms_filepath)
    #     soundfile.write('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms_16bitPCM.wav', sig, sr, subtype='PCM_16')

    # # Bad - librosa doesn't let you write to a certain Wav bitdepth?
    # # def test_convert_wav_8_to_16_librosa(self):
    # #     sig, sr = librosa.load(brahms_filepath, sr=None)
    # #     librosa.output.write_wav('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/brahms_16bitPCM_libROSA.wav', sig, sr)

    # # To test   - orig & new datatype of Brahms sig, Mary sig, piano sig
    # #           - orig * diff sampling rate of "
    # # def test_reconst_new_sig_type_8bitPCM(self):
    # #     sr, sig = wavfile.read(brahms_filepath)

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_8bitPCM_newtype.wav'
    # #         wavfile.write(out_filepath, sr, synthetic_sig)

    # def test_reconst_orig_sig_type_8bitPCM(self):
    #     sr, sig = wavfile.read(brahms_filepath)
    #     print('8-bit PCM Sig Type:', sig.dtype, 'Sig Min-Max Range (~ 0 - 255): (', np.min(sig), ',', np.max(sig), ')') 

    #     if write_flag:
    #         out_filepath = test_path + 'reconst_8bitPCM_origtype.wav'
    #         orig_sig_type = sig.dtype

    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     print('Synthetic Sig Type (New):', synthetic_sig.dtype, 'Synthetic Sig Min-Max Range: (', np.min(synthetic_sig), ',', np.max(synthetic_sig), ')') 

    #     # Correct way to convert back to 8-bit PCM (unsigned -> signed)
    #     if orig_sig_type == 'uint8':
    #         # Bring to range [-1, 1]
    #         synthetic_sig = synthetic_sig / 32768
    #         # Bring to range [0, 255]
    #         synthetic_sig = synthetic_sig * 128
    #         # Signed to unsigned
    #         synthetic_sig = synthetic_sig + 128
    #         synthetic_sig = synthetic_sig.astype('uint8')
        
    #     print('Synthetic Sig Type (Like Orig):', synthetic_sig.dtype, 'Synthetic Sig Min-Max Range: (', np.min(synthetic_sig), ',', np.max(synthetic_sig), ')') 

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

    # # def test_reconst_new_sig_type_16bitPCM(self):
    # #     sr, sig = wavfile.read(mary_filepath)

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_16bitPCM_newtypeHORRIBLYLOUD.wav'
    # #         wavfile.write(out_filepath, sr, synthetic_sig)

    # def test_reconst_orig_sig_type_16bitPCM(self):
    #     sr, sig = wavfile.read(mary_filepath)
    #     print('16-bit PCM Sig Type:', sig.dtype, 'Sig Min-Max Range (~ -32768 - 32767): (', np.min(sig), ',', np.max(sig), ')') 

    #     if write_flag:
    #         out_filepath = test_path + 'reconst_16bitPCM_origtype.wav'
    #         orig_sig_type = sig.dtype

    #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    #     print('Synthetic Sig Type (New):', synthetic_sig.dtype, 'Synthetic Sig Min-Max Range: (', np.min(synthetic_sig), ',', np.max(synthetic_sig), ')') 

    #     if write_flag:
    #         wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))
    
    # # def test_reconst_new_sig_type_32bitPCM(self):
    # #     sr, sig = wavfile.read(mary32_filepath)
    # #     print('32-bit PCM Sig Type:', sig.dtype, 'Sig Min-Max Range (~ -2147483648 - 2147483647): (', np.min(sig), ',', np.max(sig), ')') 

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_32bitPCM_newtype.wav'
    # #         wavfile.write(out_filepath, sr, synthetic_sig)

    # # def test_reconst_orig_sig_type_32bitPCM(self):
    # #     sr, sig = wavfile.read(mary32_filepath)
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_32bitPCM_origtype.wav'
    # #         orig_sig_type = sig.dtype

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    
    # #     if write_flag:
    # #         wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

    # # def test_reconst_new_sig_type_32bit_fp(self):
    # #     sr, sig = wavfile.read(piano_echo_filepath)
    # #     print('32-bit floating-point Sig Type:', sig.dtype, 'Sig Min-Max Range (~ -1.0 - 1.0): (', np.min(sig), ',', np.max(sig), ')') 

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     print('Synthetic Sig Type (New):', synthetic_sig.dtype, 'Synthetic Sig Min-Max Range: (', np.min(synthetic_sig), ',', np.max(synthetic_sig), ')') 

    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_32bitFP_newtype.wav'
    # #         wavfile.write(out_filepath, sr, synthetic_sig)

    # # def test_reconst_orig_sig_type_32bit_fp(self):
    # #     sr, sig = wavfile.read(piano_echo_filepath)
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_32bitFP_origtype.wav'
    # #         orig_sig_type = sig.dtype

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)
    
    # #     if write_flag:
    # #         wavfile.write(out_filepath, sr, synthetic_sig.astype(orig_sig_type))

    # # UNCOMMENT ALL BELOW FOR GOOD TESTS
    
    # # # As signal get's longer, the average ratio between synthetic and original gets smaller
    # # def test_ova_reconst_by_sig_diff(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_Mary_ova.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    # #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, write_file=write_flag)
    # #     # synthetic_sig *= (4/3) # Amplitude ratio made by ova

    # #     # u_bound = len(sig)
    # #     comp_synthetic_sig = synthetic_sig[: len(sig)]

    # #     # Difference between total signal values (excluding the end window halves)
    # #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] - comp_synthetic_sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)]))

    # #     ratios = sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] / comp_synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]

    # #     print('Ratio:', sum(ratios) / len(ratios))

    # #     # plt.plot(sig[PIANO_WDW_SIZE*2:(PIANO_WDW_SIZE*2) + 100])
    # #     # plt.plot(comp_synthetic_sig[PIANO_WDW_SIZE*2:(PIANO_WDW_SIZE*2) + 100])

    # #     # Uncomment for the plot
    # #     # plt.plot(sig[(len(sig) - 100):])
    # #     # plt.plot(comp_synthetic_sig[(len(sig) - 100):])
    # #     # plt.show()
    # #     # print('Diff:', sig_diff)

    # #     # self.assertEqual(len(sig), len(synthetic_sig[: u_bound]))
    # #     self.assertAlmostEqual(sig_diff, 0)

    # # def test_reconst_by_sig_diff(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'reconst_Mary.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    # #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, write_file=write_flag)

    # #     # Difference between total signal values (excluding the end window halves)
    # #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] - synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]))

    # #     # plt.plot(rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    # #     # plt.plot(synthetic_rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    # #     # plt.show()
    # #     # print('Diff:', sig_diff)

    # #     self.assertAlmostEqual(sig_diff, 0)

    # # def test_ova_reconst_by_sig_diff_2(self):
    # #     rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE
    # #     synthetic_rand_sig = reconstruct_audio(rand_sig, DEBUG_WDW_SIZE, '', 0, ova=True)
    # #     synthetic_rand_sig *= (4/3) # Amplitude ratio made by ova

    # #     # Difference between total signal values (excluding the end window halves)
    # #     sig_diff = np.sum(np.abs(rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)] - synthetic_rand_sig[(DEBUG_WDW_SIZE // 2):-(DEBUG_WDW_SIZE // 2)]))

    # #     # plt.plot(rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    # #     # plt.plot(synthetic_rand_sig[DEBUG_WDW_SIZE*2:(DEBUG_WDW_SIZE*2) + 100])
    # #     # plt.show()
    # #     # print('Diff:', sig_diff)

    # #     self.assertAlmostEqual(sig_diff, 0)

    # # def test_ova_reconst_by_sig_ratio(self):
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     # rand_sig = np.random.rand(44100)    # Original test from Dennis - length requires DEBUG_WDW_SIZE

    # #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, '', sr, ova=True)

    # #     # Difference between total signal values (excluding the end window halves)
    # #     ratios = sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)] / synthetic_sig[(PIANO_WDW_SIZE // 2):-(PIANO_WDW_SIZE // 2)]

    # #     self.assertAlmostEqual(ratios[0], 4/3)

    # # def test_ova_reconst_beginning(self):
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz

    # #     synthetic_sig = reconstruct_audio(sig, PIANO_WDW_SIZE, '', sr, ova=True)

    # #     match = (sig[: PIANO_WDW_SIZE] * np.hanning(PIANO_WDW_SIZE))[: (PIANO_WDW_SIZE // 2)]

    # #     np.testing.assert_array_equal(synthetic_sig[: (PIANO_WDW_SIZE // 2)], match)

    # # def test_reconst_beginning(self):
    # #     # sig = [5]*12
    # #     sig = np.ones(12)
    # #     wdw_size = 6
    # #     spectrogram, phases = make_spectrogram(sig, wdw_size)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size)

    # #     # print('Synthetic non-OVA sig:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])

    # #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
    # #     match = np.ones(wdw_size)[:wdw_size // 2]

    # #     np.testing.assert_array_equal(synthetic_sig[:wdw_size // 2], match)

    # # def test_overlap_add_beginning(self):
    # #     # sig = [5]*12
    # #     sig = np.ones(12)
    # #     wdw_size = 6
    # #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True, debug=debug_flag)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True, debug=debug_flag)

    # #     print('Synthetic OVA sig beginning:\n', synthetic_sig, '\n', synthetic_sig[:wdw_size // 2])
    # #     # [0.        0.3454915 0.9045085]

    # #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[:wdw_size // 2]
    # #     match = np.hanning(wdw_size)[:wdw_size // 2]

    # #     np.testing.assert_array_almost_equal(synthetic_sig[:wdw_size // 2], match)

    # # # def test_overlap_add_middle(self):
    # # #     # sig = [5]*12
    # # #     sig = np.ones(12)
    # # #     wdw_size = 6
    # # #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
    # # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

    # # #     print('Synthetic OVA sig middle:\n', synthetic_sig, '\n', synthetic_sig[wdw_size // 2: -(wdw_size // 2)])

    # # #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2: -(wdw_size // 2)]
    # # #     match = np.hanning(wdw_size)[wdw_size // 2: -(wdw_size // 2)]
    # # #     print('Match:\n', match)
    # # #     print()

    # # #     np.testing.assert_array_almost_equal(synthetic_sig[wdw_size // 2: -(wdw_size // 2)], match)

    # # def test_overlap_add_end(self):
    # #     # sig = [5]*12
    # #     sig = np.ones(12)
    # #     wdw_size = 6
    # #     spectrogram, phases = make_spectrogram(sig, wdw_size, ova=True)
    # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, wdw_size, ova=True)

    # #     # print('Synthetic OVA sig end:\n', synthetic_sig, '\n', synthetic_sig[-(wdw_size // 2):])

    # #     # match = (np.array([5]*wdw_size) * np.hanning(wdw_size))[wdw_size // 2:]
    # #     match = np.hanning(wdw_size)[-(wdw_size // 2):]
    # #     # print('Match:\n', match)
    # #     # print()

    # #     np.testing.assert_array_almost_equal(synthetic_sig[-(wdw_size // 2):], match)

    # # # # Useless test
    # # # def test_mary_activations_mary_bv(self):
    # # #     mary_activations = make_mary_bv_test_activations()
    # # #     mary_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True)

    # # #     synthetic_spgm = mary_basis_vectors @ mary_activations

    # # #     self.assertEqual(synthetic_spgm.shape, (mary_basis_vectors.shape[0], mary_activations.shape[1]))

    # # # # Can't do b/c don't make activations for full bv yet
    # # # # def test_mary_activations_full_bv(self):
    # # # #     mary_activations = make_mary_bv_test_activations()
    # # # #     norm_basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)

    # # # #     synthetic_spgm = norm_basis_vectors @ mary_activations

    # # # #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, wdw_size, ova=ova_flag)

    # # # #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
    # # # #     synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=ova_flag)

    # # # def test_another_2(self):
    # # #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
    # # #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
    # # #     # plot_matrix(basis_vectors, name="Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
    # # #     # plot_matrix(spectrogram, name="Original Spectrogram", ratio=SPGM_BRAHMS_RATIO)

    # # #     # print('Shape of Spectrogram V:', spectrogram.shape)
    # # #     # print('Shape of Basis Vectors W:', basis_vectors.shape)
    # # #     # print('Learning Activations...')
    # # #     # activations = make_activations(spectrogram, basis_vectors)
    # # #     # print('Shape of Activations H:', activations.shape)

    # # #     activations = make_mary_bv_test_activations()
    # # #     print('Shape of Hand-made Activations H:', activations.shape)

    # # #     with open('activations_my_mary.csv', 'w') as a_f:
    # # #     # with open('activations_learned_brahms_trunc.csv', 'w') as a_f:
    # # #     # with open('activations_learned_mary_trunc.csv', 'w') as a_f:
    # # #         for component in activations:
    # # #             a_f.write(','.join([('%.4f' % x) for x in component]) + '\n')

    # # #     synthetic_spgm = basis_vectors @ activations
    # # #     # plot_matrix(synthetic_spgm, name="Synthetic Spectrogram", ratio=SPGM_BRAHMS_RATIO)

    # # #     # print('---SYNTHETIC SPGM TRANSITION----')
    # # #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
    # # #     # print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
    # # #     # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

    # # #     synthetic_sig /= (4/3)

    # # #     out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
    # # #     # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
    # # #     wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
    # # #     # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)

    # # # def test_my_mary_activations(self):
    # # #     basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
    # # #     spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)

    # # #     activations = make_activations(spectrogram, basis_vectors)

    # # #     synthetic_spgm = basis_vectors @ activations

    # # #     synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
    # # #     # print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
    # # #     # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

    # # #     synthetic_sig /= (4/3)

    # # #     # out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
    # # #     # # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
    # # #     # wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
    # # #     # # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)

    # # # UNCOMMENT FOR GOOD TESTS

    # # # MAKE_BASIS_VECTOR
    # # def test_make_best_basis_vector(self):
    # #     wdw_size, sgmt_num = 3, 2
    # #     signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
    # #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

    # #     sgmt = np.array([5,0,1])
    # #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    # #     np.testing.assert_array_equal(basis_vector, match)

    # # def test_make_best_basis_vector_short(self):
    # #     wdw_size, sgmt_num = 3, 2
    # #     signal = np.array([1,3,4,5,0])
    # #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size)

    # #     sgmt = np.array([5,0,0])
    # #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    # #     np.testing.assert_array_equal(basis_vector, match)

    # # def test_make_avg_basis_vector(self):
    # #     wdw_size, sgmt_num = 3, 2
    # #     signal = np.array([1,3,4,5,0,1,4,6,2,1,1,2,0,3])
    # #     basis_vector = make_basis_vector(signal, sgmt_num, wdw_size, avg=True)

    # #     sgmt = np.array([2.75,2.5,2.25])
    # #     match = np.abs(np.fft.fft(sgmt))[: (wdw_size // 2) + 1]

    # #     np.testing.assert_array_equal(basis_vector, match)

    # # def test_unsupervised_nmf(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'restored_Mary_unsupnmf_ova.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    # #     activations, basis_vectors = nmf_learn(spectrogram, num_components=88, debug=debug_flag)
    # #     synthetic_spectrogram = basis_vectors @ activations

    # #     synthetic_sig = make_synthetic_signal(synthetic_spectrogram, phases, PIANO_WDW_SIZE, ova=True, debug=debug_flag)

    # #     sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    # #                              synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    # #     if write_flag:
    # #         wavfile.write(out_filepath, sr, synthetic_sig.astype(sig.dtype))

    # #     # self.assertEqual(sig_diff, 0)

    # # def test_restore_marybv(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'restored_Mary_marybv_ova.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=True, noisebv=True,
    # #                                   avgbv=True, write_file=write_flag, debug=debug_flag)

    # #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    # #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    # #     # self.assertEqual(sig_diff, 0)

    # # def test_restore_fullbv(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'restored_Mary_fullbv_ova.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    # #                                   avgbv=True, write_file=write_flag, debug=debug_flag)

    # #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    # #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    # #     # self.assertEqual(sig_diff, 0)

    # # # Output file not noticably worse b/c no noise in Mary.wav
    # # def test_restore_no_hanningbv(self):
    # #     if write_flag:
    # #         out_filepath = test_path + 'restored_Mary_nohanbv_ova.wav'
    # #     sig, sr = librosa.load(mary_filepath, sr=STD_SR_HZ)  # Upsample Mary to 44.1kHz
    # #     synthetic_sig = restore_audio(sig, PIANO_WDW_SIZE, out_filepath, sr, ova=True, marybv=False, noisebv=True,
    # #                                   avgbv=True, write_file=write_flag, debug=debug_flag, nohanbv=True)

    # #     # sig_diff = np.sum(np.abs(sig[(PIANO_WDW_SIZE // 2): -(PIANO_WDW_SIZE // 2)] -
    # #     #                          synthetic_sig[(PIANO_WDW_SIZE // 2): (len(sig) - (PIANO_WDW_SIZE // 2))]))

    # #     # self.assertEqual(sig_diff, 0)



    # # # GET_BASIS_VECTORS
    # # # def test_make_basis_vectors(self):
    # # #     basis_vectors = get_basis_vectors(PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=True, debug=debug_flag, num_noisebv=num_noise_bv_test)
    # # #     _, c4_sig = wavfile.read('Piano.ff.C4.wav')
    # # #     _, db4_sig = wavfile.read('Piano.ff.Db4.wav')
    # # #     _, d4_sig = wavfile.read('Piano.ff.D4.wav')
    # # #     _, eb4_sig = wavfile.read('Piano.ff.Eb4.wav')
    # # #     _, e4_sig = wavfile.read('Piano.ff.E4.wav')
    # # #     sigs = [c4_sig, db4_sig, d4_sig, eb4_sig, e4_sig]
    # # #     sigs = [np.array([((x[0] + x[1]) / 2) for x in sig]) for sig in sigs]
    # # #     # amp_thresh = max(sig) * 0.01  # ?????
    # # #     match = np.array([make_basis_vector(sig, BEST_WDW_NUM, PIANO_WDW_SIZE, avg=True) for sig in sigs])
    # # # def test_make_basis_vectors_2(self):
    # #     # get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=True, noise=True, avg=True, eq=False, debug=debug_flag)
    # # # # Mary.wav?
    # # # def test_make_spectrogram(self):
    # # #     pass

    # # # Dummy test to create visuals
    # # def test_piano_sig_spgm_plot(self):
    # #     sr, sig = wavfile.read('/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/piano.wav')
    # #     sig = np.array([((x[0] + x[1]) / 2) for x in sig.astype('float64')]).astype('int16')
    # #     plt.plot(sig)
    # #     plt.title('Piano Recording Signal')
    # #     plt.ylabel('Amplitude')
    # #     plt.show()

    # #     spectrogram, phases = make_spectrogram(sig, PIANO_WDW_SIZE, ova=True, debug=True)
    
if __name__ == '__main__':
    unittest.main()


# Just saving stuff from past version of restore_audio (main):

#     # _, example_sig = wavfile.read("piano.wav")
#     # _, brahms_sig = wavfile.read(brahms_filepath)
#     # mary_sig, _ = librosa.load("Mary.wav", sr=STD_SR_HZ)   # Upsample Mary to 44.1kHz
#     # debug_sig = np.array([0,1,1,0])
#     # debug_sig = np.array([0,1,1,0,1,0])
#     # debug_sig_2 = np.random.rand(44100)

#     # DEBUG BLOCK - True for debug
#     if mode == 'DEBUG':
#         print('\n\n')
#         spectrogram, phases = make_spectrogram(debug_sig_2, DEBUG_WDW_SIZE, ova=ova_flag)
#         print('\n---SYNTHETIC SPGM TRANSITION----\n')
#         synthetic_sig = make_synthetic_signal(spectrogram, phases, DEBUG_WDW_SIZE, ova=ova_flag)
#         print('Debug Synthetic Sig:\n', synthetic_sig[:20])

#         # Also try actual sig
#         print('\n\n')
#         spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
#         print('\n---SYNTHETIC SPGM TRANSITION----\n')
#         synthetic_sig = make_synthetic_signal(spectrogram, phases, PIANO_WDW_SIZE, ova=ova_flag)
#         print('Actual Synthetic Sig:\n', np.array(synthetic_sig).astype('uint8')[:20])
#         # Make synthetic WAV file
#         out_filepath += debug_ova_filepath if ova_flag else debug_filepath
#         wavfile.write(out_filepath, STD_SR_HZ, np.array(synthetic_sig).astype('uint8'))

#     else:
#         basis_vectors = get_basis_vectors(BEST_WDW_NUM, PIANO_WDW_SIZE, mary=False)
#         spectrogram, phases = make_spectrogram(brahms_sig, PIANO_WDW_SIZE, ova=ova_flag)
#         plot_matrix(basis_vectors, name="Basis Vectors", ratio=BASIS_VECTOR_FULL_RATIO)
#         plot_matrix(spectrogram, name="Original Spectrogram", ratio=SPGM_BRAHMS_RATIO)

#         print('Shape of Spectrogram V:', spectrogram.shape)
#         print('Shape of Basis Vectors W:', basis_vectors.shape)
#         print('Learning Activations...')
#         activations = make_activations(spectrogram, basis_vectors)
#         print('Shape of Activations H:', activations.shape)

#         # activations = make_mary_bv_test_activations()
#         # print('Shape of Hand-made Activations H:', activations.shape)

#         # with open('learned_brahms_activations_trunc.csv', 'w') as a_f:
#         # # with open('learned_mary_activations_trunc.csv', 'w') as a_f:
#         #     for component in activations:
#         #         a_f.write(','.join([('%.4f' % x) for x in component]) + '\n')

#         synthetic_spgm = basis_vectors @ activations
#         plot_matrix(synthetic_spgm, name="Synthetic Spectrogram", ratio=SPGM_BRAHMS_RATIO)

#         print('---SYNTHETIC SPGM TRANSITION----')
#         synthetic_sig = make_synthetic_signal(synthetic_spgm, phases, PIANO_WDW_SIZE, ova=ova_flag)
#         print('Synthesized signal (bad type for brahms):\n', synthetic_sig[:20])
#         # print('Synthesized signal:\n', synthetic_sig.astype('uint8')[:20])

#         synthetic_sig /= (4/3)

#         out_filepath += synthetic_ova_brahms_filepath if ova_flag else synthetic_brahms_filepath
#         # Make synthetic WAV file - for some reason, I must cast brahms signal elems to types of original signal (uint8) or else MUCH LOUDER
#         wavfile.write(out_filepath, STD_SR_HZ, synthetic_sig.astype('uint8'))
#         # wavfile.write("synthetic_Mary.wav", STD_SR_HZ, synthetic_sig)
