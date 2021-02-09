# Blind Test: anonymously plays brahms files, allows you to rank them afterwards.

import sys
from scipy.io import wavfile
import simpleaudio as sa
import random
import numpy as np

from brahms_restore_ml.nmf.nmf import PIANO_WDW_SIZE, WDW_NUM_AFTER_VOICE


def play_wav(wave_obj):
    play_obj = wave_obj.play()
    play_obj.wait_done()
    print('Done')


def main():
    test_gs = False
    if len(sys.argv) > 1 and sys.argv[1] == 'gs':
        print('Testing within top gs results')
        test_gs = True
    else:
        print('Testing against benchmarks')

    # Sample Creation Section
    drnn_sr, drnn_sig = wavfile.read('brahms_restore_ml/drnn/output_restore/restore_151of3072.wav')
    drnn_sr2, drnn_sig2 = wavfile.read('brahms_restore_ml/drnn/output_restore/restore_72of192.wav')
    drnn_sr3, drnn_sig3 = wavfile.read('brahms_restore_ml/drnn/output_restore/restore_997of2048.wav')
    # drnn_sr2, drnn_sig2 = wavfile.read('brahms_restore_ml/drnn/output_restore/restore_133of3072.wav')
    # drnn_sr3, drnn_sig3 = wavfile.read('brahms_restore_ml/drnn/output_restore/restore_149of3072.wav')

    orig_sr, orig_sig = wavfile.read('brahms.wav')
    # old below
    # nmf_sr, nmf_sig = wavfile.read('brahms_restore_ml/nmf/old/old_outputs/output_restored_wav_v3/brahms_sslrnpiano_madeinit_10nbv_l1pen1000.wav')
    # nmf_sr, nmf_sig = wavfile.read('brahms_restore_ml/nmf/output/output_restored_wav_v4/brahms_10nbv.wav')
    nmf_sr, nmf_sig = wavfile.read('brahms_restore_ml/nmf/output/output_restored_wav_v5/brahms_1nbv.wav')
    # drnn_sig & drnn_sr made above
    ccrma_sr, ccrma_sig = wavfile.read('../Benchmark Systems/ccrma/benchmark2(thebest?).wav')
    po_sen_sr, po_sen_sig = wavfile.read('../Benchmark Systems/po-sen/BrahmsResults/piano_brahms_denoising_model.wav')
    izotope_sr, izotope_sig = wavfile.read('../Benchmark Systems/izotope_rx/brahms_restore_izotope_rx.wav')
    # print('Sig lengths:', len(orig_sig), len(nmf_sig), len(drnn_sig), len(ccrma_sig), len(po_sen_sig))
    # print('Sig dtypes:', orig_sig.dtype, nmf_sig.dtype, drnn_sig.dtype, ccrma_sig.dtype, po_sen_sig.dtype)
    # print('Sigs:', orig_sig[1000:1010], nmf_sig[1000:1010], drnn_sig[1000:1010], ccrma_sig[1000:1010], po_sen_sig[1000:1010])

    # Take out voice & make-up for nmf & except po-sen
    drnn_sig = drnn_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    drnn_sig2 = drnn_sig2[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    drnn_sig3 = drnn_sig3[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]

    orig_sig = orig_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    nmf_sig = np.split(nmf_sig, 2)[0]
    nmf_sig = nmf_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    # nmf_sig = np.split(nmf_sig, 2)[1] # old
    # drnn_sig = drnn_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    ccrma_sig = ccrma_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    po_sen_sig = po_sen_sig[int(WDW_NUM_AFTER_VOICE//2.7) * PIANO_WDW_SIZE: -(5 * PIANO_WDW_SIZE)]
    izotope_sig = izotope_sig[WDW_NUM_AFTER_VOICE * PIANO_WDW_SIZE: -(20 * PIANO_WDW_SIZE)]
    # print('Length of all results:', len(orig_sig), len(nmf_sig), len(drnn_sig), len(ccrma_sig), len(po_sen_sig))

    blind_test_folder = 'blind_test_samples/'

    # Create blind test samples (small)
    sample_len, sample_start = 166666, random.randrange(0, 1833334)
    wavfile.write(blind_test_folder + 'drnn_smpl.wav', drnn_sr, drnn_sig[sample_start: sample_start+sample_len])
    wavfile.write(blind_test_folder + 'drnn_smpl2.wav', drnn_sr2, drnn_sig2[sample_start: sample_start+sample_len]) 
    wavfile.write(blind_test_folder + 'drnn_smpl3.wav', drnn_sr3, drnn_sig3[sample_start: sample_start+sample_len]) 

    wavfile.write(blind_test_folder + 'orig_smpl.wav', orig_sr, orig_sig[sample_start: sample_start+sample_len])
    wavfile.write(blind_test_folder + 'nmf_smpl.wav', nmf_sr, nmf_sig[sample_start: sample_start+sample_len])
    ccrma_sig_type = ccrma_sig.dtype
    ccrma_sig = np.average(ccrma_sig, axis=-1)
    ccrma_sig *= 0.5    # It's louder than the others 
    ccrma_sig = ccrma_sig.astype(ccrma_sig_type)
    wavfile.write(blind_test_folder + 'ccrma_smpl.wav', ccrma_sr, ccrma_sig[sample_start: sample_start+sample_len])
    po_sen_type = po_sen_sig.dtype
    po_sen_sig = po_sen_sig.astype('float64')
    po_sen_sig *= 0.3   # It's louder than the others 
    po_sen_sig = po_sen_sig.astype(po_sen_type)
    wavfile.write(blind_test_folder + 'po_sen_smpl.wav', po_sen_sr, po_sen_sig[int(sample_start//2.7): int(sample_start//2.7)+int(sample_len//2.7)])
    wavfile.write(blind_test_folder + 'izotope_smpl.wav', izotope_sr, izotope_sig[sample_start: sample_start+sample_len])

    # Blind Test Section
    # simpleaudio
    drnn_f = sa.WaveObject.from_wave_file(blind_test_folder + 'drnn_smpl.wav')
    drnn_f2 = sa.WaveObject.from_wave_file(blind_test_folder + 'drnn_smpl2.wav')
    drnn_f3 = sa.WaveObject.from_wave_file(blind_test_folder + 'drnn_smpl3.wav')

    orig_f = sa.WaveObject.from_wave_file(blind_test_folder + 'orig_smpl.wav')
    nmf_f = sa.WaveObject.from_wave_file(blind_test_folder + 'nmf_smpl.wav')
    ccrma_f = sa.WaveObject.from_wave_file(blind_test_folder + 'ccrma_smpl.wav')
    po_sen_f = sa.WaveObject.from_wave_file(blind_test_folder + 'po_sen_smpl.wav')
    izotope_f = sa.WaveObject.from_wave_file(blind_test_folder + 'izotope_smpl.wav')

    if test_gs:
        test_files = [('drnn1', drnn_f), ('drnn2', drnn_f2), ('drnn3', drnn_f3)]
    else:
        test_files = [('orig',orig_f), ('nmf', nmf_f), ('drnn', drnn_f), 
                    ('ccrma', ccrma_f), ('po-sen', po_sen_f), ('izotope rx', izotope_f)]
    random.shuffle(test_files)
    # print('DEBUG Shuffled test files:', test_files)

    # TODO: Think about an automatic ranking system, based off comparisons between pairs
    user_input = ''
    while user_input != 'q':
        if test_gs:
            user_input = input('Enter wav to play [1-3] (some are loud) or \"q\" to rank all: ')
        else:
            user_input = input('Enter wav to play [1-6] (some are loud) or \"q\" to rank all: ')

        if user_input.isdigit():
            print('Playing wav', user_input+'..')
            wav_f = test_files[int(user_input)-1][1]
            play_wav(wav_f)
    
    if test_gs:
        user_input = input('Quit playback. Enter wav ranking (best to worst) as their numbers 1-3, separated by spaces: ')
    else:
        user_input = input('Quit playback. Enter wav ranking (best to worst) as their numbers 1-6, separated by spaces: ')
    ranking = [int(x) for x in user_input.split()]
    for i, wav_num in enumerate(ranking):
        print(str(i+1)+')', test_files[wav_num-1][0])


if __name__ == '__main__':
    main()