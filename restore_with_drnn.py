# Main DRNN Restoration Script

from brahms_restore_ml.drnn.drnn import *
from brahms_restore_ml.audio_data_processing import PIANO_WDW_SIZE
import sys
import random
from scipy.io import wavfile
import numpy as np
import json
import math
import multiprocessing

def run_top_gs_result(num_str, best_config, 
                    #   train_mean, train_std, 
                      x_train_files, y1_train_files, y2_train_files,
                      x_val_files, y1_val_files, y2_val_files, num_train, num_val, train_feat, train_seq,
                      patience, epsilon, recent_model_path, pc_run, dmged_piano_artificial_noise_mix,
                      infer_output_path, 
                    #   wdw_size, 
                      brahms_path, combos_str, data_path=None, min_sig_len=None,
                      tuned_a430hz=False, use_basis_vectors=False,
                      loop_bare_noise=False, low_time_steps=False,
                      artificial_noise=False):
    train_batch_size = best_config['batch_size']
    # # Temp test for LSTM -> until can grid search
    # train_batch_size = 3 if train_batch_size < 3 else train_batch_size
    # # TEMP - until F35 back up, make managable for PC, for bv_s grid search results, no dimred & no lowtsteps
    # train_batch_size = 4
    # # TEMP - make what PC can actually handle (3072 run, but definitely 2048)
    # train_batch_size = 6
    if low_time_steps:
        train_batch_size = 20   # 50 # 50 sometimes caused OOM
    else:
        train_batch_size = 4    # no dimred & no lowtsteps
    train_loss_const = best_config['gamma']
    # EVAL CHANGE
    if low_time_steps:
        train_loss_const = 0.15 # 0.3    # bad for looped noise & normal piano data
    train_epochs = best_config['epochs']
    # # EVAL CHANGE - change back
    if low_time_steps:
        train_epochs = 150 # 10 # 100
    # train_epochs = 15 # TEMP - optimize learning
    # TEMP - exploit high epochs
    if train_epochs > 10:
        patience = 10
    train_opt_name = best_config['optimizer']
    train_opt_clipval = None if (best_config['clip value'] == -1) else best_config['clip value']
    train_opt_lr = best_config['learning rate']

    training_arch_config = {}
    training_arch_config['layers'] = best_config['layers']
    # # Temp test for LSTM -> until can grid search
    # for i in range(len(best_config['layers'])):
    #     if best_config['layers'][i]['type'] == 'RNN':
    #         training_arch_config['layers'][i]['type'] = 'LSTM'
    # TEMP - Don't allow dimred in RNNS/LSTMS, b/c not in lit
    for i in range(len(best_config['layers'])):
        if (best_config['layers'][i]['nrn_div'] != 1) and (best_config['layers'][i]['type'] == 'RNN' or 
                                                           best_config['layers'][i]['type'] == 'LSTM'):
            training_arch_config['layers'][i]['nrn_div'] = 1
    training_arch_config['scale'] = best_config['scale']
    training_arch_config['rnn_res_cntn'] = best_config['rnn_res_cntn']
    training_arch_config['bias_rnn'] = best_config['bias_rnn']
    training_arch_config['bias_dense'] = best_config['bias_dense']
    training_arch_config['bidir'] = best_config['bidir']
    training_arch_config['rnn_dropout'] = best_config['rnn_dropout']
    training_arch_config['bn'] = best_config['bn']
    # EVAL CHANGES
    training_arch_config['bidir'] = False
    training_arch_config['rnn_res_cntn'] = False
    # if use_basis_vectors:
    #     training_arch_config['bidir'] = False
    #     training_arch_config['rnn_res_cntn'] = False
    l1_reg = None
    # # EVAL CHANGE
    # l1_reg = 0.001

    print('#', num_str, 'TOP TRAIN ARCH FOR USE:')
    print(training_arch_config)
    print('#', num_str, 'TOP TRAIN HPs FOR USE:')
    print('Batch size:', train_batch_size, 'Epochs:', train_epochs,
            'Loss constant:', train_loss_const, 'Optimizer:', best_config['optimizer'], 
            'Clip value:', best_config['clip value'], 'Learning rate:', best_config['learning rate'])
    # Shouldn't need multiproccessing, limited number
    # Temp test for LSTM -> until can grid search
    process_train = multiprocessing.Process(target=evaluate_source_sep, args=(
                        x_train_files, y1_train_files, y2_train_files,
                        x_val_files, y1_val_files, y2_val_files,
                        num_train, num_val,
                        train_feat, train_seq, 
                        train_batch_size, 
                        train_loss_const, train_epochs, 
                        train_opt_name, train_opt_clipval, train_opt_lr,
                        patience, epsilon, training_arch_config, 
                        recent_model_path, pc_run,
                        # train_mean, train_std, 
                        int(num_str), None, int(combos_str), '', None,
                        dmged_piano_artificial_noise_mix, data_path, min_sig_len, True,
                        tuned_a430hz, use_basis_vectors, None, None, 
                        loop_bare_noise, low_time_steps, l1_reg, artificial_noise))
    process_train.start()
    process_train.join()

    # evaluate_source_sep(x_train_files, y1_train_files, y2_train_files, 
    #                     x_val_files, y1_val_files, y2_val_files,
    #                     num_train, num_val,
    #                     n_feat=train_feat, n_seq=train_seq, 
    #                     batch_size=train_batch_size, 
    #                     loss_const=train_loss_const, epochs=train_epochs, 
    #                     opt_name=train_opt_name, opt_clip_val=train_opt_clipval, opt_lr=train_opt_lr,
    #                     patience=patience, epsilon=epsilon,
    #                     recent_model_path=recent_model_path, pc_run=pc_run,
    #                     config=training_arch_config, t_mean=train_mean, t_std=train_std,
    #                     dataset2=dmged_piano_artificial_noise_mix)
    # Temp test for LSTM -> until can grid search
    process_infer = multiprocessing.Process(target=restore_with_drnn, args=(infer_output_path, recent_model_path, # wdw_size, epsilon,
                        # train_loss_const, 
                        train_opt_name, train_opt_clipval, train_opt_lr, min_sig_len, brahms_path, None, None,
                        # training_arch_config, 
                        # train_mean, train_std, 
                        PIANO_WDW_SIZE, EPSILON,
                        pc_run, '_'+num_str+'of'+combos_str, tuned_a430hz, use_basis_vectors, low_time_steps))
    # TEMP - old
    # process_infer = multiprocessing.Process(target=restore_with_drnn, args=(infer_output_path, recent_model_path, wdw_size, epsilon,
    #                     train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr, brahms_path, None, None,
    #                     training_arch_config, train_mean, train_std, pc_run, '_'+num+'of'+combos_str))
    process_infer.start()
    process_infer.join()

    # restore_with_drnn(infer_output_path, recent_model_path, wdw_size, epsilon,
    #                 train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr,
    #                 test_filepath=brahms_path,
    #                 config=training_arch_config, t_mean=train_mean, t_std=train_std, pc_run=pc_run,
    #                 name_addon='_'+num+'of3072_lstm')


def main():
    # PROGRAM ARGUMENTS #
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('\nUsage: restore_with_drnn.py <mode> <PC> [-f] [gs_id]')
        print('Parameter Options:')
        print('Mode     t               - Train model, then restore brahms with model')
        print('         g               - Perform grid search (default: starts where last left off)')
        print('         r               - Restore brahms with last-trained model')
        print('PC       true            - Uses HPs for lower GPU-memory consumption (< 4GB)')
        print('         false           - Uses HPs for higher GPU-memory limit (PC HPs + nonPC HPs = total for now)')
        print('-f                       - (Optional) Force restart grid search (grid search mode) OR force random HPs (train mode)')
        print('gs_id    <single digit>  - (Optional) grid search unique ID for running concurrently')
        print('\nTIP: Keep IDs different for PC/non-PC runs on same machine')
        sys.exit(1)

    mode = sys.argv[1] 
    pc_run = True if (sys.argv[2].lower() == 'true') else False
    # TRAIN DATA PIANO PARAMS
    # Differentiate PC GS from F35 GS
    # dmged_piano_artificial_noise_mix = True if pc_run else False
    dmged_piano_artificial_noise_mix = False    # TEMP while F35 down
    dmged_piano_only = False    # Promising w/ BVs
    # TRAIN DATA NOISE PARAMS
    loop_bare_noise = True     # to control bare_noise in nn_data_gen, needs curr for low_time_steps
    artificial_noise = False

    test_on_synthetic = False
    # wdw_size = PIANO_WDW_SIZE
    data_path = 'brahms_restore_ml/drnn/drnn_data/'
    arch_config_path = 'brahms_restore_ml/drnn/config/'
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search/'           # for use w/ grid search mode    
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_pc_wb/'     # PC
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_lstm/'    # F35
    gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_wb/'        # best results  
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_low_tsteps_two/'       # low tsteps   2 
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_low_tsteps_big/'       # low tsteps   3
    recent_model_path = 'brahms_restore_ml/drnn/recent_model'
    # recent_model_path = 'brahms_restore_ml/drnn/recent_model_149of3072'    # restore from curr best
    # recent_model_path = 'brahms_restore_ml/drnn/recent_model_111of144_earlystop'    # restore from curr best
    # recent_model_path = 'brahms_restore_ml/drnn/recent_model_3of4'    # restore from best in small gs
    # infer_output_path = 'brahms_restore_ml/drnn/output_restore/'
    # infer_output_path = 'brahms_restore_ml/drnn/output_restore_gs3072_loopnoise/'    # eval, do_curr_best, 3072 combos, looped noise
    infer_output_path = 'brahms_restore_ml/drnn/output_restore_151of3072_eval/'    # eval, tweaks curr_best
    brahms_path = 'brahms.wav'

    # To run best model configs, data_from_numpy == True & mode == train
    do_curr_best, curr_best_combos, curr_best_done_on_pc = True, '3072', False
    # # F35 LSTM
    # top_result_nums = [72, 128, 24, 176, 8, 192, 88, 112]
    # F35 WB
    top_result_nums = [151] # temp - do 1 run # [1488, 1568, 149, 1496, 1680, 86, 151, 152]
    # top_result_nums = [1488, 1568, 149, 1496, 1680, 86, 151, 152] 
    # # PC WB
    # top_result_nums = [997, 1184, 1312, 1310, 1311, 1736]
    # # BVS Architectures
    # top_result_nums = [6] # 13, 20, 6, 10, 23]
    # # BVS Architectures #2
    # top_result_nums = [10]
    # # low timesteps 2
    # top_result_nums = [34, 23] # [26,  # gamma order: 0.05, 0.15, 0.3
    # # low timesteps 3 (big)
    # top_result_nums = [103, 111, 5] # gamma order: 0.2, 0.3, 0.4 # [111] # [111, 76, 142]
    top_result_paths = [gs_output_path + 'result_' + str(x) + '_of_' + curr_best_combos +
                        ('.txt' if curr_best_done_on_pc else '_noPC.txt') for x in top_result_nums]
    # NEW
    output_file_addon = ''
    data_from_numpy = True
    tuned_a430hz = False # may not be helpful, as of now does A=436Hz by default
    basis_vector_features = False
    if tuned_a430hz:
        recent_model_path += '_a436hz' # tune_temp '_a430hz'
        output_file_addon += '_a436hz' # tune_temp '_a430hz'
    if basis_vector_features:
        recent_model_path += '_bvs'
        output_file_addon += '_bvs'
    if dmged_piano_only:
        recent_model_path += '_dmgedp'
        output_file_addon += '_dmgedp'
    if artificial_noise:
        recent_model_path += '_artn'
        output_file_addon += '_artn'
    if do_curr_best and (len(top_result_nums) == 1) and (mode == 't'):
        recent_model_path += ('_' + str(top_result_nums[0]) + 'of' + str(curr_best_combos))
    if do_curr_best and (len(top_result_nums) == 1) and (mode == 'r'):
        # recent_model_path must have result in name
        output_file_addon += ('_' + [x for x in recent_model_path.split('_')][-1])
    gs_write_model = False      # for small grid searches only, and for running ALL epochs - no early stop
    low_time_steps = True       # now default

    # print('RECENT MODEL PATH:', recent_model_path)
    # print('OUTPUT FILE PATH ADD-ON:', output_file_addon)

    # EMPERICALLY DERIVED HPs
    # Note: FROM PO-SEN PAPER - about loss_const
    #   Empirically, the value γ is in the range of 0.05∼0.2 in order
    #   to achieve SIR improvements and maintain SAR and SDR.
    train_batch_size = 100 if low_time_steps else (6 if pc_run else 12)
    # train_batch_size = 3 if pc_run else 12  # TEMP - for no dimreduc
    train_loss_const = 0.1
    train_epochs = 10
    train_opt_name, train_opt_clipval, train_opt_lr = 'Adam', 0.5, 0.001
    training_arch_config = None

    epsilon, patience, val_split = 10 ** (-10), train_epochs, 0.25

    # INFER ONLY
    if mode == 'r':
        restore_with_drnn(infer_output_path, recent_model_path, 
                          train_opt_name, train_opt_clipval, train_opt_lr,
                          MIN_SIG_LEN, test_filepath=brahms_path, pc_run=pc_run,
                          name_addon=output_file_addon, tuned_a430hz=tuned_a430hz,
                          use_basis_vectors=basis_vector_features, 
                          low_tsteps=low_time_steps)
    else:
        train_configs, arch_config_optns = get_hp_configs(arch_config_path, pc_run=pc_run, 
                                                          use_bv=basis_vector_features,
                                                          small_gs=gs_write_model,
                                                          low_tsteps=low_time_steps)
        # print('First arch config optn after return:', arch_config_optns[0])

        if data_from_numpy:
            # Load in train/validation numpy data
            noise_piano_filepath_prefix = (data_path + 'dmgedp_artn_mix_numpy/mixed'
                    if dmged_piano_artificial_noise_mix else 
                (data_path + 'dmged_mix_a436hz_numpy/mixed' if (dmged_piano_only and tuned_a430hz) else # tune_temp
                (data_path + 'dmged_mix_looped_noise_small_numpy/mixed' if (dmged_piano_only and loop_bare_noise and low_time_steps) else
                (data_path + 'dmged_mix_art_noise_small_numpy/mixed' if (dmged_piano_only and artificial_noise and low_time_steps) else
                (data_path + 'dmged_mix_looped_noise_numpy/mixed' if (dmged_piano_only and loop_bare_noise) else
                (data_path + 'piano_noise_looped_small_numpy/mixed' if (loop_bare_noise and low_time_steps) else
                (data_path + 'dmged_mix_art_noise_numpy/mixed' if (dmged_piano_only and artificial_noise) else
                (data_path + 'piano_noise_art_small_numpy/mixed' if (artificial_noise and low_time_steps) else
                (data_path + 'dmged_mix_small_numpy/mixed' if (dmged_piano_only and low_time_steps) else
                (data_path + 'piano_noise_small_numpy/mixed' if low_time_steps else
                (data_path + 'piano_noise_looped_numpy/mixed' if loop_bare_noise else
                (data_path + 'piano_noise_art_numpy/mixed' if artificial_noise else
                (data_path + 'dmged_mix_numpy/mixed' if dmged_piano_only else
                (data_path + 'piano_noise_a436hz_numpy/mixed' if tuned_a430hz else  # tune_temp
                (data_path + 'piano_noise_numpy/mixed')))))))))))))))
            piano_label_filepath_prefix = (data_path + 'piano_source_numpy/piano'
                    if dmged_piano_artificial_noise_mix else     
                (data_path + 'piano_source_a436hz_numpy/piano' if tuned_a430hz else # tune_temp
                (data_path + 'piano_source_small_numpy/piano' if low_time_steps else
                (data_path + 'piano_source_numpy/piano'))))
            noise_label_filepath_prefix = (data_path + 'dmged_noise_numpy/noise'
                    if dmged_piano_artificial_noise_mix else 
                (data_path + 'noise_source_looped_small_numpy/noise' if (loop_bare_noise and low_time_steps) else
                (data_path + 'noise_source_art_small_numpy/noise' if (artificial_noise and low_time_steps) else
                (data_path + 'noise_source_looped_numpy/noise' if loop_bare_noise else
                (data_path + 'noise_source_art_numpy/noise' if artificial_noise else
                (data_path + 'noise_source_small_numpy/noise' if low_time_steps else
                (data_path + 'noise_source_numpy/noise')))))))
        else:
            # Load in train/validation WAV file data
            noise_piano_filepath_prefix = (data_path + 'dmged_mix_wav/features'
                if dmged_piano_artificial_noise_mix else data_path + 'dmged_mix_wav/features')
            piano_label_filepath_prefix = (data_path + 'final_piano_wav_a=430hz/psource'
                    if dmged_piano_artificial_noise_mix else 
                (data_path + 'final_piano_wav_a=436hz/psource' if tuned_a430hz else     # tune_temp
                (data_path + 'small_piano_wav/psource' if low_time_steps else
                (data_path + 'final_piano_wav/psource'))))
            noise_label_filepath_prefix = (data_path + 'dmged_noise_wav/nsource'
                        if dmged_piano_artificial_noise_mix else
                    (data_path + 'small_art_noise_wav/nsource' if (low_time_steps and artificial_noise) else 
                    (data_path + 'small_noise_wav/nsource' if low_time_steps else 
                    (data_path + 'artificial_noise_wav/nsource' if artificial_noise else 
                    (data_path + 'final_noise_wav/nsource')))))
            if dmged_piano_only:    # new
                dmged_piano_filepath_prefix = (data_path + 'dmged_piano_wav_a=436hz/psource' # tune_temp
                    if tuned_a430hz else 
                (data_path + 'small_dmged_piano_wav/psource' if low_time_steps else
                (data_path + 'dmged_piano_wav_a=430hz/psource')))
                # print('Damaged piano filepath prefix:', dmged_piano_filepath_prefix)
            
            if loop_bare_noise and dmged_piano_only:
                print('\nTRAINING WITH DATASET 4 (DMGED PIANO, NORMAL NOISE)')
            elif artificial_noise and dmged_piano_only:
                print('\nTRAINING WITH DATASET 6 (DMGED PIANO, ARTIFICIAL NOISE)')
            elif dmged_piano_only:
                print('\nTRAINING WITH DATASET 5 (DMGED PIANO, TIME SHRINK/STRETCH NOISE)')
            elif artificial_noise:
                print('\nTRAINING WITH DATASET 3 (NORMAL PIANO, ARTIFICIAL NOISE)')
            elif loop_bare_noise:
                print('\nTRAINING WITH DATASET 1 (NORMAL PIANO, NORMAL NOISE)')
            else:
                print('\nTRAINING WITH DATASET', '2 (ARTIFICIAL DMG)' if dmged_piano_artificial_noise_mix else '2 (NORMAL PIANO, TIME SHRINK/STRETCH NOISE)')

        # if data_from_numpy:   
        #     print('Mix filepath prefix:', noise_piano_filepath_prefix)
        # print('Piano filepath prefix:', piano_label_filepath_prefix)
        # print('Noise filepath prefix:', noise_label_filepath_prefix)

        # TRAIN & INFER
        if mode == 't':
            random_hps = False
            for arg_i in range(3, 6):
                if arg_i < len(sys.argv):
                    if sys.argv[arg_i] == '-f':
                        random_hps = True
                        print('\nTRAINING TO USE RANDOM (NON-EMPIRICALLY-OPTIMAL) HP\'S\n')

            # Define which files to grab for training. Shuffle regardless.
            # (Currently sample is to test on 1 synthetic sample (not Brahms))
            sample = test_on_synthetic
            # sample = False   # If taking less than total samples
            if sample:  # Used now for testing on synthetic data
                # TOTAL_SMPLS += 1
                # actual_samples = TOTAL_SMPLS - 1  # How many to leave out (1)
                # sample_indices = list(range(TOTAL_SMPLS))
                actual_samples = TOTAL_SHORT_SMPLS if low_time_steps else TOTAL_SMPLS    # How many to leave out (1)
                sample_indices = list(range(actual_samples + 1))
                # FIX DMGED/ART DATA
                random.shuffle(sample_indices)
                
                test_index = sample_indices[actual_samples]
                sample_indices = sample_indices[:actual_samples]
                test_piano = piano_label_filepath_prefix + str(test_index) + '.wav'
                test_noise = noise_label_filepath_prefix + str(test_index) + '.wav'
                test_sr, test_piano_sig = wavfile.read(test_piano)
                _, test_noise_sig = wavfile.read(test_noise)
                test_sig = test_piano_sig + test_noise_sig
            else:
                actual_samples = TOTAL_SHORT_SMPLS if low_time_steps else TOTAL_SMPLS
                sample_indices = list(range(actual_samples))
                # comment out if debugging
                random.shuffle(sample_indices)

            x_files = np.array([(noise_piano_filepath_prefix + str(i) + ('.npy' if data_from_numpy else '.wav'))
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + ('.npy' if data_from_numpy else '.wav'))
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + ('.npy' if data_from_numpy else '.wav'))
                        for i in sample_indices])
            if dmged_piano_only and (not data_from_numpy):
                dmged_y1_files = np.array([(dmged_piano_filepath_prefix + str(i) + '.wav')
                        for i in sample_indices])
            #     print('ORDER CHECK: dmged_y1_files:', dmged_y1_files[:10])
            # print('ORDER CHECK: x_files:', x_files[:10])
            # print('ORDER CHECK: y1_files:', y1_files[:10])
            # print('ORDER CHECK: y2_files:', y2_files[:10])

            # OLD
            # # # # Temp - do to calc max len for padding - it's 3081621 (for youtube src data)
            # # # # it's 3784581 (for Spotify/Youtube Final Data)
            # # # # it's 3784581 (for damaged Spotify/YouTube Final Data)
            # # # max_sig_len = None
            # # # for x_file in x_files:
            # # #     _, sig = wavfile.read(x_file)
            # # #     if max_sig_len is None or len(sig) >max_sig_len:
            # # #         max_sig_len = len(sig)
            # # # print('NOTICE: MAX SIG LEN', max_sig_len)
            # max_sig_len = MAX_SIG_LEN
            # # # Temp - get training data dim (from dummy) (for model & data making)
            # # max_len_sig = np.ones((max_sig_len))
            # # dummy_train_spgm, _ = make_spectrogram(max_len_sig, wdw_size, epsilon,
            # #                                     ova=True, debug=False)
            # # train_seq, train_feat = dummy_train_spgm.shape
            # # print('NOTICE: TRAIN SEQ LEN', train_seq, 'TRAIN FEAT LEN', train_feat)
            # train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN

            # # NEW
            # train_seq, train_feat, min_sig_len = get_raw_data_stats(y1_files,
            #                                                         brahms_filename=brahms_path)
            # print('NOTICE: TRAIN SEQ LEN', train_seq, 'TRAIN FEAT LEN', train_feat, 'MIN SIG LEN',
            #     min_sig_len)
            if low_time_steps:
                train_seq, train_feat, min_sig_len = TRAIN_SEQ_LEN_SMALL, TRAIN_FEAT_LEN, MIN_SIG_LEN_SMALL
            else:
                train_seq, train_feat, min_sig_len = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MIN_SIG_LEN
            # broken
            # if basis_vector_features:
            #     train_seq += NUM_SCORE_NOTES

            # Validation & Training Split
            indices = list(range(actual_samples))
            val_indices = indices[:math.ceil(actual_samples * val_split)]
            x_train_files = np.delete(x_files, val_indices)
            y1_train_files = np.delete(y1_files, val_indices)
            y2_train_files = np.delete(y2_files, val_indices)
            x_val_files = x_files[val_indices]
            y1_val_files = y1_files[val_indices]
            y2_val_files = y2_files[val_indices]
            num_train, num_val = len(y1_train_files), len(y1_val_files)

            dmged_y1_train_files = np.delete(dmged_y1_files, val_indices) if (dmged_piano_only and (not data_from_numpy)) else None
            dmged_y1_val_files = dmged_y1_files[val_indices] if (dmged_piano_only and (not data_from_numpy)) else None
            
            # # DEBUG PRINT
            # print('Indices ( len =', len(indices), '):', indices[:10])
            # print('Val Indices ( len =', len(val_indices), '):', val_indices[:10])
            # print('Num train:', num_train, 'num val:', num_val)
            # print('x_train_files:', x_train_files[:10])
            # print('x_val_files:', x_val_files[:10])
            # print('y1_train_files:', y1_train_files[:10])
            # print('y1_val_files:', y1_val_files[:10])
            # print('y2_train_files:', y2_train_files[:10])
            # print('y2_val_files:', y2_val_files[:10])
            # if (dmged_y1_train_files is not None) and (dmged_y1_val_files is not None):
            #     print('dmged_y1_train_files:', dmged_y1_train_files[:10])
            #     print('dmged_y1_val_files:', dmged_y1_val_files[:10])

            # CUSTOM TRAINING Dist training needs a "global_batch_size"
            # if not pc_run:
            #     batch_size_per_replica = train_batch_size // 2
            #     train_batch_size = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync

            print('Train Input Stats:')
            if do_curr_best:
                print('N Feat:', train_feat, 'Seq Len:', train_seq)
            else:
                print('N Feat:', train_feat, 'Seq Len:', train_seq, 'Batch Size:', train_batch_size)

            # print('ORDER CHECK: y1_train_files:', y1_train_files[:10])

            if do_curr_best:
                for top_result_path in top_result_paths:
                    # TEMP - For damaged data - b/c "_noPC" is missing in txt file
                    # num = top_result_path.split('_')[-3] if dmged_piano_artificial_noise_mix else top_result_path.split('_')[-4]
                    num = top_result_path.split('_')[-3] if curr_best_done_on_pc else top_result_path.split('_')[-4]
                    gs_result_file = open(top_result_path, 'r')
                    for _ in range(4):
                        _ = gs_result_file.readline()
                    best_config = json.loads(gs_result_file.readline())
                    # Temp test for LSTM -> until can grid search
                    # # TEMP - until F35 back up, make managable for PC
                    # if (len(best_config['layers']) < 4) or (len(best_config['layers']) == 4 and best_config['layers'][0]['type'] == 'Dense'):
                    run_top_gs_result(num, best_config, 
                    # TRAIN_MEAN, TRAIN_STD, 
                                    x_train_files, y1_train_files, y2_train_files,
                                    x_val_files, y1_val_files, y2_val_files, num_train, num_val, train_feat, train_seq,
                                    patience, epsilon, recent_model_path, pc_run, dmged_piano_artificial_noise_mix,
                                    infer_output_path, 
                                    # wdw_size, 
                                    brahms_path, curr_best_combos, data_path=data_path, 
                                    min_sig_len=min_sig_len, tuned_a430hz=tuned_a430hz,
                                    use_basis_vectors=basis_vector_features,
                                    loop_bare_noise=loop_bare_noise,
                                    low_time_steps=low_time_steps, 
                                    artificial_noise=artificial_noise)
            else:
                # REPL TEST - arch config, all config, optiizer config
                if random_hps:
                    # MEM BOUND TEST
                    # arch_rand_index = 0
                    # Index into random arch config, and other random HPs
                    arch_rand_index = random.randint(0, len(arch_config_optns)-1)
                    # arch_rand_index = 0
                    # print('ARCH RAND INDEX:', arch_rand_index)
                    training_arch_config = arch_config_optns[arch_rand_index]
                    # print('ARCH CONFIGS AT PREV & NEXT INDICES:\n', arch_config_optns[arch_rand_index-1], 
                    #       '---\n', arch_config_optns[arch_rand_index+1])
                    # print('In random HPs section, rand_index:', arch_rand_index)
                    # print('FIRST ARCH CONFIG OPTION SHOULD HAVE RNN:\n', arch_config_optns[0])
                    for hp, optns in train_configs.items():
                        # print('HP:', hp, 'OPTNS:', optns)
                        # MEM BOUND TEST
                        # hp_rand_index = 0
                        hp_rand_index = random.randint(0, len(optns)-1)
                        if hp == 'batch_size':
                            # print('BATCH SIZE RAND INDEX:', hp_rand_index)
                            train_batch_size = optns[hp_rand_index]
                        elif hp == 'epochs':
                            # print('EPOCHS RAND INDEX:', hp_rand_index)
                            train_epochs = optns[hp_rand_index]
                        elif hp == 'loss_const':
                            # print('LOSS CONST RAND INDEX:', hp_rand_index)
                            train_loss_const = optns[hp_rand_index]
                        elif hp == 'optimizer':
                            # hp_rand_index = 2
                            # print('OPT RAND INDEX:', hp_rand_index)
                            train_opt_clipval, train_opt_lr, train_opt_name = (
                                optns[hp_rand_index]
                            )
                            # train_optimizer, clip_val, lr, opt_name = (
                            #     optns[hp_rand_index]
                            # )

                    # Early stop for random HPs
                    # TIME TEST
                    patience = 4
                    # training_arch_config = arch_config_optns[0]
                    print('RANDOM TRAIN ARCH FOR USE:')
                    print(training_arch_config)
                    print('RANDOM TRAIN HPs FOR USE:')
                    print('Batch size:', train_batch_size, 'Epochs:', train_epochs,
                        'Loss constant:', train_loss_const, 'Optimizer:', train_opt_name, 
                        'Clip value:', train_opt_clipval, 'Learning rate:', train_opt_lr)
                # elif do_curr_best:
                    # gs_result_file = open(top_result_path, 'r')
                    # for _ in range(4):
                    #     _ = gs_result_file.readline()
                    # best_config = json.loads(gs_result_file.readline())
                    # train_batch_size = best_config['batch_size']
                    # # # Avoid OOM
                    # # if pc_run and train_batch_size > 10:
                    # #     train_batch_size = 10
                    # train_loss_const = best_config['gamma']
                    # train_epochs = best_config['epochs']
                    # train_opt_name = best_config['optimizer']
                    # train_opt_clipval = None if (best_config['clip value'] == -1) else best_config['clip value']
                    # train_opt_lr = best_config['learning rate']

                    # training_arch_config = {}
                    # training_arch_config['layers'] = best_config['layers']
                    # training_arch_config['scale'] = best_config['scale']
                    # training_arch_config['rnn_res_cntn'] = best_config['rnn_res_cntn']
                    # training_arch_config['bias_rnn'] = best_config['bias_rnn']
                    # training_arch_config['bias_dense'] = best_config['bias_dense']
                    # training_arch_config['bidir'] = best_config['bidir']
                    # training_arch_config['rnn_dropout'] = best_config['rnn_dropout']
                    # training_arch_config['bn'] = best_config['bn']
                    # # # Avoid OOM
                    # # for i, layer in enumerate(best_config['layers']):
                    # #     if layer['type'] == 'LSTM':
                    # #         training_arch_config['layers'][i]['type'] = 'RNN'
                    # print('TOP TRAIN ARCH FOR USE:')
                    # print(training_arch_config)
                    # print('TOP TRAIN HPs FOR USE:')
                    # print('Batch size:', train_batch_size, 'Epochs:', train_epochs,
                    #       'Loss constant:', train_loss_const, 'Optimizer:', best_config['optimizer'], 
                    #       'Clip value:', best_config['clip value'], 'Learning rate:', best_config['learning rate'])
                    
                # else:
                #     print('CONFIG:', training_arch_config)

                # OLD
                # # # TEMP - update for each unique dataset
                # # Note - If not numpy, consider if dataset2. If numpy, supply x files.
                # # train_mean, train_std = get_data_stats(y1_train_files, y2_train_files, num_train,
                # #                                   train_seq=train_seq, train_feat=train_feat, 
                # #                                   wdw_size=wdw_size, epsilon=epsilon, 
                # #                                 #   pad_len=max_sig_len)
                # #                                   pad_len=max_sig_len, x_filenames=x_train_files)
                # # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
                # # # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20 - preprocess & mix final data
                # # # Train Mean: 3788.6515897900226 Train Std: 17932.36734269604 - 11/09/20 - damged piano artificial noise data
                # # FIX DMGED/ART DATA
                # if dmged_piano_artificial_noise_mix:
                #     train_mean, train_std = TRAIN_MEAN_DMGED, TRAIN_STD_DMGED
                # else:
                #     train_mean, train_std = TRAIN_MEAN, TRAIN_STD

                model = evaluate_source_sep(x_train_files, y1_train_files, y2_train_files, 
                                        x_val_files, y1_val_files, y2_val_files,
                                        num_train, num_val,
                                        n_feat=train_feat, n_seq=train_seq, 
                                        batch_size=train_batch_size, 
                                        loss_const=train_loss_const, epochs=train_epochs, 
                                        opt_name=train_opt_name, opt_clip_val=train_opt_clipval, 
                                        opt_lr=train_opt_lr,
                                        patience=patience, epsilon=epsilon,
                                        recent_model_path=recent_model_path, pc_run=pc_run,
                                        config=training_arch_config, # t_mean=train_mean, t_std=train_std,
                                        dataset2=dmged_piano_artificial_noise_mix,
                                        data_path=data_path, min_sig_len=min_sig_len, 
                                        data_from_numpy=data_from_numpy,
                                        tuned_a430hz=tuned_a430hz,
                                        use_basis_vectors=basis_vector_features,
                                        dmged_y1_train_files=dmged_y1_train_files,
                                        dmged_y1_val_files=dmged_y1_val_files,
                                        loop_bare_noise=loop_bare_noise,
                                        low_time_steps=low_time_steps,
                                        artificial_noise=artificial_noise)
                if sample:
                    restore_with_drnn(infer_output_path, recent_model_path, # wdw_size, epsilon, 
                                    # train_loss_const,
                                    train_opt_name, train_opt_clipval, train_opt_lr,
                                    min_sig_len, 
                                    # test_filepath=None, 
                                    test_sig=test_sig, test_sr=test_sr,
                                    # config=training_arch_config, t_mean=train_mean, t_std=train_std, 
                                    pc_run=pc_run, name_addon=output_file_addon,
                                    tuned_a430hz=tuned_a430hz,
                                    use_basis_vectors=basis_vector_features,
                                    low_tsteps=low_time_steps)
                else:
                    restore_with_drnn(infer_output_path, recent_model_path, # wdw_size, epsilon,
                                    # train_loss_const, 
                                    train_opt_name, train_opt_clipval, train_opt_lr,
                                    min_sig_len, 
                                    test_filepath=brahms_path,
                                    # config=training_arch_config, t_mean=train_mean, t_std=train_std, 
                                    pc_run=pc_run, name_addon=output_file_addon,
                                    tuned_a430hz=tuned_a430hz,
                                    use_basis_vectors=basis_vector_features,
                                    low_tsteps=low_time_steps)

        # GRID SEARCH
        elif mode == 'g':
            # Dennis - think of good metrics (my loss is obvious first start)
            restart, gs_id = False, ''
            for arg_i in range(3, 7):
                if arg_i < len(sys.argv):
                    if sys.argv[arg_i] == '-f':
                        restart = True
                        print('\nGRID SEARCH TO FORCE RESTART\n')
                    elif sys.argv[arg_i].isdigit() and len(sys.argv[arg_i]) == 1:
                        gs_id = sys.argv[arg_i]
                        print('GRID SEARCH ID:', gs_id, '\n')

            early_stop_pat = 10 if low_time_steps else 5
            # Define which files to grab for training. Shuffle regardless.
            actual_samples = TOTAL_SHORT_SMPLS if low_time_steps else TOTAL_SMPLS
            sample_indices = list(range(actual_samples))
            random.shuffle(sample_indices)

            x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])

            # Validation & Training Split
            indices = list(range(actual_samples))
            val_indices = indices[:math.ceil(actual_samples * val_split)]
            x_train_files = np.delete(x_files, val_indices)
            y1_train_files = np.delete(y1_files, val_indices)
            y2_train_files = np.delete(y2_files, val_indices)
            x_val_files = x_files[val_indices]
            y1_val_files = y1_files[val_indices]
            y2_val_files = y2_files[val_indices]
            # OLD
            # max_sig_len = MAX_SIG_LEN
            # Temp - get training data dim (from dummy) (for model & data making)
            # max_len_sig = np.ones((MAX_SIG_LEN))
            # dummy_train_spgm = make_spectrogram(max_len_sig, wdw_size, 
            #                                     ova=True, debug=False)[0].astype('float32').T
            # TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = dummy_train_spgm.shape
            # train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN
            
            # NEW
            # train_seq, train_feat, min_sig_len = get_raw_data_stats(y1_files, y2_files, x_files, 
            #                                                         brahms_filename=brahms_path)
            if low_time_steps:
                train_seq, train_feat, min_sig_len = TRAIN_SEQ_LEN_SMALL, TRAIN_FEAT_LEN, MIN_SIG_LEN_SMALL
            else:
                train_seq, train_feat, min_sig_len = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MIN_SIG_LEN

            # broken
            # if basis_vector_features:
            #     train_seq += NUM_SCORE_NOTES
            print('Grid Search Input Stats:')
            print('N Feat:', train_feat, 'Seq Len:', train_seq)

            # OLD
            # # TEMP - update for each unique dataset
            # # num_train, num_val = len(y1_train_files), len(y1_val_files)
            # # Note - If not numpy, consider if dataset2. If numpy, supply x files.
            # # train_mean, train_std = get_data_stats(y1_train_files, y2_train_files, num_train,
            # #                                   train_seq=train_seq, train_feat=train_feat, 
            # #                                   wdw_size=wdw_size, epsilon=epsilon, 
            # #                                   pad_len=max_sig_len)
            # # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
            # # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20
            # train_mean, train_std = TRAIN_MEAN, TRAIN_STD

            grid_search(x_train_files, y1_train_files, y2_train_files,
                        x_val_files, y1_val_files, y2_val_files,
                        n_feat=train_feat, n_seq=train_seq,
                        epsilon=epsilon,
                        # t_mean=train_mean, t_std=train_std,
                        train_configs=train_configs,
                        arch_config_optns=arch_config_optns,
                        gsres_path=gs_output_path,
                        early_stop_pat=early_stop_pat, 
                        pc_run=pc_run, gs_id=gs_id, 
                        restart=restart,
                        dataset2=dmged_piano_artificial_noise_mix,
                        tuned_a430hz=tuned_a430hz, 
                        use_basis_vectors=basis_vector_features,
                        save_model_path=recent_model_path if gs_write_model else None,  # NEW - for small grid searchse
                        low_time_steps=low_time_steps) 


if __name__ == '__main__':
    main()