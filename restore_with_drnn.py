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

def run_top_gs_result(num, best_config, train_mean, train_std, x_train_files, y1_train_files, y2_train_files,
                      x_val_files, y1_val_files, y2_val_files, num_train, num_val, train_feat, train_seq,
                      patience, epsilon, recent_model_path, pc_run, dmged_piano_artificial_noise_mix,
                      infer_output_path, wdw_size, brahms_path, combos_str, data_path=None, max_sig_len=-1):
    train_batch_size = best_config['batch_size']
    # # Temp test for LSTM -> until can grid search
    # train_batch_size = 3
    train_loss_const = best_config['gamma']
    train_epochs = best_config['epochs']
    train_opt_name = best_config['optimizer']
    train_opt_clipval = None if (best_config['clip value'] == -1) else best_config['clip value']
    train_opt_lr = best_config['learning rate']

    training_arch_config = {}
    training_arch_config['layers'] = best_config['layers']
    # # Temp test for LSTM -> until can grid search
    # for i in range(len(best_config['layers'])):
    #     if best_config['layers'][i]['type'] == 'RNN':
    #         training_arch_config['layers'][i]['type'] = 'LSTM'
    training_arch_config['scale'] = best_config['scale']
    training_arch_config['rnn_res_cntn'] = best_config['rnn_res_cntn']
    training_arch_config['bias_rnn'] = best_config['bias_rnn']
    training_arch_config['bias_dense'] = best_config['bias_dense']
    training_arch_config['bidir'] = best_config['bidir']
    training_arch_config['rnn_dropout'] = best_config['rnn_dropout']
    training_arch_config['bn'] = best_config['bn']

    print('#', num, 'TOP TRAIN ARCH FOR USE:')
    print(training_arch_config)
    print('#', num, 'TOP TRAIN HPs FOR USE:')
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
                        train_mean, train_std, None, None, None, None, None,
                        dmged_piano_artificial_noise_mix))
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
    process_infer = multiprocessing.Process(target=restore_audio_file, args=(infer_output_path, recent_model_path, wdw_size, epsilon,
                        train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr, brahms_path, None, None,
                        training_arch_config, train_mean, train_std, pc_run, '_'+num+'of'+combos_str))
    process_infer.start()
    process_infer.join()

    # restore_audio_file(infer_output_path, recent_model_path, wdw_size, epsilon,
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
    # Differentiate PC GS from F35 GS
    dmged_piano_artificial_noise_mix = True if pc_run else False
    test_on_synthetic = False
    wdw_size = PIANO_WDW_SIZE
    data_path = 'brahms_restore_ml/drnn/drnn_data/'
    arch_config_path = 'brahms_restore_ml/drnn/config/'
    gs_output_path = 'brahms_restore_ml/drnn/output_grid_search/'           # PC
    # gs_output_path = 'brahms_restore_ml/drnn/output_grid_search_lstm/'    # F35
    recent_model_path = 'brahms_restore_ml/drnn/recent_model'
    infer_output_path = 'brahms_restore_ml/drnn/output_restore/'
    brahms_path = 'brahms.wav'

    # add-on
    do_curr_best, curr_best_combos, curr_best_pc = True, '2048', True
    # # F35 LSTM
    # top_result_nums = [72, 128, 24, 176, 8, 192, 88, 112]
    # # F35 WB
    # top_result_nums = [1488, 1568, 149, 1496, 1680, 86, 151, 152]
    # PC WB
    top_result_nums = [997, 1184, 1312, 1310, 1311, 1736]
    top_result_paths = [gs_output_path + 'result_' + str(x) + '_of_' + curr_best_combos +
                        ('.txt' if curr_best_pc else '_noPC.txt') for x in top_result_nums]

    # EMPERICALLY DERIVED HPs
    # Note: FROM PO-SEN PAPER - about loss_const
    #   Empirically, the value γ is in the range of 0.05∼0.2 in order
    #   to achieve SIR improvements and maintain SAR and SDR.
    train_batch_size = 6 if pc_run else 12
    train_loss_const = 0.05
    train_epochs = 10
    train_opt_name, train_opt_clipval, train_opt_lr = 'RMSprop', 0.9, 0.001
    training_arch_config = None

    epsilon, patience, val_split = 10 ** (-10), train_epochs, 0.25

    # INFER ONLY
    # Branch temporarily broken - tf dependency
    if mode == 'r':
        restore_audio_file(infer_output_path, recent_model_path, wdw_size, epsilon,
                            train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr,
                            brahms_path, t_mean=TRAIN_MEAN, t_std=TRAIN_STD, pc_run=pc_run)
    else:
        train_configs, arch_config_optns = get_hp_configs(arch_config_path, pc_run=pc_run)
        # print('First arch config optn after return:', arch_config_optns[0])

        # Load in train/validation data
        noise_piano_filepath_prefix = ((data_path + 'dmged_mix_numpy/mixed')
            if dmged_piano_artificial_noise_mix else (data_path + 'piano_noise_numpy/mixed'))
        piano_label_filepath_prefix = ((data_path + 'piano_source_numpy/piano')
            if dmged_piano_artificial_noise_mix else (data_path + 'piano_source_numpy/piano'))
        noise_label_filepath_prefix = ((data_path + 'dmged_noise_numpy/noise')
            if dmged_piano_artificial_noise_mix else (data_path + 'noise_source_numpy/noise'))
        # # FIX DMGED/ART DATA - get wav files
        # noise_piano_filepath_prefix = ((data_path + 'dmged_mix_wav/features')
        #     if dmged_piano_artificial_noise_mix else (data_path + 'piano_noise_numpy/mixed'))
        # piano_label_filepath_prefix = ((data_path + 'final_piano_wav/psource')
        #     if dmged_piano_artificial_noise_mix else (data_path + 'piano_source_numpy/piano'))
        # noise_label_filepath_prefix = ((data_path + 'dmged_noise_wav/nsource')
        #     if dmged_piano_artificial_noise_mix else (data_path + 'noise_source_numpy/noise'))
        print('\nTRAINING WITH DATASET', '2 (ARTIFICIAL DMG)' if dmged_piano_artificial_noise_mix else '1 (ORIG)')

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
                actual_samples = TOTAL_SMPLS    # How many to leave out (1)
                sample_indices = list(range(TOTAL_SMPLS + 1))
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
                actual_samples = TOTAL_SMPLS
                sample_indices = list(range(TOTAL_SMPLS))
                # FIX DMGED/ART DATA
                random.shuffle(sample_indices)
            
            # FIX DMGED/ART DATA
            # if dmged_piano_artificial_noise_mix:
            #     x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            #     y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            #     y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.wav')
            #                 for i in sample_indices])
            # else:
            x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            
            # # # Temp - do to calc max len for padding - it's 3081621 (for youtube src data)
            # # # it's 3784581 (for Spotify/Youtube Final Data)
            # # # it's 3784581 (for damaged Spotify/YouTube Final Data)
            # # max_sig_len = None
            # # for x_file in x_files:
            # #     _, sig = wavfile.read(x_file)
            # #     if max_sig_len is None or len(sig) >max_sig_len:
            # #         max_sig_len = len(sig)
            # # print('NOTICE: MAX SIG LEN', max_sig_len)
            max_sig_len = MAX_SIG_LEN

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

            # DEBUG PRINT
            # print('y1_train_files:', y1_train_files[:10])
            # print('y1_val_files:', y1_val_files[:10])
            # print('y2_train_files:', y2_train_files[:10])
            # print('y2_val_files:', y2_val_files[:10])

            # # Temp - get training data dim (from dummy) (for model & data making)
            # max_len_sig = np.ones((max_sig_len))
            # dummy_train_spgm, _ = make_spectrogram(max_len_sig, wdw_size, epsilon,
            #                                     ova=True, debug=False)
            # train_seq, train_feat = dummy_train_spgm.shape
            # print('NOTICE: TRAIN SEQ LEN', train_seq, 'TRAIN FEAT LEN', train_feat)
            train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN

            # CUSTOM TRAINING Dist training needs a "global_batch_size"
            # if not pc_run:
            #     batch_size_per_replica = train_batch_size // 2
            #     train_batch_size = batch_size_per_replica * mirrored_strategy.num_replicas_in_sync

            print('Train Input Stats:')
            print('N Feat:', train_feat, 'Seq Len:', train_seq, 'Batch Size:', train_batch_size)

            if do_curr_best:
                for top_result_path in top_result_paths:
                    # TEMP - For damaged data
                    num = top_result_path.split('_')[-3] if dmged_piano_artificial_noise_mix else top_result_path.split('_')[-4]
                    gs_result_file = open(top_result_path, 'r')
                    for _ in range(4):
                        _ = gs_result_file.readline()
                    best_config = json.loads(gs_result_file.readline())
                    # Temp test for LSTM -> until can grid search
                    # if (len(best_config['layers']) < 4) or (len(best_config['layers']) == 4 and best_config['layers'][0]['type'] == 'Dense'):
                    run_top_gs_result(num, best_config, TRAIN_MEAN, TRAIN_STD, x_train_files, y1_train_files, y2_train_files,
                                    x_val_files, y1_val_files, y2_val_files, num_train, num_val, train_feat, train_seq,
                                    patience, epsilon, recent_model_path, pc_run, dmged_piano_artificial_noise_mix,
                                    infer_output_path, wdw_size, brahms_path, curr_best_combos)

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

                # # TEMP - update for each unique dataset
                # Note - If not numpy, consider if dataset2. If numpy, supply x files.
                # train_mean, train_std = get_data_stats(y1_train_files, y2_train_files, num_train,
                #                                   train_seq=train_seq, train_feat=train_feat, 
                #                                   wdw_size=wdw_size, epsilon=epsilon, 
                #                                 #   pad_len=max_sig_len)
                #                                   pad_len=max_sig_len, x_filenames=x_train_files)
                # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
                # # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20 - preprocess & mix final data
                # # Train Mean: 3788.6515897900226 Train Std: 17932.36734269604 - 11/09/20 - damged piano artificial noise data
                # FIX DMGED/ART DATA
                if dmged_piano_artificial_noise_mix:
                    train_mean, train_std = TRAIN_MEAN_DMGED, TRAIN_STD_DMGED
                else:
                    train_mean, train_std = TRAIN_MEAN, TRAIN_STD

                model = evaluate_source_sep(x_train_files, y1_train_files, y2_train_files, 
                                        x_val_files, y1_val_files, y2_val_files,
                                        num_train, num_val,
                                        n_feat=train_feat, n_seq=train_seq, 
                                        batch_size=train_batch_size, 
                                        loss_const=train_loss_const, epochs=train_epochs, 
                                        opt_name=train_opt_name, opt_clip_val=train_opt_clipval, opt_lr=train_opt_lr,
                                        patience=patience, epsilon=epsilon,
                                        recent_model_path=recent_model_path, pc_run=pc_run,
                                        config=training_arch_config, t_mean=train_mean, t_std=train_std,
                                        dataset2=dmged_piano_artificial_noise_mix)
                if sample:
                    restore_audio_file(infer_output_path, recent_model_path, wdw_size, epsilon, 
                                    train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr,
                                    test_filepath=None, 
                                    test_sig=test_sig, test_sr=test_sr,
                                    config=training_arch_config, t_mean=train_mean, t_std=train_std, pc_run=pc_run)
                else:
                    restore_audio_file(infer_output_path, recent_model_path, wdw_size, epsilon,
                                    train_loss_const, train_opt_name, train_opt_clipval, train_opt_lr,
                                    test_filepath=brahms_path,
                                    config=training_arch_config, t_mean=train_mean, t_std=train_std, pc_run=pc_run)

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

            early_stop_pat = 5
            # Define which files to grab for training. Shuffle regardless.
            actual_samples = TOTAL_SMPLS
            sample_indices = list(range(TOTAL_SMPLS))
            random.shuffle(sample_indices)

            x_files = np.array([(noise_piano_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y1_files = np.array([(piano_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            y2_files = np.array([(noise_label_filepath_prefix + str(i) + '.npy')
                        for i in sample_indices])
            # max_sig_len = MAX_SIG_LEN

            # Validation & Training Split
            indices = list(range(actual_samples))
            val_indices = indices[:math.ceil(actual_samples * val_split)]
            x_train_files = np.delete(x_files, val_indices)
            y1_train_files = np.delete(y1_files, val_indices)
            y2_train_files = np.delete(y2_files, val_indices)
            x_val_files = x_files[val_indices]
            y1_val_files = y1_files[val_indices]
            y2_val_files = y2_files[val_indices]

            # Temp - get training data dim (from dummy) (for model & data making)
            # max_len_sig = np.ones((MAX_SIG_LEN))
            # dummy_train_spgm = make_spectrogram(max_len_sig, wdw_size, 
            #                                     ova=True, debug=False)[0].astype('float32').T
            # TRAIN_SEQ_LEN, TRAIN_FEAT_LEN = dummy_train_spgm.shape
            train_seq, train_feat = TRAIN_SEQ_LEN, TRAIN_FEAT_LEN
            print('Grid Search Input Stats:')
            print('N Feat:', train_feat, 'Seq Len:', train_seq)

            # TEMP - update for each unique dataset
            # num_train, num_val = len(y1_train_files), len(y1_val_files)
            # Note - If not numpy, consider if dataset2. If numpy, supply x files.
            # train_mean, train_std = get_data_stats(y1_train_files, y2_train_files, num_train,
            #                                   train_seq=train_seq, train_feat=train_feat, 
            #                                   wdw_size=wdw_size, epsilon=epsilon, 
            #                                   pad_len=max_sig_len)
            # print('REMEMBER Train Mean:', train_mean, 'Train Std:', train_std, '\n')
            # Train Mean: 1728.2116672701493 Train Std: 6450.4985228518635 - 10/18/20
            train_mean, train_std = TRAIN_MEAN, TRAIN_STD

            grid_search(x_train_files, y1_train_files, y2_train_files,
                        x_val_files, y1_val_files, y2_val_files,
                        n_feat=train_feat, n_seq=train_seq,
                        epsilon=epsilon,
                        t_mean=train_mean, t_std=train_std,
                        train_configs=train_configs,
                        arch_config_optns=arch_config_optns,
                        gsres_path=gs_output_path,
                        early_stop_pat=early_stop_pat, 
                        pc_run=pc_run, gs_id=gs_id, 
                        restart=restart,
                        dataset2=dmged_piano_artificial_noise_mix)


if __name__ == '__main__':
    main()