import pandas as pd
import json
import numpy as np

def show_most_varied_hps(combos, grid_search_results_path, val_loss=False, pc=False):
    top_gs_result_vl_files, top_gs_result_l_files = [], []
    with open('brahms_restore_ml/drnn/top_gs_results_vl_' + str(combos) + '.txt') as fp:
        for line in fp:
            top_gs_result_vl_files.append(int(line.split(',')[0]))
    with open('brahms_restore_ml/drnn/top_gs_results_l_' + str(combos) + '.txt') as fp:
        for line in fp:
            top_gs_result_l_files.append(int(line.split(',')[0]))

    # print(top_gs_result_vl_files)
    # print(top_gs_result_l_files)

    # One-hot, each var is bool (1,0)
    # Unchanged HPs: nrn divisor
    if combos == 24:
        hps = [#'SmallBS', 
            'LowDenseLayers', 'MedDenseLayers', 'HighDenseLayers',
            'Scale', 'RnnDropout', 'BN']#, 
            # 'LowLossConst', 'Adam', 'ClipValNormalLR']
    elif combos == 72:
        hps = ['SmallBS', 'LowEpochs', 'LowGamma', 'MedGamma', 'HighGamma',
                'LowClipVal', 'Scale', 'LowRNNLayers']
    elif combos != 2048:
        hps = ['SmallBS',
            'LowRNNLayers',
            'DenseFirst', 'TanHInFirst',
            'Scale', 'ResCntn', 'Bidir', 'RnnDropout', 'BN', 
            'LowLossConst', 'Adam', 'ClipValNormalLR']
    else:
        hps = ['SmallBS',
            'LowRNNLayers',
            'DenseTanHFirst',
            'Scale', 'ResCntn', 'Bidir', 'RnnDropout', 'BN', 
            'LowLossConst', 'Adam', 'ClipValNormalLR']
        # hps = ['RNN', 'LSTM', 
        #     'LowRNNLayers', 'HighRNNLayers', 
        #     'DenseFirst', #'NrnDiv2', 'NrnDiv4', 
        #     'Scale', 'ResCntn', 'Bidir', 'RnnDropout', 'BN', 
        #     'Adam', 'RMSprop', 'ClipVal','LowLR']

    top_result_files = top_gs_result_vl_files if val_loss else top_gs_result_l_files
    # Initialize DF
    vl_hp_combos_df = pd.DataFrame(columns=hps, index=top_result_files) 
    # Fill DF
    pc_suffix = '.txt' if pc else '_noPC.txt'
    for file_id in top_result_files:
        file_name = grid_search_results_path + 'result_' + str(file_id) + '_of_' + str(combos) + pc_suffix
        with open(file_name, 'r') as fp:
            for _ in range(4):
                _ = fp.readline()

            hp_config = json.loads(fp.readline())
            # print('FILE ID:', file_id, '\nHP CONFIG:', hp_config)

            if combos == 2048:
                small_bs = 1 if (hp_config['batch_size'] == 1) else 0
            elif combos == 72:
                small_bs = 1 if (hp_config['batch_size'] <= 50) else 0
            else:
                small_bs = 1 if (hp_config['batch_size'] == 8) else 0
            low_epochs = 1 if (hp_config['epochs'] < 50) else 0
            low_rnn = 1 if (len(hp_config['layers']) < 4 or 
                    (len(hp_config['layers']) == 4 and 
                            hp_config['layers'][0]['type'] == 'Dense')
                    ) else 0
            dense_first = 1 if (hp_config['layers'][0]['type'] == 'Dense') else 0
            if combos == 24:
                low_dense = 1 if (len(hp_config['layers']) == 3) else 0
                med_dense = 1 if (len(hp_config['layers']) == 4) else 0
                high_dense = 1 if (len(hp_config['layers']) == 5) else 0
            scale = 1 if (hp_config['scale']) else 0
            res_cntn = 1 if (hp_config['rnn_res_cntn']) else 0
            bidir = 1 if (hp_config['bidir']) else 0
            rnn_dropout = 0 if (hp_config['rnn_dropout'] == [0.0, 0.0]) else 1
            bn = 1 if (hp_config['bn']) else 0
            if combos == 192:
                low_loss_const = 1 if (hp_config['gamma'] == 0.1) else 0
            elif combos == 72:
                low_loss_const = 1 if (hp_config['gamma'] <= 0.05) else 0
            else:
                low_loss_const = 1 if (hp_config['gamma'] == 0.05) else 0
            med_loss_const = 1 if (hp_config['gamma'] == 0.15) else 0
            high_loss_const = 1 if (hp_config['gamma'] == 0.3) else 0
            adam = 1 if (hp_config['optimizer'] == 'Adam') else 0
            clip_val_default_lr = 1 if (hp_config['clip value'] == 10) else 0
            low_clip_val = 1 if (hp_config['clip value'] is None or hp_config['clip value'] < 10) else 0
            if combos == 24:
                vl_hp_combos_df.loc[file_id] = [#small_bs, 
                                                low_dense, med_dense, high_dense,
                                                scale,
                                                rnn_dropout, bn]#, low_loss_const,
                                                # adam, clip_val_default_lr] 
            elif combos == 72:
                vl_hp_combos_df.loc[file_id] = [
                    small_bs, low_epochs, low_loss_const, med_loss_const, high_loss_const,
                    low_clip_val, scale, low_rnn]
            elif combos == 2048:
                vl_hp_combos_df.loc[file_id] = [small_bs, low_rnn,
                                                dense_first,
                                                scale, res_cntn, bidir,
                                                rnn_dropout, bn, low_loss_const,
                                                adam, clip_val_default_lr] 
            else:
                tanh_in_first = 1 if (hp_config['layers'][0]['type'] == 'Dense' and hp_config['layers'][0]['act'] == 'tanh') else 0
                vl_hp_combos_df.loc[file_id] = [small_bs, low_rnn,
                                                dense_first, tanh_in_first,
                                                scale, res_cntn, bidir,
                                                rnn_dropout, bn, low_loss_const,
                                                adam, clip_val_default_lr] 
            # else:
                # rnn = 1 if (hp_config['layers'][1]['type'] == 'RNN') else 0
                # lstm = 1 if (not rnn) else 0
                # low_rnn = 1 if (len(hp_config['layers']) < 4 or 
                #         (len(hp_config['layers']) == 4 and 
                #                 hp_config['layers'][0]['type'] == 'Dense')
                #         ) else 0
                # high_rnn = 1 if (not low_rnn) else 0
                # dense_first = 1 if (hp_config['layers'][0]['type'] == 'Dense') else 0
                # # nrn_div_2 = 1 if (hp_config['layers'][0]['nrn_div'] == 2) else 0
                # # nrn_div_4 = 1 if (not nrn_div_2) else 0
                # scale = 1 if (hp_config['scale']) else 0
                # res_cntn = 1 if (hp_config['rnn_res_cntn']) else 0
                # bidir = 1 if (hp_config['bidir']) else 0
                # rnn_dropout = 1 if (hp_config['rnn_dropout']) else 0
                # bn = 1 if (hp_config['bn']) else 0
                # adam = 1 if (hp_config['optimizer'] == 'Adam') else 0
                # rms_prop = 1 if (not adam) else 0
                # clip_val = 1 if (hp_config['clip value'] == 10) else 0
                # low_lr = 1 if (not clip_val) else 0
                # vl_hp_combos_df.loc[file_id] = [rnn, lstm,
                #                                 low_rnn, high_rnn, 
                #                                 dense_first,
                #                                 # nrn_div_2, nrn_div_4,
                #                                 scale, res_cntn, bidir,
                #                                 rnn_dropout, bn, adam, 
                #                                 rms_prop, clip_val,
                #                                 low_lr] 
    vl_hp_combos_df = vl_hp_combos_df.astype(int)
    print('HPs of Top Combos', '(Val. Loss)' if val_loss else '(Loss)', 'DF:')
    print(vl_hp_combos_df)

    vl_cov_df = vl_hp_combos_df.T.cov()
    # vl_cov_mat = vl_cov_df.values
    # print('Covariance Mat:')
    # print(vl_cov_mat)
    # bool_utri = np.triu(np.ones(vl_cov_df.shape), k=1)
    # print(bool_utri)
    # print(vl_cov_df.stack())
    # vl_cov_df.where(bool_utri.astype(np.bool))
    # print(vl_cov_df.where(np.triu(np.ones(vl_cov_df.shape), k=1).astype(np.bool)).stack())

    # https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
    vl_most_varied = (vl_cov_df.where(np.triu(np.ones(vl_cov_df.shape), k=1).astype(np.bool))
                    .stack()
                    .sort_values(ascending=True)[:20])
    print('Top files by most varied HPs', '(Val. Loss):' if val_loss else '(Loss):')
    print(vl_most_varied)


def main():
    combos = 3072
    grid_search_results_path = 'brahms_restore_ml/drnn/output_grid_search_wb/'
    do_val_loss = True
    pc = False

    show_most_varied_hps(combos, grid_search_results_path, do_val_loss, pc)

if __name__ == '__main__':
    main()