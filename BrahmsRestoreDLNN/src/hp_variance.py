import pandas as pd
import json
import numpy as np

top_gs_result_vl_files, top_gs_result_l_files = [], []
with open('../top_gs_results_vl.txt') as fp:
    for line in fp:
        top_gs_result_vl_files.append(int(line.split(',')[0]))
with open('../top_gs_results_l.txt') as fp:
    for line in fp:
        top_gs_result_l_files.append(int(line.split(',')[0]))

# print(top_gs_result_vl_files)
# print(top_gs_result_l_files)

grid_search_results_path = '../output_grid_search_save_11_19/'

# One-hot, each var is bool (1,0)
hps = ['RNN', 'LSTM', 
    'LowRNNLayers', 'HighRNNLayers', 
    'DenseFirst', #'NrnDiv2', 'NrnDiv4', 
    'Scale', 'ResCntn', 'Bidir', 'RnnDropout', 'BN', 
    'Adam', 'RMSprop', 'ClipVal','LowLR']
vl_hp_combos_df = pd.DataFrame(columns=hps, index=top_gs_result_vl_files) 
# vl_hp_combos_df = pd.DataFrame(columns=hps, index=[str(x) for x in top_gs_result_vl_files]) 
# Fill DF
for file_id in top_gs_result_vl_files:
    file_name = grid_search_results_path + 'result_' + str(file_id) + '_of_4096_noPC.txt'
    with open(file_name, 'r') as fp:
        for _ in range(4):
            _ = fp.readline()

        hp_config = json.loads(fp.readline())
        rnn = 1 if (hp_config['layers'][1]['type'] == 'RNN') else 0
        lstm = 1 if (not rnn) else 0
        low_rnn = 1 if (len(hp_config['layers']) < 4 or 
                  (len(hp_config['layers']) == 4 and 
                        hp_config['layers'][0]['type'] == 'Dense')
                  ) else 0
        high_rnn = 1 if (not low_rnn) else 0
        dense_first = 1 if (hp_config['layers'][0]['type'] == 'Dense') else 0
        # nrn_div_2 = 1 if (hp_config['layers'][0]['nrn_div'] == 2) else 0
        # nrn_div_4 = 1 if (not nrn_div_2) else 0
        scale = 1 if (hp_config['scale']) else 0
        res_cntn = 1 if (hp_config['rnn_res_cntn']) else 0
        bidir = 1 if (hp_config['bidir']) else 0
        rnn_dropout = 1 if (hp_config['rnn_dropout']) else 0
        bn = 1 if (hp_config['bn']) else 0
        adam = 1 if (hp_config['optimizer'] == 'Adam') else 0
        rms_prop = 1 if (not adam) else 0
        clip_val = 1 if (hp_config['clip value'] == 10) else 0
        low_lr = 1 if (not clip_val) else 0
        vl_hp_combos_df.loc[file_id] = [rnn, lstm,
                                        low_rnn, high_rnn, 
                                        dense_first,
                                        # nrn_div_2, nrn_div_4,
                                        scale, res_cntn, bidir,
                                        rnn_dropout, bn, adam, 
                                        rms_prop, clip_val,
                                        low_lr] 
print('VL DF:')
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
print('Most varied VL files:')
print(vl_most_varied)

