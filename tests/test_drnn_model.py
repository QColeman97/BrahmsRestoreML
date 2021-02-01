# test_drnn_data.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for drnn data preprocessing functions.

# Run with $ python -m unittest tests.test_drnn_data


from brahms_restore_ml.drnn.data import *
from brahms_restore_ml.drnn.drnn import TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MAX_SIG_LEN
import unittest
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False

data_path = 'brahms_restore_ml/drnn/drnn_data/'

class DRNNDataTests(unittest.TestCase):

    def test_sig_to_spgm(self):
        sig, wdw_size = np.ones(10), 4
        spgm = signal_to_nn_features(sig, wdw_size=wdw_size)
        with self.subTest():
            self.assertEqual(spgm.shape, (4, (wdw_size//2)+1))
        with self.subTest():
            self.assertEqual(spgm.dtype, 'float32')

    def test_preprocess_sig_mix(self):
        sig1 = np.arange(10)
        sig2 = np.ones((10,))
        src_amp_low, src_amp_high = 0.75, 1.15
        sig_mix, sig1, sig2 = preprocess_signals(sig1, sig2, len(sig1),
                                                src_amp_low=src_amp_low, 
                                                src_amp_high=src_amp_high)
        print('Src1 Amp %:', src_amp_low, 'Src2 Amp %:', src_amp_high)
        print('Mix Sum:', np.sum(np.abs(sig_mix)), 'Sig1:', np.sum(np.abs(sig1)), 'Sig2:', np.sum(np.abs(sig2)), 'Sum 1 & 2:', np.sum(np.abs(sig1+sig2)))
        with self.subTest():
            self.assertNotEqual(np.sum(sig_mix), np.sum(sig1 + sig2))
        with self.subTest():
            self.assertAlmostEqual(np.sum(np.abs(sig_mix)), (np.sum(np.abs(sig1)) + np.sum(np.abs(sig2)))/2)

    def test_preprocess_sig_mix2(self):
        sig1 = np.arange(10)
        sig2 = np.ones((10,))
        src_amp_low, src_amp_high = 0.05, 5.0
        sig_mix, sig1, sig2 = preprocess_signals(sig1, sig2, len(sig1),
                                                src_amp_low=src_amp_low, 
                                                src_amp_high=src_amp_high)
        print('Src1 Amp %:', src_amp_low, 'Src2 Amp %:', src_amp_high)
        print('Mix Sum:', np.sum(np.abs(sig_mix)), 'Sig1:', np.sum(np.abs(sig1)), 'Sig2:', np.sum(np.abs(sig2)), 'Sum 1 & 2:', np.sum(np.abs(sig1+sig2)))
        with self.subTest():
            self.assertNotEqual(np.sum(sig_mix), np.sum(sig1 + sig2))
        with self.subTest():
            self.assertAlmostEqual(np.sum(np.abs(sig_mix)), (np.sum(np.abs(sig1)) + np.sum(np.abs(sig2)))/2)

    def test_preprocess_sig(self):
        sig1 = np.arange(10)
        sig2 = np.ones((10,))
        sig_mix = np.arange(10)
        pad_len = 12
        sig_mix, sig1, sig2 = preprocess_signals(sig1, sig2, pad_len)
        with self.subTest():
            self.assertEqual(len(sig_mix), pad_len)
        with self.subTest():
            self.assertEqual(len(sig1), pad_len)
        with self.subTest():
            self.assertEqual(len(sig2), pad_len)

    def test_generator_from_numpy(self):
        n_samples, bs, n_seq, n_feat, pad_len = 2, 1, TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MAX_SIG_LEN
        x_files = [data_path+'piano_noise_numpy/mixed'+str(i)+'.npy' 
                    for i in range(n_samples)]
        y1_files = [data_path+'piano_source_numpy/piano'+str(i)+'.npy' 
                    for i in range(n_samples)]
        y2_files = [data_path+'noise_source_numpy/noise'+str(i)+'.npy' 
                    for i in range(n_samples)]
        gen = nn_data_generator(y1_files, y2_files, n_samples, bs, n_seq, n_feat,
                                pad_len=pad_len, x_files=x_files, from_numpy=True)
        yielded = next(gen)
        with self.subTest():
            self.assertEqual(len(yielded), 2)
        with self.subTest():
            self.assertEqual(yielded[0].shape, (bs, n_seq, n_feat))
        with self.subTest():
            self.assertEqual(yielded[1].shape, (bs, n_seq, n_feat*2))
        yielded = next(gen)
        with self.subTest():
            self.assertEqual(len(yielded), 2)
        with self.subTest():
            self.assertEqual(yielded[0].shape, (bs, n_seq, n_feat))
        with self.subTest():
            self.assertEqual(yielded[1].shape, (bs, n_seq, n_feat*2))
        yielded = next(gen) # infinite generator - restarted
        with self.subTest():
            self.assertEqual(len(yielded), 2)
        with self.subTest():
            self.assertEqual(yielded[0].shape, (bs, n_seq, n_feat))
        with self.subTest():
            self.assertEqual(yielded[1].shape, (bs, n_seq, n_feat*2))

    def test_generator(self):
        n_samples, bs, n_seq, n_feat, pad_len = 1, 1, TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MAX_SIG_LEN
        y1_files = [data_path+'final_piano_wav/psource'+str(i)+'.wav' 
                    for i in range(n_samples)]
        y2_files = [data_path+'final_noise_wav/nsource'+str(i)+'.wav' 
                    for i in range(n_samples)]
        gen = nn_data_generator(y1_files, y2_files, n_samples, bs, n_seq, n_feat, 
                                pad_len=pad_len, dmged_piano_artificial_noise=False,
                                data_path=data_path)
        yielded = next(gen)
        with self.subTest():
            self.assertEqual(len(yielded), 2)
        with self.subTest():
            self.assertEqual(yielded[0].shape, (bs, n_seq, n_feat))
        with self.subTest():
            self.assertEqual(yielded[1].shape, (bs, n_seq, n_feat*2))
    
    def test_generator_mix_given(self):
        n_samples, bs, n_seq, n_feat, pad_len = 1, 1, TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MAX_SIG_LEN
        x_files = [data_path+'dmged_mix_wav/features'+str(i)+'.wav' 
                    for i in range(n_samples)]
        y1_files = [data_path+'final_piano_wav/psource'+str(i)+'.wav' 
                    for i in range(n_samples)]
        y2_files = [data_path+'dmged_noise_wav/nsource'+str(i)+'.wav' 
                    for i in range(n_samples)]
        gen = nn_data_generator(y1_files, y2_files, n_samples, bs, n_seq, n_feat,
                                pad_len=pad_len, dmged_piano_artificial_noise=False,
                                data_path=data_path, x_files=x_files)
        yielded = next(gen)
        with self.subTest():
            self.assertEqual(len(yielded), 2)
        with self.subTest():
            self.assertEqual(yielded[0].shape, (bs, n_seq, n_feat))
        with self.subTest():
            self.assertEqual(yielded[1].shape, (bs, n_seq, n_feat*2))
    
    def test_get_data_stats(self):
        n_samples, n_seq, n_feat, pad_len = 6, TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MAX_SIG_LEN
        x_files = [data_path+'piano_noise_numpy/mixed'+str(i)+'.npy' 
                    for i in range(n_samples)]
        y1_files = [data_path+'piano_source_numpy/piano'+str(i)+'.npy' 
                    for i in range(n_samples)]
        y2_files = [data_path+'noise_source_numpy/noise'+str(i)+'.npy' 
                    for i in range(n_samples)]
        mean, std = get_data_stats(y1_files, y2_files, n_samples, n_seq, n_feat, 
                pad_len, data_path=data_path, x_filenames=x_files, from_numpy=True)
        with self.subTest():
            self.assertLess(0, mean)
        with self.subTest():
            self.assertLess(0, std)

        
  
if __name__ == '__main__':
    unittest.main()
