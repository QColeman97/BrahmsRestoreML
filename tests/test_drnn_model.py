# test_drnn_model.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for drnn model building function.

# Run with $ python -m unittest tests.test_drnn_model


from brahms_restore_ml.drnn.model import *
from brahms_restore_ml.drnn.drnn import TRAIN_SEQ_LEN, TRAIN_FEAT_LEN, MIN_SIG_LEN, EPSILON
import tensorflow as tf
import unittest
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False

class DRNNModelTests(unittest.TestCase):

    def test_discrim_loss(self):
        bs, n_sgmts, n_feat = 2, 4, 3
        y1 = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]),
            dtype='float32')
        y2 = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]),
            dtype='float32')
        y_true = tf.concat([y1, y2], -1)

        loss_const_tensor = tf.ones([1, n_sgmts, n_feat])
        y_pred = tf.concat([y1, y2, loss_const_tensor], axis=0)

        loss = discrim_loss(y_true, y_pred)
        # Check for proper result (shape, values)
        with self.subTest():
            self.assertEqual(loss.shape, ())
        with self.subTest():
            self.assertEqual(loss, tf.constant([0.0]))
        
    def test_standardize_layer(self):
        # Check for proper range of values
        bs, n_sgmts, n_feat = 2, 4, 3
        tensor = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]), 
            dtype='float32')
        mean, std = tf.math.reduce_mean(tensor), tf.math.reduce_std(tensor)
        out = Standardize(mean.numpy(), std.numpy()) (tensor)

        out_mean, out_std = tf.math.reduce_mean(out), tf.math.reduce_std(out)
        with self.subTest():
            self.assertAlmostEqual(out_mean.numpy(), 0, places=0)
        with self.subTest():
            self.assertAlmostEqual(out_std.numpy(), 1, places=0)

    def test_stdize_unstdize_layers(self):
        # Check for same values after conversions
        bs, n_sgmts, n_feat = 2, 4, 3
        tensor = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]), 
            dtype='float32')
        mean, std = tf.math.reduce_mean(tensor), tf.math.reduce_std(tensor)
        out = Standardize(mean.numpy(), std.numpy()) (tensor)
        out2 = UnStandardize(mean.numpy(), std.numpy()) (out)
        tf.debugging.assert_equal(tensor, out2)

    def test_time_freq_masking_layer(self):
        # Check that the src outputs of masking sum to input mixture
        bs, n_sgmts, n_feat = 2, 4, 3
        mix = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]),
            dtype='float32')
        s1 = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]),
            dtype='float32')
        s2 = tf.cast(tf.reshape(tf.range(bs * n_sgmts * n_feat), [bs, n_sgmts, n_feat]),
            dtype='float32')
        
        s1_masked = TimeFreqMasking(EPSILON) ((s1, s2, mix))
        s2_masked = TimeFreqMasking(EPSILON) ((s1, s2, mix))
        masked_mix = s1_masked + s2_masked

        self.assertEqual(tf.reduce_sum(masked_mix).numpy(), tf.reduce_sum(mix).numpy())

    def test_make_model(self):
        # Check for legit model result
        n_sgmts, n_feat = 4, 3
        model = make_model(n_feat, n_sgmts, epsilon=EPSILON, loss_const=0.15)
        # ['piano_noise_mixed', 'simple_rnn', 'simple_rnn_1', 'piano_hat', 'noise_hat', 'piano_pred', 'noise_pred']    
        for i, layer in enumerate(model.layers):
            print('Layer', i, layer.name,  'weights:', layer.count_params())
            print('Layer\'s weights:', layer.get_weights())


if __name__ == '__main__':
    unittest.main()
