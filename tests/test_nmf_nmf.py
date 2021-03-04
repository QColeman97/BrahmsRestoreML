# test_nmf_nmf.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Unit tests for nmf functions.

from brahms_restore_ml.nmf.nmf import *
import unittest
import numpy as np
import os

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10

brahms_filepath = os.getcwd() + '/brahms.wav'
mary_441kHz_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary_44100Hz_32bitfp_librosa.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_nmf/'

Wmade_scale_factor = 5

class NMFTests(unittest.TestCase):

   # What to test?
   # Make sure fixed BV's aren't touched if fixed exist
   # How to make sure learned matrices are touched ie. forming good representations?
   # - just look at them after, and listen to result

   def test_nmf_unsupervised(self):
      m, n, k = 8, 6, 5
      V = np.arange(m * n).reshape(m, n)
      W, H = extended_nmf(V, k)
      W2, H2 = extended_nmf(V, k)
      # print('H SUMS Unsup:',np.sum(H2),np.sum(H)) # irrelevant
      with self.subTest():
         self.assertNotEqual(np.sum(W2), np.sum(W))
      with self.subTest():
         self.assertNotEqual(np.sum(H2), np.sum(H))
      # TODO? test if V sum = approx V sum

   def test_nmf_supervised(self):
      m, n, k = 8, 6, 5
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy())
      with self.subTest():
         np.testing.assert_array_equal(W, W_after_nmf)
      _, H2 = extended_nmf(V, k, W.copy())
      # Assert H sum is near same each time
      with self.subTest():
         self.assertAlmostEqual(np.sum(H2), np.sum(H), places=0)

   def test_nmf_semi_supervised_learn_noise_made(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      # Assert learned (whole) changed
      with self.subTest():                         
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      # Assert H corres. w/ fixed W sum is near same each time
      with self.subTest():
         _, H2 = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
         print('H SUMS lnm - Whole:',np.sum(H2),np.sum(H),'Corres Fixed:',np.sum(H2[split_index:]),np.sum(H[split_index:]),'Corres Learned',np.sum(H2[:split_index]),np.sum(H[:split_index]))
         self.assertAlmostEqual(np.sum(H2[split_index:]), np.sum(H[split_index:]), places=0)

   def test_nmf_semi_supervised_learn_noise(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      Wfixed = np.ones((m, k - split_index)) * Wmade_scale_factor
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wlearn, Wfixed), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      # Assert H corres. w/ fixed W sum is near same each time
      with self.subTest():
         _, H2 = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
         print('H SUMS ln - Whole:',np.sum(H2),np.sum(H),'Corres Fixed:',np.sum(H2[split_index:]),np.sum(H[split_index:]),'Corres Learned',np.sum(H2[:split_index]),np.sum(H[:split_index]))
         self.assertAlmostEqual(np.sum(H2[split_index:]), np.sum(H[split_index:]), places=0)

   def test_nmf_semi_supervised_learn_piano_made(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      # Assert H corres. w/ fixed W sum is near same each time
      with self.subTest():
         _, H2 = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
         print('H SUMS lpm - Whole:',np.sum(H2),np.sum(H),'Corres Fixed:',np.sum(H2[:split_index]),np.sum(H[:split_index]),'Corres Learned',np.sum(H2[split_index:]),np.sum(H[split_index:]))
         self.assertAlmostEqual(np.sum(H2[:split_index]), np.sum(H[:split_index]), places=0)

   def test_nmf_semi_supervised_learn_piano(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      Wfixed = np.ones((m, split_index)) * Wmade_scale_factor
      Wlearn = np.random.rand(m, k - split_index)
      W = np.concatenate((Wfixed, Wlearn), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      # Assert H corres. w/ fixed W sum is near same each time
      with self.subTest():
         _, H2 = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
         print('H SUMS lp - Whole:',np.sum(H2),np.sum(H),'Corres Fixed:',np.sum(H2[:split_index]),np.sum(H[:split_index]),'Corres Learned',np.sum(H2[split_index:]),np.sum(H[split_index:]))
         self.assertAlmostEqual(np.sum(H2[:split_index]), np.sum(H[:split_index]), places=0)

   # L1-Penalty - moral, don't apply L1-Pen on H that isn't corres. to fixed W (isn't supervised)
   # Make sure H sum < unpenalized H sum
   def test_nmf_supervised_l1penalty(self):
      m, n, k = 8, 6, 5
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy())
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W.copy(), l1_pen=1000)
      with self.subTest():
         np.testing.assert_array_equal(W, W_after_nmf)
      with self.subTest():
         np.testing.assert_array_equal(W, W_after_nmf_pen)
      with self.subTest():
         self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_noise_made_l1penalty(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise', l1_pen=1000)
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf_pen[:, split_index:])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      print('(PEN) H SUMS lnm - Whole:',np.sum(H_penalized),np.sum(H),'Corres Fixed:',np.sum(H_penalized[split_index:]),np.sum(H[split_index:]),'Corres Learned',np.sum(H_penalized[:split_index]),np.sum(H[:split_index]))
      # # No b/c H only near same when corres. W is fixed
      # # Assert that H of learned W practically same, b/c only penalize when corres. W is fixed
      # with self.subTest():
      #    self.assertAlmostEqual(np.sum(H_penalized[:split_index]), np.sum(H[:split_index]), places=0)
      with self.subTest():
         self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_noise_l1penalty(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      Wfixed = np.ones((m, k - split_index)) * Wmade_scale_factor
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wlearn, Wfixed), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Noise', l1_pen=1000)
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      with self.subTest():
         np.testing.assert_array_equal(W[:, split_index:], W_after_nmf_pen[:, split_index:])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      print('(PEN) H SUMS ln - Whole:',np.sum(H_penalized),np.sum(H),'Corres Fixed:',np.sum(H_penalized[split_index:]),np.sum(H[split_index:]),'Corres Learned',np.sum(H_penalized[:split_index]),np.sum(H[:split_index]))
      # # No b/c H only near same when corres. W is fixed
      # # Assert that H of learned W practically same, b/c only penalize when corres. W is fixed
      # with self.subTest():
      #    self.assertAlmostEqual(np.sum(H_penalized[:split_index]), np.sum(H[:split_index]), places=0)
      with self.subTest():
         self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_piano_made_l1penalty(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      W = np.ones((m, k)) * Wmade_scale_factor
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano', l1_pen=1000)
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf_pen[:, :split_index])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      print('(PEN) H SUMS lpm - Whole:',np.sum(H_penalized),np.sum(H),'Corres Fixed:',np.sum(H_penalized[:split_index]),np.sum(H[:split_index]),'Corres Learned',np.sum(H_penalized[split_index:]),np.sum(H[split_index:]))
      # # No b/c H only near same when corres. W is fixed
      # # Assert that H of learned W practically same, b/c only penalize when corres. W is fixed
      # with self.subTest():
      #    self.assertAlmostEqual(np.sum(H_penalized[split_index:]), np.sum(H[split_index:]), places=0)
      with self.subTest():
         self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_piano_l1penalty(self):
      m, n, k = 8, 6, 5
      split_index = k // 2
      Wfixed = np.ones((m, split_index)) * Wmade_scale_factor
      Wlearn = np.random.rand(m, k - split_index)
      W = np.concatenate((Wfixed, Wlearn), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W.copy(), split_index=split_index, sslrn='Piano', l1_pen=1000)
      # Assert fixed didn't change
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      with self.subTest():
         np.testing.assert_array_equal(W[:, :split_index], W_after_nmf_pen[:, :split_index])
      # Assert learned (whole) changed
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      with self.subTest():
         np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      print('(PEN) H SUMS lp - Whole:',np.sum(H_penalized),np.sum(H),'Corres Fixed:',np.sum(H_penalized[:split_index]),np.sum(H[:split_index]),'Corres Learned',np.sum(H_penalized[split_index:]),np.sum(H[split_index:]))
      # # No b/c H only near same when corres. W is fixed
      # # Assert that H of learned W practically same, b/c only penalize when corres. W is fixed
      # with self.subTest():
      #    self.assertAlmostEqual(np.sum(H_penalized[split_index:]), np.sum(H[split_index:]), places=0)
      with self.subTest():
         self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_source_split_matrices(self):
      m, n, k, split_index = 8, 6, 5, 2
      W, H = np.random.rand(m,k), np.random.rand(k,n)
      H1, W1, H2, W2 = source_split_matrices(H, W, split_index)
      with self.subTest():
         self.assertEqual(H1.shape, (split_index, n)) 
      with self.subTest():
         self.assertEqual(W1.shape, (m, split_index)) 
      with self.subTest():
         self.assertEqual(H2.shape, (k - split_index, n)) 
      with self.subTest():
         self.assertEqual(W2.shape, (m, k - split_index)) 

   def test_perfect_product(self):
      mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
      # print('Actual Mary Sig:', mary_sig[50:100], '\nType:', mary_sig.dtype)
      mary_type = mary_sig.dtype
      mary_spgm, mary_phases = make_spectrogram(mary_sig, PIANO_WDW_SIZE, EPSILON, ova=True)
      # print('Real V:', mary_spgm.shape, np.sum(mary_spgm))
      maryW = get_basis_vectors(PIANO_WDW_SIZE, ova=True, mary=True, avg=True).T
      maryH = make_mary_bv_test_activations()
      _, mary_tsteps = maryH.shape
      mary_synth_phases = mary_phases[:mary_tsteps]
      maryV = maryW @ maryH
      # print('W:', maryW.shape, np.sum(maryW), 'H:', maryH.shape, np.sum(maryH))
      # plot_matrix(maryW, 'mary w', 'k', 'frequency', show=True)
      # plot_matrix(maryH, 'mary h', 'timesteps', 'k', show=True)
      # plot_matrix(maryV, 'mary v', 'timesteps', 'frequency', show=True)
      maryV = maryV.T   # orient to natural way
      # print('V before to sig:', maryV.shape, np.sum(maryV), 'phases:', mary_synth_phases.shape)
      mary_synth_sig = make_synthetic_signal(maryV, mary_synth_phases, PIANO_WDW_SIZE, mary_type, ova=True, debug=True) 
      # print('Synth Mary Sig:', mary_synth_sig[50:100], np.sum(mary_synth_sig))
      wavfile.write(test_path + 'synthetic_mary.wav', mary_sr, mary_synth_sig)


if __name__ == '__main__':
    unittest.main()
