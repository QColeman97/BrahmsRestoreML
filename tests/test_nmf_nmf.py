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

class NMFTests(unittest.TestCase):

   # What to test?
   # Make sure fixed BV's aren't touched if fixed exist
   # How to make sure learned matrices are touched ie. forming good representations?
   # - just look at them after, and listen to result

   # TODO After tests, the key to doing NMF correctly, is knowing how memory is being used in Numpy

   def test_nmf_supervised(self):
      m, n, k = 8, 6, 4
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W)
      np.testing.assert_array_equal(W, W_after_nmf)

   # Assert equal tests

   def test_nmf_semi_supervised_learn_noise_made(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise')
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])

   # Assert unequal tests

   def test_nmf_semi_supervised_learn_noise_made2(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise')
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)

   def test_nmf_semi_supervised_learn_noise(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      Wfixed = np.ones((m, split_index))
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wlearn, Wfixed), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise')
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)

   def test_nmf_semi_supervised_learn_piano_made(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano')
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)

   def test_nmf_semi_supervised_learn_piano(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      Wfixed = np.ones((m, split_index))
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wfixed, Wlearn), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano')
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)

   # Make sure H sum < unpenalized H sum
   def test_nmf_supervised_l1penalty(self):
      m, n, k = 8, 6, 4
      W = np.random.rand(m, k)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W)
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W, l1_pen=1000)
      np.testing.assert_array_equal(W, W_after_nmf)
      np.testing.assert_array_equal(W, W_after_nmf_pen)
      self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_noise_made_l1penalty(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise', l1_pen=1000)
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf_pen[:, split_index:])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      # Assert that H of learned W practically same, b/c only penalize when corr W is fixed
      self.assertAlmostEqual(np.sum(H_penalized[:split_index]), np.sum(H[:split_index]))
      self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_noise_l1penalty(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      Wfixed = np.ones((m, split_index))
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wlearn, Wfixed), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W, split_index=split_index, sslrn='Noise', l1_pen=1000)
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf[:, split_index:])
      np.testing.assert_array_equal(W[:, split_index:], W_after_nmf_pen[:, split_index:])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      # Assert that H of learned W practically same, b/c only penalize when corr W is fixed
      self.assertAlmostEqual(np.sum(H_penalized[:split_index]), np.sum(H[:split_index]))
      self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_piano_made_l1penalty(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      W = np.ones((m, k))
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano', l1_pen=1000)
     # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf_pen[:, :split_index])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      # Assert that H of learned W practically same, b/c only penalize when corr W is fixed
      self.assertAlmostEqual(np.sum(H_penalized[split_index:]), np.sum(H[split_index:]))
      self.assertLess(np.sum(H_penalized), np.sum(H))

   def test_nmf_semi_supervised_learn_piano_l1penalty(self):
      m, n, k = 8, 6, 4
      split_index = k // 2
      Wfixed = np.ones((m, split_index))
      Wlearn = np.random.rand(m, split_index)
      W = np.concatenate((Wfixed, Wlearn), axis=-1)
      V = np.arange(m * n).reshape(m, n)
      W_after_nmf, H = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano')
      W_after_nmf_pen, H_penalized = extended_nmf(V, k, W, split_index=split_index, sslrn='Piano', l1_pen=1000)
      # Assert fixed didn't change
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf[:, :split_index])
      np.testing.assert_array_equal(W[:, :split_index], W_after_nmf_pen[:, :split_index])
      # Assert learned (whole) changed
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf)
      np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, W, W_after_nmf_pen)
      # Assert that H of learned W practically same, b/c only penalize when corr W is fixed
      self.assertAlmostEqual(np.sum(H_penalized[split_index:]), np.sum(H[split_index:]))
      self.assertLess(np.sum(H_penalized), np.sum(H))

   # Failed case for mary_activations - not crucial
   # def test_perfect_product(self):
   #    mary_sr, mary_sig = wavfile.read(mary_441kHz_filepath)
   #    print('Actual Mary Sig:', mary_sig[50:100])
   #    mary_type = mary_sig.dtype
   #    mary_spgm, mary_phases = make_spectrogram(mary_sig, PIANO_WDW_SIZE, EPSILON, ova=True)
   #    print('Real V:', mary_spgm.shape, np.sum(mary_spgm))
   #    maryW = get_basis_vectors(PIANO_WDW_SIZE, ova=True, mary=True, avg=True).T
   #    maryH = make_mary_bv_test_activations()
   #    _, mary_tsteps = maryH.shape
   #    mary_synth_phases = mary_phases[:mary_tsteps]
   #    maryV = maryW @ maryH
   #    print('W:', maryW.shape, np.sum(maryW), 'H:', maryH.shape, np.sum(maryH))
   #    plot_matrix(maryW, 'mary w', 'k', 'frequency', show=True)
   #    plot_matrix(maryH, 'mary h', 'timesteps', 'k', show=True)
   #    plot_matrix(maryV, 'mary v', 'timesteps', 'frequency', show=True)
   #    maryV = maryV.T   # orient to natural way
   #    print('V before to sig:', maryV.shape, np.sum(maryV), 'phases:', mary_synth_phases.shape)
   #    mary_synth_sig = make_synthetic_signal(maryV, mary_synth_phases, PIANO_WDW_SIZE, mary_type, ova=True) 
   #    print('Synth Mary Sig:', mary_synth_sig[50:100], np.sum(mary_synth_sig))
   #    wavfile.write(test_path + 'synthetic_mary.wav', mary_sr, mary_synth_sig)

if __name__ == '__main__':
    unittest.main()
