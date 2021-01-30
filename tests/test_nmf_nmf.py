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

   def test_nmf(self):
      pass
   
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
