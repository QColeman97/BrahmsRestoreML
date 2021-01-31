# test_nmf_bvs.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for basis vector functions.

from brahms_restore_ml.nmf.nmf import NUM_PIANO_NOTES
from brahms_restore_ml.audio_data_processing import *
from brahms_restore_ml.nmf.basis_vectors import *
import unittest
from scipy.io import wavfile
import numpy as np
import os

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10

piano_note_filepath = os.getcwd() + '/brahms_restore_ml/nmf/all_notes_ff_wav/Piano.ff.A4.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_bv/'
# 'brahms_restore_ml/nmf/avged_ova_notes/avged_ova_note_' + str(num) + '.wav'
# 'brahms_restore_ml/nmf/trimmed_notes/trimmed_note_' + str(num) + '.wav'
saved_bvs_filepath = os.getcwd() + '/brahms_restore_ml/nmf/np_saves_bv/basis_vectors_ova_avg.npy'

class BasisVectorTests(unittest.TestCase):

   def test_make_basis_vector(self):
      note_sr, note_sig = wavfile.read(piano_note_filepath)
      bv = make_basis_vector(note_sig, note_sig.dtype, note_sr, 0, PIANO_WDW_SIZE, ova=True, avg=True, write_path=test_path) 
      self.assertEqual(bv.shape, ((PIANO_WDW_SIZE//2)+1,))

   def test_make_noise_basis_vectors(self):
      num_noise_bv = 12
      noise_basis_vectors = make_noise_basis_vectors(num_noise_bv, PIANO_WDW_SIZE, ova=True)
      self.assertEqual(noise_basis_vectors.shape, (num_noise_bv, (PIANO_WDW_SIZE//2)+1))
   
   # Success - takes long time
   # def test_make_basis_vectors(self):
   #    bvs = make_basis_vectors(PIANO_WDW_SIZE, saved_bvs_filepath, ova=True, avg=True)
   #    self.assertEqual(bvs.shape, (NUM_PIANO_NOTES, (PIANO_WDW_SIZE//2)+1))

   def test_get_basis_vectors(self):
      bvs = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
         num_noise=num_noise_bv_test)
      self.assertEqual(bvs.shape, (NUM_PIANO_NOTES+num_noise_bv_test, (PIANO_WDW_SIZE//2)+1))
   
   def test_get_basis_vectors_bare(self):
      bvs = get_basis_vectors(PIANO_WDW_SIZE, ova=True, avg=True)
      self.assertEqual(bvs.shape, (NUM_PIANO_NOTES, (PIANO_WDW_SIZE//2)+1))

  
if __name__ == '__main__':
    unittest.main()