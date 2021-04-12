# test_nmf_bvs.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for basis vector functions.

# Run with $ python -m unittest tests.test_bvs

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
saved_bvs_filepath = os.getcwd() + '/brahms_restore_ml/nmf/np_saves_bv/basis_vectors_ova_avg.npy'

class BasisVectorTests(unittest.TestCase):

   def test_make_basis_vector(self):
      note_sr, note_sig = wavfile.read(piano_note_filepath)
      bv = make_basis_vector(note_sig, note_sig.dtype, note_sr, str(0), PIANO_WDW_SIZE, ova=True, avg=True, write_path=test_path) 
      self.assertEqual(bv.shape, ((PIANO_WDW_SIZE//2)+1,))

   def test_make_noise_basis_vectors(self):
      num_noise_bv = 12
      noise_basis_vectors = make_noise_basis_vectors(num_noise_bv, PIANO_WDW_SIZE, ova=True)
      self.assertEqual(noise_basis_vectors.shape, (num_noise_bv, (PIANO_WDW_SIZE//2)+1))

   def test_bv_sound(self):
      num_noise = 1
      noise_bvs = make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag,
                                           precise_noise=True, 
                                          #  start=6, stop=83,   # tests voice, default 0 - 20
                                           write_path=test_path)
   
   def test_bv_sound_izotoperx(self):
      num_noise = 1
      noise_bvs = make_noise_basis_vectors(num_noise, PIANO_WDW_SIZE, ova=True, debug=debug_flag,
                                           precise_noise=False, write_path=test_path)
   
   # Success - but takes long time
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
