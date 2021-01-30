# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for dsp functions.

# Testing on Mary.wav for general purposes

from brahms_restore_ml.nmf.nmf import NUM_PIANO_NOTES
from brahms_restore_ml.audio_data_processing import *
from brahms_restore_ml.nmf.basis_vectors import *
import unittest
import soundfile
from scipy import stats
from scipy.io import wavfile
import numpy as np
import os
# from sklearn.metrics import mean_absolute_error

# Testing global vars
write_flag = True
debug_flag = False
num_noise_bv_test = 10

brahms_filepath = os.getcwd() + '/brahms.wav'
piano_note_filepath = os.getcwd() + '/brahms_restore_ml/nmf/all_notes_ff_wav/Piano.ff.A4.wav'
mary_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_dsp/'

brahms_clean_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreDLNN/HungarianDanceNo1Rec/(youtube)wav/BrahmsHungDance1_'
mary32_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/Mary_44100Hz_32bit.wav'

bvs_test_filepath = os.getcwd() + '/brahms_restore_ml/nmf/np_saves_bv/test.npy'

class NoiseBasisVectorTests(unittest.TestCase):

   def test_make_basis_vector(self):
      note_sr, note_sig = wavfile.read(piano_note_filepath)
      bv = make_basis_vector(note_sig, note_sig.dtype, note_sr, 0, PIANO_WDW_SIZE, ova=True, avg=True) 
      self.assertEqual(bv.shape, ((PIANO_WDW_SIZE//2)+1,))

   def test_make_noise_basis_vectors(self):
      num_noise_bv = 12
      noise_basis_vectors = make_noise_basis_vectors(num_noise_bv, PIANO_WDW_SIZE, ova=True)
      self.assertEqual(noise_basis_vectors.shape, (num_noise_bv, (PIANO_WDW_SIZE//2)+1))

   def test_make_basis_vectors(self):
      bvs = make_basis_vectors(PIANO_WDW_SIZE, bvs_test_filepath, ova=True, avg=True)
      self.assertEqual(bvs.shape, (NUM_PIANO_NOTES, (PIANO_WDW_SIZE//2)+1))

   def test_get_basis_vectors(self):
      bvs = get_basis_vectors(PIANO_WDW_SIZE, ova=True, noise=True, avg=True, 
         num_noise=num_noise_bv_test)
      self.assertEqual(bvs.shape, (NUM_PIANO_NOTES+num_noise_bv_test, (PIANO_WDW_SIZE//2)+1))
   
   def test_get_basis_vectors_bare(self):
      bvs = get_basis_vectors(PIANO_WDW_SIZE, ova=True, avg=True)
      self.assertEqual(bvs.shape, (NUM_PIANO_NOTES, (PIANO_WDW_SIZE//2)+1))

  
if __name__ == '__main__':
    unittest.main()
