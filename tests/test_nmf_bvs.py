# tests.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Test suite for dsp functions.

# Testing on Mary.wav for general purposes

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
mary_filepath = os.getcwd() + '/brahms_restore_ml/nmf/Mary.wav'
test_path = os.getcwd() + '/brahms_restore_ml/nmf/output/output_test/output_test_dsp/'

brahms_clean_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreDLNN/HungarianDanceNo1Rec/(youtube)wav/BrahmsHungDance1_'
mary32_filepath = '/Users/quinnmc/Desktop/BMSThesis/MusicRestoreNMF/Mary_44100Hz_32bit.wav'

class DSPTests(unittest.TestCase):
   pass
  
if __name__ == '__main__':
    unittest.main()
