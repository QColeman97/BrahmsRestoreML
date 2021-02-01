# test_drnn_data.py - Quinn Coleman - Senior Research Project / Master's Thesis
# Tests for drnn data preprocessing functions.

# Run with $ python -m unittest tests.test_drnn_data


from brahms_restore_ml.drnn.data import *
import unittest
import numpy as np

# Testing global vars
write_flag = True
debug_flag = False

class DRNNDataTests(unittest.TestCase):
    
    def test_generator(self):
        gen = preprocess_data_generator()
        yielded = next(gen)
        
  
if __name__ == '__main__':
    unittest.main()
