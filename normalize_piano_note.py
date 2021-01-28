from scipy.io import wavfile
import numpy as np
from sklearn.preprocessing import normalize
from brahms_restore_ml.audio_data_processing import *

sr, stereo_sig = wavfile.read('brahms_restore_ml/nmf/all_notes_ff_wav/Piano.ff.A4.wav')
sig_type = stereo_sig.dtype
sig = np.average(stereo_sig, axis=-1)

amp_thresh = max(sig) * 0.01
while sig[0] < amp_thresh:
    sig = sig[1:]
while sig[-1] < amp_thresh:
    sig = sig[:-1]

print(sig[400:410])

spgm, phases = make_spectrogram(sig, PIANO_WDW_SIZE, EPSILON, ova=True, return_f64=True)
print(spgm.dtype)
basis_vector = np.mean(spgm, axis=0)
print(basis_vector.shape)
print(np.sum(basis_vector))
# norm_spgm = normalize(spgm)
norm_bv = basis_vector / np.linalg.norm(basis_vector)
# norm_bv = normalize(basis_vector[:,np.newaxis], axis=0).ravel()
norm_bv *= 1000000
print(np.sum(norm_bv))

# norm_spgm = np.array([basis_vector,]*spgm.shape[0])
norm_spgm = np.array([norm_bv,]*spgm.shape[0])
print('Norm shape:', norm_spgm.shape)
# norm_spgm = np.array([norm_bv for _ in range(spgm.shape[1])]).T
synthetic_sig = make_synthetic_signal(norm_spgm, phases, PIANO_WDW_SIZE, sig_type, ova=True)
# print(synthetic_sig.dtype)
print(synthetic_sig[400:410])

wavfile.write('A4_normalized_bv.wav', sr, synthetic_sig)
