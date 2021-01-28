from brahms_restore_ml.audio_data_processing import *
from scipy.io import wavfile
import librosa 

# src_sig, src_sr = librosa.load('original_speech1.wav', sr=44100)
# src_sig, src_sr = librosa.load('original_speech1.wav', sr=16000)
src_sr, src_sig = wavfile.read('../Benchmark Systems/po-sen/deeplearningsourceseparation-master/codes/denoising/demo/wav/original_speech1.wav')

# src_sig_type = src_sig.dtype
src_wdw_size = 1024
src_hop_size_divisor = 4
# Unneeded change
# src_wdw_size = 4096
# src_hop_size_divisor = 2
# src_hop_size = src_wdw_size // src_hop_size_divisor

brahms_sig, brahms_sr = librosa.load('brahms.wav', sr=16000)
# brahms_sr, brahms_sig = wavfile.read('brahms.wav')
brahms_sig_type = brahms_sig.dtype

# Fix, just trim the waveform

src_spgm, src_phases = make_spectrogram(src_sig, src_wdw_size, EPSILON, ova=True, debug=False)#, hop_size_divisor=src_hop_size_divisor)
# brahms_spgm, brahms_phases = make_spectrogram(brahms_sig, src_wdw_size, EPSILON, ova=True, debug=False, hop_size_divisor=src_hop_size_divisor)

src_timesteps = src_spgm.shape[0]
print(src_spgm.shape, src_phases.shape, 'src timesteps:', src_timesteps)

# # brahms_new_spgm, brahms_new_phases = brahms_spgm[:src_timesteps].copy(), brahms_phases[:src_timesteps].copy()
# # print(brahms_new_spgm.shape, brahms_new_phases.shape, 'src timesteps:', src_timesteps)
# brahms_new_spgm, brahms_new_phases = brahms_spgm, brahms_phases

# synthetic_src_sig = make_synthetic_signal(src_spgm, src_phases, src_wdw_size, 
#                                         src_sig_type, ova=True, debug=False, 
#                                         hop_size_divisor=src_hop_size_divisor)
# wavfile.write('original_speech1_new.wav', src_sr, synthetic_src_sig)

# brahms_new_sig = make_synthetic_signal(brahms_new_spgm, brahms_new_phases, src_wdw_size, 
#                                         brahms_sig_type, ova=True, debug=False, 
#                                         hop_size_divisor=src_hop_size_divisor)

print(len(src_sig))

brahms_new_sig = brahms_sig[300000: (300000 + len(src_sig))]
print(len(brahms_new_sig))

# wavfile.write('brahms_1.wav', brahms_sr, brahms_new_sig)
# Works
# librosa.output.write_wav('brahms_1.wav', brahms_new_sig, brahms_sr)
librosa.output.write_wav('brahms_16kHz.wav', brahms_sig, brahms_sr)


