from brahms_restore_ml.audio_data_processing import *
from scipy.io import wavfile

def shrink_tsteps(in_path, out_path, shrink_to=100):
    out_i = 0
    for i in range(61):
        sr, sig = wavfile.read(in_path + 'psource' + str(i) + '.wav')
        sig_type = sig.dtype
        spgm, phases = make_spectrogram(sig, PIANO_WDW_SIZE, EPSILON, ova=True)
        t_step_index = 0
        while t_step_index < spgm.shape[0]:
            small_spgm = spgm[t_step_index: t_step_index + shrink_to]
            small_phases = phases[t_step_index: t_step_index + shrink_to]
            # Pad if necessary
            if small_spgm.shape[0] < shrink_to:
                deficit = shrink_to - small_spgm.shape[0]
                small_spgm = np.concatenate((small_spgm, np.zeros((deficit, small_spgm.shape[1]))))
                small_phases = np.concatenate((small_phases, np.zeros((deficit, small_phases.shape[1]))))
                small_spgm[small_spgm == 0], small_phases[small_phases == 0] = EPSILON, EPSILON
            # print('SMALL shape:', small_spgm.shape, small_phases.shape)
            out_name = out_path + 'psource' + str(out_i) + '.wav'
            synthetic_sig = make_synthetic_signal(small_spgm, small_phases, PIANO_WDW_SIZE, 
                                                  sig_type, ova=True)
            wavfile.write(out_name, sr, synthetic_sig)
            out_i += 1
            t_step_index += shrink_to

data_path = '../synthesize_train_data/all_train_data/'
in_path = data_path + 'dmged_piano_data/'
out_path = data_path + 'small_dmged_piano_data/'
shrink_tsteps(in_path, out_path)
