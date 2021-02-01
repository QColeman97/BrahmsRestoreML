from ..audio_data_processing import *
import random
import re

# May try replacing with lambda in the call in train
def make_gen_callable(_gen):
    def gen():
        for x,y in _gen:
            yield x,y
    return gen

def preprocess_signals(piano_sig, noise_sig, pad_len, mix_sig=None, src_amp_low=0.75, src_amp_high=1.15):
    if mix_sig is not None:
        assert len(mix_sig) == len(piano_sig) == len(noise_sig)
        mix_sig = mix_sig.astype('float64')
        # Stereo audio safety check
        if isinstance(mix_sig[0], np.ndarray):   # Stereo signal = 2 channels
            mix_sig = np.average(mix_sig, axis=-1)

    assert len(piano_sig) == len(noise_sig)
    piano_sig, noise_sig = piano_sig.astype('float64'), noise_sig.astype('float64')
    piano_sig, noise_sig = piano_sig.astype('float64'), noise_sig.astype('float64')
    # Stereo audio safety check
    if isinstance(piano_sig[0], np.ndarray):  # Stereo signal = 2 channels
        piano_sig = np.average(piano_sig, axis=-1)
    if isinstance(noise_sig[0], np.ndarray):  # Stereo signal = 2 channels
        noise_sig = np.average(noise_sig, axis=-1)

    if mix_sig is None:
        # Mix & vary SNR
        src_percent_1 = random.randrange(int(src_amp_low*100), int(src_amp_high*100)) / 100
        src_percent_2 = 1 / src_percent_1
        # print('src 1 percent:', src_percent_1, 'src 2 percent:', src_percent_2)
        piano_src_is_1 = bool(random.getrandbits(1))
        if piano_src_is_1:
            piano_sig *= src_percent_1
            noise_sig *= src_percent_2
        else:
            piano_sig *= src_percent_2
            noise_sig *= src_percent_1
        avg_src_sum = (np.sum(np.abs(piano_sig)) + np.sum(np.abs(noise_sig))) / 2
        mix_sig = piano_sig + noise_sig
        # Key - mixed signal should be on amplitude level of its sources
        # print('Mix sig abs sum:', np.sum(np.abs(mix_sig)), 'avg src sum:', avg_src_sum)
        mix_srcs_ratio = (avg_src_sum / np.sum(np.abs(mix_sig)))
        mix_sig *= mix_srcs_ratio
    # Pad
    deficit = pad_len - len(mix_sig)
    mix_sig = np.pad(mix_sig, (0,deficit))
    piano_sig = np.pad(piano_sig, (0,deficit))
    noise_sig = np.pad(noise_sig, (0,deficit))
    return mix_sig, piano_sig, noise_sig

def signal_to_nn_features(signal, wdw_size=PIANO_WDW_SIZE, epsilon=EPSILON):
    spgm, _ = make_spectrogram(signal, wdw_size, epsilon, ova=True, debug=False)
    # Float 32 for neural nets
    spgm = np.clip(spgm, np.finfo('float32').min, np.finfo('float32').max)
    spgm = spgm.astype('float32')
    return spgm

# Generator for NN - all audio data is too large from RAM
# Rule - If from_numpy True, x_files cant be None
# Rule - If from_numpy False, dmged_piano_art_noise must be considered for npy writes
def nn_data_generator(y1_files, y2_files, num_samples, batch_size, num_seq, num_feat,
                        pad_len, dmged_piano_artificial_noise=False,
                        src_amp_low=0.75, src_amp_high=1.15, 
                        data_path=None, x_files=None, from_numpy=False):
    while True:
        for offset in range(0, num_samples, batch_size):
            if (x_files is not None) or from_numpy:
                x_batch_labels = x_files[offset:offset+batch_size]
            y1_batch_labels = y1_files[offset:offset+batch_size]
            y2_batch_labels = y2_files[offset:offset+batch_size]
            if (num_samples / batch_size == 0):
                actual_batch_size = batch_size
                x, y1, y2 = (np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'))
            else:
                actual_batch_size = len(y1_batch_labels)
                x, y1, y2 = (np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'))
            
            for i in range(actual_batch_size):
                if from_numpy:
                    mix_filepath = x_batch_labels[i]
                    pl_filepath = y1_batch_labels[i]
                    nl_filepath = y2_batch_labels[i]
                    mix_spgm = np.load(mix_filepath, allow_pickle=True)
                    piano_spgm = np.load(pl_filepath, allow_pickle=True)
                    noise_spgm = np.load(nl_filepath, allow_pickle=True)
                else:
                    pl_filepath = y1_batch_labels[i]
                    nl_filepath = y2_batch_labels[i]
                    _, piano_label_sig = wavfile.read(pl_filepath)
                    _, noise_label_sig = wavfile.read(nl_filepath)
                    if x_files is not None:
                        mix_filepath = x_batch_labels[i]
                        _, mix_sig = wavfile.read(mix_filepath)
                        mix_sig, piano_sig, noise_sig = preprocess_signals(
                            piano_label_sig, noise_label_sig, mix_sig=mix_sig, pad_len=pad_len, 
                            src_amp_low=src_amp_low, src_amp_high=src_amp_high)
                    else:
                        mix_sig, piano_sig, noise_sig = preprocess_signals(
                            piano_label_sig, noise_label_sig, pad_len=pad_len, 
                            src_amp_low=src_amp_low, src_amp_high=src_amp_high)
                    mix_spgm = signal_to_nn_features(mix_sig)
                    piano_spgm = signal_to_nn_features(piano_sig)
                    noise_spgm = signal_to_nn_features(noise_sig)
                    # Get number from filename
                    file_num_str = list(re.findall(r'\d+', pl_filepath))[-1]
                    # Write to file for from numpy fixed data gen
                    piano_suffix = 'piano_source_numpy/piano'
                    if dmged_piano_artificial_noise:
                        mix_suffix = 'dmged_mix_numpy/mixed'
                        noise_suffix = 'dmged_noise_numpy/noise'
                    else:
                        mix_suffix = 'piano_noise_numpy/mixed'
                        noise_suffix = 'noise_source_numpy/noise'
                    np.save(data_path + mix_suffix + file_num_str, mix_spgm)
                    np.save(data_path + piano_suffix + file_num_str, piano_spgm)
                    np.save(data_path + noise_suffix + file_num_str, noise_spgm)

                x[i] = mix_spgm
                y1[i] = piano_spgm
                y2[i] = noise_spgm

            yield ([x, np.concatenate((y1, y2), axis=-1)])

# NN DATA STATS FUNC - Only used when dataset changes
# Rule - If from_numpy True, x_files cant be None
# Rule - If from_numpy False, dataset2 must be considered for npy writes
def get_data_stats(y1_filenames, y2_filenames, num_samples, train_seq, train_feat, 
            #   wdw_size, epsilon, 
              pad_len, src_amp_low=0.75, src_amp_high=1.15, dataset2=False,
              data_path=None, x_filenames=None, from_numpy=False):

    generator = nn_data_generator(y1_filenames, y2_filenames, num_samples,
            batch_size=1, num_seq=train_seq, num_feat=train_feat,
            pad_len=pad_len, dmged_piano_artificial_noise=dataset2, 
            src_amp_low=src_amp_low, src_amp_high=src_amp_high,
            data_path=data_path, x_files=x_filenames, from_numpy=from_numpy)

    samples = np.empty((num_samples, train_seq, train_feat))
    # piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # noise_samples = np.empty((num_samples, train_seq, train_feat))
    # aug_piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # aug_noise_samples = np.empty((num_samples, train_seq, train_feat))
    # New - iterate thru generator to get stats
    for i in range(num_samples):
        mix_spgm = next(generator)[0]
        samples[i] = mix_spgm

    #     # # VISUAL TEST
    #     # # print('PIANO SPGM #', i)
    #     # # for i in range(train_seq):
    #     # #     print(piano_spgm[i])
    #     # # print('DONE')

    #     # piano_samples[i] = piano_spgm
    #     # # noise_samples[i] = noise_spgm

    #     # piano_amp_factor = random.uniform(src_amp_low, src_amp_high)
    #     # noise_amp_factor = random.uniform(src_amp_low, src_amp_high)
    #     # piano_label_sig *= piano_amp_factor
    #     # noise_label_sig *= noise_amp_factor

    #     # aug_piano_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon, 
    #     #                                         ova=True, debug=False)
    #     # # aug_noise_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
    #     #                                         # ova=True, debug=False)
    #     # aug_piano_samples[i] = aug_piano_spgm
    #     # # aug_noise_samples[i] = aug_noise_spgm

    # # print('A different test: the average sum of piano and noise labels')
    # # # print('Avg sum of piano sources:', np.mean(np.sum(piano_samples, axis=-1)))
    # # print('Avg sum of noise sources:', np.mean(np.sum(noise_samples, axis=-1)))
    # # # print('Avg sum of aug piano sources:', np.mean(np.sum(aug_piano_samples, axis=-1)))
    # # print('Avg sum of aug noise sources:', np.mean(np.sum(aug_noise_samples, axis=-1)))
    # print('A different test: the average val of piano and noise labels')
    # print('Avg of piano sources:', np.mean(piano_samples))
    # # print('Avg of noise sources:', np.mean(noise_samples))
    # print('Avg of aug piano sources:', np.mean(aug_piano_samples))
    # # print('Avg of aug noise sources:', np.mean(aug_noise_samples))
    
    return np.mean(samples), np.std(samples)