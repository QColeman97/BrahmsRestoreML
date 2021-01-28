from ..audio_data_processing import *
import random
import re


def make_gen_callable(_gen):
    def gen():
        for x,y in _gen:
            yield x,y
    return gen

# Preprocesses data, and writes it for fixed_data_generator
def preprocess_data_generator(y1_files, y2_files, num_samples, batch_size, num_seq, num_feat,
                              pc_run, dmged_piano_artificial_noise=False, pad_len=-1, wdw_size=4096, 
                              epsilon=10 ** (-10), src_amp_low=0.75, src_amp_high=1.15, 
                              data_path=None):
    while True:
        for offset in range(0, num_samples, batch_size):
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
                pl_filepath = y1_batch_labels[i]
                nl_filepath = y2_batch_labels[i]
                _, piano_label_sig = wavfile.read(pl_filepath)
                _, noise_label_sig = wavfile.read(nl_filepath)
                piano_label_sig, noise_label_sig = piano_label_sig.astype('float64'), noise_label_sig.astype('float64')
                assert len(noise_label_sig) == len(piano_label_sig)   
                # Stereo audio safety check
                if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                    piano_label_sig = np.average(piano_label_sig, axis=-1)
                if isinstance(noise_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                    noise_label_sig = np.average(noise_label_sig, axis=-1)
                # Mix & vary SNR
                avg_src_sum = (np.sum(piano_label_sig) + np.sum(noise_label_sig)) / 2
                src_percent_1 = random.randrange(int((src_amp_low*100) // 2), int((src_amp_high*100) // 2)) / 100
                src_percent_2 = 1 - src_percent_1
                piano_src_is_1 = bool(random.getrandbits(1))
                if piano_src_is_1:
                    piano_label_sig *= src_percent_1
                    noise_label_sig *= src_percent_2
                else:
                    piano_label_sig *= src_percent_2
                    noise_label_sig *= src_percent_1

                noise_piano_sig = piano_label_sig + noise_label_sig
                noise_piano_sig *= (avg_src_sum / np.sum(noise_piano_sig)) 
                # Pad
                deficit = pad_len - len(noise_piano_sig)
                noise_piano_sig = np.pad(noise_piano_sig, (0,deficit))
                piano_label_sig = np.pad(piano_label_sig, (0,deficit))
                noise_label_sig = np.pad(noise_label_sig, (0,deficit))

                noise_piano_spgm, _ = make_spectrogram(noise_piano_sig, wdw_size, epsilon, 
                                                ova=True, debug=False)
                piano_label_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon,
                                                ova=True, debug=False)
                noise_label_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
                                                ova=True, debug=False)
                # Float 32 for neural nets
                noise_piano_spgm = np.clip(noise_piano_spgm, np.finfo('float32').min, np.finfo('float32').max)
                noise_piano_spgm = noise_piano_spgm.astype('float32')
                piano_label_spgm = np.clip(piano_label_spgm, np.finfo('float32').min, np.finfo('float32').max)
                piano_label_spgm = piano_label_spgm.astype('float32')
                noise_label_spgm = np.clip(noise_label_spgm, np.finfo('float32').min, np.finfo('float32').max)
                noise_label_spgm = noise_label_spgm.astype('float32')
                
                # Get number from filename
                file_num_str = list(re.findall(r'\d+', pl_filepath))[-1]
                # Write to file for fixed data gen
                np.save(data_path + 'piano_noise_numpy/mixed' + file_num_str, noise_piano_spgm)
                np.save(data_path + 'piano_label_numpy/piano' + file_num_str, piano_label_spgm)
                np.save(data_path + 'noise_label_numpy/noise' + file_num_str, noise_label_spgm)
                x[i] = noise_piano_spgm
                y1[i] = piano_label_spgm
                y2[i] = noise_label_spgm

            yield ([x, np.concatenate((y1, y2), axis=-1)])


# In order for this generator (unique output) to work w/ fit & cust training - batch it
def fixed_data_generator(x_files, y1_files, y2_files, num_samples, batch_size, num_seq, num_feat, pc_run, 
                         dmged_piano_artificial_noise=False):
    while True: # Loop forever so the generator never terminates
        # for i in range(num_samples):
        for offset in range(0, num_samples, batch_size):
            x_batch_labels = x_files[offset:offset+batch_size]
            y1_batch_labels = y1_files[offset:offset+batch_size]
            y2_batch_labels = y2_files[offset:offset+batch_size]
            if (num_samples / batch_size == 0):
                # TEST FLOAT16
                # MIXED PRECISION - hail mary try
                actual_batch_size = batch_size
                x, y1, y2 = (np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((batch_size, num_seq, num_feat)).astype('float32'))
            else:
                actual_batch_size = len(x_batch_labels)
                x, y1, y2 = (np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'),
                            np.empty((actual_batch_size, num_seq, num_feat)).astype('float32'))
            
            for i in range(actual_batch_size):
                pn_filepath = x_batch_labels[i]
                pl_filepath = y1_batch_labels[i]
                nl_filepath = y2_batch_labels[i]
           
                # MIXED PRECISION - hail mary try
                noise_piano_spgm = np.load(pn_filepath, allow_pickle=True)# .astype('float32')
                piano_label_spgm = np.load(pl_filepath, allow_pickle=True)# .astype('float32')
                noise_label_spgm = np.load(nl_filepath, allow_pickle=True)# .astype('float32')

                x[i] = noise_piano_spgm
                y1[i] = piano_label_spgm
                y2[i] = noise_label_spgm

            yield ([x, np.concatenate((y1, y2), axis=-1)])


# NN DATA STATS FUNC - Only used when dataset changes
def get_stats(y1_filenames, y2_filenames, num_samples, train_seq, train_feat, 
              wdw_size, epsilon, pad_len, src_amp_low=0.75, src_amp_high=1.15, 
              x_filenames=None):
    
    samples = np.empty((num_samples, train_seq, train_feat))
    # piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # noise_samples = np.empty((num_samples, train_seq, train_feat))
    # aug_piano_samples = np.empty((num_samples, train_seq, train_feat))
    # # aug_noise_samples = np.empty((num_samples, train_seq, train_feat))
    for i in range(num_samples):
        _, piano_label_sig = wavfile.read(y1_filenames[i])
        _, noise_label_sig = wavfile.read(y2_filenames[i])

        if x_filenames is not None:
            _, noise_piano_sig = wavfile.read(x_filenames[i])
            noise_piano_sig = noise_piano_sig.astype('float64')
            assert len(noise_piano_sig) == len(piano_label_sig) 
            assert len(noise_piano_sig) == len(noise_label_sig) 

            if isinstance(noise_piano_sig[0], np.ndarray):   # Stereo signal = 2 channels
                noise_piano_sig = np.average(noise_piano_sig, axis=-1)

        else:
            # print('Piano Sig Type:')
            # print(piano_label_sig.dtype)
            # print(piano_label_sig)
            # print('Done')

            # pl_orig_type, nl_orig_type = piano_label_sig.dtype, noise_label_sig.dtype 
            piano_label_sig, noise_label_sig = piano_label_sig.astype('float64'), noise_label_sig.astype('float64')
            assert len(noise_label_sig) == len(piano_label_sig)  
            # Stereo audio safety check
            if isinstance(piano_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                # piano_label_sig = np.array([((x[0] + x[1]) / 2) for x in piano_label_sig.astype('float32')]).astype(p_type)
                piano_label_sig = np.average(piano_label_sig, axis=-1)
            if isinstance(noise_label_sig[0], np.ndarray):   # Stereo signal = 2 channels
                # noise_label_sig = np.array([((x[0] + x[1]) / 2) for x in noise_label_sig.astype('float32')]).astype(n_type)
                noise_label_sig = np.average(noise_label_sig, axis=-1)

            avg_src_sum = (np.sum(piano_label_sig) + np.sum(noise_label_sig)) / 2
            src_percent_1 = random.randrange(int((src_amp_low*100) // 2), int((src_amp_high*100) // 2)) / 100
            src_percent_2 = 1 - src_percent_1
            piano_src_is_1 = bool(random.getrandbits(1))
            if piano_src_is_1:
                piano_label_sig *= src_percent_1
                noise_label_sig *= src_percent_2
            else:
                piano_label_sig *= src_percent_2
                noise_label_sig *= src_percent_1

            noise_piano_sig = piano_label_sig + noise_label_sig
            noise_piano_sig *= (avg_src_sum / np.sum(noise_piano_sig))     

        # # Pad up here now to support earlier tests
        # deficit = pad_len - len(piano_label_sig)
        # piano_label_sig = np.pad(piano_label_sig, (0,deficit))
        # # noise_label_sig = np.pad(noise_label_sig, (0,deficit))
        
        # piano_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon, 
        #                                         ova=True, debug=False)
        # # noise_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
        #                                         # ova=True, debug=False)

        # # VISUAL TEST
        # # print('PIANO SPGM #', i)
        # # for i in range(train_seq):
        # #     print(piano_spgm[i])
        # # print('DONE')

        # piano_samples[i] = piano_spgm
        # # noise_samples[i] = noise_spgm

        # piano_amp_factor = random.uniform(src_amp_low, src_amp_high)
        # noise_amp_factor = random.uniform(src_amp_low, src_amp_high)
        # piano_label_sig *= piano_amp_factor
        # noise_label_sig *= noise_amp_factor

        # aug_piano_spgm, _ = make_spectrogram(piano_label_sig, wdw_size, epsilon, 
        #                                         ova=True, debug=False)
        # # aug_noise_spgm, _ = make_spectrogram(noise_label_sig, wdw_size, epsilon, 
        #                                         # ova=True, debug=False)
        # aug_piano_samples[i] = aug_piano_spgm
        # # aug_noise_samples[i] = aug_noise_spgm

        # noise_piano_sig = piano_label_sig + noise_label_sig
        deficit = pad_len - len(noise_piano_sig)
        noise_piano_sig = np.pad(noise_piano_sig, (0,deficit))

        noise_piano_spgm, np_phase = make_spectrogram(noise_piano_sig, wdw_size, epsilon, 
                                                ova=True, debug=False)
        samples[i] = noise_piano_spgm

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