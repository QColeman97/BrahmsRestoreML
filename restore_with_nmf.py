# Main NMF Restoration Script

from brahms_restore_ml.nmf.nmf import *

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print('\nUsage: restore_with_nmf.py <signal> [-d] [window_size]')
        print('Parameter options:')
        print('Signal           filepath        - String denoting a WAV filepath')
        print('                 list            - Signal represented by list formatted like "[0,1,1,0]"')
        print('                 natural number  - Random element signal of this length')
        print('Debug            "-d"            - Option to print & plot matrices')
        print('Window Size      natural number (power of 2 preferably, default is for piano: 4096)\n')
        print('Currently not parameters: Sampling Rate, size of Basis Vectors\n')
        sys.exit(1)

    # Validated params (NMF hyperparameters) - deprecated or to be untouched
    # noisebv_flag = True       # perform source separation
    # avgbv_flag = True         # create basis vectors by avg'ing spectra (instead of 1 window of STFT)
    # ova_flag = True           # overlap-add (spectrogram creation)
    # learn_iter = 100
    # audible_range_bv = False  # to not include notes in range unreached in score
    score_piano_bv = True       
    a430hz_bv = False           # not confirmed to help
    # marybv_flag = False       # Special case for Mary.wav - basis vectors size optimization test
    out_filepath = 'brahms_restore_ml/nmf/output/output_restored_wav_v5/'

    # Experimental params (NMF hyperparameters)
    semi_sup_learn = 'Noise'    # Ternary flag - 'Piano', 'Noise', or 'None' (If not 'None', noisebv_flag MUST BE TRUE)
    semi_sup_made_init = False  # Only considered when semi_sup_learn != 'None', else ignored
    l1_penalty = 131072                                 # L1-penalty value
    l1pen_flag = True if (l1_penalty != 0) else False   # L1-penalize activations matrix
    top_acts = None             # Pick only highest valued activations, zero-out remaining
    top_acts_score = False      # Pick only activations correlating to notes in score
    num_noise_bv = 2            # Do not make as big as 1078 (smaller dim) - 88 (piano bv's) = 990
    dmged_piano_bv = False      # Learn w/ damaged BVs & synthesize restoration w/ quality BVs
    num_piano_bv_unlocked = None    # To be used only w/ semi-sup learn piano rand-init, default: None


    # Use command line arguments
    sig_sr = STD_SR_HZ # Initialize sr to default
    # Signal - comes as a list, WAV filepath, or a length of random-valued signal
    if sys.argv[1].startswith('['):
        sig = np.array([int(num) for num in sys.argv[1][1:-1].split(',')])
        out_filepath += 'my_sig'
    elif not sys.argv[1].endswith('.wav'):  # Work around for is a number
        sig = np.random.rand(int(sys.argv[1].replace(',', '')))
        out_filepath += 'rand_sig'
    else:
        sig_sr, sig = wavfile.read(sys.argv[1])
        if sig_sr != STD_SR_HZ:
            sig, sig_sr = librosa.load(sys.argv[1], sr=STD_SR_HZ)  # Upsample to 44.1kHz if necessary
        start_index = (sys.argv[1].rindex('/') + 1) if (sys.argv[1].find('/') != -1) else 0
        out_filepath += sys.argv[1][start_index: -4]
    # Debugging print/plot option, wdw size (for STFT)
    debug_flag, wdw_size = False, PIANO_WDW_SIZE    
    if len(sys.argv) > 2:
        if len(sys.argv) == 3:
            if sys.argv[2] == '-d':
                debug_flag = True
            else:
                wdw_size = int(sys.argv[2])
        else:
            debug_flag = True
            wdw_size = int(sys.argv[3]) if (sys.argv[2] == '-d') else int(sys.argv[2])

    # Describe hyperparameters in output filename
    # # Below Necessary & Default - no longer in name
    # if ova_flag:
    #     out_filepath += '_ova'
    if a430hz_bv:
        out_filepath += '_a436hz' # '_a430hz'
    if score_piano_bv:
        # override if num_piano_bv_unlocked
        out_filepath += ('_scorebv' if num_piano_bv_unlocked is None else 
                         ('_' + str(num_piano_bv_unlocked) + 'pbv'))
    # if audible_range_bv:
    #     out_filepath += '_arbv'
    if dmged_piano_bv:
        out_filepath += '_dmgedpbv'

    if semi_sup_learn == 'Piano':
        out_filepath += '_sslrnpiano'
        if semi_sup_made_init:
            out_filepath += '_madeinit'
    elif semi_sup_learn == 'Noise':
        out_filepath += '_sslrnnoise'
        if semi_sup_made_init:
            out_filepath += '_madeinit'
    # if noisebv_flag:
    out_filepath += ('_' + str(num_noise_bv) + 'nbv')
    # Averaging FFTs to get a BV is Optimal & Default - no longer in name
    # if avgbv_flag:
    #     out_filepath += '_avgbv'
    if l1pen_flag:
        out_filepath += ('_l1pen' + str(l1_penalty))
    if top_acts is not None:
        out_filepath += ('_top' + str(top_acts) + 'acts')
    if top_acts_score:
        out_filepath += '_nobottomnotes'
    out_filepath += '.wav'


    restore_with_nmf(sig, wdw_size, out_filepath, sig_sr, num_noisebv=num_noise_bv, 
                    semisuplearn=semi_sup_learn, semisupmadeinit=semi_sup_made_init,
                    l1_penalty=l1_penalty, debug=debug_flag, a430hz_bv=a430hz_bv,
                    scorebv=score_piano_bv,
                    dmged_pianobv=dmged_piano_bv, 
                    num_pbv_unlocked=num_piano_bv_unlocked,
                    top_acts=top_acts, top_acts_score=top_acts_score)

if __name__ == '__main__':
    main()