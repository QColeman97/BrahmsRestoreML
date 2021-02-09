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

    # Pre-configured params
    # # Confirmed helps remaining same - to be untouched
    # noisebv_flag = True 
    # avgbv_flag = True
    # ova_flag = True
    # learn_iter = 100
    # marybv_flag = False     # Special case for Mary.wav - basis vectors size optimization test
    # out_filepath = 'brahms_restore_ml/nmf/output/output_restored_wav_v4/'
    # TEMP
    out_filepath = 'brahms_restore_ml/nmf/output/output_restored_experimental/'

    # Experimental
    # Ternary flag - 'Piano', 'Noise', or 'None' (If not 'None', noisebv_flag MUST BE TRUE)
    semi_sup_learn = 'None'
    semi_sup_made_init = False   # Only considered when semi_sup_learn != 'None'
    l1_penalty = 0 # 10 ** 19 # 10^9 = 1Bill, 12 = trill, 15 = quad, 18 = quin, 19 = max for me
    l1pen_flag = True if (l1_penalty != 0) else False
    # Do not make as big as 1078 (smaller dim) - 88 (piano bv's) = 990
    num_noise_bv = 1 # 50 # 20 # 3 # 10 # 5 # 10000 is when last good # 100000 is when it gets bad, but 1000 sounds bad in tests.py
    # Put into if good results. Only use piano notes in recording
    audible_range_bv = False
    score_piano_bv = True
    a430hz_bv = True

    # Configure params    
    # Signal - comes as a list, filepath or a length
    sig_sr = STD_SR_HZ # Initialize sr to default
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

    # Debug-print/plot option, wdw size
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

    # # Overlap-Add is Necessary & Default - no longer in name
    # if ova_flag:
    #     out_filepath += '_ova'
    if a430hz_bv:
        out_filepath += '_a430hz'
    if score_piano_bv:
        out_filepath += '_scorebv'
    if audible_range_bv:
        out_filepath += '_arbv'

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
    out_filepath += '.wav'
    restore_with_nmf(sig, wdw_size, out_filepath, sig_sr, num_noisebv=num_noise_bv, 
                    semisuplearn=semi_sup_learn, semisupmadeinit=semi_sup_made_init,
                    l1_penalty=l1_penalty, debug=debug_flag, a430hz_bv=a430hz_bv,
                    scorebv=score_piano_bv, audible_range_bv=audible_range_bv)


if __name__ == '__main__':
    main()