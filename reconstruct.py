# Signal Reconstruct Script (Sig -> Spectrogram -> Sig)

from brahms_restore_ml.audio_data_processing import *
import sys
import librosa

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print('\nUsage: reconstruct.py <signal> [-d] [window_size]')
        print('Parameter options:')
        print('Signal           filepath        - String denoting a WAV filepath')
        print('                 list            - Signal represented by list formatted like "[0,1,1,0]"')
        print('                 natural number  - Random element signal of this length')
        print('Debug            "-d"            - Option to print & plot matrices')
        print('Window Size      natural number (power of 2 preferably, default is for piano: 4096)\n')
        print('Currently not parameters: Sampling Rate\n')
        sys.exit(1)

    # Pre-configured params
    # Confirmed helps remaining same - to be untouched
    ova_flag = True
    out_filepath = 'brahms_restore_ml/nmf/output/output_reconstructed_wav/'

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

    # FOR TESTING
    no_voice = False
    if no_voice:
        out_filepath += '_novoice'
    out_filepath += '.wav'
    reconstruct_audio(sig, wdw_size, out_filepath, sig_sr, ova=ova_flag, segment=no_voice, 
                        write_file=True, debug=debug_flag)


if __name__ == '__main__':
    main()