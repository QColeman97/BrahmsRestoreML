from brahms_restore_ml.nmf.nmf import WDW_NUM_AFTER_VOICE
from brahms_restore_ml.nmf.basis_vectors import *

ova = True

def get_note_avg_fft(note_choice, sig_at_first_note, sr, sig_type, note_dur, loudness_mult, lmf=1):
    # duration of first note in Brahms ~ 18721
    # note_dur = 30000    # need 13000 = 6/14ths (0.43) of 30000 b/c w/ that waveform changes notes halfway
    # note_dur = 18000    # add on 5000 for 18000 total, b/c note starts 5000 samples earlier
    sig_at_first_note = sig_at_first_note[:note_dur]
    sig_at_first_note = sig_at_first_note * (loudness_mult * lmf)
    print('Max val note trim sig:', np.amax(np.abs(sig_at_first_note)))
    wavfile.write('brahms_note_trim_' + note_choice + '.wav', sr, sig_at_first_note.astype(sig_type))
    print('Length of first note trim for bv:', len(sig_at_first_note))
    spectrogram, phases = make_spectrogram(sig_at_first_note, PIANO_WDW_SIZE, EPSILON, ova=ova)
    print('Spgm shape:', spectrogram.shape)
    print('Phases shape:', phases.shape)
    basis_vector = np.mean(spectrogram, axis=0)
    return basis_vector


def make_extended_note(note_choice):
    # File needs to be same sr as Brahms
    if note_choice == 'first_eflat4':  # Eb4, Eb5 detected
        note_file = '../Benchmark Systems/ccrma/benchmark4.wav'
        # note_offset = (PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE) + 25000   # 25000 perfect and makes perfect fist_note_dur
        note_offset = (PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE) + 15000   # for ccrma benchamrk 4
        note_dur = 18000
        loudness_mult = 1.1
        sig_sr, sig = wavfile.read(note_file)
        lmf=1.5
    elif note_choice == 'g6': # highest
        note_file = '../Benchmark Systems/ccrma/benchmark2(thebest?).wav'
        # note_file = 'brahms.wav'
        sig_sr, sig = wavfile.read(note_file)
        note_offset = round(len(sig) * 0.6) + 15000
        note_dur = 6000
        loudness_mult = 1.3
        lmf=3
    elif note_choice == 'aflat3':   # lowest, is out-of-key, nonconclusive
        note_file = '../Benchmark Systems/ccrma/benchmark4.wav'
        sig_sr, sig = wavfile.read(note_file)
        note_offset = round(len(sig) // 2) + 7000 # 9000
        note_dur = 6000
        loudness_mult = 0.9
        lmf=1
    elif note_choice == 'a3':
        note_file = '../Benchmark Systems/ccrma/benchmark4.wav'
        sig_sr, sig = wavfile.read(note_file)
        note_offset = round(len(sig) // 3) + 11500
        note_dur = 80000
        loudness_mult = 1.8
        lmf=0.7
        # note_file = 'brahms.wav'  # failed, too noisy
        # sig_sr, sig = wavfile.read(note_file)
        # note_offset = round(len(sig) // 6) + 48000
        # note_dur = 80000
        # loudness_mult = 7
    elif note_choice == 'c4':
        note_file = '../Benchmark Systems/ccrma/benchmark4.wav'
        sig_sr, sig = wavfile.read(note_file)
        note_offset = (round(len(sig) // 8) * 6) + 112000
        note_dur = 60000
        loudness_mult = 1.5
        lmf=1.3

    sig = convert_wav_format_up(sig)    # new
    int16_type = sig.dtype
    sig = sig[note_offset: ]   # new
    sig = sig.astype('float64')
    sig = np.average(sig, axis=-1) if (len(sig.shape) > 1) else sig
    
    wavfile.write('brahms_begin_from_trim.wav', sig_sr, sig.astype(int16_type))

    basis_vector = get_note_avg_fft(note_choice, sig.copy(), sig_sr, int16_type, note_dur=note_dur, 
                                    loudness_mult=loudness_mult, lmf=lmf)    # new & COPY SIG IMPORTANT
    print('Length of brahms for phases:', len(sig))
    spgm, phases = make_spectrogram(sig, PIANO_WDW_SIZE, EPSILON, ova=ova)
    print('Spgm shape:', spgm.shape)
    print('Phases shape:', phases.shape)

    bv_spgm = np.array([basis_vector for _ in range(spgm.shape[0])])
    bv_sig = make_synthetic_signal(bv_spgm, phases, PIANO_WDW_SIZE, int16_type, ova=ova, debug=False)
    bv_sig = bv_sig * loudness_mult     # needs to be decently loud (max multiply 2 for 16bit - no overflow)
    print('Max val fft sig:', np.amax(np.abs(bv_sig)))
    if note_choice == 'c4':
        bv_sig = bv_sig[:round(len(bv_sig) * 0.75)]
    wavfile.write('brahms_trim_bv_' + note_choice + '.wav', sig_sr, bv_sig.astype(int16_type))


def main():
    note = 'c4'
    make_extended_note(note)

if __name__ == '__main__':
    main()