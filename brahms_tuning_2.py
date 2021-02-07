from brahms_restore_ml.nmf.nmf import WDW_NUM_AFTER_VOICE
from brahms_restore_ml.nmf.basis_vectors import *

ova = True

def get_first_note_bv(sig_at_first_note, sr, sig_type, note_dur=18000):
    # brahms_sr, brahms_sig = wavfile.read('brahms.wav')
    # sig = convert_wav_format_up(brahms_sig)
    # int16_type = brahms_sig.dtype
    # # Capture only first note
    # offset = 45000
    # trim_sig = brahms_sig[PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE: (PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE) + offset]
    # # Trim off silence
    # trim_sig = trim_sig.astype('float64')
    # # Convert to mono signal (avg left & right channels)
    # trim_sig = np.average(trim_sig, axis=-1) if (len(trim_sig.shape) > 1) else trim_sig
    # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
    # amp_thresh = np.amax(np.abs(sig_at_first_note)) * 0.965
    # print('Amp thresh:', amp_thresh)
    # while np.abs(sig_at_first_note[0]) < amp_thresh:
    #     sig_at_first_note = sig_at_first_note[1:]

    # duration of first note in Brahms ~ 18721
    # note_dur = 30000    # need 13000 = 6/14ths (0.43) of 30000 b/c w/ that waveform changes notes halfway
    # note_dur = 18000    # add on 5000 for 18000 total, b/c note starts 5000 samples earlier
    sig_at_first_note = sig_at_first_note[:note_dur]

    wavfile.write('brahms_begin_note_trim.wav', sr, sig_at_first_note.astype(sig_type))
    print('Length of first note trim for bv:', len(sig_at_first_note))
    spectrogram, phases = make_spectrogram(sig_at_first_note, PIANO_WDW_SIZE, EPSILON, ova=ova)
    print('Spgm shape:', spectrogram.shape)
    print('Phases shape:', phases.shape)
    basis_vector = np.mean(spectrogram, axis=0)
    return basis_vector



# TODO - replicate how make_basis_vector writes out basis vector, see what makes it not work

# begin_note_offset = 25000   # 25000 perfect and makes perfect fist_note_dur
begin_note_offset = 15000   # for ccrma benchamrk 4

# File needs to be same sr as Brahms
# note_file = os.getcwd() + '/brahms_restore_ml/nmf/all_notes_ff_wav/Piano.ff.A4.wav'
# note_file = 'brahms.wav'    # new
note_file = '../Benchmark Systems/ccrma/benchmark4.wav'
sig_sr, sig = wavfile.read(note_file)
sig = convert_wav_format_up(sig)    # new
int16_type = sig.dtype
sig = sig[(PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE) + begin_note_offset: ]   # new
sig = sig.astype('float64')
sig = np.average(sig, axis=-1) if (len(sig.shape) > 1) else sig
# amp_thresh = np.amax(np.abs(sig)) * 0.01  # new
# while np.abs(sig[0]) < amp_thresh:
#     sig = sig[1:]
# while np.abs(sig[-1]) < amp_thresh:
#     sig = sig[:-1]
# print('Brahms converted up:', sig)
# for i in range(100000, 1000000, 100000):
#     print(sig[i:(i+1)*10000].astype(orig_type))
wavfile.write('brahms_begin_trim.wav', sig_sr, sig.astype(int16_type))

basis_vector = get_first_note_bv(sig.copy(), sig_sr, int16_type)    # new & COPY SIG IMPORTANT

print('Length of brahms for phases:', len(sig))
spgm, phases = make_spectrogram(sig, PIANO_WDW_SIZE, EPSILON, ova=ova)
print('Spgm shape:', spgm.shape)
print('Phases shape:', phases.shape)
# basis_vector = np.mean(spgm, axis=0)    # new

bv_spgm = np.array([basis_vector for _ in range(spgm.shape[0])])
bv_sig = make_synthetic_signal(bv_spgm, phases, PIANO_WDW_SIZE, int16_type, ova=ova, debug=False)
bv_sig = bv_sig * 2     # needs to be decently loud (max multiply 2 for 16bit - no overflow)
print('Max val:', np.amax(np.abs(bv_sig)))
wavfile.write('brahms_begin_trim_bv.wav', sig_sr, bv_sig.astype(int16_type))


# print('gettgin length of first note duration')

# brahms_sr, brahms_sig = wavfile.read('brahms.wav')
# orig_type = brahms_sig.dtype
# # Capture only first note
# offset = 45000
# trim_sig = brahms_sig[PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE: (PIANO_WDW_SIZE * WDW_NUM_AFTER_VOICE) + offset]

# # Trim off silence
# trim_sig = trim_sig.astype('float64')
# # Convert to mono signal (avg left & right channels)
# trim_sig = np.average(trim_sig, axis=-1) if (len(trim_sig.shape) > 1) else trim_sig

# # Need to trim beginning/end silence off signal for basis vectors - achieve best frequency signature
# amp_thresh = np.amax(np.abs(trim_sig)) * 0.965
# print('Amp thresh:', amp_thresh)
# while np.abs(trim_sig[0]) < amp_thresh:
#     trim_sig = trim_sig[1:]
# while np.abs(trim_sig[-1]) < amp_thresh:
#     trim_sig = trim_sig[:-1]

# # # optional - write trimmed piano note signals to WAV - check if trim is good
# wavfile.write('brahms_note_exact_trim.wav', brahms_sr, trim_sig.astype(orig_type))
# print('Length of first note trim:', len(trim_sig))

# Make signal smaller to hopefully make for a better sounding bv,
# Else take a pos mag fft of a part of it

# print('Length of first note trim for bv:', len(trim_sig))
# spectrogram, phases = make_spectrogram(trim_sig, PIANO_WDW_SIZE, EPSILON, ova=ova)
# print('Spgm shape:', spectrogram.shape)
# print('Phases shape:', phases.shape)
# basis_vector = np.mean(spectrogram, axis=0)


# bv_phases = phases
# # write basis vector magnitude-repeated out to wav file
# bv_spgm = np.array([basis_vector for _ in range(bv_phases.shape[0])])

# print('BV spgm shape:', bv_spgm.shape)
# print('BV phases shape:', bv_phases.shape)
# bv_sig = make_synthetic_signal(bv_spgm, bv_phases, PIANO_WDW_SIZE, orig_type, ova=ova, debug=False)
# wavfile.write('brahms_begin_trim_bv.wav', brahms_sr, bv_sig)


