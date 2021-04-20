import pyaudio
import wave
from scipy.interpolate import pchip_interpolate
from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

sig = np.array([])

# callback function
def callback(in_data, frame_count, time_info, flag):
    global start, stop, buf_sound, buf_noise, sig

    max_sample = 2 ** 15

    # sound
    audio_data = sound_file.readframes(overlap)
    sound = np.frombuffer(audio_data, dtype=np.int16)
    sound = sound / max_sample

    # noise
    noise = np.frombuffer(in_data, dtype=np.int16)
    noise = noise / max_sample

    # overlapping buffers
    buf_sound = shift(buf_sound, overlap)
    buf_sound[-overlap:] = sound[:]

    buf_noise = shift(buf_noise, overlap)
    buf_noise[-overlap:] = noise[:]

    # set the same rms
    # buf_sound = buf_sound * rms_energy(buf_noise) / rms_energy(buf_sound)

    # Hanning windowing
    # win = np.hanning(FRAMESIZE)
    # win_sound = buf_sound * win[:]
    # win_noise = buf_noise * win[:]

    # FFT --> frequency domain
    Sound = np.fft.rfft(buf_sound, FRAMESIZE)
    Noise = np.fft.rfft(buf_noise, FRAMESIZE)
    f = np.fft.rfftfreq(FRAMESIZE, 1 / RATE)

    # filter banks and central frequencies
    scale = 'bark'                                                   # frequency scale
    n_bands = 24
    fb, cnt_scale = getFilterBank(f, scale=scale, nband=n_bands)    # filter bank for signal

    # spectral filtering
    signal_bands = applyFilterBank(abs(Sound), fb)
    noise_bands = applyFilterBank(abs(Noise), fb)

    # convert center frequencies into Hz
    cnt_hz = scale2f(cnt_scale, scale)

    # computation of modulation factors
    snr_target = 2                                                  # desired SNR
    limits = [0.2, 4.0]                                             # limits for the modulation factor

    k = setSNR(signal_bands, noise_bands, snr_target, limits)  # compute modulation factor matrix

    # interpolation
    k, cnt_hz_s = addLimitBands(k, cnt_hz, f)  # add the lower and the upper bands to avoid under/overshooting
    k_interp = pchip_interpolate(cnt_hz_s, k, f)

    # signal spectrum equalization
    Signal_eq = Sound * k_interp

    # IFFT --> back to time domain
    signal_eq = np.fft.irfft(Signal_eq, FRAMESIZE)

    # equalized signal is windowed again with hanning window, to remove modulation artifacts
    # signal_eq = signal_eq * win[:]

    signal_eq = signal_eq * max_sample

    # convert back to int16
    y = signal_eq.astype(np.int16)

    # update start and stop indexes
    start += overlap
    stop += overlap

    return y.tobytes(), pyaudio.paContinue


# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano_mono.wav'     # audio must be mono

# Open the sound file
sound_file = wave.open(sonification_path, 'rb')

# audio settings
FRAMESIZE = 1024
FORMAT = p.get_format_from_width(sound_file.getsampwidth())
CHANNELS = sound_file.getnchannels()
RATE = sound_file.getframerate()

# overlapping buffers
buf_sound = np.zeros(FRAMESIZE)                             # sonification buffer
buf_noise = np.zeros(FRAMESIZE)                             # noise buffer
n_overlap = 4                                               # number of overlapping frames
overlap = FRAMESIZE // n_overlap                            # overlapping samples

# start and stop indexes for sonification buffer
start = 0
stop = overlap

# Open a stream object to write the WAV file to
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=overlap,
                input=True,
                output=True,
                stream_callback=callback)

# start stream
stream.start_stream()

# Play the sound by writing the audio data to the stream
while stream.is_active():
    time.sleep(getDuration(sonification_path))
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
