import pyaudio
import wave

from matplotlib.widgets import Slider, Button
from scipy.interpolate import pchip_interpolate
from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

# The function to be called anytime a slider's value changes
def update(val):
    global snr_target
    snr_target = gain_slider.val


def reset(event):
    gain_slider.reset()


# callback function
def callback(in_data, frame_count, time_info, flag):
    global start, stop, buf_sig_cur, buf_sig_next, buf_noi_cur, buf_sig_next

    # sound
    # sound = sonification[start:stop]

    # sound
    audio_data = sound_file.readframes(FRAMESIZE)
    sound = np.frombuffer(audio_data, dtype=np.int16)
    sound = sound.astype(np.float64)

    # noise
    noise = np.frombuffer(in_data, dtype=np.int16)
    noise = noise.astype(np.float64)

    # overlapping buffers
    buf_sig_cur[:] = buf_sig_next[:]                            # set the next buffer as the current buffer
    buf_sig_cur[overlap:] = sound[:overlap]                     # the second half of buffer is the first half of frame
    buf_sig_next[:overlap] = sound[overlap:]                    # the first half of buffer is the second half of frame

    buf_noi_cur[:] = buf_noi_next[:]                            # set the next buffer as the current buffer
    buf_noi_cur[overlap:] = noise[:overlap]                     # the second half of buffer is the first half of frame
    buf_noi_next[:overlap] = noise[overlap:]                    # the first half of buffer is the second half of frame

    # set the same rms
    # buf_sig_cur = buf_noi_cur * rms_energy(buf_noi_cur) / rms_energy(buf_sig_cur)

    # Hanning windowing
    # win = np.hanning(FRAMESIZE)
    # win_sound = buf_sig_cur * win[:]
    # win_noise = buf_noi_cur * win[:]

    # FFT --> frequency domain
    Sound = np.fft.rfft(buf_sig_cur, FRAMESIZE)
    Noise = np.fft.rfft(buf_noi_cur, FRAMESIZE)
    f = np.fft.rfftfreq(FRAMESIZE, 1 / RATE)

    # filter banks and central frequencies
    scale = 'erb'                                                   # frequency scale
    n_bands = 21
    fb, cnt_scale = getFilterBank(f, scale=scale, nband=n_bands)    # filter bank for signal

    # spectral filtering
    signal_bands = applyFilterBank(abs(Sound), fb)
    noise_bands = applyFilterBank(abs(Noise), fb)

    # convert center frequencies into Hz
    cnt_hz = scale2f(cnt_scale, scale)

    # computation of modulation factors
    limits = [0.2, 4.0]                                             # limits for the modulation factor
    k = set_snr(signal_bands, noise_bands, snr_target, limits)      # compute modulation factor matrix

    # interpolation
    k, cnt_hz_s = add_limit_bands(k, cnt_hz, f)         # add the lower and the upper bands to avoid under/overshooting
    k_interp = pchip_interpolate(cnt_hz_s, k, f)

    # signal spectrum equalization
    Signal_eq = Sound * k_interp

    # IFFT --> back to time domain
    signal_eq = np.fft.irfft(Signal_eq, FRAMESIZE)

    # equalized signal is windowed again with hanning window, to remove modulation artifacts
    # signal_eq = signal_eq * win[:]

    # convert back to int16
    y = signal_eq.astype(np.int16)

    # update start and stop indexes
    start += FRAMESIZE
    stop += FRAMESIZE

    return y.tobytes(), pyaudio.paContinue


# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano_mono.wav'     # audio must be mono

# Open the sound file
sound_file = wave.open(sonification_path, 'rb')
# sonification, _ = audioread(sonification_path)

# audio settings
FRAMESIZE = 1024
FORMAT = p.get_format_from_width(sound_file.getsampwidth())
CHANNELS = sound_file.getnchannels()
RATE = sound_file.getframerate()
snr_target = 2

# overlapping buffers
buf_sig_cur = np.zeros(FRAMESIZE)                                           # current sonification buffer
buf_sig_next = np.zeros(FRAMESIZE)                                          # next sonification buffer
buf_noi_cur = np.zeros(FRAMESIZE)                                           # current noise buffer
buf_noi_next = np.zeros(FRAMESIZE)                                          # next noise buffer
n_overlap = 2                                                               # number of overlapping frames
overlap = FRAMESIZE // n_overlap                                            # overlapping samples

# start and stop indexes for sonification buffer
start = 0
stop = FRAMESIZE

# Make a horizontal slider to control the gain
axcolor = 'lightgoldenrodyellow'
init_gain = 2
axgain = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
gain_slider = Slider(
    ax=axgain,
    label='SNR TARGET',
    valmin=-10,
    valmax=10,
    valinit=init_gain,
)

# register the update function with each slider
gain_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.4, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
button.on_clicked(reset)

# Open a stream object to write the WAV file to
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=FRAMESIZE,
                input=True,
                output=True,
                stream_callback=callback)

# start stream
stream.start_stream()

# Play the sound by writing the audio data to the stream
while stream.is_active():
    plt.show()
    time.sleep(get_duration(sonification_path))
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
