import time

import numpy as np
import pyaudio
from matplotlib.widgets import Slider, Button
from scipy.interpolate import pchip_interpolate

from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

def update(val):
    global snr_target
    snr_target = gain_slider.val


def reset(event):
    gain_slider.reset()

# callback function
def callback(in_data, frame_count, time_info, flag):
    global start, stop, buf_sound, buf_noise, adapt_sonif, w

    # sound is read from array for faster performances
    sound = audio_data[start:stop]

    # noise is given by the microphone
    noise = byte2audio(in_data)

    # make sure that sound and noise have the same length
    sound, noise = setSameLength(sound, noise, padding=True)

    # overlapping buffers
    buf_sound = shift(buf_sound, overlap)
    buf_sound[-overlap:] = sound[:]

    buf_noise = shift(buf_noise, overlap)
    buf_noise[-overlap:] = noise[:]

    # set the same rms
    # buf_sound = buf_sound * rmsEnergy(buf_noise) / rmsEnergy(buf_sound)

    # Hanning windowing
    win = np.hanning(FRAMESIZE)
    win_sound = buf_sound * win[:]
    win_noise = buf_noise * win[:]

    # FFT --> frequency domain
    Sound = np.fft.rfft(win_sound, FRAMESIZE)
    Noise = np.fft.rfft(win_noise, FRAMESIZE)
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
    signal_eq = signal_eq * win[:]

    # adaptive sonification and window array shift
    adapt_sonif = shift(adapt_sonif, overlap)
    w = shift(w, overlap)

    # set adaptive sonification and window array last quarter to 0 (eps for window)
    adapt_sonif[-overlap:] = 0
    w[-overlap:] = 0.1

    # add equalized signal to adaptive sonification array
    adapt_sonif[:] += signal_eq

    # add squared Hanning window to window array
    w[:] += win ** 2

    adapt_sonif[:] = adapt_sonif / w

    # signal_eq = signal_eq[:overlap]
    out = adapt_sonif[:overlap]

    # y = audio2byte(signal_eq)
    y = audio2byte(out)

    # update start and stop indexes
    start += overlap
    stop += overlap

    return y, pyaudio.paContinue


# sonification sound: piano samples or speech
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/voice.wav'
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/piano.wav'

# audio settings
FRAMESIZE = 1024
FORMAT = 8
CHANNELS = 1
RATE = 44100
snr_target = 2                                              # desired SNR

audio_data, _ = audioread(sonification_path, RATE)
audio_data = audio_data / 2 ** 15

# overlapping buffers
buf_sound = np.zeros(FRAMESIZE)                             # sonification buffer
buf_noise = np.zeros(FRAMESIZE)                             # noise buffer
n_overlap = 4                                               # number of overlapping frames
overlap = FRAMESIZE // n_overlap                            # overlapping samples

# start and stop indexes for sonification buffer
start = 0
stop = overlap

# adaptive sonification array
adapt_sonif = np.zeros(FRAMESIZE)

# window array
w = np.zeros(FRAMESIZE) + 0.1

# Make a horizontal slider to control the gain
axcolor = 'lightgoldenrodyellow'
init_gain = 2.0
axgain = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
gain_slider = Slider(
    ax=axgain,
    label='GAIN [amp]',
    valmin=0.01,
    valmax=10.0,
    valinit=init_gain
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
                frames_per_buffer=overlap,
                input=True,
                output=True,
                stream_callback=callback)

# start stream
stream.start_stream()

# Play the sound by writing the audio data to the stream
while stream.is_active():
    plt.show()
    # time.sleep(getDuration(sonification_path))
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
