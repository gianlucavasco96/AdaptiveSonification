import pyaudio
import sys
import time
# from matplotlib.widgets import Slider, Button
from scipy.interpolate import pchip_interpolate

from functions import *

# CONTROLLI TEST
fast = sys.argv[1] == 'True'        # if True: "frasi veloci"; if False: "frasi lente"
num_sentence = sys.argv[2]          # number of the sentence (from 1 to 64 --> test; from 65 to 100 --> training)

# STAMPA INFO PROGRAMMA
speed = " veloce" if fast else " lenta"
print("Equalizzazione adattiva 10 dB, Frase " + str(num_sentence) + speed)

calibration_gain = db2amp(-10)

# initialize pyaudio
p = pyaudio.PyAudio()

# def update(val):
#     global snr_target
#     snr_target = gain_slider.val
#
#
# def reset(event):
#     gain_slider.reset()

# callback function
def callback(in_data, frame_count, time_info, flag):
    global start, stop, buf_sound, buf_noise, adapt_sonif, w, limits, a, b, zi

    # sound is read from array for faster performances
    sound = audio_data[start:stop] * calibration_gain

    # noise is given by the microphone
    noise = byte2audio(in_data)

    # make sure that sound and noise have the same length
    sound, noise = setSameLength(sound, noise, padding=True)

    # print(amp2db(rmsEnergy(noise)), amp2db(rmsEnergy(sound)))

    # overlapping buffers
    buf_sound = shift(buf_sound, overlap)
    buf_sound[-overlap:] = sound[:]

    buf_noise = shift(buf_noise, overlap)
    buf_noise[-overlap:] = noise[:]

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
    k = setSNR(signal_bands, noise_bands, snr_target, limits, offset=db2amp(22))

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

    # set adaptive sonification and window array last quarter to 0
    adapt_sonif[-overlap:] = 0
    w[-overlap:] = 0

    # add equalized signal to adaptive sonification array
    adapt_sonif[:] += signal_eq

    # add squared Hanning window to window array and set to eps all zero values
    w[:] += win ** 2
    w[w == 0.0] = eps

    # since overlap samples have been taken as input, first overlap samples have to be taken as output
    out = adapt_sonif[:overlap] / w[:overlap]

    out = boneConductingFilter(out, a, b, zi)

    # limiter
    out = limiter(out)

    # convert audio to bytes
    y = audio2byte(out)

    # update start and stop indexes
    start += overlap
    stop += overlap

    return y, pyaudio.paContinue


# sonification sound: piano samples or speech
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/voice.wav'
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/piano.wav'
if fast:
    folder = "frasi veloci/"
else:
    folder = "frasi lente/"

sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/' + folder + 'frase ' \
                    + str(num_sentence) + '.wav'

# audio settings
FRAMESIZE = 1024                                            # buffer-frame size
FORMAT = 8                                                  # int16 format
CHANNELS = 1                                                # mono
RATE = 44100                                                # sampling rate
snr_target = db2amp(10)                                     # desired SNR: +10 dB
limits = [db2amp(-12), db2amp(18)]                          # limits for the modulation factor

# sonification reading and normalization
audio_data, _ = audioread(sonification_path, RATE)
audio_data = addPadding(audio_data, RATE, 10)
audio_data = audio_data / 2 ** 15

# bone conducting filter
a = [[1, -1.98717373, 0.98724634], [1, -1.96713666, 0.96794841],
     [1, -1.38158037, 0.49588455], [1, -1.01411009, 0.35962489]]
b = [[0.99360502, -1.98721004, 0.99360502], [0.99685135, -1.96713666, 0.97109706],
     [0.86998166, -1.38158037, 0.62590289], [1.55411604, -1.91366498, 0.70506374]]
zi = np.zeros((4, 2))

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
w = np.zeros(FRAMESIZE)

# # Make a horizontal slider to control the gain
# axcolor = 'lightgoldenrodyellow'
# init_gain = snr_target
# axgain = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# gain_slider = Slider(
#     ax=axgain,
#     label='GAIN [amp]',
#     valmin=0.01,
#     valmax=10.0,
#     valinit=init_gain
# )
#
# # register the update function with each slider
# gain_slider.on_changed(update)
#
# # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
# resetax = plt.axes([0.4, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
# button.on_clicked(reset)

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
    print("START")
    # plt.show()
    time.sleep(getDuration(sonification_path) + 15)
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
