import pyaudio
import sys
import time
# from matplotlib.widgets import Button, Slider
from functions import *

# CONTROLLI TEST
fast = sys.argv[1] == 'True'        # if True: "frasi veloci"; if False: "frasi lente"
num_sentence = sys.argv[2]          # number of the sentence (from 1 to 64 --> test; from 65 to 100 --> training)

# STAMPA INFO PROGRAMMA
speed = " veloce" if fast else " lenta"
print("Volume adattivo, Frase " + str(num_sentence) + speed)

calibration_gain = db2amp(-10)

# initialize pyaudio
p = pyaudio.PyAudio()

# # The function to be called anytime a slider's value changes
# def update(val):
#     global snr_target
#     snr_target = gain_slider.val
#
#
# def reset(event):
#     gain_slider.reset()


def callback(in_data, frame_count, time_info, flag):
    global start, stop, mod_previous, a, b, zi

    # sound is read from array for faster performances
    sound = audio_data[start:stop] * calibration_gain

    # noise
    noise = byte2audio(in_data)

    sound, noise = setSameLength(sound, noise, padding=True)

    # deltaRmsIN = amp2db(rmsEnergy(sound)) - amp2db(rmsEnergy(noise))

    # volume adaptivity: SNR is kept constant inside a specified sound intensity range,
    # outside of witch the signal volume is kept constant

    # get modulation term
    modulation, mod_factor = getRealTimeModulation(sound, noise, snr_target, mod_previous, limits, offset=db2amp(22))

    # apply the modulation factor to the signal
    sound = sound * modulation

    sound = boneConductingFilter(sound, a, b, zi)

    # deltaRmsOUT = amp2db(rmsEnergy(sound)) - amp2db(rmsEnergy(noise))
    # print(deltaRmsIN, deltaRmsOUT)

    # limiter
    sound = limiter(sound)

    y = audio2byte(sound)

    start += CHUNK
    stop += CHUNK

    mod_previous = mod_factor

    return y, pyaudio.paContinue


# sonification sound: piano samples, C major scale
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/voice.wav'
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/piano.wav'
if fast:
    folder = "frasi veloci/"
else:
    folder = "frasi lente/"

sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/' + folder + 'frase ' \
                    + str(num_sentence) + '.wav'

# audio settings
CHUNK = 1024
FORMAT = 8                                                          # int16
CHANNELS = 1
RATE = 44100
snr_target = db2amp(6)
mod_previous = -1
limits = [db2amp(-12), db2amp(12)]                                  # limits for the modulation factor

# start and stop indexes for sonification reading
start = 0
stop = CHUNK

# store sonification samples in array
audio_data, _ = audioread(sonification_path, RATE)
audio_data = addPadding(audio_data, RATE, 10)
audio_data = audio_data / 2 ** 15

# bone conducting filter
a = [[1, -1.98717373, 0.98724634], [1, -1.96713666, 0.96794841],
     [1, -1.38158037, 0.49588455], [1, -1.01411009, 0.35962489]]
b = [[0.99360502, -1.98721004, 0.99360502], [0.99685135, -1.96713666, 0.97109706],
     [0.86998166, -1.38158037, 0.62590289], [1.55411604, -1.91366498, 0.70506374]]
zi = np.zeros((4, 2))

# # Make a horizontal slider to control the gain
# axcolor = 'lightgoldenrodyellow'
# init_gain = snr_target
# axgain = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# gain_slider = Slider(
#     ax=axgain,
#     label='GAIN [amp]',
#     valmin=0.1,
#     valmax=10,
#     valinit=init_gain,
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
                frames_per_buffer=CHUNK,
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
