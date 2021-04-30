import pyaudio

from matplotlib.widgets import Button, Slider

from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

# The function to be called anytime a slider's value changes
def update(val):
    global gain
    gain = gain_slider.val


def reset(event):
    gain_slider.reset()


def callback(in_data, frame_count, time_info, flag):
    global start, stop

    # sound is read from array for faster performances
    sound = audio_data[start:stop]

    # noise
    noise = byte2audio(in_data)

    sound, noise = setSameLength(sound, noise, padding=True)

    # volume adaptivity: SNR is kept constant inside a specified sound intensity range,
    # outside of witch the signal volume is kept constant

    limits = [0.2, 2.0]  # limits for the modulation factor

    # get modulation term
    modulation = getModulation(sound, noise, gain, limits)

    # apply the modulation factor to the signal
    sound = sound * modulation

    y = audio2byte(sound)

    start += CHUNK
    stop += CHUNK

    return y, pyaudio.paContinue


# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/voice.wav'
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/piano.wav'

# audio settings
CHUNK = 1024
FORMAT = 8                                                                      # int16
CHANNELS = 1
RATE = 44100
gain = 2

# start and stop indexes for sonification reading
start = 0
stop = CHUNK

# store sonification samples in array
audio_data, _ = audioread(sonification_path, RATE)
audio_data = audio_data / 2 ** 15

# Make a horizontal slider to control the gain
axcolor = 'lightgoldenrodyellow'
init_gain = 2
axgain = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
gain_slider = Slider(
    ax=axgain,
    label='GAIN [amp]',
    valmin=0.1,
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
                frames_per_buffer=CHUNK,
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
