import time
import pyaudio
from functions import *

# CONTROLLI TEST
test = True                     # if True: training; if False: test --> add padding
fast = False                         # if True: "frasi veloci"; if False: "frasi lente"
num_sentence = 1                    # number of the sentence (from 1 to 64 --> test; from 65 to 100 --> training)

calibration_gain = db2amp(-10)

# initialize pyaudio
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, flag):
    global start, stop, a, b, zi

    # sound is read from array for faster performances
    sound = audio_data[start:stop] * calibration_gain
    sound = boneConductingFilter(sound, a, b, zi)

    start += CHUNK
    stop += CHUNK

    return audio2byte(sound), pyaudio.paContinue


# sonification sound: Siri's voice
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
FORMAT = 8                                                                      # int16
CHANNELS = 1
RATE = 44100

# start and stop indexes for sonification reading
start = 0
stop = CHUNK

# store sonification samples in array
audio_data, _ = audioread(sonification_path, RATE)
if test:
    audio_data = addPadding(audio_data, RATE, 10)
audio_data = audio_data / 2 ** 15

# bone conducting filter
a = [[1, -1.98717373, 0.98724634], [1, -1.96713666, 0.96794841],
     [1, -1.38158037, 0.49588455], [1, -1.01411009, 0.35962489]]
b = [[0.99360502, -1.98721004, 0.99360502], [0.99685135, -1.96713666, 0.97109706],
     [0.86998166, -1.38158037, 0.62590289], [1.55411604, -1.91366498, 0.70506374]]
zi = np.zeros((4, 2))

# Open a stream object to write the WAV file to
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                frames_per_buffer=CHUNK,
                output=True,
                stream_callback=callback)

# start stream
stream.start_stream()

# Play the sound by writing the audio data to the stream
while stream.is_active():
    print("start")
    time.sleep(getDuration(sonification_path) + 15)
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
