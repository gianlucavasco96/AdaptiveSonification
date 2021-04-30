import time

import pyaudio
import wave
from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, flag):
    global start, stop

    # sound is read from array for faster performances
    sound = audio_data[start:stop]

    start += CHUNK
    stop += CHUNK

    return audio2byte(sound), pyaudio.paContinue


# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/piano.wav'
# sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/sonifications/voice.wav'

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
audio_data = audio_data / 2 ** 15

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
    time.sleep(getDuration(sonification_path))
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
