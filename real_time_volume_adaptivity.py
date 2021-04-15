import pyaudio
import wave
from functions import *

# initialize pyaudio
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, flag):
    # sound
    audio_data = sound_file.readframes(CHUNK)
    sound = np.frombuffer(audio_data, dtype=np.int16)
    sound = sound.astype(np.float64)

    # noise
    noise = np.frombuffer(in_data, dtype=np.int16)
    noise = noise.astype(np.float64)

    # volume adaptivity: SNR is kept constant inside a specified sound intensity range,
    # outside of witch the signal volume is kept constant

    gain = 5  # gain factor: SNR target (amplitude value, not dB)
    limits = [0.2, 1.0]  # limits for the modulation factor

    # get modulation term
    modulation = get_modulation(sound, noise, gain, limits)

    # apply the modulation factor to the signal
    sound = sound * modulation

    y = sound.astype(np.int16)

    return y.tobytes(), pyaudio.paContinue


# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano_mono.wav'     # audio must be mono

# Open the sound file
sound_file = wave.open(sonification_path, 'rb')

# audio settings
CHUNK = 1024
FORMAT = p.get_format_from_width(sound_file.getsampwidth())
CHANNELS = sound_file.getnchannels()
RATE = sound_file.getframerate()

# start and stop indexes for sonification buffer
start = 0
stop = CHUNK

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
    time.sleep(get_duration(sonification_path))
    stream.stop_stream()

# Close stream and terminate pyaudio
stream.close()

p.terminate()
