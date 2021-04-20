from functions import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram, welch

# ambience sound path: Urban Sound dataset has been used to simulate enviromental sounds
noise_path = 'D:/Gianluca/Università/Magistrale/Dataset/UrbanSound/data/'
sound_type = 'air_conditioner'
noise_name = '47160.wav'

# sonification sound: piano samples, C major scale
sound_path = 'D:/Gianluca/Università/Magistrale/Tesi/'
sound_name = 'piano.wav'

# audio reading
amb_sound, fs = audioread(noise_path + sound_type + '/' + noise_name)
sonification, sr = audioread(sound_path + sound_name)

# setup parameters
n_bit = 16                                      # number of quantization bits
n_samples = len(amb_sound)                      # number of samples of the ambience sound
max_sample = 2 ** (n_bit - 1)                   # biggest sample
dur = n_samples / fs                            # ambience sound duration in seconds
dt = 1 / fs                                     # smallest step, period

# sound normalization
amb_sound = amb_sound / max_sample
sonification = sonification / max_sample

# signals adjusting: they must have the same length
sonification, amb_sound = setSameLength(sonification, amb_sound)

# the noisy signal is the simple sum of the sonification sound and the ambience sound
noisy_signal = sonification + amb_sound

# spectrum plot
plt.figure()
plt.subplot(121)
f, t, Sxx = spectrogram(noisy_signal, fs, nperseg=int(fs))
pic = plt.pcolormesh(t, f, np.log10(Sxx), shading='gouraud')
plt.colorbar(pic)
plt.yscale('symlog')
plt.ylabel('Log Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of the noisy signal')

plt.subplot(122)

# power spectrum
f1, P1 = welch(amb_sound, fs, 'flattop', 1024, scaling='spectrum')
f2, P2 = welch(sonification, fs, 'flattop', 1024, scaling='spectrum')

plt.semilogy(f1, np.sqrt(P1), label='ambience')
plt.xscale('symlog')
plt.semilogy(f2, np.sqrt(P2), label='sonification')
plt.xscale('symlog')
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum')
plt.legend()

# full screen
plt.get_current_fig_manager().full_screen_toggle()
plt.show()

# playback the equalized audio signal plus the noisy signal
sound(noisy_signal, fs)
