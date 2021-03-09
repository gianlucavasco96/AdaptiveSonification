from scipy.signal import spectrogram, welch
from functions import *

# ambience sound path: Urban Sound dataset has been used to simulate enviromental sounds
path = 'D:/Gianluca/Università/Magistrale/Dataset/UrbanSound/data/'
sound_type = 'air_conditioner'
file_name = '47160.wav'

# audio reading
amb_sound, fs = audioread(path + sound_type + '/' + file_name)

# setup parameters
n_bit = 16                                      # number of quantization bits
n_samples = len(amb_sound)                      # number of samples of the ambience sound
dur = n_samples / fs                            # ambience sound duration in seconds

# ambience sound normalization
max_sample = 2 ** (n_bit - 1)
amb_sound = amb_sound / max_sample

# sine tone
A = 0.5                                         # sound wave amplitude
freq = 440                                      # sound wave frequency in Hz
t = np.linspace(0, dur, n_samples)              # time vector
tone = A * np.sin(2 * np.pi * freq * t)         # sine tone

# the noisy signal is the simple sum of the sonification sine and the ambience sound
noisy_signal = tone + amb_sound

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
f2, P2 = welch(tone, fs, 'flattop', 1024, scaling='spectrum')

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

sound(noisy_signal, fs)                         # playback for perceptive feedback
