import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
from functions import *

# ambience sound path: Urban Sound dataset has been used to simulate enviromental sounds
path = 'D:/Gianluca/Università/Magistrale/Dataset/UrbanSound/data/'
sound_type = 'air_conditioner'
file_name = '47160.wav'

# audio reading
# sonification sound: piano samples, C major scale
sound_path = 'D:/Gianluca/Università/Magistrale/Tesi/'
sound_name = 'piano.wav'

# audio reading
sonification, fs = audioread(sound_path + sound_name)
amb_sound, _ = audioread(path + sound_type + '/' + file_name, fs)

# setup parameters
n_bit = 16                                      # number of quantization bits
n_samples = len(amb_sound)                      # number of samples of the ambience sound
dur = n_samples / fs                            # ambience sound duration in seconds

# ambience sound normalization
max_sample = 2 ** (n_bit - 1)
sonification = sonification / max_sample
amb_sound = amb_sound / max_sample

# signals adjusting: they must have the same length
sonification, amb_sound = setSameLength(sonification, amb_sound)

# set the same rms
sonification = sonification * rmsEnergy(amb_sound) / rmsEnergy(sonification)

# volume adaptivity: SNR is kept constant inside a specified sound intensity range,
# outside of witch the signal volume is kept constant

gain = 3                                                # gain factor: SNR target (amplitude value, not dB)
limits = [0.2, 2.0]                                     # limits for the modulation factor

# get modulation term
modulation = getModulation(sonification, amb_sound, gain, limits)

adaptive_sonif = sonification * modulation              # apply the modulation factor to the signal
noisy_signal = adaptive_sonif + amb_sound               # sum the adaptive sonification to the noise signal

plt.plot(adaptive_sonif, label='sonification')
plt.plot(amb_sound, label='ambience noise')
plt.plot(modulation, label='modulation factor')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


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
f2, P2 = welch(adaptive_sonif, fs, 'flattop', 1024, scaling='spectrum')

plt.semilogy(f1, np.sqrt(P1), label='ambience')
plt.xscale('symlog')
plt.semilogy(f2, np.sqrt(P2), label='sonification')
plt.xscale('symlog')
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.title('Power spectrum')
plt.legend()

# full screen
# plt.get_current_fig_manager().full_screen_toggle()
plt.show()

sound(noisy_signal, fs)                         # playback for perceptive feedback
