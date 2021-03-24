from functions import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.io.wavfile import write

# ambience sound path: Urban Sound dataset has been used to simulate enviromental sounds
noise_path = 'D:/Gianluca/Università/Magistrale/Dataset/UrbanSound/data/'
sound_type = 'drilling'
noise_name = '71529.wav'

# sonification sound: piano samples, C major scale
sound_path = 'D:/Gianluca/Università/Magistrale/Tesi/'
sound_name = 'piano.wav'

# audio reading
sonification, fs = audioread(sound_path + sound_name)
amb_sound, fs = audioread(noise_path + sound_type + '/' + noise_name, fs=fs)

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
sonification, amb_sound = set_same_length(sonification, amb_sound)

# set the same loudness, with rms
sonification = sonification * rms_energy(amb_sound) / rms_energy(sonification)

# Masking adaptivity

# setup parameters for STFT
frame_size = 1024
hop_size = frame_size // 4
win = np.hanning(frame_size)

# signal splitting into columns
buf_signal = buffer(sonification, frame_size, hop_size)
buf_noise = buffer(amb_sound, frame_size, hop_size)

# signal windowing, hanning window
win_signal = buf_signal * win[:, None]
win_noise = buf_noise * win[:, None]

# Short Time Fourier Trasform --> frequency domain
Signal, f_signal, t_signal = stft(win_signal, hop_size, fs)
Noise, f_noise, t_noise = stft(win_noise, hop_size, fs)

# filter banks and central frequencies
scale = 'erb'                                                         # frequency scale
fb_signal, cnt_signal = getFilterBank(f_signal, scale=scale)          # filter bank for signal
fb_noise, cnt_noise = getFilterBank(f_noise, scale=scale)             # filter bank for noise

# spectral filtering
signal_bands = applyFilterBank(abs(Signal), fb_signal)
noise_bands = applyFilterBank(abs(Noise), fb_noise)

# maximum dB values for plot comparison
max_db_signal = maximum_db(Signal, signal_bands)
max_db_noise = maximum_db(Noise, noise_bands)

# plot spectra
plt.subplot(221)
plot_spectrum(Signal, f_signal, t_signal, title='signal spectrum', max_db=max_db_signal)
plt.subplot(222)
plot_spectrum(Noise, f_noise, t_noise, title='noise spectrum', max_db=max_db_noise)

# band splitted signal plots
plt.subplot(223)
plot_spectrum(signal_bands, f_signal, t_signal, title='band splitted signal', max_db=max_db_signal, freq_scale=scale)
plt.subplot(224)
plot_spectrum(noise_bands, f_noise, t_noise, title='band splitted noise', max_db=max_db_noise, freq_scale=scale)
plt.show()

# convert center frequencies into Hz
cnt_hz_s = scale2f(cnt_signal, scale)
cnt_hz_n = scale2f(cnt_noise, scale)

# computation of modulation factors
snr_target = 2                                                      # desired SNR
n_bands, n_frames = signal_bands.shape                              # number of bands and number of frames
limits = [0.2, 4.0]                                                 # limits for the modulation factor
min_thres = db2amp(-30)                                             # minimum threshold not to emphasize irrelevant freq

k = set_snr_matrix(signal_bands, noise_bands, snr_target, limits, min_thres)   # compute modulation factor matrix
k, cnt_hz_s = add_limit_bands(k, cnt_hz_s, f_signal)    # add the lower and the upper bands to avoid under/overshooting

# interpolation
k_interp = pchip_interpolate(cnt_hz_s, k, f_signal)

# interpolated equalization mask plot
plot_spectrum(k_interp, f_signal, t_signal, title='equalization mask', min_db=None)
plt.show()

# signal spectrum equalization
Signal_eq = Signal * k_interp

# plot new spectrum
max_db = maximum_db(Signal, Signal_eq)
plt.subplot(121)
plot_spectrum(Signal, f_signal, t_signal, title='original signal spectrum', max_db=max_db)
plt.subplot(122)
plot_spectrum(Signal_eq, f_signal, t_signal, title='equalized signal spectrum', max_db=max_db)
plt.show()

# inverse STFT to get equalized audio signal
signal_eq = istft(Signal_eq)

# equalized signal is windowed again with hanning window, to remove modulation artifacts
signal_eq = signal_eq * win[:, None]

# unbuffer the matrix to obtain the audio signal vector
signal_eq = unbuffer(signal_eq, hop_size, w=win**2, ln=sonification.size)

# playback the equalized audio signal plus the noisy signal
noisy_signal = signal_eq + amb_sound
sound(noisy_signal, fs)

# write the audio file
# path = "D:/Gianluca/Università/Magistrale/Tesi/test/drilling/solo attenuazione/audio/"
# write(path + "piano.wav", fs, (max_sample * sonification).astype(np.int16))
# write(path + "piano_eq.wav", fs, (max_sample * signal_eq).astype(np.int16))
# write(path + "both_signals.wav", fs, (max_sample * noisy_signal).astype(np.int16))
