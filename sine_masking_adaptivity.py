from functions import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate

# ambience sound path: Urban Sound dataset has been used to simulate enviromental sounds
path_noise = 'D:/Gianluca/Università/Magistrale/Dataset/UrbanSound/data/'
sound_type = 'air_conditioner'
file_name = '47160.wav'

# audio reading
amb_sound, fs = audioread(path_noise + sound_type + '/' + file_name)

# setup parameters
n_bit = 16                                      # number of quantization bits
n_samples = len(amb_sound)                      # number of samples of the ambience sound
dur = n_samples / fs                            # ambience sound duration in seconds
dt = 1 / fs                                     # smallest step, period

# ambience sound normalization
max_sample = 2 ** (n_bit - 1)
amb_sound = amb_sound / max_sample

# sine tone
A = 0.5                                         # sound wave amplitude
freq = 440                                      # sound wave frequency in Hz
t = np.linspace(0, dur, n_samples)              # time vector
tone = A * np.sin(2 * np.pi * freq * t)         # sine tone

# Masking adaptivity

# setup parameters for STFT
frame_size = 1024
hop_size = frame_size // 4
win = np.hanning(frame_size)

# signal splitting into columns
buf_sound = buffer(tone, frame_size, hop_size)
buf_noise = buffer(amb_sound, frame_size, hop_size)

# signal windowing, hanning window
win_signal = buf_sound * win[:, None]
win_noise = buf_noise * win[:, None]

# Short Time Fourier Trasform --> frequency domain
Signal, f_signal, t_signal = stft(win_signal, hop_size, fs)
Noise, f_noise, t_noise = stft(win_noise, hop_size, fs)

# filter banks and central frequencies
scale = 'erb'                                                       # frequency scale
fb_signal, cnt_signal = getFilterBank(f_signal, scale=scale)        # filter bank for signal
fb_noise, cnt_noise = getFilterBank(f_noise, scale=scale)           # filter bank for noise

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
snr_target = 5                                                      # desired SNR
n_bands, n_frames = signal_bands.shape                              # number of bands and number of frames
limits = [0.9, 4.0]                                                 # limits for the modulation factor

k = set_snr(signal_bands, noise_bands, snr_target, limits)  # compute modulation factor matrix

# interpolation
k_interp = pchip_interpolate(cnt_hz_s, k, f_signal)

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
signal_eq = unbuffer(signal_eq, hop_size, w=win**2, ln=tone.size)

# playback the equalized audio signal plus the noisy signal
noisy_signal = signal_eq + amb_sound
sound(noisy_signal, fs)
