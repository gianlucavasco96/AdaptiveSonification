from matplotlib import pyplot as plt
from functions import *

# setup parameters
fs = 44100                                      # sampling frequency in Hz
dur = 10                                        # sound duration in seconds
n_samples = fs * dur                            # number of samples of the sound

# sine tone
A = 0.5                                         # sound wave amplitude
freq = 500                                      # sound wave frequency in Hz
t = np.linspace(0, dur, n_samples)              # time vector
tone = A * np.sin(2 * np.pi * freq * t)         # sine tone

# white gaussian noise
noise = wgn(A, n_samples, freq, fs, filt=False)

# signal to noise ratio
print('Initial SNR is ' + str(snr(tone, noise)) + ' dB')

# intellegibility test
# for i in range(-50, 50, 5):
#     adaptive_sonif = set_snr(tone, noise, i)
#     sound(adaptive_sonif + noise, fs)

# adaptive adjustment: constant SNR
slope = drawSlope(fs)  # compute variable slope of the noise signal
noise = slope * noise                           # apply it to the noise signal

fixed_snr = 10                                  # SNR value we want to keep constant
limits = [0.2, 1.0]                             # limits for the modulation factor
w_size = 0.1 * fs                               # window size
adj_tone = np.asarray([])                       # initialize the empty array of the new sine tone

for i in range(int(n_samples/w_size)):          # for each window
    start = int(i * w_size)                     # set the start point
    stop = int((i+1) * w_size)                  # set the stop point

    win_signal = tone[start:stop]               # consider the windowed signal
    win_noise = noise[start:stop]               # consider the windowed noise

    adj_signal = setSnrTime(win_signal, win_noise, fixed_snr, limits)  # adjust windowed signal amplitude
    adj_tone = np.append(adj_tone, adj_signal)                  # append it to the new sine tone array

noisy_signal = adj_tone + noise                 # sum the new adjusted sine tone and the noise signal

# spectrum plot
plt.magnitude_spectrum(noisy_signal)
plt.show()

sound(adj_tone + noise, fs)                     # playback for perceptive feedback
