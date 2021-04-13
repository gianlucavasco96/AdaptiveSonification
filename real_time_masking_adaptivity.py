from pyo import *

from functions import *

s = Server().boot()

# sonification sound: piano samples, C major scale
sonification_path = 'D:/Gianluca/Università/Magistrale/Tesi/piano_mono.wav'
dur = get_duration(sonification_path)

sound = SfPlayer(sonification_path)                             # soundfile player

# Recorded noise
noise = Input().play().mix(2).out(dur=dur)

# frequency axis
frame_size = 512
fs = int(sound.getSamplingRate())
freq = np.fft.rfftfreq(frame_size, 1/fs)

# FFT
Sound = FFT(sound, frame_size, overlaps=4, wintype=2)           # wintype=2 is Hanning window
Noise = FFT(noise, frame_size, overlaps=4, wintype=2)

# Magnitudes
pol_sound = CarToPol(Sound['real'], Sound['imag'])
pol_noise = CarToPol(Noise['real'], Noise['imag'])

mag_sound = pol_sound['mag']
mag_noise = pol_noise['mag']

# modulation factor
gain = Sig(2.0)
k = (gain * mag_noise / (mag_sound + eps)).range(0.2, 4.0)
k = AToDB(k)

# filter banks and central frequencies
scale = 'erb'                                                   # frequency scale
nband = 5
fb, cnt_scale = getFilterBank(freq, scale=scale, nband=nband)   # filter bank for signal
cnt_hz = scale2f(cnt_scale, scale)                              # convert center frequencies into Hz
bw_hz = get_bandwidth(cnt_hz)                                   # get the corrispondent bandwidths

# signal split in multiple frequency bands
q = list(cnt_hz/bw_hz)                                          # compute q factors
signal_bands = BandSplit(sound, num=nband, max=fs/2, q=q)
noise_bands = BandSplit(noise, num=nband, max=fs/2, q=q)

# equalizer
eq = EQ(sound, freq=list(cnt_hz), q=q, boost=k).out()

s.gui(locals())
