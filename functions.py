import numpy as np
import sounddevice as sd
import scipy as sp
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from pydub import AudioSegment

eps = 10e-9                                         # dummy variable to avoid divide by zero errors


def amp2db(amp, thres=-np.inf):
    """This function converts amplitude values into dB values"""

    amp = np.abs(amp) + eps                         # make sure that amp is strictly positive
    t = db2amp(thres)
    db = 20 * np.log10(np.maximum(t, amp))          # compute dB value

    return db


def applyFilterBank(psd, fbank):
    """This function reduces psd to n lines, given n filters in fbank"""

    fltc = lambda ps, fb: [np.sum(ps * fb[:, n]) for n in range(fb.shape[1])]

    return np.asarray([fltc(psd[:, c], fbank) for c in range(psd.shape[1])]).T


def audioread(file_path):
    """This function reads the input file as an audio file, using the AudioSegment library"""

    audio_format = file_path[-3:]

    if audio_format == 'wav':
        aud_seg = AudioSegment.from_wav(file_path)      # read the wav file
    else:
        aud_seg = AudioSegment.from_mp3(file_path)

    fs = aud_seg.frame_rate                             # sampling frequency
    aud_seg = aud_seg.set_channels(1)                   # conversion to mono
    samples = aud_seg.get_array_of_samples()            # get the array of samples

    return np.asarray(samples), fs


def bark2f(bark):
    """This function converts Bark scale values into frequency values"""

    if isinstance(bark, np.ndarray):                    # if bark is an array
        for i in range(len(bark)):
            if bark[i] < 2:
                bark[i] = (bark[i] - 0.3) / 0.85
            if bark[i] > 20.1:
                bark[i] = (bark[i] + 4.422) / 1.22
    else:
        if bark < 2:
            bark = (bark - 0.3) / 0.85
        if bark > 20.1:
            bark = (bark + 4.422) / 1.22

    f = 1960 * (bark + 0.53) / (26.28 - bark)

    # return 600 * np.sinh(bark/6)
    return f


def buffer(signal, framesize, hopsize):
    """This function splits an array into columns"""

    if len(signal) % framesize != 0:                    # if signal length is not a framesize multiple
        xpad = np.append(signal, np.zeros(framesize))   # the padded signal is the signal itself plus a 0s vector
    else:
        xpad = signal                                   # otherwise, the padded signal is the signal itself

    return np.asarray([xpad[i:i + framesize] for i in range(0, len(xpad) - framesize, hopsize)]).T


def db2amp(db):
    """This funtion converts dB values into amplitude values"""

    amp = 10 ** (db / 20)

    return amp


def draw_slope(fs):
    """This function computes the slope of the noise signal"""

    slope = np.linspace(0.1, 1, 3 * fs)
    slope = np.append(slope, np.ones(2 * fs))
    slope = np.append(slope, np.linspace(1, 0.6, fs))
    slope = np.append(slope, np.ones(2 * fs) - 0.4)
    slope = np.append(slope, np.linspace(0.6, 0.1, 2 * fs))

    return slope


def envelope(x, method='rms', param=512):
    """Extract envelope using 'rms','smooth','peak' or 'hilbert"""

    def pk(sig, r):
        y = np.zeros(sig.shape)
        rel = np.exp(-1 / r)
        y[0] = np.abs(sig[0])
        for n, e in enumerate(np.abs(sig[1:])):
            y[n + 1] = np.maximum(e, y[n] * rel)
        return y

    def rms(sig, w, mode):
        return np.sqrt(np.convolve(np.power(sig, 2), w / np.sum(w), mode))  # Not very efficient...

    m = {
        'rms': lambda sig, par: rms(x, np.ones(param), 'full')[:1 - param],
        'smooth': lambda sig, par: rms(x, np.hanning(param), 'same'),
        'peak': pk,
        'hilbert': lambda sig, par: np.abs(sp.signal.hilbert(x))
    }.get(method.lower())
    return m(x, param)


def erb2f(erb):
    """This function converts ERB scale values into frequency values"""

    # f = (100000 * (10 * erb - 247)) / 107939                              # linear approximation
    # f = (500/623) * (np.sqrt(249200 * erb + 80109737) - 9339)             # polynomial approximation

    A = 1000 * np.log(10) / (24.7 * 4.37)
    f = (10 ** (erb / A) - 1) / 0.00437

    return f


def f2bark(f):
    """This function converts frequency values into Bark scale values"""

    bark = (26.81 * f / (1960 + f)) - 0.53

    if isinstance(bark, np.ndarray):                            # if bark is an array
        for i in range(len(bark)):
            if bark[i] < 2:
                bark[i] += 0.15 * (2 - bark[i])
            if bark[i] > 20.1:
                bark[i] += 0.22 * (bark[i] - 20.1)
    else:                                                       # if bark is a scalar
        if bark < 2:
            bark += 0.15 * (2 - bark)
        if bark > 20.1:
            bark += 0.22 * (bark - 20.1)

    # return 6 * np.arcsinh(f / 600)
    return bark


def f2erb(f):
    """This function converts frequency values into ERB scale values"""

    # erb = 24.7 * (4.37 * f / 1000 + 1)                                    # linear approximation
    # erb = 6.23 * (f/1000) ** 2 + 93.39 * (f/1000) + 28.52                   # polynomial approximation

    A = 1000 * np.log(10) / (24.7 * 4.37)
    erb = A * np.log10(1 + f * 0.00437)

    return erb


def f2mel(f):
    """This function converts frequency values into Mel scale values"""

    return 2595.0 * np.log10(1 + f / 700.0)


def f2scale(signal, scale):
    """This function converts the input signal from the input scale to frequencies"""

    f2x = {
        'mel': f2mel,
        'bark': f2bark,
        'st': linf2logf,
        'erb': f2erb
    }.get(scale.lower())

    return f2x(signal)


def fade(x, leng, typ='inout', shape=2):
    """This function applies fade-in, fade-out or both (typ =('in','out','inout')),
    of length 'leng' and exponent 'shape' """

    typ = typ.lower()

    if typ == 'in' or typ == 'inout':
        x[0:leng] = x[0:leng] * (np.linspace(0, 1, leng)**shape)
    if typ == 'out' or typ == 'inout':
        x[-leng:] = x[-leng:] * (np.linspace(1, 0, leng)**shape)

    return x


def getFilterBank(F, bw=100, scale='mel', wfunc=np.bartlett):
    """This function returns a matrix of overlapping masks of size bw*2+1 and center freq. array
    Masks have wfunc shape and are equally spaced in scale-space
    scale can be: 'mel', 'bark', 'st', 'erb' """

    findx = lambda array, x: (np.abs(array - x)).argmin()
    f2x, x2f = {
        'mel': (f2mel, mel2f),
        'bark': (f2bark, bark2f),
        'st': (linf2logf, logf2linf),
        'erb': (f2erb, erb2f)
    }.get(scale.lower())
    mstrt = f2x(F[0]) if F[0] > 0 else f2x(F[1])
    mstop = f2x(F[-1])
    bw = f2x(bw)
    nband = int((mstop - mstrt) // bw - 1)
    cnt = np.linspace(1, nband, nband) * bw
    low = x2f(cnt - bw)
    hig = x2f(cnt + bw)
    # cnt = x2f(cnt)
    fbank = np.zeros((len(F), nband))
    for b in range(nband):
        il = findx(F, low[b])
        ih = findx(F, hig[b])
        fbank[il:ih, b] = wfunc(ih - il)
    return fbank, cnt


def get_modulation(signal, noise, gain, limits=None):
    """This function computes the modulation factor that has to be applied to the sonification signal"""

    env_sig = envelope(signal, 'rms')               # get the mobile RMS envelope of signal
    env_noise = envelope(noise, 'rms')              # get the mobile RMS envelope of noise

    modulation = gain * env_noise / env_sig         # modulation factor: difference in dB is division in amplitude

    if limits is not None:
        min_value = limits[0]                       # set minimum modulation value
        max_value = limits[1]                       # set maximum modulation value

        for i in range(len(modulation)):
            if modulation[i] < min_value:           # if modulation factor is too small
                modulation[i] = min_value           # limit its value to minimum value, in order not to silence signal
            if modulation[i] > max_value:           # if modulation factor is too big
                modulation[i] = max_value           # limit its value to maximum value, in order to avoid clipping

    return modulation


def interpolate(new_x, old_x, y):
    """This function computes the interpolation of a vector y on a new axis new_x"""

    f = interp1d(old_x, y, axis=0, kind='linear')

    return f(new_x)


def istft(X):
    """This function computes the real iFFT along columns"""

    return np.fft.irfft(X, axis=0)


def linf2logf(f, fref=440, nref=69, edo=12):
    """This function takes frequency values expressed in linear Hz to a log scale"""
    # fref=440, nref=69, edo=12 for typical MIDI nn setup

    return edo * np.log2(f / fref + eps) + nref


def logf2linf(nn, fref=440, nref=69, edo=12):
    """This function takes frequency values expressed in a log scale to linear Hz"""
    # fref=440, nref=69, edo=12 for typical MIDI nn setup

    return np.exp2((nn - nref) / edo) * fref


def maximum_db(X, Y):
    """This function computes the maximum value in dB of the two input spectra"""

    # dB convertion
    X = amp2db(abs(X))
    Y = amp2db(abs(Y))

    # find the maxima of the matrices
    max_x = np.max(X)
    max_y = np.max(Y)

    return max(max_x, max_y)


def mel2f(mel):
    """This function converts Mel scale values into frequency values"""

    return 700.0 * ((10.0 ** (np.asarray(mel) / 2595)) - 1)


def plot_spectrum(S, f, t, title=None, max_db=None, freq_scale='Hz'):
    """This function plots the spectrum of the computed STFT, as an image"""

    data = amp2db(abs(S))

    if freq_scale != 'Hz':
        f = f2scale(f, freq_scale)

    plt.imshow(data, extent=[t[0], t[-1], f[0], f[-1]], aspect='auto', origin='lower', vmin=-48, vmax=max_db)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (' + freq_scale + ')')
    plt.colorbar()
    plt.tight_layout()

    if title is not None:
        plt.title(title)

    plt.draw()


def rms_energy(signal):
    """This function computes the root mean square energy of the input signal"""

    return np.sqrt(np.mean(signal ** 2))


def scale2f(signal, scale):
    """This function converts the input signal from the input scale to frequencies"""

    x2f = {
        'mel': mel2f,
        'bark': bark2f,
        'st': logf2linf,
        'erb': erb2f
    }.get(scale.lower())

    return x2f(signal)


def set_same_length(signal, noise):
    """This function compares the two input signals and modifies them, such that they have the same length"""

    s_len = len(signal)
    n_len = len(noise)

    if s_len == n_len:                                      # if signals have the same length
        return signal, noise                                # return them, without changes
    elif s_len > n_len:                                     # if signal is longer than noise
        signal = signal[:n_len]                             # signal is cut, such that it matches the noise length
    else:                                                   # if noise is longer than signal
        diff = n_len - s_len                                # compute the samples difference
        signal = np.append(signal, np.zeros((diff,)))       # append to signal as many zeros as the samples difference

    return signal, noise


def set_snr(signal, noise, snr_target, limits=None, onlyk=False):
    """This function takes in input a clear signal and a noise signal and modifies the clear one so that the signal
    to noise ratio is the one specified by the 3rd parameter"""

    snr_real = snr(signal, noise)                   # compute real signal to noise ratio
    k = snr_target / (snr_real + eps)                 # compute the ratio between the target and the real snr

    if limits is not None:
        min_value = limits[0]                       # set minimum modulation value
        max_value = limits[1]                       # set maximum modulation value

        if k < min_value:                           # if modulation factor is too small
            k = min_value                           # limit its value to minimum value, in order not to silence signal
        if k > max_value:                           # if modulation factor is too big
            k = max_value                           # limit its value to maximum value, in order to avoid clipping

    if onlyk:
        return k
    else:
        mod_signal = signal * k                      # apply the corrective factor to the clear signal
        print('New SNR is ' + str(snr(mod_signal, noise)) + ' dB')
        return mod_signal


def snr(x, y):
    """This function computes the signal to noise ratio between the two input signals x and y"""

    return rms_energy(x) / (rms_energy(y) + eps)


def stft(x, hopsize=None, fs=None):
    """This function computes the real FFT along columns"""

    wsize, n = x.shape
    X = np.fft.rfft(x, axis=0)
    if fs is not None:
        F = np.fft.rfftfreq(wsize, 1.0 / fs)
        T = np.linspace(0, n-1, n) * hopsize / fs
        return X, F, T
    else:
        return X


def sound(signal, fs):
    """This function plays the input signal vector at the given sampling frequency"""

    signal = signal / max(abs(signal))        # normalization

    sd.play(signal, fs)
    sd.wait()


def surface_graph(S, f, t):
    data = amp2db(abs(S))

    fig = go.Figure(data=[go.Surface(x=t, y=f, z=data, cmin=-48, cmax=np.max(data), colorscale='Greens')])
    fig.update_layout(title='signal spectrum surface')
    fig.layout.scene.xaxis.title = 'time frames'
    fig.layout.scene.yaxis.title = 'frequency bands'
    fig.layout.scene.yaxis.type = 'log'
    fig.layout.scene.zaxis.title = 'dB energy'
    fig.show()


def unbuffer(buf, hopsize, w=1, ln=-1, fades=True):
    """This function overlaps and adds columns
    w is the window function and ln is the final length"""

    framesize, n = buf.shape
    l = framesize + hopsize * (n-1)
    x = np.zeros(l)
    e = np.zeros(l)
    for n, i in enumerate(range(0, 1 + l - framesize, hopsize)):
        x[i:i + framesize] += buf[:, n]
        e[i:i + framesize] += w
    e[e == 0] = 0.1
    x = x[:ln]
    e = e[:ln]
    if fades:
        return fade(x/e, hopsize)
    else:
        return x/e


def wgn(A, n_samples, freq, fs, filt=False):
    """This function creates a white gaussian noise signal and filters it with a butterworth bandstop filter if the
    "filt" flag is set to True"""

    noise = np.random.normal(0, 1, n_samples)
    noise = A * noise / np.max(np.abs(noise))       # normalization and same amplitude as the sine tone

    if filt:
        nyquist = 0.5 * fs                          # nyquist frequency is half the sample rate
        order = 3
        low_cut = (freq - 50) / nyquist             # low cut frequency normalization
        high_cut = (freq + 50) / nyquist            # high cut frequency normalization

        # noise filtering
        bandstop = butter(order, [low_cut, high_cut], 'bandstop', analog=False)
        noise = lfilter(bandstop[0], bandstop[1], noise)

    return noise


def window_idx(i, w_size, max_length):
    """This function computes the start and the stop indexes of the windowed signals"""

    start = int(i * w_size)                         # set the start point
    stop = int((i + 1) * w_size)                    # set the stop point

    if stop > max_length:                           # if the stop point exceeds the maximum length
        stop = max_length                           # it is set to the last sample

    return start, stop
