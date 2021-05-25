import random
import plotly.express as pt
import numpy as np
import pandas as pd
import sounddevice as sd
import scipy as sp
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from pydub import AudioSegment

eps = 10e-9                                         # dummy variable to avoid divide by zero errors


def addLimitBands(k, cnt, f):
    """This function adds to (matrix) k two rows or elements, one at the top and one at the bottom, such that k first
    and last frequency values match f vector ones"""

    if k.ndim == 1:
        first_el = k[0]                                 # get the first element of k
        last_el = k[-1]                                 # get the last element of k

        k = np.hstack((first_el, k))                    # append the duplicate of the first element of k in 1st position
        k = np.hstack((k, last_el))                     # append the duplicate of the last element of k in last position
    else:
        first_row = k[0, :]                             # get the first row of k
        last_row = k[-1, :]                             # get the last row of k

        k = np.vstack((first_row, k))                   # append the duplicate of the first row of k in 1st position
        k = np.vstack((k, last_row))                    # append the duplicate of the last row of k in last position

    cnt = np.hstack((f[0], cnt))                        # append first freq value to center freq vector
    cnt = np.hstack((cnt, f[-1]))                       # append last freq value to center freq vector

    return k, cnt


def addPadding(data, fs, time):
    """This function adds time seconds of silence to the input data"""

    padding = np.zeros(fs * time)
    data = np.hstack((padding, data))

    return data


def amp2db(amp, thres=-np.inf):
    """This function converts amplitude values into dB values"""

    amp = np.abs(amp) + eps                         # make sure that amp is strictly positive
    t = db2amp(thres)
    db = 20 * np.log10(np.maximum(t, amp))          # compute dB value

    return db


def applyFilterBank(psd, fbank):
    """This function reduces psd to n lines, given n filters in fbank"""

    fltc = lambda ps, fb: [np.sum(ps * fb[:, n]) for n in range(fb.shape[1])]

    if psd.ndim == 1:
        return np.asarray(fltc(psd, fbank)).T
    else:
        return np.asarray([fltc(psd[:, c], fbank) for c in range(psd.shape[1])]).T


def audio2byte(audio_samples, max_sample=2**15):
    """This function converts audio samples between -1 and 1 into bytes"""

    audio_samples = audio_samples * max_sample
    int_samples = audio_samples.astype(np.int16)

    return int_samples.tobytes()


def audioread(file_path, fs=None):
    """This function reads the input file as an audio file, using the AudioSegment library"""

    aud_format = file_path[-3:]
    aud_seg = AudioSegment.from_file(file_path, aud_format)         # open file and create AudioSegment object

    if fs is not None:
        aud_seg.set_frame_rate(fs)
    else:
        fs = aud_seg.frame_rate                         # sampling frequency

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


def boundInterpolation(k, limits):
    """This function takes a matrix k and makes sure that all of its values are bounded between the input limits"""

    # find undershooting indexes, caused by the fact that np.min(f_signal) < np.min(cnt_hz_s)
    idx_under = np.where(k < limits[0])
    k[idx_under] = limits[0]                                # bound the undershooting value with the lower limit

    # find overshooting indexes, caused by the fact that np.max(f_signal) > np.max(cnt_hz_s)
    idx_over = np.where(k > limits[1])
    k[idx_over] = limits[1]                                 # bound the overshooting value with the upper limit

    return k


def buffer(signal, framesize, hopsize):
    """This function splits an array into columns"""

    if len(signal) % framesize != 0:                    # if signal length is not a framesize multiple
        xpad = np.append(signal, np.zeros(framesize))   # the padded signal is the signal itself plus a 0s vector
    else:
        xpad = signal                                   # otherwise, the padded signal is the signal itself

    return np.asarray([xpad[i:i + framesize] for i in range(0, len(xpad) - framesize, hopsize)]).T


def byte2audio(byte, max_sample=2**15):
    """This function computes audio samples between -1 and 1, starting from a bytes input signal"""

    int_samples = np.frombuffer(byte, dtype=np.int16)
    audio_samples = int_samples / max_sample

    return audio_samples


def db2amp(db):
    """This funtion converts dB values into amplitude values"""

    amp = 10 ** (db / 20)

    return amp


def drawSlope(fs):
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


def erb(f):
    """This function computes the equivalent rectangular bandwidth, approximating the bandwidths of the filters in human
     hearing, using the unrealistic but convenient simplification of modeling the filters as rectangular band-pass
     filters"""

    return 24.7 * (0.00437 * f + 1)


def erb2f(erb):
    """This function converts ERB scale values into frequency values"""

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


def boneConductingFilter(audio, a, b, zi):
    """This function filters the input audio to compensate the bone conducting headphone equalization"""

    n_filt = 4                                                      # number of filters
    for i in range(n_filt):
        audio, zi[i, :] = lfilter(b[i], a[i], audio, zi=zi[i, :])

    return audio * db2amp(2.4)


def getDuration(path):
    """This function returns the duration in seconds of the input audio file"""

    audio, fs = audioread(path)
    duration = len(audio) / fs

    return duration


def getFilterBank(F, nband=12, scale='mel', wfunc=np.bartlett):
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
    bw = (mstop - mstrt) / nband
    cnt = np.linspace(1, nband, nband) * bw
    low = x2f(cnt - bw)
    hig = x2f(cnt + bw)
    fbank = np.zeros((len(F), nband))
    for b in range(nband):
        il = findx(F, low[b])
        ih = findx(F, hig[b])
        fbank[il:ih, b] = wfunc(ih - il)
    return fbank, cnt


def getNoiseMask(bands, start, stop, type='linear'):
    """This function computes the corrective mask to apply to the band splitted noise matrix, in order not to equalize
    the signal (in next steps) where it contains silence"""

    if bands.ndim == 1:
        r = len(bands)                                  # length of band splitted input signal
        mask = np.zeros(r)                              # initialize mask vector with 0s
    else:
        r, c = bands.shape                              # shape of band splitted input signal
        mask = np.zeros((r, c))                         # initialize mask matrix with 0s
    start = db2amp(start)                               # convert start value into amplitude
    stop = db2amp(stop)                                 # convert stop value into amplitude

    idx_between = np.where((bands >= start) & (bands <= stop))  # (a) indexes where bands is between start and stop
    idx_1 = np.where(bands > stop)                      # (b) indexes where bands is more than stop

    if type == 'linear':
        m = 1 / (stop - start)                           # slope of the straght line
        q = - m * start                                  # stoichiometric coefficient
        mask[idx_between] = m * bands[idx_between] + q   # set linear slope for (a)

    if type == 'cos':
        a = (bands[idx_between] - bands[idx_between].min()) / bands[idx_between].max()
        mask[idx_between] = 0.5 * (-np.cos(a * np.pi) + 1)

    mask[idx_1] = 1                                     # set 1 for (b)

    return mask


def getModulation(signal, noise, gain, limits=None):
    """This function computes the modulation factor that has to be applied to the sonification signal"""

    env_sig = envelope(signal, 'rms')                   # get the mobile RMS envelope of signal
    env_noise = envelope(noise, 'rms')                  # get the mobile RMS envelope of noise

    modulation = gain * env_noise / (env_sig + eps)     # modulation factor: difference in dB is division in amplitude

    if limits is not None:
        min_value = limits[0]                           # set minimum modulation value
        max_value = limits[1]                           # set maximum modulation value

        idx_under = np.where(modulation < min_value)    # find indexes of values under the minimum threshold
        idx_over = np.where(modulation > max_value)     # find indexes of values under the maximum threshold

        modulation[idx_under] = min_value               # exchange under threshold values with the minimum
        modulation[idx_over] = max_value                # exchange over threshold values with the maximum

    return modulation


def getRealTimeModulation(sound, noise, gain, previous, limits=None, leng=1024, rate=0.5, offset=db2amp(0)):
    """This function computes the real time modulation factor that has to be applied to the sonification signal"""

    # compute sonification and soundscape rms energy
    rms_sig = rmsEnergy(sound)
    rms_noi = rmsEnergy(noise)

    # initialize the modulation array
    modulation = np.zeros(leng)

    # compute the modulation factor (this time it is a number)
    mod_factor = offset * gain * rms_noi / (rms_sig + eps)

    if limits is not None:
        min_value = limits[0]                                           # set minimum modulation value
        max_value = limits[1]                                           # set maximum modulation value

        if mod_factor < min_value:
            mod_factor = min_value
        if mod_factor > max_value:
            mod_factor = max_value

    # compute ramp and set the remaining samples with the modulation factor
    if previous == -1:                                                  # first iteration
        modulation[:] = mod_factor                                      # there is no previous mod factor, so no ramp
    else:
        # compute stop index of the ramp
        stop = int(leng * rate)

        modulation[:stop] = np.linspace(previous, mod_factor, stop)     # from previous to mod factor in stop samples
        modulation[stop:] = mod_factor                                  # remaining samples are set to mod factor

    return modulation, mod_factor


def interpolate(new_x, old_x, y):
    """This function computes the interpolation of a vector y on a new axis new_x"""

    f = interp1d(old_x, y, axis=0, kind='linear')

    return f(new_x)


def istft(X):
    """This function computes the real iFFT along columns"""

    return np.fft.irfft(X, axis=0)


def limiter(audio):
    """This function prevents audio clipping, rescaling the audio vector"""

    max_value = np.max(np.abs(audio))
    if max_value > 1:
        audio = audio / max_value

    return audio


def linf2logf(f, fref=440, nref=69, edo=12):
    """This function takes frequency values expressed in linear Hz to a log scale"""
    # fref=440, nref=69, edo=12 for typical MIDI nn setup

    return edo * np.log2(f / fref + eps) + nref


def logf2linf(nn, fref=440, nref=69, edo=12):
    """This function takes frequency values expressed in a log scale to linear Hz"""
    # fref=440, nref=69, edo=12 for typical MIDI nn setup

    return np.exp2((nn - nref) / edo) * fref


def maximumDB(X, Y):
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


def plotSpectrum(S, f, t, title=None, min_db=-48, max_db=None, freq_scale='Hz', interp='antialiased'):
    """This function plots the spectrum of the computed STFT, as an image"""

    data = amp2db(abs(S))

    if freq_scale != 'Hz':
        f = f2scale(f, freq_scale)

    plt.imshow(data, extent=[t[0], t[-1], f[0], f[-1]], aspect='auto', origin='lower',
               vmin=min_db, vmax=max_db, interpolation=interp)
    plt.xlabel('time (s)')
    plt.ylabel('frequency (' + freq_scale + ')')
    plt.colorbar()
    plt.tight_layout()

    if title is not None:
        plt.title(title)

    plt.draw()


def printBoxplot(data, labels, column):
    """This function plots the boxplot of the input distribution"""

    if column == 'Parole corrette':
        column = '% Parole corrette'

    plt.boxplot(data, labels=labels, patch_artist=True, flierprops=dict(alpha=.1), showmeans=True)
    plt.ylabel(column)
    plt.show()


def printViolinplot(df, y='Parole corrette'):
    """This function plots the violin plots of the input distributions"""

    fig = pt.violin(df, x='Adattamento', y=y, box=True, range_y=[0, 100] if y == 'Parole corrette' else [1, 5])
    fig.show()

def rmsEnergy(signal):
    """This function computes the root mean square energy of the input signal"""

    if isinstance(signal, np.ndarray):
        return np.sqrt(np.mean(signal ** 2, axis=0))
    else:
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


def setControlVariables(user=1, repetition=1, task=1, test=True):
    """This function initialises the control variables for the real time test"""

    adaptivity = {"nessuno": "real_time_no_adaptivity.py",
                  "volume": "real_time_volume_adaptivity.py",
                  "eq 0": "real_time_masking_adaptivity.py",
                  "eq positivo": "real_time_masking_adaptivity_pos.py"}

    if test:
        path = 'D:/Gianluca/Università/Magistrale/Tesi/test/ordine.xlsx'
        df = pd.read_excel(path)

        if repetition == 1:
            df = df[:16]
            if user in [1, 5, 9, 13]:
                start = 0
            elif user in [2, 6, 10, 14]:
                start = 4
            elif user in [3, 7, 11, 15]:
                start = 8
            else:
                start = 12
        elif repetition == 2:
            df = df[16:32]
            if user in [1, 5, 9, 13]:
                start = 16
            elif user in [2, 6, 10, 14]:
                start = 20
            elif user in [3, 7, 11, 15]:
                start = 24
            else:
                start = 28
        elif repetition == 3:
            df = df[32:48]
            if user in [1, 5, 9, 13]:
                start = 32
            elif user in [2, 6, 10, 14]:
                start = 36
            elif user in [3, 7, 11, 15]:
                start = 40
            else:
                start = 44
        else:
            df = df[48:]
            if user in [1, 5, 9, 13]:
                start = 48
            elif user in [2, 6, 10, 14]:
                start = 52
            elif user in [3, 7, 11, 15]:
                start = 56
            else:
                start = 60

        idx = start + (task - 1)
        if idx >= 16 * repetition:
            idx -= 16

        row = df.loc[idx]
        script = adaptivity[row['Adattamento']]
        fast = row['Velocità']
        sentence = row['# Frase']
        soundscape = row['Soundscape']
    else:
        script = adaptivity['nessuno']
        fast = random.choice([True, False])
        sentence = random.randint(64, 99)
        soundscape = 'nessuno'

    return script, fast, sentence, soundscape


def setSameLength(signal, noise, padding=False):
    """This function compares the two input signals and modifies them, such that they have the same length"""

    s_len = len(signal)
    n_len = len(noise)

    if s_len == n_len:                                      # if signals have the same length
        return signal, noise                                # return them, without changes
    elif s_len > n_len:                                     # if signal is longer than noise
        signal = signal[:n_len]                             # signal is cut, such that it matches the noise length
    else:                                                   # if noise is longer than signal
        if padding:
            diff = n_len - s_len                            # compute the samples difference
            signal = np.append(signal, np.zeros((diff,)))   # append to signal as many zeros as the samples difference
        else:
            noise = noise[:s_len]                               # noise is cut, such that it matches the signal length

    return signal, noise


def setSNR(signal, noise, snr_target, limits=None, offset=db2amp(0)):
    """This function takes in input a clear signal and a noise signal (matrices) and modifies
    the clear one so that the signal to noise ratio is the one specified by the 3rd parameter.
    This function works in frequency domain"""

    # correction mask: it ensures that noise is lower than signal, where signal itself is very low
    noise_mask = getNoiseMask(signal, start=-30, stop=-20, type='linear')
    noise = noise_mask * noise

    # compute the ratio between the snr target and the real
    k = offset * snr_target * np.abs(noise) / (np.abs(signal) + eps)

    if limits is not None:
        min_value = limits[0]                       # set minimum modulation value
        max_value = limits[1]                       # set maximum modulation value

        idx_min = np.where(k < min_value)           # check if any value is less than the minimum limit
        k[idx_min] = min_value                      # set the minumum value for these ones

        idx_max = np.where(k > max_value)           # check if any value is greater than the maximum limit
        k[idx_max] = max_value                      # set the maximum value for these ones

    return k


def setSnrTime(signal, noise, snr_target, limits=None, onlyk=False):
    """This function takes in input a clear signal and a noise signal and modifies the clear one so that the signal
    to noise ratio is the one specified by the 3rd parameter.
    This function works in time domain"""

    snr_real = snr(signal, noise)                   # compute real signal to noise ratio

    if snr_real == 0:                               # if it equals zero
        snr_real += eps                             # add epsilon to avoid zero division

    k = snr_target / snr_real                       # compute the ratio between the target and the real snr

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

    rms_x = rmsEnergy(x)
    rms_y = rmsEnergy(y)
    rms_y = rms_y if rms_y != 0 else rms_y + eps

    return rms_x / rms_y


def shift(array, n):
    """This function left shifts the input array of n position."""
    """Example:
        array: [0, 1, 2, 3]
        n: 1
        --> return [1, 2, 3, 0]"""

    return np.hstack((array[n:], array[:n]))


def starMatrix(result):
    """This function prints the p-value matrix of the post hoc test and the corresponding star matrix"""

    idx_ns = result > .05
    idx_s = (result > .01) & (result <= .05)
    idx_ss = (result > .001) & (result <= .01)
    idx_sss = (result > .0001) & (result <= .001)

    result_star = result.copy()
    result_star[idx_ns] = 'ns'
    result_star[idx_s] = '*'
    result_star[idx_ss] = '**'
    result_star[idx_sss] = '***'

    print(result)
    print()
    print(result_star)


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

    max_sig = np.max(np.abs(signal))            # max absolute value in the signal to play

    if max_sig > 1:
        signal /= max_sig                       # normalization to avoid clipping

    sd.play(signal, fs)
    sd.wait()


def surfaceGraph(S, f, t):
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


def windowIdx(i, w_size, max_length):
    """This function computes the start and the stop indexes of the windowed signals"""

    start = int(i * w_size)                         # set the start point
    stop = int((i + 1) * w_size)                    # set the stop point

    if stop > max_length:                           # if the stop point exceeds the maximum length
        stop = max_length                           # it is set to the last sample

    return start, stop
