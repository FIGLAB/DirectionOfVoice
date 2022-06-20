from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
import scipy.io.wavfile as wf
from scipy.signal import welch
from scipy.signal import correlate
from scipy import signal, fftpack
import scipy.stats as stats
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fftpack import fft
import scipy
from srmrpy import srmr

SOUND_SPEED = 343.2
MIC_DISTANCE_4 = 0.08127
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)
 
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def dbfft(x, fs,segment_size = 1024):
    if max(x) > 1:
        x = x / 32768.0 
        # x = x / np.max(x)
    noverlap = segment_size / 2
    f, Pxx = welch(x,                           # signal
                      fs=fs,                    # sample rate
                      nperseg=segment_size,     # segment size
                      window='hanning',         # window type to use
                      nfft=segment_size,        # num. of samples in FFT
                      detrend=False,            # remove DC part
                      scaling='spectrum',       # return power spectrum [V^2]
                      noverlap=0)        # overlap between segments

    ref = (1/np.sqrt(2)**2)
    p = 10 * np.log10(Pxx/ref)
    return f, p

def calc_fft(signal,fs_rate):
    N = signal.shape[0]
    secs = N / float(fs_rate)
    Ts = 1.0/fs_rate 
    t = scipy.arange(0, secs, Ts) 
    FFT = abs(scipy.fft(signal))
    FFT_side = FFT[range(N//2)] 
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] 
    fft_freqs_side = np.array(freqs_side)
    return fft_freqs_side, FFT_side

def get_max_correlation(original, match):
    # Shift though convolution
    z = signal.fftconvolve(original, match[::-1])
    lags = np.arange(z.size) - (match.size - 1)
    return ( lags[np.argmax(np.abs(z))] )

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
    
    return shift, cc

def time_shift_correlation(sig, refsig, fs, p_len):

    n = len(sig)

    cc = correlate(refsig, sig, mode='same') #/ np.sqrt(correlate(sig, sig, mode='same')[int(n/2)] * correlate(refsig, refsig, mode='same')[int(n/2)])
    delay_arr = np.linspace(-0.5*n/fs, 0.5*n/fs, n)
    delay = delay_arr[np.argmax(cc)]

    max_shift = int(fs * MAX_TDOA_4) * p_len
    cc = cc[len(cc)//2 - max_shift:len(cc)//2 + max_shift + 1]

    return delay, cc

def get_peason_correlation(sig, refsig):
    r, p = stats.pearsonr(sig, refsig)
    return r, p

def get_csd(sig1, sig2, fs):
    f, Pxy = signal.csd(sig1, sig2, fs)
    Pxy = np.abs(Pxy)
    return f, Pxy
    
def create_chunks(l, n):
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))

def featurize_signals(sig1, sig2, fs):
    
    gcc_shift, gcc = gcc_phat(sig1, sig2, fs, max_tau=MAX_TDOA_4, interp=1)
    
    gcc = np.abs(gcc)
    
    f, Pxy = get_csd(sig1, sig2, fs)
    
    Pxy = np.abs(Pxy)
    
    return np.abs(gcc_shift), np.array(gcc), np.array(Pxy)


def simple_cross_corr_feat(sig1, sig2, fs, pad_len):

    gcc_shift, gcc =  time_shift_correlation(sig1, sig2, fs, pad_len)

    return np.array(np.abs(gcc))
    

def feat_sig(sig1, sig2, fs, chunks = None):
    if chunks is None:
        return featurize_signals(sig1, sig2, fs)
    else:
        sig_list_1 = create_chunks(sig1, chunks)
        sig_list_2 = create_chunks(sig2, chunks)
        gcc_shift_arr = []
        gcc_arr = []
        csd_arr = []
        for i in range(len(sig_list_1)):
            sig1 = sig_list_1[i]
            sig2 = sig_list_2[i]
            if len(sig1) != chunks:
                continue
            gcc_shift, gcc, csd = featurize_signals(sig1, sig2, fs)
            gcc_shift_arr.append(gcc_shift)
            gcc_arr.append(gcc)
            csd_arr.append(csd)
        
        return np.mean(gcc_shift_arr,axis=0), np.mean(gcc_arr,axis=0), np.mean(csd_arr,axis=0)


def features_per_wav_channel(wav_files, fs):
    output = []
    for wav_file in wav_files:
        output.append(get_amp_feat(wav_file,fs))
    return output

def get_amp_feat(wav_file,fs):
    # f, p = dbfft(wav_file, fs,segment_size = 128)
    # f, p = calc_fft(wav_file, fs)
    f, p = get_fft_values(wav_file, 1/fs, wav_file.shape[0], fs)

    split_index = next(x[0] for x in enumerate(f) if x[1] > 5000)
    lower_freq_p = sum(p[:split_index])
    higher_freq_p = sum(p[split_index:])
    ratio = higher_freq_p/lower_freq_p
    curve_coef1 = np.polyfit(p, f, 3).tolist()
    curve_coef2 = np.polyfit(p, f, 1).tolist()
    output = [lower_freq_p] + [higher_freq_p] + [ratio] + curve_coef2 + curve_coef1
    return output

def autocorr_echo_feat(signal,fs):
    gcc_shift, gcc =  time_shift_correlation(signal, signal, fs, 100)
    gcc = gcc.tolist()
    gcc = gcc[:len(gcc)//2 - 200] + gcc[len(gcc)//2 + 200:]
    return np.abs(gcc)

def get_reverb_feat(path, filename, fs):
    wav = []
    for i in [0, 1, 2, 3, 4]:
        rate, raw_signal = wf.read(path + str(filename)+"_"+str(i)+".wav")
        if len(raw_signal.shape) == 2:
            raw_signal = raw_signal.T[0]
        raw_signal = raw_signal/32768.0
        wav.append(raw_signal)
        
    wav_channel = wav[1:]

    reverb_ratio = []

    for sig in wav_channel:
        ratio, energy = srmr(sig,fs)
        reverb_ratio.append(ratio)

    reverb_sl = stat_feat(reverb_ratio)

    r0, e0 = srmr(wav[0],fs)

    return reverb_sl + [r0]

          
def features_for_mic_group(path, filename, fs, mic_group, chunk_size, avg_channels):
    wav = []
    xcorr_gcc_list = []
    shift_list = []
    gcc_list = []
    csd_list = []
    
    for i in [0, 1, 2, 3, 4]:
        rate, raw_signal = wf.read(path + str(filename)+"_"+str(i)+".wav")
        if len(raw_signal.shape) == 2:
            raw_signal = raw_signal.T[0]
        raw_signal = raw_signal/32768.0
        # raw_signal = raw_signal/np.max(raw_signal)
        wav.append(raw_signal)
        
    wav_channel = wav[1:]

    reverb_ratio = []

    for sig in wav_channel:
        ratio, energy = srmr(sig,fs)
        reverb_ratio.append(ratio)

    reverb_sl = stat_feat(reverb_ratio)

    r0, e0 = srmr(wav[0],fs)

    reverb_sl = reverb_sl + [r0]
    
    # out_per_wav = features_per_wav_channel(wav_channel, fs)

    out_per_wav = get_amp_feat(wav[0],fs)

    acorr_gcc = autocorr_echo_feat(wav_channel[0],fs)
        
    for i, v in enumerate(mic_group):
        sig1 = wav_channel[v[0]]
        sig2 = wav_channel[v[1]]
        gcc_shift, gcc, csd = feat_sig(sig1, sig2, fs, chunks = chunk_size)
        xcorr_gcc = simple_cross_corr_feat(sig1, sig2, fs, 10)
        xcorr_gcc_list.append(xcorr_gcc)
        shift_list.append(gcc_shift)
        gcc_list.append(gcc)
        csd_list.append(csd)

    autocorr_feat = stat_autocorr(acorr_gcc)

    stat_sl = stat_feat(shift_list)
    stat_gcc_sum = stat_feat(gcc_list,mode='sum')
    stat_gcc_max = stat_feat(gcc_list,mode='max')
    stat_csd_sum = stat_feat(csd_list,mode='sum')

    xcorr_gcc_list = np.mean(xcorr_gcc_list,axis=0)
    
    extended_fv = stat_sl + stat_gcc_sum + stat_gcc_max + stat_csd_sum + [np.std(xcorr_gcc_list)] + [simps(xcorr_gcc_list,dx=5)] 

    if avg_channels == True:
        gcc_list = np.mean(gcc_list,axis=0)
        
        
    return wav_channel, gcc_list, xcorr_gcc_list, acorr_gcc, out_per_wav, extended_fv, autocorr_feat, reverb_sl


def stat_autocorr(acorr_gcc):

    dy = np.gradient(acorr_gcc)

    peaks, _ = signal.find_peaks(acorr_gcc, height=0)
    vals = acorr_gcc[peaks].tolist()
    vals.sort(reverse=True)
    p1 = np.mean(vals[:2])
    p2 = np.mean(vals[2:10])
    p3 = np.mean(vals[10:15])

    p_ratio = p1/p2
    p_diff1 = p1 - p2
    p_diff2 = p1 - p3 
    a_std = np.std(acorr_gcc)
    dy_sum = np.sum(np.abs(dy))
    dy_std = np.std(dy)
    a_max = np.max(acorr_gcc)
    a_sum = np.sum(acorr_gcc)

    return [p_ratio, p_diff1, p_diff2, a_std, dy_sum, dy_std, a_max, a_sum]

def stat_feat(fvec,mode='sum'):
    fvec = np.array(fvec)
    if len(fvec.shape) > 1:
        if mode == 'sum':
            for l in range(len(fvec)):
                fvec[l] = sum(fvec[l])
        elif mode == 'max':
            for l in range(len(fvec)):
                fvec[l] = np.max(fvec[l])
        fvec = fvec[:,0] 
        
    std = np.std(fvec)
    y_range = np.max(fvec)-np.min(fvec)
    avg = np.mean(fvec)
    out_arr = [std] + [y_range] + [avg]
    return out_arr

def get_peaks(sig,n_peaks=2):
    peaks, _ = signal.find_peaks(sig, height=0)
    peaks = peaks[:n_peaks]
    vals = sig[peaks]
    return np.array(sig[peaks])

def get_concat_fv(features_gcc, features_xcorr, features_per_wav, extended_fv, avg_channels):
        
    out_features_gcc = np.array(features_gcc)
    out_features_xcorr = np.array(features_xcorr)
    features_per_wav = np.array(features_per_wav)
    extended_fv = np.array(extended_fv)

    out_features_gcc = out_features_gcc.flatten()
    out_features_xcorr = out_features_xcorr.flatten()
    features_per_wav = features_per_wav.flatten()
    extended_fv = extended_fv.flatten()
    
    fv = out_features_gcc.tolist() + out_features_xcorr.tolist() + features_per_wav.tolist() + extended_fv.tolist()

    return fv