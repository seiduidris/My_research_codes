# import numpy as np
# import soundfile as sf
# from scipy.signal import bilinear, lfilter
# import matplotlib.pyplot as plt

# # ---------- A-weighting filter ----------
# def a_weighting(fs):
#     f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
#     w1, w2, w3, w4 = 2 * np.pi * np.array([f1, f2, f3, f4])
#     num = [1, 0, 0, 0, 0]
#     den = np.polymul([1, w1],
#                      np.polymul([1, w2], np.polymul([1, w3],
#                                                     [1, 2 * w4, w4**2])))
#     w0 = 2 * np.pi * 1000.0
#     k = np.abs(np.polyval(den, 1j * w0) / (1j * w0)**4)
#     num = [k, 0, 0, 0, 0]
#     return bilinear(num, den, fs)

# # ---------- Compute A-weighted SPL ----------
# def compute_A_weighted_SPL(wav_file):
#     data, fs = sf.read(wav_file)
#     if data.ndim > 1:
#         data = data[:, 0]  # use first channel if stereo
#     b, a = a_weighting(fs)
#     data_A = lfilter(b, a, data)
#     p_rms = np.sqrt(np.mean(data_A**2))
#     p0 = 2e-5  # reference 20 µPa
#     L_Aeq = 20 * np.log10(p_rms / p0)
#     return L_Aeq, data_A, fs

# # ---------- Files ----------
# wav1_file = '/Users/idrisseidu/Documents/RAY TRACING/Tester.wav'
# wav8_file = '/Users/idrisseidu/Documents/RAY TRACING/8m_sound.wav'
# # wav1_file = '/Users/idrisseidu/Documents/RAY TRACING/6m_sound.wav'
# # wav8_file = '/Users/idrisseidu/Documents/RAY TRACING/7m_sound.wav'

# # ---------- Compute ----------
# L1, data1A, fs1 = compute_A_weighted_SPL(wav1_file)
# L8, data8A, fs8 = compute_A_weighted_SPL(wav8_file)

# print(f"A-weighted SPL for 1m file: {L1:.2f} dBA")
# print(f"A-weighted SPL for 8m file: {L8:.2f} dBA")

# # ---------- Plot ----------
# plt.figure(figsize=(10, 5))
# time1 = np.arange(len(data1A)) / fs1
# time8 = np.arange(len(data8A)) / fs8

# plt.plot(time1, data1A, label=f'1m_sound.wav ({L1:.1f} dBA)')
# plt.plot(time8, data8A, label=f'8m_sound.wav ({L8:.1f} dBA)', alpha=0.8)

# plt.xlabel('Time [s]')
# plt.ylabel('A-weighted Amplitude (relative)')
# plt.title('A-weighted Sound Pressure Signals')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, bilinear

# File paths
wav1_file = '/Users/idrisseidu/Documents/RAY TRACING/Tester.wav'
wav8_file = '/Users/idrisseidu/Documents/RAY TRACING/8m_sound.wav'


def A_weighting(fs):
    """
    Design of A-weighting filter, returns filter coefficients b, a.
    Reference: IEC/CD 1672.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    # Pre-warped angular frequencies
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    w3 = 2 * np.pi * f3
    w4 = 2 * np.pi * f4
    w = 2 * np.pi * 1000

    # Analog A-weighting filter coefficients
    NUMs = [(w4**2) * (10**(A1000 / 20)), 0, 0, 0, 0]
    DENs = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4)**2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1)**2])
    DENs = np.polymul(np.polymul(DENs, [1, 2 * np.pi * f3]),
                      [1, 2 * np.pi * f2])

    # Bilinear transform
    b, a = bilinear(NUMs, DENs, fs)
    return b, a


def rms_flat(a):
    return np.sqrt(np.mean(np.absolute(a)**2))


def read_wav(filename):
    fs, data = wavfile.read(filename)
    # If stereo, take one channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # Normalize data (assuming 16-bit PCM)
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0
    return fs, data


def compute_a_weighted_SPL(filename):
    fs, data = read_wav(filename)

    b, a = A_weighting(fs)
    weighted = lfilter(b, a, data)

    # Reference pressure in Pa (20 uPa)
    p0 = 20e-6

    # Calculate instantaneous SPL in dB
    # We calculate RMS over sliding windows to get time evolution

    window_size = int(0.02 * fs)  # 20 ms window
    step = window_size // 2  # 50% overlap

    spl = []
    times = []
    for start in range(0, len(weighted) - window_size, step):
        window_data = weighted[start:start + window_size]
        p_rms = rms_flat(window_data)
        spl_db = 20 * np.log10(
            p_rms / p0 + 1e-12)  # add small number to avoid log(0)
        spl.append(spl_db)
        times.append(start / fs)

    return np.array(times), np.array(spl)


# Compute A-weighted SPL for both files
times1, spl1 = compute_a_weighted_SPL(wav1_file)
times8, spl8 = compute_a_weighted_SPL(wav8_file)
L_Aeq_1 = 10 * np.log10(np.mean(10**(spl1 / 10)))
L_Aeq_8 = 10 * np.log10(np.mean(10**(spl8 / 10)))
print(f"Overall LAeq 1m: {L_Aeq_1:.2f} dBA")
print(f"Overall LAeq 8m: {L_Aeq_8:.2f} dBA")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times1, spl1, label='1m sound')
plt.plot(times8, spl8, label='8m sound')
plt.xlabel('Time [s]')
plt.ylabel('A-weighted SPL [dB]')
plt.title('A-weighted Sound Pressure Level')
plt.legend()
plt.grid(True)
plt.show()



Index: 1 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.60 m , A-weighted SPL: 61.15 dB(A)
Index: 2 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.54 m , A-weighted SPL: 64.93 dB(A)
Index: 3 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.23 m , A-weighted SPL: 64.45 dB(A)
Index: 4 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.28 m , A-weighted SPL: 60.03 dB(A)
Index: 5 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.51 m , A-weighted SPL: 63.44 dB(A)
Index: 6 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.54 m , A-weighted SPL: 63.28 dB(A)
Index: 7 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.23 m , A-weighted SPL: 61.54 dB(A)
Index: 8 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.20 m , A-weighted SPL: 63.54 dB(A)
Index: 9 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.71 m , A-weighted SPL: 62.91 dB(A)
Index: 10 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.80 m , A-weighted SPL: 62.20 dB(A)
Index: 11 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.46 m , A-weighted SPL: 61.69 dB(A)
Index: 12 , MicPos: [0.000, 0.000, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.38 m , A-weighted SPL: 62.79 dB(A)

Index: 1 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.65 m , A-weighted SPL: 62.53 dB(A)
Index: 2 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.56 m , A-weighted SPL: 66.68 dB(A)
Index: 3 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.24 m , A-weighted SPL: 64.53 dB(A)
Index: 4 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.32 m , A-weighted SPL: 60.17 dB(A)
Index: 5 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.52 m , A-weighted SPL: 65.15 dB(A)
Index: 6 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.56 m , A-weighted SPL: 64.72 dB(A)
Index: 7 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.24 m , A-weighted SPL: 62.01 dB(A)
Index: 8 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.21 m , A-weighted SPL: 63.63 dB(A)
Index: 9 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.82 m , A-weighted SPL: 64.88 dB(A)
Index: 10 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.96 m , A-weighted SPL: 64.33 dB(A)
Index: 11 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.58 m , A-weighted SPL: 62.50 dB(A)
Index: 12 , MicPos: [0.000, 2.000, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.47 m , A-weighted SPL: 62.80 dB(A)

Index: 1 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.81 m , A-weighted SPL: 65.21 dB(A)
Index: 2 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.63 m , A-weighted SPL: 68.08 dB(A)
Index: 3 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.28 m , A-weighted SPL: 66.48 dB(A)
Index: 4 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.42 m , A-weighted SPL: 62.72 dB(A)
Index: 5 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.53 m , A-weighted SPL: 67.61 dB(A)
Index: 6 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.62 m , A-weighted SPL: 66.59 dB(A)
Index: 7 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.27 m , A-weighted SPL: 64.11 dB(A)
Index: 8 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.21 m , A-weighted SPL: 65.82 dB(A)
Index: 9 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.14 m , A-weighted SPL: 66.01 dB(A)
Index: 10 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.37 m , A-weighted SPL: 65.49 dB(A)
Index: 11 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.86 m , A-weighted SPL: 63.68 dB(A)
Index: 12 , MicPos: [0.000, 4.000, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.67 m , A-weighted SPL: 64.55 dB(A)

Index: 1 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.69 m , A-weighted SPL: 62.34 dB(A)
Index: 2 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.56 m , A-weighted SPL: 65.62 dB(A)
Index: 3 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.90 m , A-weighted SPL: 67.51 dB(A)
Index: 4 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.06 m , A-weighted SPL: 64.68 dB(A)
Index: 5 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.51 m , A-weighted SPL: 65.13 dB(A)
Index: 6 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.56 m , A-weighted SPL: 63.54 dB(A)
Index: 7 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.89 m , A-weighted SPL: 66.06 dB(A)
Index: 8 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.82 m , A-weighted SPL: 66.76 dB(A)
Index: 9 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.92 m , A-weighted SPL: 63.96 dB(A)
Index: 10 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.09 m , A-weighted SPL: 63.51 dB(A)
Index: 11 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.57 m , A-weighted SPL: 65.72 dB(A)
Index: 12 , MicPos: [0.000, 8.000, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.36 m , A-weighted SPL: 65.76 dB(A)



New:
Index: 12 , MicPos: [0.000, -0.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.87 m , A-weighted SPL: 63.32 dB(A)
Index: 12 , MicPos: [0.000, 0.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.90 m , A-weighted SPL: 64.34 dB(A)
Index: 12 , MicPos: [0.000, 1.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.94 m , A-weighted SPL: 64.04 dB(A)
Index: 12 , MicPos: [0.000, 2.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.00 m , A-weighted SPL: 63.87 dB(A)
Index: 12 , MicPos: [0.000, 3.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.10 m , A-weighted SPL: 64.09 dB(A)
Index: 12 , MicPos: [0.000, 4.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.28 m , A-weighted SPL: 65.73 dB(A)
Index: 12 , MicPos: [0.000, 5.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.67 m , A-weighted SPL: 66.04 dB(A)
Index: 12 , MicPos: [0.000, 6.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.55 m , A-weighted SPL: 68.05 dB(A)

------------------------------------------------------
Index: 12 , MicPos: [1.520, -0.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 61.83 dB(A)
Index: 12 , MicPos: [1.520, 0.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 62.37 dB(A)
Index: 12 , MicPos: [1.520, 1.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 62.31 dB(A)
Index: 12 , MicPos: [1.520, 2.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 62.54 dB(A)
Index: 12 , MicPos: [1.520, 3.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 62.69 dB(A)
Index: 12 , MicPos: [1.520, 4.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 65.16 dB(A)
Index: 12 , MicPos: [1.520, 5.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 72.96 dB(A)
Index: 12 , MicPos: [1.520, 6.500, 1.254] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.66 dB(A)


Index: 1 , MicPos: [-1.000, -0.600, 4.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 55.12 dB(A)
Index: 1 , MicPos: [-1.000, -0.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 57.31 dB(A)
Index: 1 , MicPos: [-1.000, 0.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 58.92 dB(A)
Index: 1 , MicPos: [-1.000, 1.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 58.67 dB(A)
Index: 1 , MicPos: [-1.000, 2.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 59.91 dB(A)
Index: 1 , MicPos: [-1.000, 3.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 62.74 dB(A)
Index: 1 , MicPos: [-1.000, 4.500, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 68.68 dB(A)
Index: 1 , MicPos: [-1.000, 5.000, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.71 dB(A)
Index: 1 , MicPos: [-1.000, 5.200, 1.254] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.42 dB(A)


Index: 4 , MicPos: [-1.000, -0.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 56.75 dB(A)
Index: 4 , MicPos: [-1.000, 0.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 57.96 dB(A)
Index: 4 , MicPos: [-1.000, 1.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 58.48 dB(A)
Index: 4 , MicPos: [-1.000, 2.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 59.24 dB(A)
Index: 4 , MicPos: [-1.000, 3.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 60.68 dB(A)
Index: 4 , MicPos: [-1.000, 4.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 63.99 dB(A)
Index: 4 , MicPos: [-1.000, 5.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 72.80 dB(A)
Index: 4 , MicPos: [-1.000, 6.500, 1.254] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.64 dB(A)


Index: 2 , MicPos: [-0.606, -0.600, 4.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 63.29 dB(A)
Index: 2 , MicPos: [-0.606, -0.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 64.87 dB(A)
Index: 2 , MicPos: [-0.606, 0.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 65.79 dB(A)
Index: 2 , MicPos: [-0.606, 1.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 65.18 dB(A)
Index: 2 , MicPos: [-0.606, 2.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 65.85 dB(A)
Index: 2 , MicPos: [-0.606, 3.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 67.30 dB(A)
Index: 2 , MicPos: [-0.606, 4.500, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 70.34 dB(A)
Index: 2 , MicPos: [-0.606, 5.000, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.84 dB(A)
Index: 2 , MicPos: [-0.606, 5.200, 1.254] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.43 dB(A)


Index: 3 , MicPos: [-0.606, -0.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 65.04 dB(A)
Index: 3 , MicPos: [-0.606, 0.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 65.67 dB(A)
Index: 3 , MicPos: [-0.606, 1.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 65.50 dB(A)
Index: 3 , MicPos: [-0.606, 2.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 65.82 dB(A)
Index: 3 , MicPos: [-0.606, 3.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 65.89 dB(A)
Index: 3 , MicPos: [-0.606, 4.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 68.07 dB(A)
Index: 3 , MicPos: [-0.606, 5.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 73.36 dB(A)
Index: 3 , MicPos: [-0.606, 6.500, 1.254] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.66 dB(A)


Index: 5 , MicPos: [0.257, -0.600, 4.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 60.92 dB(A)
Index: 5 , MicPos: [0.257, -0.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 64.07 dB(A)
Index: 5 , MicPos: [0.257, 0.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 64.45 dB(A)
Index: 5 , MicPos: [0.257, 1.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 63.79 dB(A)
Index: 5 , MicPos: [0.257, 2.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 64.10 dB(A)
Index: 5 , MicPos: [0.257, 3.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 65.88 dB(A)
Index: 5 , MicPos: [0.257, 4.500, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 69.90 dB(A)
Index: 5 , MicPos: [0.257, 5.000, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.87 dB(A)
Index: 5 , MicPos: [0.257, 5.200, 1.254] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.43 dB(A)

Index: 8 , MicPos: [0.257, -0.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 62.80 dB(A)
Index: 8 , MicPos: [0.257, 0.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 63.53 dB(A)
Index: 8 , MicPos: [0.257, 1.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 63.09 dB(A)
Index: 8 , MicPos: [0.257, 2.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 63.38 dB(A)
Index: 8 , MicPos: [0.257, 3.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 63.54 dB(A)
Index: 8 , MicPos: [0.257, 4.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 65.82 dB(A)
Index: 8 , MicPos: [0.257, 5.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 72.98 dB(A)
Index: 8 , MicPos: [0.257, 6.500, 1.254] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.65 dB(A)

Index: 7 , MicPos: [0.580, -0.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 63.67 dB(A)
Index: 7 , MicPos: [0.580, 0.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 64.26 dB(A)
Index: 7 , MicPos: [0.580, 1.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 64.63 dB(A)
Index: 7 , MicPos: [0.580, 2.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 64.27 dB(A)
Index: 7 , MicPos: [0.580, 3.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 64.92 dB(A)
Index: 7 , MicPos: [0.580, 4.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 67.28 dB(A)
Index: 7 , MicPos: [0.580, 5.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 73.40 dB(A)
Index: 7 , MicPos: [0.580, 6.500, 1.254] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.65 dB(A)


Index: 6 , MicPos: [0.580, -0.600, 4.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 60.63 dB(A)
Index: 6 , MicPos: [0.580, -0.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 63.21 dB(A)
Index: 6 , MicPos: [0.580, 0.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 64.09 dB(A)
Index: 6 , MicPos: [0.580, 1.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 63.68 dB(A)
Index: 6 , MicPos: [0.580, 2.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 63.95 dB(A)
Index: 6 , MicPos: [0.580, 3.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 65.91 dB(A)
Index: 6 , MicPos: [0.580, 4.500, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 69.93 dB(A)
Index: 6 , MicPos: [0.580, 5.000, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.90 dB(A)
Index: 6 , MicPos: [0.580, 5.200, 1.254] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.43 dB(A)



Index: 10 , MicPos: [1.830, -0.600, 4.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 61.05 dB(A)
Index: 10 , MicPos: [1.830, -0.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 63.68 dB(A)
Index: 10 , MicPos: [1.830, 0.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 64.17 dB(A)
Index: 10 , MicPos: [1.830, 1.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 63.48 dB(A)
Index: 10 , MicPos: [1.830, 2.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 64.61 dB(A)
Index: 10 , MicPos: [1.830, 3.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 65.85 dB(A)
Index: 10 , MicPos: [1.830, 4.500, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 69.97 dB(A)
Index: 10 , MicPos: [1.830, 5.000, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.88 dB(A)
Index: 10 , MicPos: [1.830, 5.200, 1.254] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.44 dB(A)


Index: 11 , MicPos: [1.830, -0.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 63.65 dB(A)
Index: 11 , MicPos: [1.830, 0.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 64.25 dB(A)
Index: 11 , MicPos: [1.830, 1.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 64.03 dB(A)
Index: 11 , MicPos: [1.830, 2.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 64.42 dB(A)
Index: 11 , MicPos: [1.830, 3.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 64.33 dB(A)
Index: 11 , MicPos: [1.830, 4.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 66.74 dB(A)
Index: 11 , MicPos: [0.580, 5.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.43 m , A-weighted SPL: 67.00 dB(A)
Index: 11 , MicPos: [1.830, 6.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.65 dB(A)


Index: 9 , MicPos: [1.520, -0.600, 4.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.70 m , A-weighted SPL: 60.50 dB(A)
Index: 9 , MicPos: [1.520, -0.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.01 m , A-weighted SPL: 62.56 dB(A)
Index: 9 , MicPos: [1.520, 0.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.01 m , A-weighted SPL: 63.33 dB(A)
Index: 9 , MicPos: [1.520, 1.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.01 m , A-weighted SPL: 63.26 dB(A)
Index: 9 , MicPos: [1.520, 2.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 63.51 dB(A)
Index: 9 , MicPos: [1.520, 3.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.01 m , A-weighted SPL: 65.45 dB(A)
Index: 9 , MicPos: [1.520, 4.500, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.01 m , A-weighted SPL: 69.77 dB(A)
Index: 9 , MicPos: [1.520, 5.000, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 0.51 m , A-weighted SPL: 76.86 dB(A)
Index: 9 , MicPos: [1.520, 5.200, 1.254] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 0.31 m , A-weighted SPL: 82.44 dB(A)

Index: 11 , MicPos: [1.830, -0.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 63.65 dB(A)
Index: 11 , MicPos: [1.830, 0.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.70 m , A-weighted SPL: 64.25 dB(A)
Index: 11 , MicPos: [1.830, 1.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.70 m , A-weighted SPL: 64.03 dB(A)
Index: 11 , MicPos: [1.830, 2.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.70 m , A-weighted SPL: 64.42 dB(A)
Index: 11 , MicPos: [1.830, 3.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.70 m , A-weighted SPL: 64.33 dB(A)
Index: 11 , MicPos: [1.830, 4.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.70 m , A-weighted SPL: 66.74 dB(A)
Index: 11 , MicPos: [1.830, 5.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 0.70 m , A-weighted SPL: 73.29 dB(A)
Index: 11 , MicPos: [1.830, 6.500, 1.254] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 0.30 m , A-weighted SPL: 82.65 dB(A)  