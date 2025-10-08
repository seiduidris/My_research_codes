# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt

# def compute_dB(wav_file, chunk_duration=1.0, reference_amplitude=1.0):
#     data, sample_rate = sf.read(wav_file)
#     if data.ndim > 1:
#         data = data[:, 0]  # Use first channel if stereo

#     samples_per_chunk = int(chunk_duration * sample_rate)
#     num_chunks = len(data) // samples_per_chunk

#     timestamps = []
#     db_values = []

#     for i in range(num_chunks):
#         chunk = data[i * samples_per_chunk:(i + 1) * samples_per_chunk]
#         rms = np.sqrt(np.mean(np.square(chunk)))
#         db = 20 * np.log10(rms /
#                            reference_amplitude) + 60 if rms > 0 else -np.inf
#         timestamps.append(i * chunk_duration)
#         db_values.append(db)

#     return timestamps, db_values

# # File paths
# wav1_file = '/Users/idrisseidu/Documents/RAY TRACING/mic1_6m_trimmed_Outt3.wav'
# wav2_file = '/Users/idrisseidu/Documents/RAY TRACING/mic1_1m_trimmed_Outt3.wav'

# # Compute dB values
# t1, db1 = compute_dB(wav1_file)
# t2, db2 = compute_dB(wav2_file)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(t1, db1, label='6m', marker='o')
# plt.plot(t2, db2, label='1m', marker='s')
# print(np.mean(db1))
# print(np.mean(db2))

# plt.title('Sound Level Comparison (dB)')
# plt.xlabel('Time (s)')
# plt.ylabel('Sound Level (dB)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Read wav files
# data1, sr1 = sf.read(wav1_file)
# if data1.ndim > 1:
#     data1 = data1[:, 0]
# time1 = np.arange(len(data1)) / sr1

# data2, sr2 = sf.read(wav2_file)
# if data2.ndim > 1:
#     data2 = data2[:, 0]
# time2 = np.arange(len(data2)) / sr2

# # Plot both on the same figure
# plt.figure(figsize=(10, 4))
# plt.plot(time2, data2, label='1m waveform', alpha=0.7)
# plt.plot(time1, data1, label='6m waveform')
# plt.title('Waveform Comparison (6m vs 1m)')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# t1, db1 = compute_dB(
#     wav1_file, chunk_duration=0.1)  # shorter chunks = smoother loudness curve
# t2, db2 = compute_dB(wav2_file, chunk_duration=0.1)
# print(np.mean(db1))
# print(np.mean(db2))

# # Plot loudness comparison
# plt.figure(figsize=(10, 4))
# plt.plot(t1, db1, label='6m loudness (dB)')
# plt.plot(t2, db2, label='1m loudness (dB)')
# plt.title('Loudness Comparison (RMS in dB)')
# plt.xlabel('Time (s)')
# plt.ylabel('Loudness (dB)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt

# def compute_avg_db_per_window(wav_file,
#                               window_duration=30.0,
#                               reference_amplitude=1.0):
#     data, sample_rate = sf.read(wav_file)
#     if data.ndim > 1:
#         data = data[:, 0]  # mono only

#     samples_per_window = int(window_duration * sample_rate)
#     num_windows = len(data) // samples_per_window

#     times = []
#     avg_dbs = []

#     for i in range(num_windows):
#         chunk = data[i * samples_per_window:(i + 1) * samples_per_window]
#         rms = np.sqrt(np.mean(np.square(chunk)))
#         db = 20 * np.log10(rms /
#                            reference_amplitude) + 60 if rms > 0 else -np.inf
#         times.append(i * window_duration)
#         avg_dbs.append(db)

#     return times, avg_dbs

# # File paths
# wav1 = '/Users/idrisseidu/Documents/RAY TRACING/mic1_6m_trimmed_Outt3.wav'
# wav2 = '/Users/idrisseidu/Documents/RAY TRACING/mic1_1m_trimmed_Outt3.wav'

# # Get averaged dB values every 5s
# t1, db1 = compute_avg_db_per_window(wav1, window_duration=35.0)
# t2, db2 = compute_avg_db_per_window(wav2, window_duration=35.0)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(t1, db1, marker='o', label='6m')
# plt.plot(t2, db2, marker='s', label='1m')
# print(len(t1))
# plt.xlabel('Time (s)')
# plt.ylabel('Average Sound Level (dB)')
# plt.title('Average dB Every 5 Seconds')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def compute_dB(wav_file, chunk_duration=0.1, reference_amplitude=1.0):
    data, sample_rate = sf.read(wav_file)
    if data.ndim > 1:
        data = data[:, 0]  # Use first channel if stereo

    samples_per_chunk = int(chunk_duration * sample_rate)
    num_chunks = len(data) // samples_per_chunk

    timestamps = []
    db_values = []

    for i in range(num_chunks):
        chunk = data[i * samples_per_chunk:(i + 1) * samples_per_chunk]
        rms = np.sqrt(np.mean(np.square(chunk)))
        db = 20 * np.log10(rms /
                           reference_amplitude) + 60 if rms > 0 else -np.inf
        timestamps.append(i * chunk_duration)
        db_values.append(db)

    return timestamps, db_values


# File paths
wav1_file = '/Users/idrisseidu/Documents/RAY TRACING/1m_sound.wav'
wav2_file = '/Users/idrisseidu/Documents/RAY TRACING/2m_sound.wav'
wav3_file = '/Users/idrisseidu/Documents/RAY TRACING/3m_sound.wav'
wav4_file = '/Users/idrisseidu/Documents/RAY TRACING/4m_sound.wav'
wav5_file = '/Users/idrisseidu/Documents/RAY TRACING/5m_sound.wav'
wav6_file = '/Users/idrisseidu/Documents/RAY TRACING/6m_sound.wav'
wav7_file = '/Users/idrisseidu/Documents/RAY TRACING/7m_sound.wav'
wav8_file = '/Users/idrisseidu/Documents/RAY TRACING/8m_sound.wav'

# Compute dB values
t1, db1 = compute_dB(wav1_file)
t2, db2 = compute_dB(wav2_file)
t3, db3 = compute_dB(wav3_file)
t4, db4 = compute_dB(wav4_file)
t5, db5 = compute_dB(wav5_file)
t6, db6 = compute_dB(wav6_file)
t7, db7 = compute_dB(wav7_file)
t8, db8 = compute_dB(wav8_file)

# Compute means
mean_db1 = np.mean(db1)
mean_db2 = np.mean(db2)
mean_db3 = np.mean(db3)
mean_db4 = np.mean(db4)
mean_db5 = np.mean(db5)
mean_db6 = np.mean(db6)
mean_db7 = np.mean(db7)
mean_db8 = np.mean(db8)
print("Mean dB - 1m:", mean_db1)
print("Mean dB - 2m:", mean_db2)
print("Mean dB - 3m:", mean_db3)
print("Mean dB - 4m:", mean_db4)
print("Mean dB - 5m:", mean_db5)
print("Mean dB - 6m:", mean_db6)
print("Mean dB - 7m:", mean_db7)
print("Mean dB - 8m:", mean_db8)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t1, db1, label='1m', marker='o')
plt.plot(t2, db2, label='2m', marker='s')
plt.plot(t3, db3, label='3m', marker='^')
plt.plot(t4, db4, label='4m', marker='d')
plt.plot(t5, db5, label='5m', marker='v')
plt.plot(t6, db6, label='6m', marker='p')
plt.plot(t7, db7, label='7m', marker='h')
plt.plot(t8, db8, label='8m', marker='x')

# Add mean lines
plt.axhline(mean_db1, color='blue', linestyle='--', label='Mean 1m')
plt.axhline(mean_db2, color='orange', linestyle='--', label='Mean 2m')
plt.axhline(mean_db3, color='green', linestyle='--', label='Mean 3m')
plt.axhline(mean_db4, color='red', linestyle='--', label='Mean 4m')
plt.axhline(mean_db5, color='purple', linestyle='--', label='Mean 5m')
plt.axhline(mean_db6, color='brown', linestyle='--', label='Mean 6m')
plt.axhline(mean_db7, color='pink', linestyle='--', label='Mean 7m')
plt.axhline(mean_db8, color='gray', linestyle='--', label='Mean 8m')

plt.title('Sound Level Comparison (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Sound Level (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot (mean lines only)
plt.figure(figsize=(10, 5))
plt.axhline(mean_db1, color='blue', linestyle='--', label='Mean 1m')
plt.axhline(mean_db2, color='orange', linestyle='--', label='Mean 2m')
plt.axhline(mean_db3, color='green', linestyle='--', label='Mean 3m')
plt.axhline(mean_db4, color='red', linestyle='--', label='Mean 4m')
plt.axhline(mean_db5, color='purple', linestyle='--', label='Mean 5m')
plt.axhline(mean_db6, color='brown', linestyle='--', label='Mean 6m')
plt.axhline(mean_db7, color='pink', linestyle='--', label='Mean 7m')
plt.axhline(mean_db8, color='gray', linestyle='--', label='Mean 8m')

plt.title('Mean Sound Levels (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Sound Level (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Heat map of distance vs mean dB ---
distances = np.arange(1, 9)  # 1m ... 8m
mean_dbs = np.array([
    mean_db1, mean_db2, mean_db3, mean_db4, mean_db5, mean_db6, mean_db7,
    mean_db8
])
# mean_dbs = np.array([mean_db3, mean_db8])
# Put it as a single-row "image"
heat_data = mean_dbs.reshape(1, -1)

plt.figure(figsize=(10, 2.5))
im = plt.imshow(heat_data, aspect='auto', interpolation='nearest')
plt.colorbar(im, label='Mean dB')

plt.xticks(np.arange(len(distances)), distances)
plt.yticks([0], ['Mean dB'])
plt.xlabel('Distance (m)')
plt.title('Heat Map: Mean dB vs Distance')
plt.tight_layout()
plt.show()
