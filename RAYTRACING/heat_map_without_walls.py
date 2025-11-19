import numpy as np
import matplotlib.pyplot as plt

# --- Data (Mic x, Mic y, SPL) ---
mic_positions = np.array([[-1, 0], [-1, 1], [-1, 2], [-1, 3], [-1, 4], [-1, 5],
                          [-1, 6], [-1, 7], [1, 0], [1, 1], [1, 2], [1, 3],
                          [1, 4], [1, 5], [1, 6], [1, 7], [3, 0], [3,
                                                                   1], [3, 2],
                          [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [5,
                                                                   0], [5, 1],
                          [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]])
# Box 1 (left)
box1 = np.array([[-1.000, 5.508], [-0.606, 5.508], [-0.606, 6.199],
                 [-1.000, 6.199]])
# Box 2 (middle)
box2 = np.array([[0.257, 5.508], [0.580, 5.508], [0.580, 6.199],
                 [0.257, 6.199]])
# Box 3 (right)
box3 = np.array([[1.520, 5.508], [1.830, 5.508], [1.830, 6.199],
                 [1.520, 6.199]])

boxes = [box1, box2, box3]

SPL1 = np.array([
    54.34, 55.62, 57.41, 60.64, 64.38, 69.91, 69.76, 63.31, 53.97, 55.04,
    56.39, 58.25, 60.65, 62.11, 60.74, 59.16, 52.79, 53.85, 54.65, 55.38,
    55.98, 56.46, 55.10, 54.60, 50.86, 51.76, 52.57, 53.19, 53.63, 53.86,
    51.65, 51.42
])
speaker_pos1 = np.array([-1.000, 5.508])

SPL2 = np.array([
    54.32, 55.59, 57.36, 60.53, 64.12, 69.05, 68.82, 63.16, 54.09, 55.21,
    56.68, 59.08, 61.67, 63.55, 62.74, 61.72, 53.09, 54.11, 54.99, 55.82,
    56.64, 57.19, 55.97, 56.08, 51.24, 52.19, 53.02, 53.69, 54.07, 54.27,
    52.23, 52.72
])
speaker_pos2 = np.array([-0.606, 5.508])

SPL3 = np.array([
    51.38, 52.89, 54.71, 57.00, 60.07, 64.56, 70.21, 67.46, 53.26, 54.43,
    55.64, 57.26, 59.95, 61.13, 63.08, 63.07, 52.21, 53.43, 54.41, 55.26,
    54.69, 56.01, 56.03, 57.10, 50.65, 51.46, 50.34, 51.04, 53.18, 52.33,
    52.25, 54.23
])
speaker_pos3 = np.array([-0.606, 6.199])

SPL4 = np.array([
    51.39, 52.91, 54.74, 57.06, 60.20, 64.94, 71.45, 68.06, 50.97, 52.32,
    53.87, 55.66, 57.69, 59.72, 60.94, 61.84, 49.89, 50.90, 51.97, 53.04,
    54.03, 54.79, 55.15, 56.38, 48.54, 49.25, 49.95, 50.59, 51.13, 51.51,
    51.67, 53.81
])
speaker_pos4 = np.array([-1.000, 6.199])

SPL5 = np.array([
    54.17, 55.38, 56.97, 59.69, 62.48, 65.14, 64.36, 62.51, 54.34, 55.59,
    57.30, 60.34, 63.60, 67.44, 66.94, 62.57, 53.68, 54.71, 55.77, 57.02,
    58.55, 59.78, 58.23, 57.67, 52.11, 53.14, 53.99, 54.58, 55.08, 55.30,
    53.65, 53.70
])
speaker_pos5 = np.array([0.257, 5.508])

SPL6 = np.array([
    54.10, 55.27, 56.67, 59.18, 61.75, 63.65, 62.59, 60.55, 54.39, 55.65,
    57.41, 60.57, 64.16, 68.98, 69.30, 64.19, 53.86, 54.86, 56.01, 57.55,
    59.62, 60.81, 59.75, 58.42, 52.40, 53.45, 54.29, 54.92, 55.42, 55.77,
    54.25, 54.33
])
speaker_pos6 = np.array([0.580, 5.508])

SPL7 = np.array([
    51.12, 52.53, 54.18, 56.14, 59.04, 61.19, 62.89, 63.16, 53.60, 54.76,
    56.08, 58.26, 61.60, 65.53, 70.52, 67.41, 53.01, 54.17, 55.22, 56.51,
    58.11, 58.91, 59.91, 60.55, 51.50, 52.77, 53.75, 52.49, 54.29, 54.32,
    54.30, 55.71
])
speaker_pos7 = np.array([0.580, 6.199])

SPL8 = np.array([
    53.38, 54.54, 55.82, 57.66, 60.55, 63.30, 64.85, 64.46, 51.33, 52.82,
    54.61, 56.84, 59.76, 63.77, 67.81, 66.29, 50.62, 51.85, 53.22, 56.24,
    57.22, 57.83, 58.42, 59.55, 49.40, 50.30, 53.15, 53.93, 52.86, 53.67,
    53.69, 55.28
])
speaker_pos8 = np.array([0.257, 6.199])

SPL9 = np.array([
    53.81, 54.79, 55.95, 57.37, 59.23, 60.49, 59.12, 58.19, 54.41, 55.63,
    57.38, 60.52, 63.98, 68.54, 68.86, 64.07, 54.21, 55.34, 56.84, 59.36,
    62.00, 64.16, 63.04, 60.80, 53.18, 54.22, 55.09, 55.96, 56.87, 57.45,
    56.26, 56.06
])
speaker_pos9 = np.array([1.520, 5.508])

SPL10 = np.array([
    53.63, 54.68, 55.69, 56.86, 58.31, 59.51, 57.98, 57.06, 54.36, 55.56,
    57.27, 60.29, 63.40, 66.97, 66.42, 62.26, 54.27, 55.44, 57.06, 59.89,
    62.69, 65.51, 65.54, 62.73, 53.39, 54.44, 55.37, 56.45, 57.46, 58.30,
    58.30, 57.52
])
speaker_pos10 = np.array([1.830, 5.508])

SPL11 = np.array([
    50.58, 51.79, 53.14, 54.60, 56.10, 57.40, 58.08, 59.29, 51.32, 52.80,
    54.58, 56.79, 59.65, 63.48, 67.18, 66.00, 53.48, 54.64, 55.88, 57.78,
    60.74, 63.63, 65.86, 64.84, 52.56, 53.77, 54.74, 55.68, 56.74, 57.82,
    58.38, 58.11
])
speaker_pos11 = np.array([1.830, 6.199])

SPL12 = np.array([
    52.92, 54.11, 55.14, 56.39, 57.93, 58.63, 59.26, 60.26, 53.59, 54.75,
    56.06, 58.18, 61.53, 65.38, 69.91, 67.09, 51.15, 52.57, 54.24, 56.24,
    59.24, 61.49, 63.38, 63.60, 50.21, 51.31, 52.50, 55.24, 55.79, 56.14,
    56.34, 57.34
])
speaker_pos12 = np.array([1.520, 6.199])

SPL = SPL12
speaker_pos = speaker_pos12

# --- Create grid for heatmap ---
x = np.unique(mic_positions[:, 0])
y = np.unique(mic_positions[:, 1])
X, Y = np.meshgrid(x, y)
Z = SPL.reshape(len(x), len(y)).T

# --- Plot heatmap ---
plt.figure(figsize=(7, 6))
heatmap = plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')
plt.colorbar(heatmap, label='A-weighted SPL [dB(A)]')

# for i, box in enumerate(boxes):
#     # Close the loop for plotting
#     x_box = np.append(box[:, 0], box[0, 0])
#     y_box = np.append(box[:, 1], box[0, 1])
#     label = 'Box' if i == 0 else None  # only label first one
#     plt.plot(x_box, y_box, 'b-', linewidth=2, label=label, zorder=2)

# --- Plot mic positions ---
plt.scatter(mic_positions[:, 0],
            mic_positions[:, 1],
            c='red',
            s=20,
            label='Mic Positions',
            zorder=3)

# --- Plot speaker position ---
plt.scatter(speaker_pos[0],
            speaker_pos[1],
            c='cyan',
            s=120,
            edgecolor='black',
            marker='*',
            label='Speaker',
            zorder=4)

plt.xlabel('Mic X Position [m]')
plt.ylabel('Mic Y Position [m]')
plt.title('A-weighted SPL (No wall) Heatmap Speaker 12')

# --- Move legend outside the plot ---
plt.legend(loc='center left', bbox_to_anchor=(1.22, 0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# #..................

# Index: 1 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.56 m , A-weighted SPL: 54.34 dB(A)
# Index: 1 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.57 m , A-weighted SPL: 55.62 dB(A)
# Index: 1 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 57.41 dB(A)
# Index: 1 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.62 m , A-weighted SPL: 60.64 dB(A)
# Index: 1 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.68 m , A-weighted SPL: 64.38 dB(A)
# Index: 1 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.90 m , A-weighted SPL: 69.91 dB(A)
# Index: 1 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.89 m , A-weighted SPL: 69.76 dB(A)
# Index: 1 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.67 m , A-weighted SPL: 63.31 dB(A)
# Index: 1 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.91 m , A-weighted SPL: 53.97 dB(A)
# Index: 1 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.99 m , A-weighted SPL: 55.04 dB(A)
# Index: 1 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.11 m , A-weighted SPL: 56.39 dB(A)
# Index: 1 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 58.25 dB(A)
# Index: 1 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.61 m , A-weighted SPL: 60.65 dB(A)
# Index: 1 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.19 m , A-weighted SPL: 62.11 dB(A)
# Index: 1 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.19 m , A-weighted SPL: 60.74 dB(A)
# Index: 1 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.60 m , A-weighted SPL: 59.16 dB(A)
# Index: 1 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.85 m , A-weighted SPL: 52.79 dB(A)
# Index: 1 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 53.85 dB(A)
# Index: 1 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.37 m , A-weighted SPL: 54.65 dB(A)
# Index: 1 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.78 m , A-weighted SPL: 55.38 dB(A)
# Index: 1 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.34 m , A-weighted SPL: 55.98 dB(A)
# Index: 1 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.10 m , A-weighted SPL: 56.46 dB(A)
# Index: 1 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.10 m , A-weighted SPL: 55.10 dB(A)
# Index: 1 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.33 m , A-weighted SPL: 54.60 dB(A)
# Index: 1 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 8.18 m , A-weighted SPL: 50.86 dB(A)
# Index: 1 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 7.54 m , A-weighted SPL: 51.76 dB(A)
# Index: 1 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.99 m , A-weighted SPL: 52.57 dB(A)
# Index: 1 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.55 m , A-weighted SPL: 53.19 dB(A)
# Index: 1 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.23 m , A-weighted SPL: 53.63 dB(A)
# Index: 1 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 53.86 dB(A)
# Index: 1 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 51.65 dB(A)
# Index: 1 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.23 m , A-weighted SPL: 51.42 dB(A)

# Index: 2 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.57 m , A-weighted SPL: 54.32 dB(A)
# Index: 2 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.59 m , A-weighted SPL: 55.59 dB(A)
# Index: 2 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.61 m , A-weighted SPL: 57.36 dB(A)
# Index: 2 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.65 m , A-weighted SPL: 60.53 dB(A)
# Index: 2 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 64.12 dB(A)
# Index: 2 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.98 m , A-weighted SPL: 69.05 dB(A)
# Index: 2 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.98 m , A-weighted SPL: 68.82 dB(A)
# Index: 2 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.71 m , A-weighted SPL: 63.16 dB(A)
# Index: 2 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.79 m , A-weighted SPL: 54.09 dB(A)
# Index: 2 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.84 m , A-weighted SPL: 55.21 dB(A)
# Index: 2 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.93 m , A-weighted SPL: 56.68 dB(A)
# Index: 2 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.07 m , A-weighted SPL: 59.08 dB(A)
# Index: 2 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.33 m , A-weighted SPL: 61.67 dB(A)
# Index: 2 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 63.55 dB(A)
# Index: 2 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 62.74 dB(A)
# Index: 2 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.32 m , A-weighted SPL: 61.72 dB(A)
# Index: 2 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.63 m , A-weighted SPL: 53.09 dB(A)
# Index: 2 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.82 m , A-weighted SPL: 54.11 dB(A)
# Index: 2 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.09 m , A-weighted SPL: 54.99 dB(A)
# Index: 2 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.46 m , A-weighted SPL: 55.82 dB(A)
# Index: 2 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.98 m , A-weighted SPL: 56.64 dB(A)
# Index: 2 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.72 m , A-weighted SPL: 57.19 dB(A)
# Index: 2 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.72 m , A-weighted SPL: 55.97 dB(A)
# Index: 2 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.97 m , A-weighted SPL: 56.08 dB(A)
# Index: 2 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 7.89 m , A-weighted SPL: 51.24 dB(A)
# Index: 2 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 7.23 m , A-weighted SPL: 52.19 dB(A)
# Index: 2 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.66 m , A-weighted SPL: 53.02 dB(A)
# Index: 2 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.19 m , A-weighted SPL: 53.69 dB(A)
# Index: 2 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.85 m , A-weighted SPL: 54.07 dB(A)
# Index: 2 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 54.27 dB(A)
# Index: 2 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 52.23 dB(A)
# Index: 2 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.85 m , A-weighted SPL: 52.72 dB(A)

# Index: 3 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.26 m , A-weighted SPL: 51.38 dB(A)
# Index: 3 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.27 m , A-weighted SPL: 52.89 dB(A)
# Index: 3 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.28 m , A-weighted SPL: 54.71 dB(A)
# Index: 3 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.31 m , A-weighted SPL: 57.00 dB(A)
# Index: 3 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.36 m , A-weighted SPL: 60.07 dB(A)
# Index: 3 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.47 m , A-weighted SPL: 64.56 dB(A)
# Index: 3 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 0.87 m , A-weighted SPL: 70.21 dB(A)
# Index: 3 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.16 m , A-weighted SPL: 67.46 dB(A)
# Index: 3 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.45 m , A-weighted SPL: 53.26 dB(A)
# Index: 3 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.49 m , A-weighted SPL: 54.43 dB(A)
# Index: 3 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.56 m , A-weighted SPL: 55.64 dB(A)
# Index: 3 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.66 m , A-weighted SPL: 57.26 dB(A)
# Index: 3 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.82 m , A-weighted SPL: 59.95 dB(A)
# Index: 3 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.14 m , A-weighted SPL: 61.13 dB(A)
# Index: 3 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.78 m , A-weighted SPL: 63.08 dB(A)
# Index: 3 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.94 m , A-weighted SPL: 63.07 dB(A)
# Index: 3 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.21 m , A-weighted SPL: 52.21 dB(A)
# Index: 3 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.37 m , A-weighted SPL: 53.43 dB(A)
# Index: 3 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.58 m , A-weighted SPL: 54.41 dB(A)
# Index: 3 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.88 m , A-weighted SPL: 55.26 dB(A)
# Index: 3 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.29 m , A-weighted SPL: 54.69 dB(A)
# Index: 3 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.87 m , A-weighted SPL: 56.01 dB(A)
# Index: 3 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.69 m , A-weighted SPL: 56.03 dB(A)
# Index: 3 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.77 m , A-weighted SPL: 57.10 dB(A)
# Index: 3 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 8.39 m , A-weighted SPL: 50.65 dB(A)
# Index: 3 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.68 m , A-weighted SPL: 51.46 dB(A)
# Index: 3 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.04 m , A-weighted SPL: 50.34 dB(A)
# Index: 3 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.50 m , A-weighted SPL: 51.04 dB(A)
# Index: 3 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.07 m , A-weighted SPL: 53.18 dB(A)
# Index: 3 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.78 m , A-weighted SPL: 52.33 dB(A)
# Index: 3 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.66 m , A-weighted SPL: 52.25 dB(A)
# Index: 3 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.71 m , A-weighted SPL: 54.23 dB(A)

# Index: 4 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.24 m , A-weighted SPL: 51.39 dB(A)
# Index: 4 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.25 m , A-weighted SPL: 52.91 dB(A)
# Index: 4 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.26 m , A-weighted SPL: 54.74 dB(A)
# Index: 4 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.28 m , A-weighted SPL: 57.06 dB(A)
# Index: 4 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.32 m , A-weighted SPL: 60.20 dB(A)
# Index: 4 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 1.41 m , A-weighted SPL: 64.94 dB(A)
# Index: 4 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 0.77 m , A-weighted SPL: 71.45 dB(A)
# Index: 4 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 1.09 m , A-weighted SPL: 68.06 dB(A)
# Index: 4 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.56 m , A-weighted SPL: 50.97 dB(A)
# Index: 4 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.62 m , A-weighted SPL: 52.32 dB(A)
# Index: 4 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.71 m , A-weighted SPL: 53.87 dB(A)
# Index: 4 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.85 m , A-weighted SPL: 55.66 dB(A)
# Index: 4 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.06 m , A-weighted SPL: 57.69 dB(A)
# Index: 4 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.45 m , A-weighted SPL: 59.72 dB(A)
# Index: 4 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.14 m , A-weighted SPL: 60.94 dB(A)
# Index: 4 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.28 m , A-weighted SPL: 61.84 dB(A)
# Index: 4 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.42 m , A-weighted SPL: 49.89 dB(A)
# Index: 4 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.60 m , A-weighted SPL: 50.90 dB(A)
# Index: 4 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.85 m , A-weighted SPL: 51.97 dB(A)
# Index: 4 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.18 m , A-weighted SPL: 53.04 dB(A)
# Index: 4 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.63 m , A-weighted SPL: 54.03 dB(A)
# Index: 4 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.24 m , A-weighted SPL: 54.79 dB(A)
# Index: 4 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.07 m , A-weighted SPL: 55.15 dB(A)
# Index: 4 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.15 m , A-weighted SPL: 56.38 dB(A)
# Index: 4 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 8.66 m , A-weighted SPL: 48.54 dB(A)
# Index: 4 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.97 m , A-weighted SPL: 49.25 dB(A)
# Index: 4 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.36 m , A-weighted SPL: 49.95 dB(A)
# Index: 4 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.84 m , A-weighted SPL: 50.59 dB(A)
# Index: 4 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.43 m , A-weighted SPL: 51.13 dB(A)
# Index: 4 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.16 m , A-weighted SPL: 51.51 dB(A)
# Index: 4 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.05 m , A-weighted SPL: 51.67 dB(A)
# Index: 4 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.10 m , A-weighted SPL: 53.81 dB(A)

# Index: 5 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.70 m , A-weighted SPL: 54.17 dB(A)
# Index: 5 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.74 m , A-weighted SPL: 55.38 dB(A)
# Index: 5 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.80 m , A-weighted SPL: 56.97 dB(A)
# Index: 5 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.90 m , A-weighted SPL: 59.69 dB(A)
# Index: 5 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.10 m , A-weighted SPL: 62.48 dB(A)
# Index: 5 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.55 m , A-weighted SPL: 65.14 dB(A)
# Index: 5 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.54 m , A-weighted SPL: 64.36 dB(A)
# Index: 5 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.09 m , A-weighted SPL: 62.51 dB(A)
# Index: 5 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.61 m , A-weighted SPL: 54.34 dB(A)
# Index: 5 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.63 m , A-weighted SPL: 55.59 dB(A)
# Index: 5 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.66 m , A-weighted SPL: 57.30 dB(A)
# Index: 5 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.72 m , A-weighted SPL: 60.34 dB(A)
# Index: 5 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 63.60 dB(A)
# Index: 5 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.17 m , A-weighted SPL: 67.44 dB(A)
# Index: 5 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.16 m , A-weighted SPL: 66.94 dB(A)
# Index: 5 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.83 m , A-weighted SPL: 62.57 dB(A)
# Index: 5 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.20 m , A-weighted SPL: 53.68 dB(A)
# Index: 5 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.33 m , A-weighted SPL: 54.71 dB(A)
# Index: 5 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.52 m , A-weighted SPL: 55.77 dB(A)
# Index: 5 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.79 m , A-weighted SPL: 57.02 dB(A)
# Index: 5 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.22 m , A-weighted SPL: 58.55 dB(A)
# Index: 5 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.89 m , A-weighted SPL: 59.78 dB(A)
# Index: 5 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.88 m , A-weighted SPL: 58.23 dB(A)
# Index: 5 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.21 m , A-weighted SPL: 57.67 dB(A)
# Index: 5 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 7.31 m , A-weighted SPL: 52.11 dB(A)
# Index: 5 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.59 m , A-weighted SPL: 53.14 dB(A)
# Index: 5 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.95 m , A-weighted SPL: 53.99 dB(A)
# Index: 5 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.42 m , A-weighted SPL: 54.58 dB(A)
# Index: 5 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.03 m , A-weighted SPL: 55.08 dB(A)
# Index: 5 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 55.30 dB(A)
# Index: 5 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 53.65 dB(A)
# Index: 5 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.03 m , A-weighted SPL: 53.70 dB(A)

# Index: 6 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.78 m , A-weighted SPL: 54.10 dB(A)
# Index: 6 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 55.27 dB(A)
# Index: 6 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.92 m , A-weighted SPL: 56.67 dB(A)
# Index: 6 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.06 m , A-weighted SPL: 59.18 dB(A)
# Index: 6 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.31 m , A-weighted SPL: 61.75 dB(A)
# Index: 6 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.82 m , A-weighted SPL: 63.65 dB(A)
# Index: 6 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.82 m , A-weighted SPL: 62.59 dB(A)
# Index: 6 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.30 m , A-weighted SPL: 60.55 dB(A)
# Index: 6 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.57 m , A-weighted SPL: 54.39 dB(A)
# Index: 6 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.59 m , A-weighted SPL: 55.65 dB(A)
# Index: 6 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.61 m , A-weighted SPL: 57.41 dB(A)
# Index: 6 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.65 m , A-weighted SPL: 60.57 dB(A)
# Index: 6 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 64.16 dB(A)
# Index: 6 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.00 m , A-weighted SPL: 68.98 dB(A)
# Index: 6 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 0.99 m , A-weighted SPL: 69.30 dB(A)
# Index: 6 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.72 m , A-weighted SPL: 64.19 dB(A)
# Index: 6 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.06 m , A-weighted SPL: 53.86 dB(A)
# Index: 6 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.17 m , A-weighted SPL: 54.86 dB(A)
# Index: 6 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.33 m , A-weighted SPL: 56.01 dB(A)
# Index: 6 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.56 m , A-weighted SPL: 57.55 dB(A)
# Index: 6 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.95 m , A-weighted SPL: 59.62 dB(A)
# Index: 6 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.58 m , A-weighted SPL: 60.81 dB(A)
# Index: 6 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.58 m , A-weighted SPL: 59.75 dB(A)
# Index: 6 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.94 m , A-weighted SPL: 58.42 dB(A)
# Index: 6 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 7.10 m , A-weighted SPL: 52.40 dB(A)
# Index: 6 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.36 m , A-weighted SPL: 53.45 dB(A)
# Index: 6 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.69 m , A-weighted SPL: 54.29 dB(A)
# Index: 6 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.14 m , A-weighted SPL: 54.92 dB(A)
# Index: 6 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.73 m , A-weighted SPL: 55.42 dB(A)
# Index: 6 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.51 m , A-weighted SPL: 55.77 dB(A)
# Index: 6 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.51 m , A-weighted SPL: 54.25 dB(A)
# Index: 6 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.72 m , A-weighted SPL: 54.33 dB(A)

# Index: 7 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.44 m , A-weighted SPL: 51.12 dB(A)
# Index: 7 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.48 m , A-weighted SPL: 52.53 dB(A)
# Index: 7 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.55 m , A-weighted SPL: 54.18 dB(A)
# Index: 7 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.65 m , A-weighted SPL: 56.14 dB(A)
# Index: 7 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.81 m , A-weighted SPL: 59.04 dB(A)
# Index: 7 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.12 m , A-weighted SPL: 61.19 dB(A)
# Index: 7 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.76 m , A-weighted SPL: 62.89 dB(A)
# Index: 7 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.92 m , A-weighted SPL: 63.16 dB(A)
# Index: 7 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.26 m , A-weighted SPL: 53.60 dB(A)
# Index: 7 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.27 m , A-weighted SPL: 54.76 dB(A)
# Index: 7 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.29 m , A-weighted SPL: 56.08 dB(A)
# Index: 7 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.31 m , A-weighted SPL: 58.26 dB(A)
# Index: 7 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.36 m , A-weighted SPL: 61.60 dB(A)
# Index: 7 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.47 m , A-weighted SPL: 65.53 dB(A)
# Index: 7 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 0.88 m , A-weighted SPL: 70.52 dB(A)
# Index: 7 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.17 m , A-weighted SPL: 67.41 dB(A)
# Index: 7 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 53.01 dB(A)
# Index: 7 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.78 m , A-weighted SPL: 54.17 dB(A)
# Index: 7 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.90 m , A-weighted SPL: 55.22 dB(A)
# Index: 7 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.08 m , A-weighted SPL: 56.51 dB(A)
# Index: 7 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.35 m , A-weighted SPL: 58.11 dB(A)
# Index: 7 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.80 m , A-weighted SPL: 58.91 dB(A)
# Index: 7 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.54 m , A-weighted SPL: 59.91 dB(A)
# Index: 7 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.66 m , A-weighted SPL: 60.55 dB(A)
# Index: 7 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 7.65 m , A-weighted SPL: 51.50 dB(A)
# Index: 7 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.86 m , A-weighted SPL: 52.77 dB(A)
# Index: 7 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.14 m , A-weighted SPL: 53.75 dB(A)
# Index: 7 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.51 m , A-weighted SPL: 52.49 dB(A)
# Index: 7 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.99 m , A-weighted SPL: 54.29 dB(A)
# Index: 7 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.64 m , A-weighted SPL: 54.32 dB(A)
# Index: 7 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.49 m , A-weighted SPL: 54.30 dB(A)
# Index: 7 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.55 m , A-weighted SPL: 55.71 dB(A)

# Index: 8 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.37 m , A-weighted SPL: 53.38 dB(A)
# Index: 8 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.40 m , A-weighted SPL: 54.54 dB(A)
# Index: 8 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.45 m , A-weighted SPL: 55.82 dB(A)
# Index: 8 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.52 m , A-weighted SPL: 57.66 dB(A)
# Index: 8 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.64 m , A-weighted SPL: 60.55 dB(A)
# Index: 8 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.89 m , A-weighted SPL: 63.30 dB(A)
# Index: 8 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.48 m , A-weighted SPL: 64.85 dB(A)
# Index: 8 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.67 m , A-weighted SPL: 64.46 dB(A)
# Index: 8 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.29 m , A-weighted SPL: 51.33 dB(A)
# Index: 8 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.30 m , A-weighted SPL: 52.82 dB(A)
# Index: 8 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.33 m , A-weighted SPL: 54.61 dB(A)
# Index: 8 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.37 m , A-weighted SPL: 56.84 dB(A)
# Index: 8 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.44 m , A-weighted SPL: 59.76 dB(A)
# Index: 8 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.60 m , A-weighted SPL: 63.77 dB(A)
# Index: 8 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.07 m , A-weighted SPL: 67.81 dB(A)
# Index: 8 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.32 m , A-weighted SPL: 66.29 dB(A)
# Index: 8 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.82 m , A-weighted SPL: 50.62 dB(A)
# Index: 8 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.93 m , A-weighted SPL: 51.85 dB(A)
# Index: 8 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.07 m , A-weighted SPL: 53.22 dB(A)
# Index: 8 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.28 m , A-weighted SPL: 56.24 dB(A)
# Index: 8 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.59 m , A-weighted SPL: 57.22 dB(A)
# Index: 8 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.09 m , A-weighted SPL: 57.83 dB(A)
# Index: 8 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.85 m , A-weighted SPL: 58.42 dB(A)
# Index: 8 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.95 m , A-weighted SPL: 59.55 dB(A)
# Index: 8 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 7.84 m , A-weighted SPL: 49.40 dB(A)
# Index: 8 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 7.08 m , A-weighted SPL: 50.30 dB(A)
# Index: 8 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.38 m , A-weighted SPL: 53.15 dB(A)
# Index: 8 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.77 m , A-weighted SPL: 53.93 dB(A)
# Index: 8 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.28 m , A-weighted SPL: 52.86 dB(A)
# Index: 8 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.95 m , A-weighted SPL: 53.67 dB(A)
# Index: 8 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.81 m , A-weighted SPL: 53.69 dB(A)
# Index: 8 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.87 m , A-weighted SPL: 55.28 dB(A)

# Index: 9 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.10 m , A-weighted SPL: 53.81 dB(A)
# Index: 9 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.22 m , A-weighted SPL: 54.79 dB(A)
# Index: 9 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.38 m , A-weighted SPL: 55.95 dB(A)
# Index: 9 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.63 m , A-weighted SPL: 57.37 dB(A)
# Index: 9 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.03 m , A-weighted SPL: 59.23 dB(A)
# Index: 9 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.68 m , A-weighted SPL: 60.49 dB(A)
# Index: 9 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.67 m , A-weighted SPL: 59.12 dB(A)
# Index: 9 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.02 m , A-weighted SPL: 58.19 dB(A)
# Index: 9 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.58 m , A-weighted SPL: 54.41 dB(A)
# Index: 9 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.60 m , A-weighted SPL: 55.63 dB(A)
# Index: 9 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.62 m , A-weighted SPL: 57.38 dB(A)
# Index: 9 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.67 m , A-weighted SPL: 60.52 dB(A)
# Index: 9 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.76 m , A-weighted SPL: 63.98 dB(A)
# Index: 9 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.04 m , A-weighted SPL: 68.54 dB(A)
# Index: 9 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.03 m , A-weighted SPL: 68.86 dB(A)
# Index: 9 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.75 m , A-weighted SPL: 64.07 dB(A)
# Index: 9 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.75 m , A-weighted SPL: 54.21 dB(A)
# Index: 9 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.80 m , A-weighted SPL: 55.34 dB(A)
# Index: 9 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.88 m , A-weighted SPL: 56.84 dB(A)
# Index: 9 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 59.36 dB(A)
# Index: 9 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.24 m , A-weighted SPL: 62.00 dB(A)
# Index: 9 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 64.16 dB(A)
# Index: 9 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 63.04 dB(A)
# Index: 9 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.23 m , A-weighted SPL: 60.80 dB(A)
# Index: 9 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.56 m , A-weighted SPL: 53.18 dB(A)
# Index: 9 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.74 m , A-weighted SPL: 54.22 dB(A)
# Index: 9 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.00 m , A-weighted SPL: 55.09 dB(A)
# Index: 9 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.35 m , A-weighted SPL: 55.96 dB(A)
# Index: 9 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.87 m , A-weighted SPL: 56.87 dB(A)
# Index: 9 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.60 m , A-weighted SPL: 57.45 dB(A)
# Index: 9 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 56.26 dB(A)
# Index: 9 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.86 m , A-weighted SPL: 56.06 dB(A)

# Index: 10 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.24 m , A-weighted SPL: 53.63 dB(A)
# Index: 10 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.37 m , A-weighted SPL: 54.68 dB(A)
# Index: 10 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.57 m , A-weighted SPL: 55.69 dB(A)
# Index: 10 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.85 m , A-weighted SPL: 56.86 dB(A)
# Index: 10 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 58.31 dB(A)
# Index: 10 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.97 m , A-weighted SPL: 59.51 dB(A)
# Index: 10 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.97 m , A-weighted SPL: 57.98 dB(A)
# Index: 10 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 57.06 dB(A)
# Index: 10 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.62 m , A-weighted SPL: 54.36 dB(A)
# Index: 10 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.64 m , A-weighted SPL: 55.56 dB(A)
# Index: 10 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.68 m , A-weighted SPL: 57.27 dB(A)
# Index: 10 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.75 m , A-weighted SPL: 60.29 dB(A)
# Index: 10 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.88 m , A-weighted SPL: 63.40 dB(A)
# Index: 10 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.23 m , A-weighted SPL: 66.97 dB(A)
# Index: 10 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.22 m , A-weighted SPL: 66.42 dB(A)
# Index: 10 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.86 m , A-weighted SPL: 62.26 dB(A)
# Index: 10 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 54.27 dB(A)
# Index: 10 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.72 m , A-weighted SPL: 55.44 dB(A)
# Index: 10 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.77 m , A-weighted SPL: 57.06 dB(A)
# Index: 10 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.87 m , A-weighted SPL: 59.89 dB(A)
# Index: 10 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.05 m , A-weighted SPL: 62.69 dB(A)
# Index: 10 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.48 m , A-weighted SPL: 65.51 dB(A)
# Index: 10 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.47 m , A-weighted SPL: 65.54 dB(A)
# Index: 10 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.04 m , A-weighted SPL: 62.73 dB(A)
# Index: 10 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.40 m , A-weighted SPL: 53.39 dB(A)
# Index: 10 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.56 m , A-weighted SPL: 54.44 dB(A)
# Index: 10 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.79 m , A-weighted SPL: 55.37 dB(A)
# Index: 10 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.11 m , A-weighted SPL: 56.45 dB(A)
# Index: 10 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 57.46 dB(A)
# Index: 10 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.30 m , A-weighted SPL: 58.30 dB(A)
# Index: 10 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 58.30 dB(A)
# Index: 10 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.58 m , A-weighted SPL: 57.52 dB(A)

# Index: 11 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.86 m , A-weighted SPL: 50.58 dB(A)
# Index: 11 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.97 m , A-weighted SPL: 51.79 dB(A)
# Index: 11 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.12 m , A-weighted SPL: 53.14 dB(A)
# Index: 11 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.34 m , A-weighted SPL: 54.60 dB(A)
# Index: 11 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.66 m , A-weighted SPL: 56.10 dB(A)
# Index: 11 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.16 m , A-weighted SPL: 57.40 dB(A)
# Index: 11 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.93 m , A-weighted SPL: 58.08 dB(A)
# Index: 11 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.03 m , A-weighted SPL: 59.29 dB(A)
# Index: 11 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.30 m , A-weighted SPL: 51.32 dB(A)
# Index: 11 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.32 m , A-weighted SPL: 52.80 dB(A)
# Index: 11 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.34 m , A-weighted SPL: 54.58 dB(A)
# Index: 11 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.39 m , A-weighted SPL: 56.79 dB(A)
# Index: 11 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.47 m , A-weighted SPL: 59.65 dB(A)
# Index: 11 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.64 m , A-weighted SPL: 63.48 dB(A)
# Index: 11 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.13 m , A-weighted SPL: 67.18 dB(A)
# Index: 11 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.37 m , A-weighted SPL: 66.00 dB(A)
# Index: 11 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.35 m , A-weighted SPL: 53.48 dB(A)
# Index: 11 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.38 m , A-weighted SPL: 54.64 dB(A)
# Index: 11 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.42 m , A-weighted SPL: 55.88 dB(A)
# Index: 11 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.49 m , A-weighted SPL: 57.78 dB(A)
# Index: 11 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.60 m , A-weighted SPL: 60.74 dB(A)
# Index: 11 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.83 m , A-weighted SPL: 63.63 dB(A)
# Index: 11 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.40 m , A-weighted SPL: 65.86 dB(A)
# Index: 11 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.60 m , A-weighted SPL: 64.84 dB(A)
# Index: 11 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 7.00 m , A-weighted SPL: 52.56 dB(A)
# Index: 11 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.13 m , A-weighted SPL: 53.77 dB(A)
# Index: 11 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.31 m , A-weighted SPL: 54.74 dB(A)
# Index: 11 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.56 m , A-weighted SPL: 55.68 dB(A)
# Index: 11 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.93 m , A-weighted SPL: 56.74 dB(A)
# Index: 11 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.47 m , A-weighted SPL: 57.82 dB(A)
# Index: 11 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.26 m , A-weighted SPL: 58.38 dB(A)
# Index: 11 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.35 m , A-weighted SPL: 58.11 dB(A)

# Index: 12 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.73 m , A-weighted SPL: 52.92 dB(A)
# Index: 12 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.83 m , A-weighted SPL: 54.11 dB(A)
# Index: 12 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.95 m , A-weighted SPL: 55.14 dB(A)
# Index: 12 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.14 m , A-weighted SPL: 56.39 dB(A)
# Index: 12 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.43 m , A-weighted SPL: 57.93 dB(A)
# Index: 12 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.89 m , A-weighted SPL: 58.63 dB(A)
# Index: 12 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.64 m , A-weighted SPL: 59.26 dB(A)
# Index: 12 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.75 m , A-weighted SPL: 60.26 dB(A)
# Index: 12 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.27 m , A-weighted SPL: 53.59 dB(A)
# Index: 12 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.28 m , A-weighted SPL: 54.75 dB(A)
# Index: 12 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.30 m , A-weighted SPL: 56.06 dB(A)
# Index: 12 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.33 m , A-weighted SPL: 58.18 dB(A)
# Index: 12 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.38 m , A-weighted SPL: 61.53 dB(A)
# Index: 12 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.50 m , A-weighted SPL: 65.38 dB(A)
# Index: 12 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 0.93 m , A-weighted SPL: 69.91 dB(A)
# Index: 12 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.21 m , A-weighted SPL: 67.09 dB(A)
# Index: 12 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.42 m , A-weighted SPL: 51.15 dB(A)
# Index: 12 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.46 m , A-weighted SPL: 52.57 dB(A)
# Index: 12 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.51 m , A-weighted SPL: 54.24 dB(A)
# Index: 12 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.60 m , A-weighted SPL: 56.24 dB(A)
# Index: 12 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.75 m , A-weighted SPL: 59.24 dB(A)
# Index: 12 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.05 m , A-weighted SPL: 61.49 dB(A)
# Index: 12 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.67 m , A-weighted SPL: 63.38 dB(A)
# Index: 12 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.84 m , A-weighted SPL: 63.60 dB(A)
# Index: 12 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 7.15 m , A-weighted SPL: 50.21 dB(A)
# Index: 12 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.30 m , A-weighted SPL: 51.31 dB(A)
# Index: 12 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.50 m , A-weighted SPL: 52.50 dB(A)
# Index: 12 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.79 m , A-weighted SPL: 55.24 dB(A)
# Index: 12 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.18 m , A-weighted SPL: 55.79 dB(A)
# Index: 12 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.76 m , A-weighted SPL: 56.14 dB(A)
# Index: 12 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.56 m , A-weighted SPL: 56.34 dB(A)
# Index: 12 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.65 m , A-weighted SPL: 57.34 dB(A)
