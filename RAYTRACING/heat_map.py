# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata

# # -------------------------------
# # Room outline
# # -------------------------------
# room = np.array([[-1.000, -0.700], [5.000, -0.700], [5.000, 8.300],
#                  [-1.000, 8.300]])

# # -------------------------------
# # Box corners (buildings)
# # -------------------------------
# boxes = {
#     "box1":
#     np.array([[-1.000, 5.508], [-0.606, 5.508], [-0.606, 6.199],
#               [-1.000, 6.199]]),
#     "box2":
#     np.array([[0.257, 5.508], [0.580, 5.508], [0.580, 6.199], [0.257, 6.199]]),
#     "box3":
#     np.array([[1.520, 5.508], [1.830, 5.508], [1.830, 6.199], [1.520, 6.199]])
# }

# # -------------------------------
# # Speaker and mic data
# # -------------------------------
# speaker_pos = np.array([[-1.000, 5.508], [-0.606, 5.508], [-0.606, 6.199],
#                         [-1.000, 6.199], [0.257, 5.508], [0.580, 5.508],
#                         [0.580, 6.199], [0.257, 6.199], [1.520, 5.508],
#                         [1.830, 5.508], [1.830, 6.199], [1.520, 6.199]])
# spl = np.array([
#     61.15, 64.93, 64.45, 60.03, 63.44, 63.28, 61.54, 63.54, 62.91, 62.20,
#     61.69, 62.79
# ])
# dist = np.array(
#     [5.60, 5.54, 6.23, 6.28, 5.51, 5.54, 6.23, 6.20, 5.71, 5.80, 6.46, 6.38])

# # Microphone positions (drone positions)
# mic_pos = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 8.0]])

# # -------------------------------
# # Heatmap interpolation
# # -------------------------------
# Xq, Yq = np.meshgrid(np.linspace(-1, 5, 200), np.linspace(-1, 9, 200))
# Zq = griddata(speaker_pos, spl, (Xq, Yq), method='cubic')

# # -------------------------------
# # Plot setup
# # -------------------------------
# plt.figure(figsize=(8, 7))
# plt.title('A-weighted SPL Heat Map (dB(A))')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.axis('equal')
# plt.grid(True)

# # Room outline
# plt.plot(*np.vstack((room, room[0])).T, 'k-', linewidth=2, label='Room')

# # Boxes (buildings)
# for name, box in boxes.items():
#     plt.fill(*np.vstack((box, box[0])).T,
#              color='lightgray',
#              edgecolor='k',
#              label=name if name == "box1" else None)

# # Heatmap
# plt.contourf(Xq, Yq, Zq, levels=30, cmap='jet', alpha=0.7)
# cbar = plt.colorbar(label='SPL (dB(A))')

# # Speakers
# plt.scatter(speaker_pos[:, 0],
#             speaker_pos[:, 1],
#             c='red',
#             s=60,
#             label='Speakers')

# # -------------------------------
# # Drone markers (at each mic position)
# # -------------------------------
# arm = 0.15  # half arm length for the "X" shape
# for (x, y) in mic_pos:
#     plt.plot([x - arm, x + arm], [y - arm, y + arm], 'g-', lw=2)
#     plt.plot([x - arm, x + arm], [y + arm, y - arm], 'g-', lw=2)
#     plt.text(x + 0.1, y, 'Drone', color='green', fontsize=8, va='center')

# # -------------------------------
# # Annotate SPL and distance
# # -------------------------------
# for i, (x, y) in enumerate(speaker_pos):
#     plt.text(x + 0.05,
#              y,
#              f'{spl[i]:.1f} dB\n{dist[i]:.2f} m',
#              fontsize=8,
#              color='black')

# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------------
# # Example speaker positions and SPL
# # -----------------------------------
# speaker_pos = np.array([[-1.000, 5.508], [-0.606, 5.508], [-0.606, 6.199],
#                         [-1.000, 6.199], [0.257, 5.508], [0.580, 5.508],
#                         [0.580, 6.199], [0.257, 6.199], [1.520, 5.508],
#                         [1.830, 5.508], [1.830, 6.199], [1.520, 6.199]])
# spl = np.array([
#     61.15, 64.93, 64.45, 60.03, 63.44, 63.28, 61.54, 63.54, 62.91, 62.20,
#     61.69, 62.79
# ])

# # Microphone (drone) positions
# mic_pos = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 8.0]])

# # -----------------------------------
# # Create grid
# # -----------------------------------
# x_min, x_max = -1, 2.5
# y_min, y_max = 5, 6.5
# grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

# # Interpolate SPL onto grid
# from scipy.interpolate import griddata

# grid_z = griddata(speaker_pos, spl, (grid_x, grid_y), method='linear')

# # -----------------------------------
# # Plot heatmap
# # -----------------------------------
# plt.figure(figsize=(8, 6))
# plt.title('Sound Pressure Level Heatmap (dB(A))')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.axis('equal')

# # Plot as grid-style heatmap
# im = plt.imshow(grid_z.T,
#                 extent=(x_min, x_max, y_min, y_max),
#                 origin='lower',
#                 cmap='jet',
#                 aspect='auto')
# plt.colorbar(im, label='SPL (dB(A))')

# # Overlay speaker and drone positions
# plt.scatter(speaker_pos[:, 0],
#             speaker_pos[:, 1],
#             c='red',
#             s=50,
#             label='Speakers')
# plt.scatter(mic_pos[:, 0],
#             mic_pos[:, 1],
#             c='green',
#             s=80,
#             marker='x',
#             label='Drones')

# # Annotate each speaker SPL
# for i, (x, y) in enumerate(speaker_pos):
#     plt.text(x + 0.05, y, f'{spl[i]:.1f}', fontsize=8, color='black')

# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import interp1d

# # Each index data: (Distance, SPL)
# data = {
#     1:
#     ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51,
#       0.31], [55.12, 57.31, 58.92, 58.67, 59.91, 62.74, 68.68, 76.71, 82.42]),
#     2:
#     ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51,
#       0.31], [63.29, 64.87, 65.79, 65.18, 65.85, 67.30, 70.34, 76.84, 82.43]),
#     3: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#          0.30], [65.04, 65.67, 65.50, 65.82, 65.89, 68.07, 73.36, 82.66]),
#     4: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#          0.30], [56.75, 57.96, 58.48, 59.24, 60.68, 63.99, 72.80, 82.64]),
#     5: ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51, 0.31],
#         [60.92, 64.07, 64.45, 63.79, 64.10, 65.88, 69.90, 76.87, 82.43]),
#     6: ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51, 0.31],
#         [60.63, 63.21, 64.09, 63.68, 63.95, 65.91, 69.93, 76.90, 82.43]),
#     7: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#          0.30], [63.67, 64.26, 64.63, 64.27, 64.92, 67.28, 73.40, 82.65]),
#     8: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#          0.30], [62.80, 63.53, 63.09, 63.38, 63.54, 65.82, 72.98, 82.65]),
#     9: ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51, 0.31],
#         [60.50, 62.56, 63.33, 63.26, 63.51, 65.45, 69.77, 76.86, 82.44]),
#     10: ([6.70, 6.01, 5.01, 4.01, 3.01, 2.01, 1.01, 0.51, 0.31],
#          [61.05, 63.68, 64.17, 63.48, 64.61, 65.85, 69.97, 76.88, 82.44]),
#     11: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#           0.30], [63.65, 64.25, 64.03, 64.42, 64.33, 66.74, 73.29, 82.65]),
#     12: ([6.70, 5.70, 4.70, 3.70, 2.70, 1.70, 0.70,
#           0.30], [61.83, 62.37, 62.31, 62.54, 62.69, 65.16, 72.96, 82.66])
# }

# # Combine all SPL values to get global color scale
# all_spl = np.concatenate([v[1] for v in data.values()])
# vmin, vmax = min(all_spl), max(all_spl)

# plt.figure(figsize=(7, 8))

# for i, (idx, (dist, spl)) in enumerate(data.items()):
#     # Interpolate for smooth color transitions
#     dist = np.array(dist)
#     spl = np.array(spl)
#     f = interp1d(dist, spl, kind='cubic', fill_value="extrapolate")

#     # Create smooth distance grid
#     dist_smooth = np.linspace(min(dist), max(dist), 200)
#     spl_smooth = f(dist_smooth)

#     # Make a smooth rectangle (using gradient row)
#     spl_array = np.tile(spl_smooth, (10, 1))  # 10 pixels tall rectangle
#     extent = [min(dist), max(dist), i, i + 1]

#     plt.imshow(spl_array,
#                extent=extent,
#                vmin=vmin,
#                vmax=vmax,
#                cmap='jet',
#                aspect='auto',
#                origin='lower')

# plt.colorbar(label='A-weighted SPL (dB(A))')
# plt.xlabel('Distance (m)')
# plt.ylabel('Microphone Index Position')
# plt.title('Heatmap Across 12 mic positions')
# plt.yticks(np.arange(len(data)) + 0.5, list(data.keys()))
# plt.tight_layout()
# plt.show()

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

# SPL1 = np.array([
#     57.92, 58.94, 59.53, 61.00, 64.35, 69.93, 70.03, 64.34, 59.91, 60.94,
#     61.14, 61.88, 63.54, 64.46, 63.28, 62.09, 59.47, 60.35, 60.90, 61.16,
#     61.49, 62.05, 60.30, 60.42, 57.39, 57.83, 58.98, 58.90, 59.45, 58.80,
#     58.01, 57.02
# ])
# SPL2 = np.array([
#     59.53, 60.63, 61.10, 61.55, 65.16, 69.14, 69.23, 64.53, 62.25, 63.51,
#     63.47, 64.39, 65.51, 66.36, 66.22, 64.92, 61.43, 62.55, 62.76, 63.05,
#     64.13, 64.71, 63.18, 63.21, 59.84, 60.96, 61.39, 61.10, 62.15, 61.41,
#     60.26, 59.92
# ])

# SPL3 = np.array([
#     58.38, 59.49, 60.57, 60.79, 62.47, 65.72, 70.58, 67.51, 62.91, 62.53,
#     63.64, 63.77, 64.09, 65.41, 66.65, 66.72, 61.15, 61.29, 62.48, 62.99,
#     62.78, 63.85, 63.86, 65.53, 59.59, 60.86, 60.24, 60.43, 61.67, 60.64,
#     61.03, 63.70
# ])

# SPL4 = np.array([
#     57.49, 58.19, 59.30, 59.88, 61.91, 65.63, 71.67, 67.98, 59.09, 59.74,
#     60.45, 60.02, 61.03, 62.73, 63.68, 65.00, 58.44, 58.81, 60.34, 60.28,
#     60.69, 61.26, 61.73, 63.67, 56.64, 56.71, 57.86, 57.74, 59.04, 58.14,
#     58.56, 60.59
# ])

# SPL5 = np.array([
#     60.42, 61.34, 60.64, 63.11, 64.54, 65.57, 65.59, 64.00, 61.94, 63.45,
#     64.04, 64.76, 66.48, 68.83, 68.23, 65.27, 61.80, 62.17, 63.40, 63.39,
#     64.50, 64.86, 64.39, 63.99, 60.25, 60.67, 60.65, 61.46, 61.85, 61.21,
#     61.07, 60.41
# ])

# SPL6 = np.array([
#     60.21, 60.59, 61.80, 61.62, 64.31, 64.76, 64.14, 63.45, 62.16, 63.40,
#     63.79, 64.52, 65.90, 69.40, 69.53, 65.84, 61.56, 62.54, 63.21, 63.40,
#     64.76, 65.52, 65.49, 64.44, 59.81, 60.95, 61.02, 61.15, 62.07, 62.20,
#     60.75, 61.29
# ])

# SPL7 = np.array([
#     59.89, 59.82, 60.82, 61.42, 61.97, 63.71, 64.78, 65.49, 62.11, 61.90,
#     63.41, 63.80, 64.62, 66.88, 70.86, 68.71, 60.99, 61.62, 63.12, 63.00,
#     63.91, 65.96, 65.15, 66.04, 60.09, 60.49, 60.42, 59.75, 61.74, 61.48,
#     61.30, 62.79
# ])
# SPL8 = np.array([
#     60.26, 59.84, 60.97, 61.85, 62.87, 64.94, 65.98, 65.55, 62.10, 61.83,
#     62.58, 63.42, 64.42, 66.10, 69.17, 68.84, 61.45, 61.78, 62.72, 64.00,
#     63.56, 64.85, 64.86, 66.51, 59.41, 59.36, 60.81, 61.06, 61.08, 61.30,
#     61.71, 63.22
# ])

# SPL9 = np.array([
#     58.91, 61.31, 61.77, 62.23, 62.67, 62.88, 62.43, 61.77, 62.32, 62.72,
#     63.82, 64.32, 65.97, 69.30, 69.38, 65.87, 61.17, 62.63, 63.20, 63.32,
#     65.00, 66.71, 66.39, 64.67, 59.68, 61.10, 60.95, 61.50, 61.93, 61.47,
#     60.72, 60.62
# ])

# SPL10 = np.array([
#     60.41, 61.60, 61.65, 61.66, 62.48, 62.30, 61.77, 61.11, 61.97, 62.83,
#     63.37, 63.82, 65.83, 68.18, 67.68, 64.73, 61.52, 62.33, 62.66, 63.43,
#     65.48, 67.39, 67.30, 65.25, 60.75, 61.36, 61.83, 61.44, 62.22, 62.45,
#     62.17, 61.66
# ])

# SPL11 = np.array([
#     59.87, 60.52, 61.41, 61.64, 61.68, 61.79, 62.37, 63.61, 62.30, 62.40,
#     62.61, 62.56, 63.77, 65.72, 68.60, 67.90, 61.63, 61.66, 62.67, 63.30,
#     64.73, 66.36, 68.02, 67.39, 60.45, 61.01, 61.70, 61.41, 62.02, 62.12,
#     62.19, 63.36
# ])

# SPL12 = np.array([
#     60.31, 60.52, 61.39, 61.15, 62.16, 61.82, 62.39, 63.88, 63.44, 61.84,
#     63.02, 63.78, 64.93, 66.92, 70.60, 68.61, 61.43, 61.45, 61.72, 61.69,
#     64.33, 65.72, 66.99, 67.16, 59.20, 59.77, 61.22, 60.89, 61.50, 61.13,
#     61.73, 62.85
# ])

SPLB1 = np.array([
    56.33, 57.03, 58.53, 59.35, 60.86, 61.99, 52.60, 51.83, 61.52, 62.28,
    63.05, 63.36, 64.15, 64.97, 57.90, 56.78, 60.55, 62.09, 62.31, 62.91,
    63.19, 63.70, 56.76, 56.78, 57.97, 59.43, 60.84, 60.64, 60.41, 60.57,
    55.65, 55.30
])

SPLB2 = np.array([
    61.49, 59.74, 62.57, 63.97, 65.62, 66.28, 63.93, 62.97, 64.56, 65.40,
    67.03, 66.37, 67.04, 67.81, 63.80, 66.31, 63.29, 64.90, 66.38, 66.02,
    66.07, 67.13, 63.12, 63.77, 60.81, 62.56, 63.01, 63.19, 62.79, 63.80,
    61.25, 59.44
])
SPLB3 = np.array([
    58.43, 58.15, 59.66, 57.86, 57.18, 63.17, 64.01, 67.18, 63.99, 64.21,
    64.02, 65.71, 66.31, 66.07, 64.97, 67.89, 63.52, 63.78, 64.37, 64.47,
    63.60, 65.23, 64.52, 66.67, 62.24, 62.23, 59.36, 60.41, 61.61, 61.38,
    62.12, 64.46
])

SPLB4 = np.array([
    50.92, 54.64, 50.53, 51.31, 54.58, 55.51, 55.37, 62.98, 59.77, 59.14,
    59.41, 58.97, 59.05, 59.97, 60.64, 65.47, 57.25, 57.91, 58.43, 58.14,
    59.17, 60.72, 60.93, 64.35, 54.89, 55.61, 57.05, 57.87, 58.27, 59.37,
    59.31, 61.32
])

SPLB5 = np.array([
    60.82, 61.54, 64.32, 64.77, 65.55, 65.98, 66.67, 64.80, 63.54, 65.20,
    65.59, 65.96, 67.15, 68.37, 66.10, 64.93, 63.36, 63.98, 65.25, 65.85,
    66.56, 67.77, 62.70, 62.70, 62.02, 63.51, 64.28, 64.78, 63.81, 64.21,
    60.22, 59.70
])
SPLB6 = np.array([
    61.40, 63.27, 62.71, 63.97, 65.03, 65.96, 59.26, 61.97, 64.07, 65.06,
    65.89, 66.20, 67.62, 68.10, 67.25, 67.14, 63.62, 64.14, 65.08, 66.23,
    67.12, 67.12, 66.04, 62.92, 61.98, 62.92, 63.82, 63.13, 62.98, 64.76,
    62.09, 61.50
])

SPLB7 = np.array([
    58.68, 59.17, 58.64, 59.60, 62.20, 58.45, 60.82, 65.84, 64.06, 63.65,
    63.86, 65.52, 66.39, 67.06, 68.02, 68.60, 63.56, 63.69, 63.63, 65.23,
    65.91, 64.98, 66.61, 67.46, 62.93, 62.90, 61.80, 59.71, 61.63, 61.99,
    61.63, 63.87
])

SPLB8 = np.array([
    61.15, 61.96, 61.82, 64.20, 64.84, 65.61, 66.89, 66.43, 62.49, 63.10,
    62.58, 63.69, 63.59, 65.48, 66.55, 68.71, 62.78, 62.33, 61.89, 63.18,
    64.96, 63.32, 64.13, 67.80, 60.75, 59.16, 61.44, 61.73, 61.97, 62.67,
    61.56, 64.33
])

SPLB9 = np.array([
    62.23, 62.91, 63.15, 64.21, 64.51, 65.04, 64.25, 61.82, 63.88, 64.78,
    65.86, 66.02, 67.30, 67.91, 67.31, 67.22, 63.10, 64.27, 64.86, 65.50,
    66.61, 68.01, 62.74, 63.77, 62.42, 63.13, 63.91, 63.14, 64.38, 64.19,
    58.58, 59.40
])

SPLB10 = np.array([
    61.78, 62.25, 63.33, 63.29, 64.26, 64.63, 59.91, 59.84, 63.86, 63.86,
    64.79, 65.14, 66.89, 67.89, 61.92, 62.35, 63.68, 64.33, 65.40, 65.62,
    66.96, 68.15, 67.05, 66.34, 62.40, 62.85, 64.71, 64.61, 64.15, 64.73,
    63.39, 63.54
])

SPLB11 = np.array([
    60.19, 59.75, 59.27, 60.93, 61.45, 61.33, 61.49, 64.92, 62.03, 62.15,
    62.17, 62.22, 62.75, 62.46, 62.37, 68.08, 63.78, 63.65, 63.73, 64.82,
    66.19, 66.87, 68.43, 67.99, 61.95, 62.52, 62.92, 63.07, 63.94, 64.19,
    64.73, 65.08
])

SPLB12 = np.array([
    61.07, 61.79, 62.09, 61.95, 64.13, 62.70, 64.83, 65.31, 65.12, 64.30,
    64.55, 65.77, 66.86, 66.84, 67.89, 68.29, 61.51, 62.46, 60.77, 62.20,
    65.11, 62.07, 62.59, 68.00, 59.58, 59.77, 59.74, 62.66, 62.93, 62.72,
    62.93, 64.72
])

SPL = SPLB12

# --- Speaker position ---
# speaker_pos1 = np.array([-1.0, 5.508])
# speaker_pos2 = np.array([-0.606, 5.508])
# speaker_pos3 = np.array([-0.606, 6.199])
# speaker_pos4 = np.array([-1.000, 6.199])
# speaker_pos5 = np.array([0.257, 5.508])
# speaker_pos6 = np.array([0.580, 5.508])
# speaker_pos7 = np.array([0.580, 6.199])
# speaker_pos8 = np.array([0.257, 6.199])
# speaker_pos9 = np.array([1.520, 5.508])
# speaker_pos10 = np.array([1.830, 5.508])
# speaker_pos11 = np.array([1.830, 6.199])
# speaker_pos12 = np.array([1.520, 6.199])
speaker_posB1 = np.array([-1.000, 5.508])
speaker_posB2 = np.array([-0.606, 5.508])
speaker_posB3 = np.array([-0.606, 6.199])
speaker_posB4 = np.array([-1.000, 6.199])
speaker_posB5 = np.array([0.257, 5.508])
speaker_posB6 = np.array([0.580, 5.508])
speaker_posB7 = np.array([0.580, 6.199])
speaker_posB8 = np.array([0.257, 6.199])
speaker_posB9 = np.array([1.520, 5.508])
speaker_posB10 = np.array([1.830, 5.508])
speaker_posB11 = np.array([1.830, 6.199])
speaker_posB12 = np.array([1.520, 6.199])

speaker_pos = speaker_posB12

# --- Create grid for heatmap ---
x = np.unique(mic_positions[:, 0])
y = np.unique(mic_positions[:, 1])
X, Y = np.meshgrid(x, y)
Z = SPL.reshape(len(x), len(y)).T

# --- Plot heatmap ---
plt.figure(figsize=(7, 6))
heatmap = plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')
plt.colorbar(heatmap, label='A-weighted SPL [dB(A)]')

for i, box in enumerate(boxes):
    # Close the loop for plotting
    x_box = np.append(box[:, 0], box[0, 0])
    y_box = np.append(box[:, 1], box[0, 1])
    label = 'Box' if i == 0 else None  # only label first one
    plt.plot(x_box, y_box, 'b-', linewidth=2, label=label, zorder=2)

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
plt.title('A-weighted SPL Heatmap Speaker B_12')

# --- Move legend outside the plot ---
plt.legend(loc='center left', bbox_to_anchor=(1.22, 0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# # -------------------------------------------------

# Index: 1 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.56 m , A-weighted SPL: 57.92 dB(A)
# Index: 1 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.57 m , A-weighted SPL: 58.94 dB(A)
# Index: 1 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 59.53 dB(A)
# Index: 1 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.62 m , A-weighted SPL: 61.00 dB(A)
# Index: 1 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.68 m , A-weighted SPL: 64.35 dB(A)
# Index: 1 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.90 m , A-weighted SPL: 69.93 dB(A)
# Index: 1 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 0.89 m , A-weighted SPL: 70.03 dB(A)
# Index: 1 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 1.67 m , A-weighted SPL: 64.34 dB(A)
# Index: 1 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.91 m , A-weighted SPL: 59.91 dB(A)
# Index: 1 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.99 m , A-weighted SPL: 60.94 dB(A)
# Index: 1 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.11 m , A-weighted SPL: 61.14 dB(A)
# Index: 1 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 61.88 dB(A)
# Index: 1 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.61 m , A-weighted SPL: 63.54 dB(A)
# Index: 1 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.19 m , A-weighted SPL: 64.46 dB(A)
# Index: 1 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.19 m , A-weighted SPL: 63.28 dB(A)
# Index: 1 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 2.60 m , A-weighted SPL: 62.09 dB(A)
# Index: 1 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.85 m , A-weighted SPL: 59.47 dB(A)
# Index: 1 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 60.35 dB(A)
# Index: 1 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 5.37 m , A-weighted SPL: 60.90 dB(A)
# Index: 1 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.78 m , A-weighted SPL: 61.16 dB(A)
# Index: 1 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.34 m , A-weighted SPL: 61.49 dB(A)
# Index: 1 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.10 m , A-weighted SPL: 62.05 dB(A)
# Index: 1 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.10 m , A-weighted SPL: 60.30 dB(A)
# Index: 1 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 4.33 m , A-weighted SPL: 60.42 dB(A)
# Index: 1 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 8.18 m , A-weighted SPL: 57.39 dB(A)
# Index: 1 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 7.54 m , A-weighted SPL: 57.83 dB(A)
# Index: 1 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.99 m , A-weighted SPL: 58.98 dB(A)
# Index: 1 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.55 m , A-weighted SPL: 58.90 dB(A)
# Index: 1 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.23 m , A-weighted SPL: 59.45 dB(A)
# Index: 1 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 58.80 dB(A)
# Index: 1 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.07 m , A-weighted SPL: 58.01 dB(A)
# Index: 1 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 1.254] , Distance: 6.23 m , A-weighted SPL: 57.02 dB(A)

# Index: 2 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.57 m , A-weighted SPL: 59.53 dB(A)
# Index: 2 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.59 m , A-weighted SPL: 60.63 dB(A)
# Index: 2 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.61 m , A-weighted SPL: 61.10 dB(A)
# Index: 2 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.65 m , A-weighted SPL: 61.55 dB(A)
# Index: 2 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 65.16 dB(A)
# Index: 2 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.98 m , A-weighted SPL: 69.14 dB(A)
# Index: 2 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 0.98 m , A-weighted SPL: 69.23 dB(A)
# Index: 2 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.71 m , A-weighted SPL: 64.53 dB(A)
# Index: 2 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.79 m , A-weighted SPL: 62.25 dB(A)
# Index: 2 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.84 m , A-weighted SPL: 63.51 dB(A)
# Index: 2 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.93 m , A-weighted SPL: 63.47 dB(A)
# Index: 2 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.07 m , A-weighted SPL: 64.39 dB(A)
# Index: 2 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.33 m , A-weighted SPL: 65.51 dB(A)
# Index: 2 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 66.36 dB(A)
# Index: 2 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 66.22 dB(A)
# Index: 2 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 2.32 m , A-weighted SPL: 64.92 dB(A)
# Index: 2 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.63 m , A-weighted SPL: 61.43 dB(A)
# Index: 2 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.82 m , A-weighted SPL: 62.55 dB(A)
# Index: 2 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.09 m , A-weighted SPL: 62.76 dB(A)
# Index: 2 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 4.46 m , A-weighted SPL: 63.05 dB(A)
# Index: 2 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.98 m , A-weighted SPL: 64.13 dB(A)
# Index: 2 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.72 m , A-weighted SPL: 64.71 dB(A)
# Index: 2 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.72 m , A-weighted SPL: 63.18 dB(A)
# Index: 2 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 3.97 m , A-weighted SPL: 63.21 dB(A)
# Index: 2 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 7.89 m , A-weighted SPL: 59.84 dB(A)
# Index: 2 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 7.23 m , A-weighted SPL: 60.96 dB(A)
# Index: 2 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.66 m , A-weighted SPL: 61.39 dB(A)
# Index: 2 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 6.19 m , A-weighted SPL: 61.10 dB(A)
# Index: 2 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.85 m , A-weighted SPL: 62.15 dB(A)
# Index: 2 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 61.41 dB(A)
# Index: 2 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 60.26 dB(A)
# Index: 2 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 1.254] , Distance: 5.85 m , A-weighted SPL: 59.92 dB(A)

# Index: 3 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.26 m , A-weighted SPL: 58.38 dB(A)
# Index: 3 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.27 m , A-weighted SPL: 59.49 dB(A)
# Index: 3 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.28 m , A-weighted SPL: 60.57 dB(A)
# Index: 3 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.31 m , A-weighted SPL: 60.79 dB(A)
# Index: 3 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.36 m , A-weighted SPL: 62.47 dB(A)
# Index: 3 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.47 m , A-weighted SPL: 65.72 dB(A)
# Index: 3 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 0.87 m , A-weighted SPL: 70.58 dB(A)
# Index: 3 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.16 m , A-weighted SPL: 67.51 dB(A)
# Index: 3 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.45 m , A-weighted SPL: 62.91 dB(A)
# Index: 3 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.49 m , A-weighted SPL: 62.53 dB(A)
# Index: 3 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.56 m , A-weighted SPL: 63.64 dB(A)
# Index: 3 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.66 m , A-weighted SPL: 63.77 dB(A)
# Index: 3 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.82 m , A-weighted SPL: 64.09 dB(A)
# Index: 3 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 2.14 m , A-weighted SPL: 65.41 dB(A)
# Index: 3 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.78 m , A-weighted SPL: 66.65 dB(A)
# Index: 3 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 1.94 m , A-weighted SPL: 66.72 dB(A)
# Index: 3 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.21 m , A-weighted SPL: 61.15 dB(A)
# Index: 3 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.37 m , A-weighted SPL: 61.29 dB(A)
# Index: 3 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.58 m , A-weighted SPL: 62.48 dB(A)
# Index: 3 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.88 m , A-weighted SPL: 62.99 dB(A)
# Index: 3 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 4.29 m , A-weighted SPL: 62.78 dB(A)
# Index: 3 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.87 m , A-weighted SPL: 63.85 dB(A)
# Index: 3 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.69 m , A-weighted SPL: 63.86 dB(A)
# Index: 3 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 3.77 m , A-weighted SPL: 65.53 dB(A)
# Index: 3 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 8.39 m , A-weighted SPL: 59.59 dB(A)
# Index: 3 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.68 m , A-weighted SPL: 60.86 dB(A)
# Index: 3 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 7.04 m , A-weighted SPL: 60.24 dB(A)
# Index: 3 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.50 m , A-weighted SPL: 60.43 dB(A)
# Index: 3 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 6.07 m , A-weighted SPL: 61.67 dB(A)
# Index: 3 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.78 m , A-weighted SPL: 60.64 dB(A)
# Index: 3 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.66 m , A-weighted SPL: 61.03 dB(A)
# Index: 3 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 1.254] , Distance: 5.71 m , A-weighted SPL: 63.70 dB(A)

# Index: 4 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.24 m , A-weighted SPL: 57.49 dB(A)
# Index: 4 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.25 m , A-weighted SPL: 58.19 dB(A)
# Index: 4 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.26 m , A-weighted SPL: 59.30 dB(A)
# Index: 4 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.28 m , A-weighted SPL: 59.88 dB(A)
# Index: 4 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.32 m , A-weighted SPL: 61.91 dB(A)
# Index: 4 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 1.41 m , A-weighted SPL: 65.63 dB(A)
# Index: 4 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 0.77 m , A-weighted SPL: 71.67 dB(A)
# Index: 4 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 1.09 m , A-weighted SPL: 67.98 dB(A)
# Index: 4 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.56 m , A-weighted SPL: 59.09 dB(A)
# Index: 4 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.62 m , A-weighted SPL: 59.74 dB(A)
# Index: 4 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.71 m , A-weighted SPL: 60.45 dB(A)
# Index: 4 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.85 m , A-weighted SPL: 60.02 dB(A)
# Index: 4 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 3.06 m , A-weighted SPL: 61.03 dB(A)
# Index: 4 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.45 m , A-weighted SPL: 62.73 dB(A)
# Index: 4 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.14 m , A-weighted SPL: 63.68 dB(A)
# Index: 4 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 2.28 m , A-weighted SPL: 65.00 dB(A)
# Index: 4 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.42 m , A-weighted SPL: 58.44 dB(A)
# Index: 4 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.60 m , A-weighted SPL: 58.81 dB(A)
# Index: 4 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.85 m , A-weighted SPL: 60.34 dB(A)
# Index: 4 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 5.18 m , A-weighted SPL: 60.28 dB(A)
# Index: 4 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.63 m , A-weighted SPL: 60.69 dB(A)
# Index: 4 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.24 m , A-weighted SPL: 61.26 dB(A)
# Index: 4 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.07 m , A-weighted SPL: 61.73 dB(A)
# Index: 4 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 4.15 m , A-weighted SPL: 63.67 dB(A)
# Index: 4 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 8.66 m , A-weighted SPL: 56.64 dB(A)
# Index: 4 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.97 m , A-weighted SPL: 56.71 dB(A)
# Index: 4 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 7.36 m , A-weighted SPL: 57.86 dB(A)
# Index: 4 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.84 m , A-weighted SPL: 57.74 dB(A)
# Index: 4 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.43 m , A-weighted SPL: 59.04 dB(A)
# Index: 4 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.16 m , A-weighted SPL: 58.14 dB(A)
# Index: 4 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.05 m , A-weighted SPL: 58.56 dB(A)
# Index: 4 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 1.254] , Distance: 6.10 m , A-weighted SPL: 60.59 dB(A)

# Index: 5 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.70 m , A-weighted SPL: 60.42 dB(A)
# Index: 5 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.74 m , A-weighted SPL: 61.34 dB(A)
# Index: 5 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.80 m , A-weighted SPL: 60.64 dB(A)
# Index: 5 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.90 m , A-weighted SPL: 63.11 dB(A)
# Index: 5 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.10 m , A-weighted SPL: 64.54 dB(A)
# Index: 5 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.55 m , A-weighted SPL: 65.57 dB(A)
# Index: 5 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.54 m , A-weighted SPL: 65.59 dB(A)
# Index: 5 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.09 m , A-weighted SPL: 64.00 dB(A)
# Index: 5 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.61 m , A-weighted SPL: 61.94 dB(A)
# Index: 5 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.63 m , A-weighted SPL: 63.45 dB(A)
# Index: 5 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.66 m , A-weighted SPL: 64.04 dB(A)
# Index: 5 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.72 m , A-weighted SPL: 64.76 dB(A)
# Index: 5 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.84 m , A-weighted SPL: 66.48 dB(A)
# Index: 5 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.17 m , A-weighted SPL: 68.83 dB(A)
# Index: 5 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.16 m , A-weighted SPL: 68.23 dB(A)
# Index: 5 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 1.83 m , A-weighted SPL: 65.27 dB(A)
# Index: 5 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.20 m , A-weighted SPL: 61.80 dB(A)
# Index: 5 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.33 m , A-weighted SPL: 62.17 dB(A)
# Index: 5 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.52 m , A-weighted SPL: 63.40 dB(A)
# Index: 5 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.79 m , A-weighted SPL: 63.39 dB(A)
# Index: 5 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.22 m , A-weighted SPL: 64.50 dB(A)
# Index: 5 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.89 m , A-weighted SPL: 64.86 dB(A)
# Index: 5 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 2.88 m , A-weighted SPL: 64.39 dB(A)
# Index: 5 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 3.21 m , A-weighted SPL: 63.99 dB(A)
# Index: 5 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 7.31 m , A-weighted SPL: 60.25 dB(A)
# Index: 5 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 6.59 m , A-weighted SPL: 60.67 dB(A)
# Index: 5 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.95 m , A-weighted SPL: 60.65 dB(A)
# Index: 5 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.42 m , A-weighted SPL: 61.46 dB(A)
# Index: 5 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.03 m , A-weighted SPL: 61.85 dB(A)
# Index: 5 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 61.21 dB(A)
# Index: 5 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 61.07 dB(A)
# Index: 5 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 1.254] , Distance: 5.03 m , A-weighted SPL: 60.41 dB(A)

# Index: 6 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.78 m , A-weighted SPL: 60.21 dB(A)
# Index: 6 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.83 m , A-weighted SPL: 60.59 dB(A)
# Index: 6 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.92 m , A-weighted SPL: 61.80 dB(A)
# Index: 6 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.06 m , A-weighted SPL: 61.62 dB(A)
# Index: 6 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.31 m , A-weighted SPL: 64.31 dB(A)
# Index: 6 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.82 m , A-weighted SPL: 64.76 dB(A)
# Index: 6 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.82 m , A-weighted SPL: 64.14 dB(A)
# Index: 6 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.30 m , A-weighted SPL: 63.45 dB(A)
# Index: 6 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.57 m , A-weighted SPL: 62.16 dB(A)
# Index: 6 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.59 m , A-weighted SPL: 63.40 dB(A)
# Index: 6 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.61 m , A-weighted SPL: 63.79 dB(A)
# Index: 6 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.65 m , A-weighted SPL: 64.52 dB(A)
# Index: 6 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 65.90 dB(A)
# Index: 6 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.00 m , A-weighted SPL: 69.40 dB(A)
# Index: 6 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 0.99 m , A-weighted SPL: 69.53 dB(A)
# Index: 6 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 1.72 m , A-weighted SPL: 65.84 dB(A)
# Index: 6 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.06 m , A-weighted SPL: 61.56 dB(A)
# Index: 6 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.17 m , A-weighted SPL: 62.54 dB(A)
# Index: 6 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.33 m , A-weighted SPL: 63.21 dB(A)
# Index: 6 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 3.56 m , A-weighted SPL: 63.40 dB(A)
# Index: 6 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.95 m , A-weighted SPL: 64.76 dB(A)
# Index: 6 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.58 m , A-weighted SPL: 65.52 dB(A)
# Index: 6 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.58 m , A-weighted SPL: 65.49 dB(A)
# Index: 6 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 2.94 m , A-weighted SPL: 64.44 dB(A)
# Index: 6 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 7.10 m , A-weighted SPL: 59.81 dB(A)
# Index: 6 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 6.36 m , A-weighted SPL: 60.95 dB(A)
# Index: 6 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.69 m , A-weighted SPL: 61.02 dB(A)
# Index: 6 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 5.14 m , A-weighted SPL: 61.15 dB(A)
# Index: 6 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.73 m , A-weighted SPL: 62.07 dB(A)
# Index: 6 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.51 m , A-weighted SPL: 62.20 dB(A)
# Index: 6 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.51 m , A-weighted SPL: 60.75 dB(A)
# Index: 6 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 1.254] , Distance: 4.72 m , A-weighted SPL: 61.29 dB(A)

# Index: 7 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.44 m , A-weighted SPL: 59.89 dB(A)
# Index: 7 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.48 m , A-weighted SPL: 59.82 dB(A)
# Index: 7 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.55 m , A-weighted SPL: 60.82 dB(A)
# Index: 7 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.65 m , A-weighted SPL: 61.42 dB(A)
# Index: 7 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.81 m , A-weighted SPL: 61.97 dB(A)
# Index: 7 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.12 m , A-weighted SPL: 63.71 dB(A)
# Index: 7 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.76 m , A-weighted SPL: 64.78 dB(A)
# Index: 7 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.92 m , A-weighted SPL: 65.49 dB(A)
# Index: 7 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.26 m , A-weighted SPL: 62.11 dB(A)
# Index: 7 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.27 m , A-weighted SPL: 61.90 dB(A)
# Index: 7 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.29 m , A-weighted SPL: 63.41 dB(A)
# Index: 7 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.31 m , A-weighted SPL: 63.80 dB(A)
# Index: 7 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.36 m , A-weighted SPL: 64.62 dB(A)
# Index: 7 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.47 m , A-weighted SPL: 66.88 dB(A)
# Index: 7 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 0.88 m , A-weighted SPL: 70.86 dB(A)
# Index: 7 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 1.17 m , A-weighted SPL: 68.71 dB(A)
# Index: 7 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.70 m , A-weighted SPL: 60.99 dB(A)
# Index: 7 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.78 m , A-weighted SPL: 61.62 dB(A)
# Index: 7 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.90 m , A-weighted SPL: 63.12 dB(A)
# Index: 7 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.08 m , A-weighted SPL: 63.00 dB(A)
# Index: 7 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 3.35 m , A-weighted SPL: 63.91 dB(A)
# Index: 7 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.80 m , A-weighted SPL: 65.96 dB(A)
# Index: 7 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.54 m , A-weighted SPL: 65.15 dB(A)
# Index: 7 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 2.66 m , A-weighted SPL: 66.04 dB(A)
# Index: 7 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 7.65 m , A-weighted SPL: 60.09 dB(A)
# Index: 7 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.86 m , A-weighted SPL: 60.49 dB(A)
# Index: 7 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 6.14 m , A-weighted SPL: 60.42 dB(A)
# Index: 7 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 5.51 m , A-weighted SPL: 59.75 dB(A)
# Index: 7 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.99 m , A-weighted SPL: 61.74 dB(A)
# Index: 7 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.64 m , A-weighted SPL: 61.48 dB(A)
# Index: 7 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.49 m , A-weighted SPL: 61.30 dB(A)
# Index: 7 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 1.254] , Distance: 4.55 m , A-weighted SPL: 62.79 dB(A)

# Index: 8 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.37 m , A-weighted SPL: 60.26 dB(A)
# Index: 8 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.40 m , A-weighted SPL: 59.84 dB(A)
# Index: 8 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.45 m , A-weighted SPL: 60.97 dB(A)
# Index: 8 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.52 m , A-weighted SPL: 61.85 dB(A)
# Index: 8 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.64 m , A-weighted SPL: 62.87 dB(A)
# Index: 8 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.89 m , A-weighted SPL: 64.94 dB(A)
# Index: 8 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.48 m , A-weighted SPL: 65.98 dB(A)
# Index: 8 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.67 m , A-weighted SPL: 65.55 dB(A)
# Index: 8 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.29 m , A-weighted SPL: 62.10 dB(A)
# Index: 8 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.30 m , A-weighted SPL: 61.83 dB(A)
# Index: 8 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.33 m , A-weighted SPL: 62.58 dB(A)
# Index: 8 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.37 m , A-weighted SPL: 63.42 dB(A)
# Index: 8 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.44 m , A-weighted SPL: 64.42 dB(A)
# Index: 8 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.60 m , A-weighted SPL: 66.10 dB(A)
# Index: 8 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.07 m , A-weighted SPL: 69.17 dB(A)
# Index: 8 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 1.32 m , A-weighted SPL: 68.84 dB(A)
# Index: 8 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.82 m , A-weighted SPL: 61.45 dB(A)
# Index: 8 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.93 m , A-weighted SPL: 61.78 dB(A)
# Index: 8 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.07 m , A-weighted SPL: 62.72 dB(A)
# Index: 8 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.28 m , A-weighted SPL: 64.00 dB(A)
# Index: 8 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.59 m , A-weighted SPL: 63.56 dB(A)
# Index: 8 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 3.09 m , A-weighted SPL: 64.85 dB(A)
# Index: 8 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.85 m , A-weighted SPL: 64.86 dB(A)
# Index: 8 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 2.95 m , A-weighted SPL: 66.51 dB(A)
# Index: 8 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 7.84 m , A-weighted SPL: 59.41 dB(A)
# Index: 8 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 7.08 m , A-weighted SPL: 59.36 dB(A)
# Index: 8 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 6.38 m , A-weighted SPL: 60.81 dB(A)
# Index: 8 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.77 m , A-weighted SPL: 61.06 dB(A)
# Index: 8 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 5.28 m , A-weighted SPL: 61.08 dB(A)
# Index: 8 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.95 m , A-weighted SPL: 61.30 dB(A)
# Index: 8 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.81 m , A-weighted SPL: 61.71 dB(A)
# Index: 8 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 1.254] , Distance: 4.87 m , A-weighted SPL: 63.22 dB(A)

# Index: 9 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.10 m , A-weighted SPL: 58.91 dB(A)
# Index: 9 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.22 m , A-weighted SPL: 61.31 dB(A)
# Index: 9 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.38 m , A-weighted SPL: 61.77 dB(A)
# Index: 9 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.63 m , A-weighted SPL: 62.23 dB(A)
# Index: 9 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.03 m , A-weighted SPL: 62.67 dB(A)
# Index: 9 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.68 m , A-weighted SPL: 62.88 dB(A)
# Index: 9 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.67 m , A-weighted SPL: 62.43 dB(A)
# Index: 9 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.02 m , A-weighted SPL: 61.77 dB(A)
# Index: 9 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.58 m , A-weighted SPL: 62.32 dB(A)
# Index: 9 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.60 m , A-weighted SPL: 62.72 dB(A)
# Index: 9 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.62 m , A-weighted SPL: 63.82 dB(A)
# Index: 9 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.67 m , A-weighted SPL: 64.32 dB(A)
# Index: 9 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.76 m , A-weighted SPL: 65.97 dB(A)
# Index: 9 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.04 m , A-weighted SPL: 69.30 dB(A)
# Index: 9 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.03 m , A-weighted SPL: 69.38 dB(A)
# Index: 9 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.75 m , A-weighted SPL: 65.87 dB(A)
# Index: 9 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.75 m , A-weighted SPL: 61.17 dB(A)
# Index: 9 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.80 m , A-weighted SPL: 62.63 dB(A)
# Index: 9 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.88 m , A-weighted SPL: 63.20 dB(A)
# Index: 9 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.01 m , A-weighted SPL: 63.32 dB(A)
# Index: 9 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.24 m , A-weighted SPL: 65.00 dB(A)
# Index: 9 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 66.71 dB(A)
# Index: 9 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 1.73 m , A-weighted SPL: 66.39 dB(A)
# Index: 9 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 2.23 m , A-weighted SPL: 64.67 dB(A)
# Index: 9 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 6.56 m , A-weighted SPL: 59.68 dB(A)
# Index: 9 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.74 m , A-weighted SPL: 61.10 dB(A)
# Index: 9 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 5.00 m , A-weighted SPL: 60.95 dB(A)
# Index: 9 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 4.35 m , A-weighted SPL: 61.50 dB(A)
# Index: 9 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.87 m , A-weighted SPL: 61.93 dB(A)
# Index: 9 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.60 m , A-weighted SPL: 61.47 dB(A)
# Index: 9 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 60.72 dB(A)
# Index: 9 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 1.254] , Distance: 3.86 m , A-weighted SPL: 60.62 dB(A)

# Index: 10 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.24 m , A-weighted SPL: 60.41 dB(A)
# Index: 10 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.37 m , A-weighted SPL: 61.60 dB(A)
# Index: 10 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.57 m , A-weighted SPL: 61.65 dB(A)
# Index: 10 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.85 m , A-weighted SPL: 61.66 dB(A)
# Index: 10 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 62.48 dB(A)
# Index: 10 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.97 m , A-weighted SPL: 62.30 dB(A)
# Index: 10 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.97 m , A-weighted SPL: 61.77 dB(A)
# Index: 10 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 61.11 dB(A)
# Index: 10 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.62 m , A-weighted SPL: 61.97 dB(A)
# Index: 10 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.64 m , A-weighted SPL: 62.83 dB(A)
# Index: 10 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.68 m , A-weighted SPL: 63.37 dB(A)
# Index: 10 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.75 m , A-weighted SPL: 63.82 dB(A)
# Index: 10 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.88 m , A-weighted SPL: 65.83 dB(A)
# Index: 10 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.23 m , A-weighted SPL: 68.18 dB(A)
# Index: 10 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.22 m , A-weighted SPL: 67.68 dB(A)
# Index: 10 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.86 m , A-weighted SPL: 64.73 dB(A)
# Index: 10 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.68 m , A-weighted SPL: 61.52 dB(A)
# Index: 10 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.72 m , A-weighted SPL: 62.33 dB(A)
# Index: 10 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.77 m , A-weighted SPL: 62.66 dB(A)
# Index: 10 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.87 m , A-weighted SPL: 63.43 dB(A)
# Index: 10 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.05 m , A-weighted SPL: 65.48 dB(A)
# Index: 10 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.48 m , A-weighted SPL: 67.39 dB(A)
# Index: 10 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 1.47 m , A-weighted SPL: 67.30 dB(A)
# Index: 10 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 2.04 m , A-weighted SPL: 65.25 dB(A)
# Index: 10 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 6.40 m , A-weighted SPL: 60.75 dB(A)
# Index: 10 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 5.56 m , A-weighted SPL: 61.36 dB(A)
# Index: 10 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.79 m , A-weighted SPL: 61.83 dB(A)
# Index: 10 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 4.11 m , A-weighted SPL: 61.44 dB(A)
# Index: 10 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.59 m , A-weighted SPL: 62.22 dB(A)
# Index: 10 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.30 m , A-weighted SPL: 62.45 dB(A)
# Index: 10 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.29 m , A-weighted SPL: 62.17 dB(A)
# Index: 10 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 1.254] , Distance: 3.58 m , A-weighted SPL: 61.66 dB(A)

# Index: 11 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.86 m , A-weighted SPL: 59.87 dB(A)
# Index: 11 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.97 m , A-weighted SPL: 60.52 dB(A)
# Index: 11 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.12 m , A-weighted SPL: 61.41 dB(A)
# Index: 11 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.34 m , A-weighted SPL: 61.64 dB(A)
# Index: 11 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.66 m , A-weighted SPL: 61.68 dB(A)
# Index: 11 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.16 m , A-weighted SPL: 61.79 dB(A)
# Index: 11 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.93 m , A-weighted SPL: 62.37 dB(A)
# Index: 11 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.03 m , A-weighted SPL: 63.61 dB(A)
# Index: 11 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.30 m , A-weighted SPL: 62.30 dB(A)
# Index: 11 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.32 m , A-weighted SPL: 62.40 dB(A)
# Index: 11 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.34 m , A-weighted SPL: 62.61 dB(A)
# Index: 11 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.39 m , A-weighted SPL: 62.56 dB(A)
# Index: 11 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.47 m , A-weighted SPL: 63.77 dB(A)
# Index: 11 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.64 m , A-weighted SPL: 65.72 dB(A)
# Index: 11 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.13 m , A-weighted SPL: 68.60 dB(A)
# Index: 11 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.37 m , A-weighted SPL: 67.90 dB(A)
# Index: 11 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.35 m , A-weighted SPL: 61.63 dB(A)
# Index: 11 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.38 m , A-weighted SPL: 61.66 dB(A)
# Index: 11 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.42 m , A-weighted SPL: 62.67 dB(A)
# Index: 11 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.49 m , A-weighted SPL: 63.30 dB(A)
# Index: 11 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 2.60 m , A-weighted SPL: 64.73 dB(A)
# Index: 11 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.83 m , A-weighted SPL: 66.36 dB(A)
# Index: 11 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.40 m , A-weighted SPL: 68.02 dB(A)
# Index: 11 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 1.60 m , A-weighted SPL: 67.39 dB(A)
# Index: 11 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 7.00 m , A-weighted SPL: 60.45 dB(A)
# Index: 11 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 6.13 m , A-weighted SPL: 61.01 dB(A)
# Index: 11 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 5.31 m , A-weighted SPL: 61.70 dB(A)
# Index: 11 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 4.56 m , A-weighted SPL: 61.41 dB(A)
# Index: 11 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.93 m , A-weighted SPL: 62.02 dB(A)
# Index: 11 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.47 m , A-weighted SPL: 62.12 dB(A)
# Index: 11 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.26 m , A-weighted SPL: 62.19 dB(A)
# Index: 11 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 1.254] , Distance: 3.35 m , A-weighted SPL: 63.36 dB(A)

# Index: 12 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.73 m , A-weighted SPL: 60.31 dB(A)
# Index: 12 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.83 m , A-weighted SPL: 60.52 dB(A)
# Index: 12 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.95 m , A-weighted SPL: 61.39 dB(A)
# Index: 12 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.14 m , A-weighted SPL: 61.15 dB(A)
# Index: 12 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.43 m , A-weighted SPL: 62.16 dB(A)
# Index: 12 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.89 m , A-weighted SPL: 61.82 dB(A)
# Index: 12 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.64 m , A-weighted SPL: 62.39 dB(A)
# Index: 12 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.75 m , A-weighted SPL: 63.88 dB(A)
# Index: 12 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.27 m , A-weighted SPL: 63.44 dB(A)
# Index: 12 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.28 m , A-weighted SPL: 61.84 dB(A)
# Index: 12 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.30 m , A-weighted SPL: 63.02 dB(A)
# Index: 12 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.33 m , A-weighted SPL: 63.78 dB(A)
# Index: 12 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.38 m , A-weighted SPL: 64.93 dB(A)
# Index: 12 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.50 m , A-weighted SPL: 66.92 dB(A)
# Index: 12 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 0.93 m , A-weighted SPL: 70.60 dB(A)
# Index: 12 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.21 m , A-weighted SPL: 68.61 dB(A)
# Index: 12 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.42 m , A-weighted SPL: 61.43 dB(A)
# Index: 12 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.46 m , A-weighted SPL: 61.45 dB(A)
# Index: 12 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.51 m , A-weighted SPL: 61.72 dB(A)
# Index: 12 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.60 m , A-weighted SPL: 61.69 dB(A)
# Index: 12 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.75 m , A-weighted SPL: 64.33 dB(A)
# Index: 12 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 2.05 m , A-weighted SPL: 65.72 dB(A)
# Index: 12 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.67 m , A-weighted SPL: 66.99 dB(A)
# Index: 12 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 1.84 m , A-weighted SPL: 67.16 dB(A)
# Index: 12 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 7.15 m , A-weighted SPL: 59.20 dB(A)
# Index: 12 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 6.30 m , A-weighted SPL: 59.77 dB(A)
# Index: 12 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 5.50 m , A-weighted SPL: 61.22 dB(A)
# Index: 12 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.79 m , A-weighted SPL: 60.89 dB(A)
# Index: 12 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 4.18 m , A-weighted SPL: 61.50 dB(A)
# Index: 12 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.76 m , A-weighted SPL: 61.13 dB(A)
# Index: 12 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.56 m , A-weighted SPL: 61.73 dB(A)
# Index: 12 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 1.254] , Distance: 3.65 m , A-weighted SPL: 62.85 dB(A)

# #---------------------------------------------

# Index: 1 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 5.86 m , A-weighted SPL: 56.33 dB(A)
# Index: 1 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.93 m , A-weighted SPL: 57.03 dB(A)
# Index: 1 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.04 m , A-weighted SPL: 58.53 dB(A)
# Index: 1 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 3.21 m , A-weighted SPL: 59.35 dB(A)
# Index: 1 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.50 m , A-weighted SPL: 60.86 dB(A)
# Index: 1 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.06 m , A-weighted SPL: 61.99 dB(A)
# Index: 1 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.06 m , A-weighted SPL: 52.60 dB(A)
# Index: 1 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.50 m , A-weighted SPL: 51.83 dB(A)
# Index: 1 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.19 m , A-weighted SPL: 61.52 dB(A)
# Index: 1 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 5.32 m , A-weighted SPL: 62.28 dB(A)
# Index: 1 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.51 m , A-weighted SPL: 63.05 dB(A)
# Index: 1 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 3.78 m , A-weighted SPL: 63.36 dB(A)
# Index: 1 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 3.21 m , A-weighted SPL: 64.15 dB(A)
# Index: 1 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.87 m , A-weighted SPL: 64.97 dB(A)
# Index: 1 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 2.87 m , A-weighted SPL: 57.90 dB(A)
# Index: 1 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 3.20 m , A-weighted SPL: 56.78 dB(A)
# Index: 1 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 7.09 m , A-weighted SPL: 60.55 dB(A)
# Index: 1 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.35 m , A-weighted SPL: 62.09 dB(A)
# Index: 1 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 5.68 m , A-weighted SPL: 62.31 dB(A)
# Index: 1 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 5.13 m , A-weighted SPL: 62.91 dB(A)
# Index: 1 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.72 m , A-weighted SPL: 63.19 dB(A)
# Index: 1 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.50 m , A-weighted SPL: 63.70 dB(A)
# Index: 1 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.50 m , A-weighted SPL: 56.76 dB(A)
# Index: 1 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 4.71 m , A-weighted SPL: 56.78 dB(A)
# Index: 1 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 8.39 m , A-weighted SPL: 57.97 dB(A)
# Index: 1 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 7.77 m , A-weighted SPL: 59.43 dB(A)
# Index: 1 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 7.23 m , A-weighted SPL: 60.84 dB(A)
# Index: 1 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.80 m , A-weighted SPL: 60.64 dB(A)
# Index: 1 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.50 m , A-weighted SPL: 60.41 dB(A)
# Index: 1 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.34 m , A-weighted SPL: 60.57 dB(A)
# Index: 1 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.34 m , A-weighted SPL: 55.65 dB(A)
# Index: 1 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 5.508, 0.000] , Distance: 6.50 m , A-weighted SPL: 55.30 dB(A)

# Index: 2 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 5.87 m , A-weighted SPL: 61.49 dB(A)
# Index: 2 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.95 m , A-weighted SPL: 59.74 dB(A)
# Index: 2 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.06 m , A-weighted SPL: 62.57 dB(A)
# Index: 2 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 3.23 m , A-weighted SPL: 63.97 dB(A)
# Index: 2 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.54 m , A-weighted SPL: 65.62 dB(A)
# Index: 2 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.10 m , A-weighted SPL: 66.28 dB(A)
# Index: 2 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.10 m , A-weighted SPL: 63.93 dB(A)
# Index: 2 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.53 m , A-weighted SPL: 62.97 dB(A)
# Index: 2 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.08 m , A-weighted SPL: 64.56 dB(A)
# Index: 2 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 5.19 m , A-weighted SPL: 65.40 dB(A)
# Index: 2 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.35 m , A-weighted SPL: 67.03 dB(A)
# Index: 2 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 3.59 m , A-weighted SPL: 66.37 dB(A)
# Index: 2 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.98 m , A-weighted SPL: 67.04 dB(A)
# Index: 2 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.61 m , A-weighted SPL: 67.81 dB(A)
# Index: 2 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.61 m , A-weighted SPL: 63.80 dB(A)
# Index: 2 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 2.97 m , A-weighted SPL: 66.31 dB(A)
# Index: 2 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.88 m , A-weighted SPL: 63.29 dB(A)
# Index: 2 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.11 m , A-weighted SPL: 64.90 dB(A)
# Index: 2 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 5.41 m , A-weighted SPL: 66.38 dB(A)
# Index: 2 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.83 m , A-weighted SPL: 66.02 dB(A)
# Index: 2 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.39 m , A-weighted SPL: 66.07 dB(A)
# Index: 2 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.15 m , A-weighted SPL: 67.13 dB(A)
# Index: 2 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.15 m , A-weighted SPL: 63.12 dB(A)
# Index: 2 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 4.39 m , A-weighted SPL: 63.77 dB(A)
# Index: 2 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 8.11 m , A-weighted SPL: 60.81 dB(A)
# Index: 2 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 7.47 m , A-weighted SPL: 62.56 dB(A)
# Index: 2 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.91 m , A-weighted SPL: 63.01 dB(A)
# Index: 2 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.46 m , A-weighted SPL: 63.19 dB(A)
# Index: 2 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.14 m , A-weighted SPL: 62.79 dB(A)
# Index: 2 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 5.97 m , A-weighted SPL: 63.80 dB(A)
# Index: 2 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 5.97 m , A-weighted SPL: 61.25 dB(A)
# Index: 2 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 5.508, 0.000] , Distance: 6.14 m , A-weighted SPL: 59.44 dB(A)

# Index: 3 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.53 m , A-weighted SPL: 58.43 dB(A)
# Index: 3 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 5.58 m , A-weighted SPL: 58.15 dB(A)
# Index: 3 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.67 m , A-weighted SPL: 59.66 dB(A)
# Index: 3 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 3.79 m , A-weighted SPL: 57.86 dB(A)
# Index: 3 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 3.00 m , A-weighted SPL: 57.18 dB(A)
# Index: 3 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.36 m , A-weighted SPL: 63.17 dB(A)
# Index: 3 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.05 m , A-weighted SPL: 64.01 dB(A)
# Index: 3 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.19 m , A-weighted SPL: 67.18 dB(A)
# Index: 3 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.71 m , A-weighted SPL: 63.99 dB(A)
# Index: 3 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 5.80 m , A-weighted SPL: 64.21 dB(A)
# Index: 3 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.92 m , A-weighted SPL: 64.02 dB(A)
# Index: 3 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.10 m , A-weighted SPL: 65.71 dB(A)
# Index: 3 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 3.38 m , A-weighted SPL: 66.31 dB(A)
# Index: 3 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.83 m , A-weighted SPL: 66.07 dB(A)
# Index: 3 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.57 m , A-weighted SPL: 64.97 dB(A)
# Index: 3 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 2.69 m , A-weighted SPL: 67.89 dB(A)
# Index: 3 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 7.45 m , A-weighted SPL: 63.52 dB(A)
# Index: 3 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.64 m , A-weighted SPL: 63.78 dB(A)
# Index: 3 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 5.89 m , A-weighted SPL: 64.37 dB(A)
# Index: 3 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 5.22 m , A-weighted SPL: 64.47 dB(A)
# Index: 3 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.67 m , A-weighted SPL: 63.60 dB(A)
# Index: 3 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.29 m , A-weighted SPL: 65.23 dB(A)
# Index: 3 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.13 m , A-weighted SPL: 64.52 dB(A)
# Index: 3 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 4.20 m , A-weighted SPL: 66.67 dB(A)
# Index: 3 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 8.59 m , A-weighted SPL: 62.24 dB(A)
# Index: 3 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 7.90 m , A-weighted SPL: 62.23 dB(A)
# Index: 3 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 7.28 m , A-weighted SPL: 59.36 dB(A)
# Index: 3 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.76 m , A-weighted SPL: 60.41 dB(A)
# Index: 3 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.35 m , A-weighted SPL: 61.61 dB(A)
# Index: 3 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.07 m , A-weighted SPL: 61.38 dB(A)
# Index: 3 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 5.96 m , A-weighted SPL: 62.12 dB(A)
# Index: 3 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-0.606, 6.199, 0.000] , Distance: 6.01 m , A-weighted SPL: 64.46 dB(A)

# Index: 4 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.51 m , A-weighted SPL: 50.92 dB(A)
# Index: 4 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 5.57 m , A-weighted SPL: 54.64 dB(A)
# Index: 4 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.65 m , A-weighted SPL: 50.53 dB(A)
# Index: 4 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 3.77 m , A-weighted SPL: 51.31 dB(A)
# Index: 4 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.97 m , A-weighted SPL: 54.58 dB(A)
# Index: 4 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.33 m , A-weighted SPL: 55.51 dB(A)
# Index: 4 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.01 m , A-weighted SPL: 55.37 dB(A)
# Index: 4 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.15 m , A-weighted SPL: 62.98 dB(A)
# Index: 4 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.81 m , A-weighted SPL: 59.77 dB(A)
# Index: 4 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 5.92 m , A-weighted SPL: 59.14 dB(A)
# Index: 4 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 5.06 m , A-weighted SPL: 59.41 dB(A)
# Index: 4 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.27 m , A-weighted SPL: 58.97 dB(A)
# Index: 4 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 3.58 m , A-weighted SPL: 59.05 dB(A)
# Index: 4 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 3.07 m , A-weighted SPL: 59.97 dB(A)
# Index: 4 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.84 m , A-weighted SPL: 60.64 dB(A)
# Index: 4 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 2.94 m , A-weighted SPL: 65.47 dB(A)
# Index: 4 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 7.64 m , A-weighted SPL: 57.25 dB(A)
# Index: 4 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.86 m , A-weighted SPL: 57.91 dB(A)
# Index: 4 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.13 m , A-weighted SPL: 58.43 dB(A)
# Index: 4 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 5.50 m , A-weighted SPL: 58.14 dB(A)
# Index: 4 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.98 m , A-weighted SPL: 59.17 dB(A)
# Index: 4 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.63 m , A-weighted SPL: 60.72 dB(A)
# Index: 4 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.48 m , A-weighted SPL: 60.93 dB(A)
# Index: 4 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 4.54 m , A-weighted SPL: 64.35 dB(A)
# Index: 4 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 8.86 m , A-weighted SPL: 54.89 dB(A)
# Index: 4 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 8.19 m , A-weighted SPL: 55.61 dB(A)
# Index: 4 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 7.59 m , A-weighted SPL: 57.05 dB(A)
# Index: 4 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 7.09 m , A-weighted SPL: 57.87 dB(A)
# Index: 4 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.70 m , A-weighted SPL: 58.27 dB(A)
# Index: 4 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.44 m , A-weighted SPL: 59.37 dB(A)
# Index: 4 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.33 m , A-weighted SPL: 59.31 dB(A)
# Index: 4 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [-1.000, 6.199, 0.000] , Distance: 6.38 m , A-weighted SPL: 61.32 dB(A)

# Index: 5 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.99 m , A-weighted SPL: 60.82 dB(A)
# Index: 5 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.09 m , A-weighted SPL: 61.54 dB(A)
# Index: 5 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 4.23 m , A-weighted SPL: 64.32 dB(A)
# Index: 5 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.45 m , A-weighted SPL: 64.77 dB(A)
# Index: 5 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.80 m , A-weighted SPL: 65.55 dB(A)
# Index: 5 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.42 m , A-weighted SPL: 65.98 dB(A)
# Index: 5 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.41 m , A-weighted SPL: 66.67 dB(A)
# Index: 5 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.79 m , A-weighted SPL: 64.80 dB(A)
# Index: 5 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.91 m , A-weighted SPL: 63.54 dB(A)
# Index: 5 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 4.99 m , A-weighted SPL: 65.20 dB(A)
# Index: 5 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 4.11 m , A-weighted SPL: 65.59 dB(A)
# Index: 5 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.29 m , A-weighted SPL: 65.96 dB(A)
# Index: 5 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.61 m , A-weighted SPL: 67.15 dB(A)
# Index: 5 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.19 m , A-weighted SPL: 68.37 dB(A)
# Index: 5 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.19 m , A-weighted SPL: 66.10 dB(A)
# Index: 5 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 2.60 m , A-weighted SPL: 64.93 dB(A)
# Index: 5 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 6.47 m , A-weighted SPL: 63.36 dB(A)
# Index: 5 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.64 m , A-weighted SPL: 63.98 dB(A)
# Index: 5 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 4.88 m , A-weighted SPL: 65.25 dB(A)
# Index: 5 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 4.22 m , A-weighted SPL: 65.85 dB(A)
# Index: 5 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.71 m , A-weighted SPL: 66.56 dB(A)
# Index: 5 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.43 m , A-weighted SPL: 67.77 dB(A)
# Index: 5 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.43 m , A-weighted SPL: 62.70 dB(A)
# Index: 5 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 3.71 m , A-weighted SPL: 62.70 dB(A)
# Index: 5 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 7.54 m , A-weighted SPL: 62.02 dB(A)
# Index: 5 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 6.84 m , A-weighted SPL: 63.51 dB(A)
# Index: 5 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 6.23 m , A-weighted SPL: 64.28 dB(A)
# Index: 5 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.73 m , A-weighted SPL: 64.78 dB(A)
# Index: 5 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.36 m , A-weighted SPL: 63.81 dB(A)
# Index: 5 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.17 m , A-weighted SPL: 64.21 dB(A)
# Index: 5 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.17 m , A-weighted SPL: 60.22 dB(A)
# Index: 5 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 5.508, 0.000] , Distance: 5.36 m , A-weighted SPL: 59.70 dB(A)

# Index: 6 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 6.07 m , A-weighted SPL: 61.40 dB(A)
# Index: 6 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.18 m , A-weighted SPL: 63.27 dB(A)
# Index: 6 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.34 m , A-weighted SPL: 62.71 dB(A)
# Index: 6 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.58 m , A-weighted SPL: 63.97 dB(A)
# Index: 6 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.96 m , A-weighted SPL: 65.03 dB(A)
# Index: 6 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.60 m , A-weighted SPL: 65.96 dB(A)
# Index: 6 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.60 m , A-weighted SPL: 59.26 dB(A)
# Index: 6 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.95 m , A-weighted SPL: 61.97 dB(A)
# Index: 6 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.87 m , A-weighted SPL: 64.07 dB(A)
# Index: 6 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.95 m , A-weighted SPL: 65.06 dB(A)
# Index: 6 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.06 m , A-weighted SPL: 65.89 dB(A)
# Index: 6 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.24 m , A-weighted SPL: 66.20 dB(A)
# Index: 6 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.54 m , A-weighted SPL: 67.62 dB(A)
# Index: 6 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.11 m , A-weighted SPL: 68.10 dB(A)
# Index: 6 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.10 m , A-weighted SPL: 67.25 dB(A)
# Index: 6 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 2.53 m , A-weighted SPL: 67.14 dB(A)
# Index: 6 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 6.34 m , A-weighted SPL: 63.62 dB(A)
# Index: 6 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.49 m , A-weighted SPL: 64.14 dB(A)
# Index: 6 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.71 m , A-weighted SPL: 65.08 dB(A)
# Index: 6 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.02 m , A-weighted SPL: 66.23 dB(A)
# Index: 6 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.48 m , A-weighted SPL: 67.12 dB(A)
# Index: 6 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.18 m , A-weighted SPL: 67.12 dB(A)
# Index: 6 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.18 m , A-weighted SPL: 66.04 dB(A)
# Index: 6 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 3.48 m , A-weighted SPL: 62.92 dB(A)
# Index: 6 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 7.34 m , A-weighted SPL: 61.98 dB(A)
# Index: 6 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 6.62 m , A-weighted SPL: 62.92 dB(A)
# Index: 6 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.99 m , A-weighted SPL: 63.82 dB(A)
# Index: 6 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.46 m , A-weighted SPL: 63.13 dB(A)
# Index: 6 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.08 m , A-weighted SPL: 62.98 dB(A)
# Index: 6 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.88 m , A-weighted SPL: 64.76 dB(A)
# Index: 6 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 4.88 m , A-weighted SPL: 62.09 dB(A)
# Index: 6 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 5.508, 0.000] , Distance: 5.08 m , A-weighted SPL: 61.50 dB(A)

# Index: 7 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 6.70 m , A-weighted SPL: 58.68 dB(A)
# Index: 7 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.79 m , A-weighted SPL: 59.17 dB(A)
# Index: 7 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.91 m , A-weighted SPL: 58.64 dB(A)
# Index: 7 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.09 m , A-weighted SPL: 59.60 dB(A)
# Index: 7 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.37 m , A-weighted SPL: 62.20 dB(A)
# Index: 7 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.82 m , A-weighted SPL: 58.45 dB(A)
# Index: 7 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.56 m , A-weighted SPL: 60.82 dB(A)
# Index: 7 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.67 m , A-weighted SPL: 65.84 dB(A)
# Index: 7 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 6.53 m , A-weighted SPL: 64.06 dB(A)
# Index: 7 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.59 m , A-weighted SPL: 63.65 dB(A)
# Index: 7 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.67 m , A-weighted SPL: 63.86 dB(A)
# Index: 7 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.80 m , A-weighted SPL: 65.52 dB(A)
# Index: 7 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.00 m , A-weighted SPL: 66.39 dB(A)
# Index: 7 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.37 m , A-weighted SPL: 67.06 dB(A)
# Index: 7 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.05 m , A-weighted SPL: 68.02 dB(A)
# Index: 7 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 2.19 m , A-weighted SPL: 68.60 dB(A)
# Index: 7 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 6.95 m , A-weighted SPL: 63.56 dB(A)
# Index: 7 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 6.07 m , A-weighted SPL: 63.69 dB(A)
# Index: 7 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.24 m , A-weighted SPL: 63.63 dB(A)
# Index: 7 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.48 m , A-weighted SPL: 65.23 dB(A)
# Index: 7 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.83 m , A-weighted SPL: 65.91 dB(A)
# Index: 7 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.36 m , A-weighted SPL: 64.98 dB(A)
# Index: 7 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.15 m , A-weighted SPL: 66.61 dB(A)
# Index: 7 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 3.24 m , A-weighted SPL: 67.46 dB(A)
# Index: 7 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 7.87 m , A-weighted SPL: 62.93 dB(A)
# Index: 7 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 7.11 m , A-weighted SPL: 62.90 dB(A)
# Index: 7 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 6.42 m , A-weighted SPL: 61.80 dB(A)
# Index: 7 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.81 m , A-weighted SPL: 59.71 dB(A)
# Index: 7 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.33 m , A-weighted SPL: 61.63 dB(A)
# Index: 7 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 5.00 m , A-weighted SPL: 61.99 dB(A)
# Index: 7 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.86 m , A-weighted SPL: 61.63 dB(A)
# Index: 7 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.580, 6.199, 0.000] , Distance: 4.92 m , A-weighted SPL: 63.87 dB(A)

# Index: 8 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 6.63 m , A-weighted SPL: 61.15 dB(A)
# Index: 8 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.71 m , A-weighted SPL: 61.96 dB(A)
# Index: 8 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 4.82 m , A-weighted SPL: 61.82 dB(A)
# Index: 8 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.98 m , A-weighted SPL: 64.20 dB(A)
# Index: 8 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.23 m , A-weighted SPL: 64.84 dB(A)
# Index: 8 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.65 m , A-weighted SPL: 65.61 dB(A)
# Index: 8 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.37 m , A-weighted SPL: 66.89 dB(A)
# Index: 8 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.49 m , A-weighted SPL: 66.43 dB(A)
# Index: 8 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 6.56 m , A-weighted SPL: 62.49 dB(A)
# Index: 8 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.62 m , A-weighted SPL: 63.10 dB(A)
# Index: 8 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 4.71 m , A-weighted SPL: 62.58 dB(A)
# Index: 8 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.85 m , A-weighted SPL: 63.69 dB(A)
# Index: 8 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.06 m , A-weighted SPL: 63.59 dB(A)
# Index: 8 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.45 m , A-weighted SPL: 65.48 dB(A)
# Index: 8 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.14 m , A-weighted SPL: 66.55 dB(A)
# Index: 8 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 2.28 m , A-weighted SPL: 68.71 dB(A)
# Index: 8 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 7.07 m , A-weighted SPL: 62.78 dB(A)
# Index: 8 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 6.21 m , A-weighted SPL: 62.33 dB(A)
# Index: 8 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.40 m , A-weighted SPL: 61.89 dB(A)
# Index: 8 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 4.66 m , A-weighted SPL: 63.18 dB(A)
# Index: 8 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 4.04 m , A-weighted SPL: 64.96 dB(A)
# Index: 8 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.60 m , A-weighted SPL: 63.32 dB(A)
# Index: 8 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.40 m , A-weighted SPL: 64.13 dB(A)
# Index: 8 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 3.49 m , A-weighted SPL: 67.80 dB(A)
# Index: 8 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 8.06 m , A-weighted SPL: 60.75 dB(A)
# Index: 8 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 7.32 m , A-weighted SPL: 59.16 dB(A)
# Index: 8 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 6.64 m , A-weighted SPL: 61.44 dB(A)
# Index: 8 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 6.06 m , A-weighted SPL: 61.73 dB(A)
# Index: 8 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.60 m , A-weighted SPL: 61.97 dB(A)
# Index: 8 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.29 m , A-weighted SPL: 62.67 dB(A)
# Index: 8 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.15 m , A-weighted SPL: 61.56 dB(A)
# Index: 8 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [0.257, 6.199, 0.000] , Distance: 5.21 m , A-weighted SPL: 64.33 dB(A)

# Index: 9 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 6.38 m , A-weighted SPL: 62.23 dB(A)
# Index: 9 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 5.54 m , A-weighted SPL: 62.91 dB(A)
# Index: 9 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.76 m , A-weighted SPL: 63.15 dB(A)
# Index: 9 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.08 m , A-weighted SPL: 64.21 dB(A)
# Index: 9 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.55 m , A-weighted SPL: 64.51 dB(A)
# Index: 9 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.26 m , A-weighted SPL: 65.04 dB(A)
# Index: 9 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.25 m , A-weighted SPL: 64.25 dB(A)
# Index: 9 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.55 m , A-weighted SPL: 61.82 dB(A)
# Index: 9 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 5.88 m , A-weighted SPL: 63.88 dB(A)
# Index: 9 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.96 m , A-weighted SPL: 64.78 dB(A)
# Index: 9 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.07 m , A-weighted SPL: 65.86 dB(A)
# Index: 9 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.25 m , A-weighted SPL: 66.02 dB(A)
# Index: 9 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.56 m , A-weighted SPL: 67.30 dB(A)
# Index: 9 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.13 m , A-weighted SPL: 67.91 dB(A)
# Index: 9 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.12 m , A-weighted SPL: 67.31 dB(A)
# Index: 9 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.55 m , A-weighted SPL: 67.22 dB(A)
# Index: 9 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 6.04 m , A-weighted SPL: 63.10 dB(A)
# Index: 9 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 5.15 m , A-weighted SPL: 64.27 dB(A)
# Index: 9 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.30 m , A-weighted SPL: 64.86 dB(A)
# Index: 9 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 3.53 m , A-weighted SPL: 65.50 dB(A)
# Index: 9 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.91 m , A-weighted SPL: 66.61 dB(A)
# Index: 9 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.54 m , A-weighted SPL: 68.01 dB(A)
# Index: 9 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.54 m , A-weighted SPL: 62.74 dB(A)
# Index: 9 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 2.90 m , A-weighted SPL: 63.77 dB(A)
# Index: 9 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 6.82 m , A-weighted SPL: 62.42 dB(A)
# Index: 9 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 6.04 m , A-weighted SPL: 63.13 dB(A)
# Index: 9 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 5.33 m , A-weighted SPL: 63.91 dB(A)
# Index: 9 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.73 m , A-weighted SPL: 63.14 dB(A)
# Index: 9 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.29 m , A-weighted SPL: 64.38 dB(A)
# Index: 9 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.05 m , A-weighted SPL: 64.19 dB(A)
# Index: 9 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.04 m , A-weighted SPL: 58.58 dB(A)
# Index: 9 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 5.508, 0.000] , Distance: 4.28 m , A-weighted SPL: 59.40 dB(A)

# Index: 10 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 6.51 m , A-weighted SPL: 61.78 dB(A)
# Index: 10 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.69 m , A-weighted SPL: 62.25 dB(A)
# Index: 10 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.93 m , A-weighted SPL: 63.33 dB(A)
# Index: 10 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.28 m , A-weighted SPL: 63.29 dB(A)
# Index: 10 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.78 m , A-weighted SPL: 64.26 dB(A)
# Index: 10 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.50 m , A-weighted SPL: 64.63 dB(A)
# Index: 10 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.50 m , A-weighted SPL: 59.91 dB(A)
# Index: 10 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.77 m , A-weighted SPL: 59.84 dB(A)
# Index: 10 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.92 m , A-weighted SPL: 63.86 dB(A)
# Index: 10 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.00 m , A-weighted SPL: 63.86 dB(A)
# Index: 10 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.12 m , A-weighted SPL: 64.79 dB(A)
# Index: 10 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.31 m , A-weighted SPL: 65.14 dB(A)
# Index: 10 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.64 m , A-weighted SPL: 66.89 dB(A)
# Index: 10 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.22 m , A-weighted SPL: 67.89 dB(A)
# Index: 10 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.22 m , A-weighted SPL: 61.92 dB(A)
# Index: 10 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.63 m , A-weighted SPL: 62.35 dB(A)
# Index: 10 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.98 m , A-weighted SPL: 63.68 dB(A)
# Index: 10 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.07 m , A-weighted SPL: 64.33 dB(A)
# Index: 10 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.20 m , A-weighted SPL: 65.40 dB(A)
# Index: 10 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.41 m , A-weighted SPL: 65.62 dB(A)
# Index: 10 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.76 m , A-weighted SPL: 66.96 dB(A)
# Index: 10 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.37 m , A-weighted SPL: 68.15 dB(A)
# Index: 10 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.37 m , A-weighted SPL: 67.05 dB(A)
# Index: 10 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 2.76 m , A-weighted SPL: 66.34 dB(A)
# Index: 10 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 6.66 m , A-weighted SPL: 62.40 dB(A)
# Index: 10 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.86 m , A-weighted SPL: 62.85 dB(A)
# Index: 10 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 5.13 m , A-weighted SPL: 64.71 dB(A)
# Index: 10 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.51 m , A-weighted SPL: 64.61 dB(A)
# Index: 10 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.04 m , A-weighted SPL: 64.15 dB(A)
# Index: 10 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.78 m , A-weighted SPL: 64.73 dB(A)
# Index: 10 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 3.78 m , A-weighted SPL: 63.39 dB(A)
# Index: 10 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 5.508, 0.000] , Distance: 4.03 m , A-weighted SPL: 63.54 dB(A)

# Index: 11 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 7.10 m , A-weighted SPL: 60.19 dB(A)
# Index: 11 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 6.25 m , A-weighted SPL: 59.75 dB(A)
# Index: 11 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 5.44 m , A-weighted SPL: 59.27 dB(A)
# Index: 11 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.72 m , A-weighted SPL: 60.93 dB(A)
# Index: 11 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.10 m , A-weighted SPL: 61.45 dB(A)
# Index: 11 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.67 m , A-weighted SPL: 61.33 dB(A)
# Index: 11 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.47 m , A-weighted SPL: 61.49 dB(A)
# Index: 11 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.56 m , A-weighted SPL: 64.92 dB(A)
# Index: 11 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 6.57 m , A-weighted SPL: 62.03 dB(A)
# Index: 11 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 5.63 m , A-weighted SPL: 62.15 dB(A)
# Index: 11 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.72 m , A-weighted SPL: 62.17 dB(A)
# Index: 11 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.86 m , A-weighted SPL: 62.22 dB(A)
# Index: 11 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.09 m , A-weighted SPL: 62.75 dB(A)
# Index: 11 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.48 m , A-weighted SPL: 62.46 dB(A)
# Index: 11 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.17 m , A-weighted SPL: 62.37 dB(A)
# Index: 11 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.31 m , A-weighted SPL: 68.08 dB(A)
# Index: 11 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 6.62 m , A-weighted SPL: 63.78 dB(A)
# Index: 11 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 5.69 m , A-weighted SPL: 63.65 dB(A)
# Index: 11 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.80 m , A-weighted SPL: 63.73 dB(A)
# Index: 11 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.95 m , A-weighted SPL: 64.82 dB(A)
# Index: 11 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.19 m , A-weighted SPL: 66.19 dB(A)
# Index: 11 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.61 m , A-weighted SPL: 66.87 dB(A)
# Index: 11 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.33 m , A-weighted SPL: 68.43 dB(A)
# Index: 11 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 2.45 m , A-weighted SPL: 67.99 dB(A)
# Index: 11 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 7.24 m , A-weighted SPL: 61.95 dB(A)
# Index: 11 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 6.41 m , A-weighted SPL: 62.52 dB(A)
# Index: 11 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 5.63 m , A-weighted SPL: 62.92 dB(A)
# Index: 11 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.93 m , A-weighted SPL: 63.07 dB(A)
# Index: 11 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 4.35 m , A-weighted SPL: 63.94 dB(A)
# Index: 11 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.94 m , A-weighted SPL: 64.19 dB(A)
# Index: 11 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.75 m , A-weighted SPL: 64.73 dB(A)
# Index: 11 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.830, 6.199, 0.000] , Distance: 3.83 m , A-weighted SPL: 65.08 dB(A)

# Index: 12 , MicPos: [-1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 6.98 m , A-weighted SPL: 61.07 dB(A)
# Index: 12 , MicPos: [-1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 6.11 m , A-weighted SPL: 61.79 dB(A)
# Index: 12 , MicPos: [-1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 5.29 m , A-weighted SPL: 62.09 dB(A)
# Index: 12 , MicPos: [-1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.54 m , A-weighted SPL: 61.95 dB(A)
# Index: 12 , MicPos: [-1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.90 m , A-weighted SPL: 64.13 dB(A)
# Index: 12 , MicPos: [-1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.43 m , A-weighted SPL: 62.70 dB(A)
# Index: 12 , MicPos: [-1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.22 m , A-weighted SPL: 64.83 dB(A)
# Index: 12 , MicPos: [-1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.32 m , A-weighted SPL: 65.31 dB(A)
# Index: 12 , MicPos: [1.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 6.53 m , A-weighted SPL: 65.12 dB(A)
# Index: 12 , MicPos: [1.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 5.59 m , A-weighted SPL: 64.30 dB(A)
# Index: 12 , MicPos: [1.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.68 m , A-weighted SPL: 64.55 dB(A)
# Index: 12 , MicPos: [1.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.81 m , A-weighted SPL: 65.77 dB(A)
# Index: 12 , MicPos: [1.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.02 m , A-weighted SPL: 66.86 dB(A)
# Index: 12 , MicPos: [1.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.39 m , A-weighted SPL: 66.84 dB(A)
# Index: 12 , MicPos: [1.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.08 m , A-weighted SPL: 67.89 dB(A)
# Index: 12 , MicPos: [1.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.22 m , A-weighted SPL: 68.29 dB(A)
# Index: 12 , MicPos: [3.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 6.68 m , A-weighted SPL: 61.51 dB(A)
# Index: 12 , MicPos: [3.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 5.76 m , A-weighted SPL: 62.46 dB(A)
# Index: 12 , MicPos: [3.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.88 m , A-weighted SPL: 60.77 dB(A)
# Index: 12 , MicPos: [3.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.05 m , A-weighted SPL: 62.20 dB(A)
# Index: 12 , MicPos: [3.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 3.32 m , A-weighted SPL: 65.11 dB(A)
# Index: 12 , MicPos: [3.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.76 m , A-weighted SPL: 62.07 dB(A)
# Index: 12 , MicPos: [3.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.50 m , A-weighted SPL: 62.59 dB(A)
# Index: 12 , MicPos: [3.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 2.61 m , A-weighted SPL: 68.00 dB(A)
# Index: 12 , MicPos: [5.000, 0.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 7.38 m , A-weighted SPL: 59.58 dB(A)
# Index: 12 , MicPos: [5.000, 1.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 6.57 m , A-weighted SPL: 59.77 dB(A)
# Index: 12 , MicPos: [5.000, 2.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 5.81 m , A-weighted SPL: 59.74 dB(A)
# Index: 12 , MicPos: [5.000, 3.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 5.13 m , A-weighted SPL: 62.66 dB(A)
# Index: 12 , MicPos: [5.000, 4.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.58 m , A-weighted SPL: 62.93 dB(A)
# Index: 12 , MicPos: [5.000, 5.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.19 m , A-weighted SPL: 62.72 dB(A)
# Index: 12 , MicPos: [5.000, 6.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.02 m , A-weighted SPL: 62.93 dB(A)
# Index: 12 , MicPos: [5.000, 7.000, 2.000] , SpeakerPos: [1.520, 6.199, 0.000] , Distance: 4.09 m , A-weighted SPL: 64.72 dB(A)
