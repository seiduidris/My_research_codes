import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# New integration function for variable time steps
def J_x_t_funct_variable(time_points, gaussian):
    """
    Approximates the integral using the trapezoidal rule for variable time steps.
    'time_points' is a list of time stamps corresponding to the 'gaussian' values.
    """
    if len(gaussian) <= 1 or len(time_points) != len(gaussian):
        return 0.0
    total = 0.0
    for i in range(1, len(gaussian)):
        dt = time_points[i] - time_points[i - 1]
        total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
    return total


# Parameter
gamma = 0.9

# The "normal" time instants at which you want to compute J(q,t)
normal_times = [
    0, 3.2, 11.4, 20.4, 27.3, 33.7, 35.0, 38.1, 39.9, 46.8, 48.5, 53.4, 55.4,
    64, 65.5
]

# Read the Excel file (update the file_path as needed)
file_path = "/Users/idrisseidu/Documents/Second_take.xls"
df = pd.read_excel(file_path)

# Ensure the data is sorted by timestamp
df.sort_values(by='Timestamp', inplace=True)

# Initialize lists to store the computed cost for each microphone at the normal times.
J_mic1 = []
J_mic2 = []

# At time 0, we set J = 0.
prev_time = normal_times[0]
J_mic1_current = 0.0
J_mic2_current = 0.0
J_mic1.append(J_mic1_current)
J_mic2.append(J_mic2_current)

# Loop over each time segment defined by the normal times.
for t in normal_times[1:]:
    # Select all rows with timestamps in the current segment (prev_time, t]
    segment = df[(df['Timestamp'] > prev_time) & (df['Timestamp'] <= t)]

    # Lists to store weighted values and the corresponding time stamps from this segment.
    weights_mic1 = []
    weights_mic2 = []
    time_points_segment = []

    # For each row in the segment, compute the weighted p(q,x(tau)) values.
    # (Here, p is the microphone sound level in dB.)
    for _, row in segment.iterrows():
        tau = row['Timestamp']
        time_points_segment.append(tau)
        # Weight factor: gamma^(t - tau)
        weight_factor = gamma**(t - tau)
        weights_mic1.append(weight_factor * row['Mic1 Sound (dB)'])
        weights_mic2.append(weight_factor * row['Mic2 Sound (dB)'])

    # Compute the integral over [prev_time, t] using the variable dt trapezoidal rule.
    # Handle cases where the segment has 0 or 1 data point.
    if len(time_points_segment) == 0:
        increment_mic1 = 0.0
        increment_mic2 = 0.0
    elif len(time_points_segment) == 1:
        # If only one data point exists, assume the function is constant over the interval.
        dt_total = t - prev_time
        increment_mic1 = weights_mic1[0] * dt_total
        increment_mic2 = weights_mic2[0] * dt_total
    else:
        increment_mic1 = J_x_t_funct_variable(time_points_segment,
                                              weights_mic1)
        increment_mic2 = J_x_t_funct_variable(time_points_segment,
                                              weights_mic2)

    # Update the cost recursively:
    # J(q, t) = gamma^(t - prev_time)*J(q, prev_time) + ∫[prev_time,t] gamma^(t-tau)*p(q,x(tau)) d tau
    J_mic1_current = (gamma**(t - prev_time)) * J_mic1_current + increment_mic1
    J_mic2_current = (gamma**(t - prev_time)) * J_mic2_current + increment_mic2

    J_mic1.append(J_mic1_current)
    J_mic2.append(J_mic2_current)

    # Update the previous time for the next segment.
    prev_time = t

# -------------------------------
# Animation of the plot over time
# -------------------------------
# Create a figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], label='Mic1 J(q,t)')
line2, = ax.plot([], [], label='Mic2 J(q,t)')
ax.axhline(y=195, color='red', linestyle='--', label='J_limit')
ax.set_xlim(0, normal_times[-1])
ax.set_ylim(0, max(max(J_mic1), max(J_mic2), 200))
ax.set_xlabel('Time')
ax.set_ylabel('J(q,t)')
ax.set_title('J(q,t) vs Time for Mic1 and Mic2')
ax.legend()
ax.grid(True)

# To produce a smooth animation, we interpolate the computed points.
# Here, we create a fine time grid from 0 to 65.5 seconds.
t_fine = np.linspace(normal_times[0], normal_times[-1], 500)
J1_fine = np.interp(t_fine, normal_times, J_mic1)
J2_fine = np.interp(t_fine, normal_times, J_mic2)

# Animation parameters: total duration 65.5s, and set frames per second (fps)
total_time = normal_times[-1]
fps = 30
total_frames = int(total_time * fps)


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2


def animate(frame):
    # current animation time in seconds:
    current_time = frame / fps
    # select the portion of the fine grid that is <= current_time
    mask = t_fine <= current_time
    line1.set_data(t_fine[mask], J1_fine[mask])
    line2.set_data(t_fine[mask], J2_fine[mask])
    return line1, line2


ani = animation.FuncAnimation(fig,
                              animate,
                              frames=total_frames,
                              init_func=init,
                              interval=1000 / fps,
                              blit=True)

# Save the animation as an MP4 video (requires ffmpeg)
ani.save('J_q_t_animation.mp4', writer='ffmpeg', fps=fps)

plt.show()
