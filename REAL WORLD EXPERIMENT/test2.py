import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
#file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth_third_version.xls"
#file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth_second_version.xls"

#file_path = "/Users/idrisseidu/Documents/Second_Tracking_3.xls"

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

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(normal_times, J_mic1, label='Mic1 J(q,t)')
plt.plot(normal_times, J_mic2, label='Mic2 J(q,t)')

# Draw a horizontal line at y = 195 with label "J_limit"
plt.axhline(y=195, color='red', linestyle='--', label='J_limit')
plt.ylim(0, max(max(J_mic1), max(J_mic2), 200))

plt.xlabel('Time')
plt.ylabel('J(q,t)')
plt.title('J(q,t) vs Time for Mic1 and Mic2')
plt.legend()
plt.grid(True)
plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Provided integration function (trapezoidal rule)
# def J_x_t_funct(gaussian, dt=0.1):
#     """
#     Approximates the integral using the trapezoidal rule.
#     'gaussian' is a list of function values at discretized time points.
#     """
#     if len(gaussian) <= 1:
#         return 0.0
#     else:
#         total = 0.0
#         for i in range(1, len(gaussian)):
#             total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
#         return total

# # Parameter
# gamma = 0.9

# # The "normal" time instants at which you want to compute J(q,t)
# normal_times = [
#     0, 3.2, 11.4, 20.4, 27.3, 33.7, 35.0, 38.1, 39.9, 46.8, 48.5, 53.4, 55.4,
#     64, 65.5
# ]

# # Read the Excel file (replace 'data.xlsx' with your file name if needed)
# #file_path = "/Users/idrisseidu/Documents/Idris_Tracking_real_audio_distance_log.xlsx"
# #file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth.xls"
# #file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth_second_version.xls"
# file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth_third_version.xls"

# df = pd.read_excel(file_path)

# # Ensure the data is sorted by timestamp
# df.sort_values(by='Timestamp', inplace=True)

# # Initialize lists to store the computed cost for each microphone at the normal times.
# J_mic1 = []
# J_mic2 = []

# # At time 0, we set J = 0.
# prev_time = normal_times[0]
# J_mic1_current = 0.0
# J_mic2_current = 0.0
# J_mic1.append(J_mic1_current)
# J_mic2.append(J_mic2_current)

# # Loop over each time segment defined by the normal times.
# for t in normal_times[1:]:
#     # Select all rows with timestamps in the current segment (prev_time, t]
#     segment = df[(df['Timestamp'] > prev_time) & (df['Timestamp'] <= t)]

#     # For each row in the segment, compute the weighted p(q,x(tau)) values.
#     # (Here, p is the microphone sound level in dB.)
#     weights_mic1 = []
#     weights_mic2 = []
#     for _, row in segment.iterrows():
#         tau = row['Timestamp']
#         # The weighting factor gamma^(t - tau)
#         weight_factor = gamma**(t - tau)
#         weights_mic1.append(weight_factor * row['Mic1 Sound (dB)'])
#         weights_mic2.append(weight_factor * row['Mic2 Sound (dB)'])

#     # Approximate the integral over [prev_time, t] using the trapezoidal rule.
#     increment_mic1 = J_x_t_funct(weights_mic1, dt=0.1)
#     increment_mic2 = J_x_t_funct(weights_mic2, dt=0.1)

#     # Update the cost recursively:
#     # J(q, t) = gamma^(t - prev_time)*J(q, prev_time) + ∫[prev_time,t] gamma^(t-tau)*p(q,x(tau)) d tau
#     J_mic1_current = (gamma**(t - prev_time)) * J_mic1_current + increment_mic1
#     J_mic2_current = (gamma**(t - prev_time)) * J_mic2_current + increment_mic2

#     J_mic1.append(J_mic1_current)
#     J_mic2.append(J_mic2_current)

#     # Update the previous time for the next segment.
#     prev_time = t

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(normal_times, J_mic1, marker='o', label='Mic1 J(q,t)')
# plt.plot(normal_times, J_mic2, marker='o', label='Mic2 J(q,t)')

# # Draw a horizontal line at y = 12 with label "J_limit"
# plt.axhline(y=195, color='red', linestyle='--', label='J_limit')

# plt.xlabel('Time')
# plt.ylabel('J(q,t)')
# plt.title('J(q,t) vs Time for Mic1 and Mic2')
# plt.legend()
# plt.grid(True)
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Provided integration function (trapezoidal rule)
# def J_x_t_funct(gaussian, dt=0.1):
#     """
#     Approximates the integral using the trapezoidal rule.
#     'gaussian' is a list of function values at discretized time points.
#     """
#     if len(gaussian) <= 1:
#         return 0.0
#     total = 0.0
#     for i in range(1, len(gaussian)):
#         total += ((gaussian[i] + gaussian[i - 1]) / 2.0) * dt
#     return total

# # Helper function to get the value corresponding to the nearest timestamp
# def get_closest_value(t, ts, values):
#     idx = np.argmin(np.abs(ts - t))
#     return values[idx]

# # Parameter
# gamma = 0.9
# dt = 0.1

# # The "normal" time instants at which you want to compute J(q,t)
# normal_times = [0, 3.5, 11.9, 15.2, 25.1, 35.1, 37.0, 46.9, 56.9, 65.7]

# # Read the Excel file (replace with your file path if needed)
# file_path = "/Users/idrisseidu/Documents/Idris_Tracking_fourth.xls"
# df = pd.read_excel(file_path)

# # Ensure the data is sorted by timestamp
# df.sort_values(by='Timestamp', inplace=True)

# # Extract arrays for timestamps and dB values for each microphone
# timestamps = df['Timestamp'].values
# mic1_values = df['Mic1 Sound (dB)'].values
# mic2_values = df['Mic2 Sound (dB)'].values

# # Initialize lists to store the computed cost for each microphone at the normal times.
# J_mic1 = []
# J_mic2 = []

# # At time 0, we set J = 0.
# prev_time = normal_times[0]
# J_mic1_current = 0.0
# J_mic2_current = 0.0
# J_mic1.append(J_mic1_current)
# J_mic2.append(J_mic2_current)

# # Loop over each time segment defined by the normal times.
# for t in normal_times[1:]:
#     # Instead of using all rows in the segment, we now discretize the time interval.
#     n_steps = max(1, int((t - prev_time)))
#     tau_values = [prev_time + i * dt for i in range(n_steps)]
#     tau_values.append(t)  # Ensure the endpoint is included

#     # For each tau, compute the weighted p(q,x(tau)) values using the closest available dB value.
#     weights_mic1 = []
#     weights_mic2 = []
#     for tau in tau_values:
#         weight_factor = gamma**(t - tau)
#         # Get the nearest dB value for each microphone
#         db_mic1 = get_closest_value(tau, timestamps, mic1_values)
#         db_mic2 = get_closest_value(tau, timestamps, mic2_values)
#         weights_mic1.append(weight_factor * db_mic1)
#         weights_mic2.append(weight_factor * db_mic2)

#     # Approximate the integral over [prev_time, t] using the trapezoidal rule.
#     increment_mic1 = J_x_t_funct(weights_mic1, dt=dt)
#     increment_mic2 = J_x_t_funct(weights_mic2, dt=dt)

#     # Update the cost recursively:
#     # J(q, t) = gamma^(t - prev_time) * J(q, prev_time) + ∫[prev_time,t] gamma^(t-tau)*p(q,x(tau)) d tau
#     delta_t = t - prev_time
#     J_mic1_current = (gamma**delta_t) * J_mic1_current + increment_mic1
#     J_mic2_current = (gamma**delta_t) * J_mic2_current + increment_mic2

#     J_mic1.append(J_mic1_current)
#     J_mic2.append(J_mic2_current)

#     # Update the previous time for the next segment.
#     prev_time = t

# # Plotting the results
# plt.figure(figsize=(10, 6))
# plt.plot(normal_times, J_mic1, marker='o', label='Mic1 J(q,t)')
# plt.plot(normal_times, J_mic2, marker='o', label='Mic2 J(q,t)')

# # Draw a horizontal line at y = 12 with label "J_limit"
# plt.axhline(y=12, color='red', linestyle='--', label='J_limit')

# plt.xlabel('Time')
# plt.ylabel('J(q,t)')
# plt.title('J(q,t) vs Time for Mic1 and Mic2')
# plt.legend()
# plt.grid(True)
# plt.show()
