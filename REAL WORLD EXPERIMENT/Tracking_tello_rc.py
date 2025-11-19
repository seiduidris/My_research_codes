
# #TO DEBUG

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Float32MultiArray, Bool
# import numpy as np
# import time
# import csv
# import threading
# from my_robot_controller.QP_for_tracking_node import QPTrajectoryTracker

# class TelloControllerNode(Node):
#     def __init__(self):
#         super().__init__('tello_controller_node')

#         # Initialize the trajectory tracker
#         self.qp_tracker = QPTrajectoryTracker()

#         # Define trajectory points and reference velocities with timings
#         # self.trajectory_points = [
#         #     [-0.3, 2.3], [-0.13, 2.79], [0.098, 3.23], [0.2312, 3.567],
#         #     [0.4584, 4.084], [0.742, 4.92], [0.78, 5.36], [0.78, 6.05],
#         #     [0.468, 6.62], [0.294, 6.94], [-0.211, 7.79], [-0.3, 7.95]
#         # ]
#         self.trajectory_points = [
#              [0.695, 2.266], [0.760, 8.203]]
#         self.reference_velocities_times = [54.0, 10.2]

#         # self.trajectory_points = [
#         #     [-0.3, 2.3], [-0.18, 2.63], [0.11, 3.42], [0.23, 3.73], [0.49, 4.68],
#         #     [0.89, 5.59], [1.01, 5.74], [0.87, 6.73], [0.54, 7.66],
#         #     [-0.30, 7.95]
#         # ]

#         # self.reference_velocities_times = [3.5, 11.9, 15.2, 25.1, 35.1, 37.0, 46.9, 56.9, 65.7]
#         #self.reference_velocities_times = [5.2, 10.2, 13.8, 19.4, 28.5, 32.9, 39.8, 46.3, 50, 59.8, 61.6]
#         # self.reference_velocities = [
#         #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
#         #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
#         #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
#         #     (-0.0512, 0.0859), (-0.0498, 0.0867)
#         # ]
        
#         self.reference_velocities = [(0.0343, 0.0943), (0.0345, 0.0940), (0.0364, 0.0939),
#             (0.0263, 0.0960), (0.0400, 0.0910), (0.0632, 0.0789),
#             (-0.0141, 0.1000), (-0.0330, 0.0930), (-0.0955, 0.0330)
#         ]
#         # self.reference_velocities = [(0.0343*35, 0.0943*20), (0.0345*35, 0.0940*20), (0.0364*35, 0.0939*20),
#         #     (0.0263*35, 0.0960*20), (0.0400*35, 0.0910*20), (0.0632*35, 0.0789*20),
#         #     (-0.0141*35, 0.1000*20), (-0.0330*35, 0.0930*20), (-0.0955*35, 0.0330*20)
#         # ]


#         # self.reference_velocities = [
#         #     (0.0319*165, 0.0948*90), (0.0465*155, 0.0885*100), (0.0373*145, 0.0928*60),
#         #     (0.0402*165, 0.0916*85), (0.0312*165, 0.0927*85), (0.0101*450, 0.0995*100),
#         #     (-0.0007*5714, 0.1000*110), (-0.0476*120, 0.0879*125), (-0.0479*135, 0.0878*95),
#         #     (-0.0512*135, 0.0859*140), (-0.0498*130, 0.0867*140)
#         # ]


#         # self.trajectory_points = [
#         #     [0.695, 2.266], 
#         #     [0.760, 8.203]
#         # ]
#         # self.reference_velocities_times = [60.0, 10.2]
#         # self.reference_velocities = [
#         #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
#         #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
#         #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
#         #     (-0.0512, 0.0859), (-0.0498, 0.0867)
#         # ]

#         # Initialize trajectory and target settings
#         self.current_trajectory_index = 0
#         self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         self.movement_start_time = None

#         # Flags & variables for control flow
#         self.ready_to_publish = True #DEBUG    # Will be True after /tello_ready
#         self.global_start_time = time.time()
#         self.last_control_time = 0.0      # Last time we actually ran control

#         # Initialize position & mic data
#         self.current_position = None
#         self.mic_position = None   # Mic1 position
#         self.mic_position2 = None  # Mic2 position
#         self.distance1 = None
#         self.distance2 = None
#         self.db_mic1 = None
#         self.db_mic2 = None

#         # Keep the full sound history (fine-grained)
#         self.db_history = []  # List of (timestamp, dB) for mic1, etc.

#         # Subscribe to position, sound, distance data, and ready signal
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', 
#                                  self.pose_callback, 10)
#         self.create_subscription(Float32MultiArray, 'sound_data', 
#                                  self.sound_callback, 10)
#         self.create_subscription(Float32MultiArray, 'distance_data', 
#                                  self.distance_callback, 10)
#         self.create_subscription(Bool, 'tello_ready', self.ready_callback, 10)

#         # Publishers for control commands
#         self.control_publisher = self.create_publisher(
#             Float32MultiArray, 'tello_control', 10)
#         self.stop_publisher = self.create_publisher(
#             Float32MultiArray, 'tello_stop', 10)
#         self.jqt_publisher = self.create_publisher(
#             Float32MultiArray, 'Jqt_value', 10)
#         self.debug_data_publisher = self.create_publisher(Float32MultiArray, 'debug_data', 10)
        

#         # CSV setup for logging data
#         self.sound_data_folder = '/home/idris/ros2_ws/drone_data/'
#         self.sound_log_file = open(
#             self.sound_data_folder + 'Idris_Tracking_real_audio_distance_log.csv', 
#             'w', newline=''
#         )
#         self.csv_writer = csv.writer(self.sound_log_file)
#         self.csv_writer.writerow([
#             'Timestamp', 'Mic1 Pos', 'Mic2 Pos', 'Tello Pos', 
#             'Distance1', 'Distance2', 'Mic1 Sound (dB)', 'Mic2 Sound (dB)'
#         ])

#         self.lock = threading.Lock()

#     def ready_callback(self, msg: Bool):
#         """Called once TelloMovementNode says it's ready."""
#         if msg.data:
#             self.get_logger().info("Received ready signal from TelloMovementNode.")
#             self.ready_to_publish = True
#             self.global_start_time = time.time()
#             self.last_control_time = 0.0   # reset whenever we truly start

#     def pose_callback(self, msg: PoseStamped):
#         """Updates Tello's current position (x, y)."""
#         self.current_position = np.array([msg.pose.position.x, -msg.pose.position.y])

#     def sound_callback(self, msg: Float32MultiArray):
#         """Called whenever new sound data arrives. We store it in db_history
#         and run control if >=0.1s has passed since last control step."""
#         # 1. Capture fine-grained mic timestamp
#         timestamp = time.time() - self.global_start_time

#         # 2. Update mic data
#         self.db_mic1 = msg.data[0]
#         self.db_mic2 = msg.data[1]

#         # 3. Store in the fine-grained history
#         self.db_history.append((timestamp, self.db_mic1))

#         # 4. If Tello is ready and 0.1s has passed since last control step...
#         if self.ready_to_publish and (timestamp - self.last_control_time >= 0.1):
#             self.perform_control_step(timestamp)
#             self.last_control_time = timestamp

#     def distance_callback(self, msg: Float32MultiArray):
#         """Updates microphone positions and distances."""
#         self.mic_position = np.array(msg.data[3:6])
#         self.mic_position2 = np.array(msg.data[6:9])
#         self.distance1 = msg.data[9]
#         self.distance2 = msg.data[10]

#     def perform_control_step(self, current_t: float):
#         """
#         This does what 'publish_control_command' used to do, but now
#         it is called directly from sound_callback at ~0.1 s intervals.
#         """
#         # Safety check
#         if self.current_position is None:
#             return

#         # Initialize "movement_start_time" at the first step
#         if self.movement_start_time is None:
#             self.movement_start_time = current_t

#         # See how long we've been on the current leg
#         time_elapsed_on_leg = current_t - self.movement_start_time
#         distance_to_goal = np.linalg.norm(self.current_position - self.goal)

#         # The time threshold for each trajectory leg
#         if self.current_trajectory_index == 0:
#             time_threshold = self.reference_velocities_times[0]
#         else:
#             # For subsequent legs, subtract the previous sum of times
#             time_threshold = (self.reference_velocities_times[self.current_trajectory_index] -
#                               self.reference_velocities_times[self.current_trajectory_index - 1])

#         # Check if it's time to switch to the next goal or we're close enough
#         if False: #DEBUG #(time_elapsed_on_leg >= time_threshold) or (distance_to_goal < 0.15):
#             if self.current_trajectory_index < len(self.trajectory_points) - 2:
#                 self.current_trajectory_index += 1
#                 self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#                 self.movement_start_time = current_t
#                 self.get_logger().info(f"Switching to next goal: {self.goal}")
#                 print(f"Switching to next goal: {self.goal}")
#             else:
#                 self.get_logger().info('Final goal reached')
#                 # Send stop command & close CSV
#                 stop_msg = Float32MultiArray(data=[])
#                 self.stop_publisher.publish(stop_msg)
#                 self.sound_log_file.close()
#                 return

#         # Now run the QP tracker to get velocity
#         x_ref = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         dx_ref = np.array(self.reference_velocities[self.current_trajectory_index])
       
#         # Grab your mic1 position (q_mic1) and mic1 dB
#         q_mic1 = self.mic_position  # or None if not known yet
#         p_value = self.db_mic1

#         # Generate the control using QP
#         velocity, x_current, jqt_value, grad_v_flat, grad_p_T_flat = self.qp_tracker.generate_control(
#             self.current_position,  # Tello current position
#             x_ref,                  # Next reference point
#             dx_ref,                 # Next reference velocity
#             q_mic1,                 # Mic1 position
#             current_t,              # Current time
#             p_value,                # dB mic1
#             self.db_history,        # Fine-grained history
#             self.reference_velocities_times
#         )
#         print(f"x_current: {x_current}")
#         print(f"Tello_position: {self.current_position}")
#         print(f"second_q_mic1: {q_mic1}")
#         print(f"x_ref: {x_ref}")
#         print(f"u_ref: {dx_ref}")
#         # Publish the calculated velocity
#         control_msg = Float32MultiArray(data=[velocity[0], velocity[1]])
#         print(f"control: {control_msg}")
#         self.control_publisher.publish(control_msg)

#         #Publish gradient and velocity debug data
#         debug_data = Float32MultiArray(
#             data=[velocity[0], velocity[1], grad_v_flat[0], grad_v_flat[1], grad_p_T_flat[0], grad_p_T_flat[1], x_current[0], x_current[1]]
#         )
#         print (f"gradient_data: {debug_data}")
#         self.debug_data_publisher.publish(debug_data)

#         # Publish Jqt_value
#         jqt_msg = Float32MultiArray(data=[jqt_value])
#         self.jqt_publisher.publish(jqt_msg)

#         # Log to CSV
#         self.record_data_to_csv(current_t)

#     def record_data_to_csv(self, current_t: float):
#         with self.lock:
#             # If you only want one line for each control step, do it here
#             self.csv_writer.writerow([
#                 round(current_t, 3),
#                 list(self.mic_position) if self.mic_position is not None else None,
#                 list(self.mic_position2) if self.mic_position2 is not None else None,
#                 list(self.current_position),
#                 self.distance1,
#                 self.distance2,
#                 self.db_mic1,
#                 self.db_mic2
#             ])

# def main(args=None):
#     rclpy.init(args=args)
#     tello_controller = TelloControllerNode()
#     rclpy.spin(tello_controller)
#     tello_controller.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()





















#DRONE_TRACKING_RC

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
import time
import csv
import threading
from my_robot_controller.QP_for_tracking_node import QPTrajectoryTracker

class TelloControllerNode(Node):
    def __init__(self):
        super().__init__('tello_controller_node')

        # Initialize the trajectory tracker
        self.qp_tracker = QPTrajectoryTracker()

        # Define trajectory points and reference velocities with timings
        # self.trajectory_points = [
        #     [-0.3, 2.3], [-0.13, 2.79], [0.098, 3.23], [0.2312, 3.567],
        #     [0.4584, 4.084], [0.742, 4.92], [0.78, 5.36], [0.78, 6.05],
        #     [0.468, 6.62], [0.294, 6.94], [-0.211, 7.79], [-0.3, 7.95]
        # ]


        # self.trajectory_points = [
        #     [-0.3, 2.3], [-0.18, 2.63], [0.11, 3.42], [0.23, 3.73], [0.49, 4.68],
        #     [0.89, 5.59], [1.01, 5.74], [0.87, 6.73], [0.54, 7.66],
        #     [-0.30, 7.95]
        # ]


        self.trajectory_points = [
            [-0.3, 2.3], [-0.18, 2.60], [0.13, 3.36], [0.44, 4.2], [0.7, 4.85],
            [0.84, 5.47], [0.89, 5.59], [1.0, 5.88], [1.1, 6.02],
            [1.09, 6.72], [0.99, 6.85], [0.62, 7.17], [0.48, 7.30], [-0.17, 7.86], [-0.3, 7.95]
        ]
        

        self.reference_velocities_times = [3.2, 11.4, 20.4, 27.3, 33.7, 35.0, 38.1, 39.9, 46.8, 48.5, 53.4, 55.4, 64.0, 65.5]
        #self.reference_velocities_times = [3.5, 11.9, 15.2, 25.1, 35.1, 37.0, 46.9, 56.9, 65.7]
        #self.reference_velocities_times = [5.2, 10.2, 13.8, 19.4, 28.5, 32.9, 39.8, 46.3, 50, 59.8, 61.6]
        # self.reference_velocities = [
        #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
        #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
        #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
        #     (-0.0512, 0.0859), (-0.0498, 0.0867)
        # ]


    #     self.reference_velocities = [(0.0375*1.4, 0.0938),(0.0378, 0.0927*1.5),(0.0344, 0.0933*1.5),(0.0377, 0.0942*2.5),(0.0219, 0.0969*3.5),
    # (0.0385, 0.0923*4.5),(0.0355, 0.0936*4.5),(0.0556, 0.0778*4.5),(-0.0015, 0.1014*3.5),(-0.0588, 0.0765*3.5),(-0.0755*0.2, 0.0653*0.8),
    # (-0.0700*0.7, 0.0650*2.0),(-0.0756*0.2, 0.0651*1.0),(-0.0867, 0.0600*3.0)] #used
        
        self.reference_velocities = [(0.0375, 0.0938),(0.0378, 0.0927),(0.0344, 0.0933),(0.0377, 0.0942),(0.0219, 0.0969),
    (0.0385, 0.0923),(0.0355, 0.0936),(0.0556, 0.0778),(-0.0015, 0.1014),(-0.0588, 0.0765),(-0.0755, 0.0653),
    (-0.0700, 0.0650),(-0.0756, 0.0651),(-0.0867, 0.0600)]

        
        # self.reference_velocities = [(0.0343*2.0, 0.0943), (0.0345, 0.0940), (0.0364, 0.0939),
        #     (0.0263, 0.0960), (0.0400*1.5, 0.0910*2.0), (0.0632*3.8, 0.0789*3.8),
        #     (-0.0141, 0.1000), (-0.0330, 0.0930*2.5), (-0.0955* 0.5, 0.0330*2.5)
        # ]
        # self.reference_velocities = [(0.0343*250, 0.0943*60), (0.0345*200, 0.0940*80), (0.0364*200, 0.0939*80),
        #     (0.0263*200, 0.0960*80), (0.0400*115, 0.0910*90), (0.0632*105, 0.0789*90),
        #     (-0.0141*165, 0.1000*80), (-0.0330*165, 0.0930*80), (-0.0955*105, 0.0330*80)
        # ]


        # self.reference_velocities = [
        #     (0.0319*165, 0.0948*90), (0.0465*155, 0.0885*100), (0.0373*145, 0.0928*60),
        #     (0.0402*165, 0.0916*85), (0.0312*165, 0.0927*85), (0.0101*450, 0.0995*100),
        #     (-0.0007*5714, 0.1000*110), (-0.0476*120, 0.0879*125), (-0.0479*135, 0.0878*95),
        #     (-0.0512*135, 0.0859*140), (-0.0498*130, 0.0867*140)
        # ]


        # self.trajectory_points = [
        #     [0.695, 2.266], 
        #     [0.760, 8.203]
        # ]
        # self.reference_velocities_times = [60.0, 10.2]
        # self.reference_velocities = [
        #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
        #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
        #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
        #     (-0.0512, 0.0859), (-0.0498, 0.0867)
        # ]

        # Initialize trajectory and target settings
        self.current_trajectory_index = 0
        self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
        self.movement_start_time = None

        # Flags & variables for control flow
        self.ready_to_publish = False     # Will be True after /tello_ready
        self.global_start_time = time.time()
        self.last_control_time = 0.0      # Last time we actually ran control

        # Initialize position & mic data
        self.current_position = None
        self.mic_position = None   # Mic1 position
        self.mic_position2 = None  # Mic2 position
        self.distance1 = None
        self.distance2 = None
        self.db_mic1 = None
        self.db_mic2 = None

        # Keep the full sound history (fine-grained)
        self.db_history = []  # List of (timestamp, dB) for mic1, etc.

        # Subscribe to position, sound, distance data, and ready signal
        self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', 
                                 self.pose_callback, 10)
        self.create_subscription(Float32MultiArray, 'sound_data', 
                                 self.sound_callback, 10)
        self.create_subscription(Float32MultiArray, 'distance_data', 
                                 self.distance_callback, 10)
        self.create_subscription(Bool, 'tello_ready', self.ready_callback, 10)

        # Publishers for control commands
        self.control_publisher = self.create_publisher(
            Float32MultiArray, 'tello_control', 10)
        self.stop_publisher = self.create_publisher(
            Float32MultiArray, 'tello_stop', 10)
        self.jqt_publisher = self.create_publisher(
            Float32MultiArray, 'Jqt_value', 10)

        # CSV setup for logging data
        self.sound_data_folder = '/home/idris/ros2_ws/drone_data/'
        self.sound_log_file = open(
            self.sound_data_folder + 'Idris_Tracking_real_audio_distance_log_test_7.csv', 
            'w', newline=''
        )
        self.csv_writer = csv.writer(self.sound_log_file)
        self.csv_writer.writerow([
            'Timestamp', 'Mic1 Pos', 'Mic2 Pos', 'Tello Pos', 
            'Distance1', 'Distance2', 'Mic1 Sound (dB)', 'Mic2 Sound (dB)'
        ])

        self.lock = threading.Lock()

    def ready_callback(self, msg: Bool):
        """Called once TelloMovementNode says it's ready."""
        if msg.data:
            self.get_logger().info("Received ready signal from TelloMovementNode.")
            self.ready_to_publish = True
            self.global_start_time = time.time()
            self.last_control_time = 0.0   # reset whenever we truly start

    def pose_callback(self, msg: PoseStamped):
        """Updates Tello's current position (x, y)."""
        self.current_position = np.array([msg.pose.position.x, -msg.pose.position.y])

    def sound_callback(self, msg: Float32MultiArray):
        """Called whenever new sound data arrives. We store it in db_history
        and run control if >=0.1s has passed since last control step."""
        # 1. Capture fine-grained mic timestamp
        timestamp = time.time() - self.global_start_time

        # 2. Update mic data
        self.db_mic1 = msg.data[0]
        self.db_mic2 = msg.data[1]

        # 3. Store in the fine-grained history
        self.db_history.append((timestamp, self.db_mic1))

        # 4. If Tello is ready and 0.1s has passed since last control step...
        if self.ready_to_publish and (timestamp - self.last_control_time >= 0.1):
            self.perform_control_step(timestamp)
            self.last_control_time = timestamp

    def distance_callback(self, msg: Float32MultiArray):
        """Updates microphone positions and distances."""
        self.mic_position = np.array(msg.data[:3])
        self.mic_position2 = np.array(msg.data[3:6])
        self.distance1 = msg.data[6]
        self.distance2 = msg.data[7]

    def perform_control_step(self, current_t: float):
        """
        This does what 'publish_control_command' used to do, but now
        it is called directly from sound_callback at ~0.1 s intervals.
        """
        # Safety check
        if self.current_position is None:
            return

        # Initialize "movement_start_time" at the first step
        if self.movement_start_time is None:
            self.movement_start_time = current_t

        # See how long we've been on the current leg
        time_elapsed_on_leg = current_t - self.movement_start_time
        distance_to_goal = np.linalg.norm(self.current_position - self.goal)

        # The time threshold for each trajectory leg
        if self.current_trajectory_index == 0:
            time_threshold = self.reference_velocities_times[0]
        else:
            # For subsequent legs, subtract the previous sum of times
            time_threshold = (self.reference_velocities_times[self.current_trajectory_index] -
                              self.reference_velocities_times[self.current_trajectory_index - 1])

        # Check if it's time to switch to the next goal or we're close enough
        if (time_elapsed_on_leg >= time_threshold) or (distance_to_goal < 0.02):
            if self.current_trajectory_index < len(self.trajectory_points) - 2:
                self.current_trajectory_index += 1
                self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
                self.movement_start_time = current_t
                self.get_logger().info(f"Switching to next goal: {self.goal}")
                print(f"Switching to next goal: {self.goal}")
            else:
                self.get_logger().info('Final goal reached')
                # Send stop command & close CSV
                stop_msg = Float32MultiArray(data=[])
                self.stop_publisher.publish(stop_msg)
                self.sound_log_file.close()
                return

        # Now run the QP tracker to get velocity
        x_ref = np.array(self.trajectory_points[self.current_trajectory_index + 1])
        dx_ref = np.array(self.reference_velocities[self.current_trajectory_index])
       
        # Grab your mic1 position (q_mic1) and mic1 dB
        q_mic1 = self.mic_position  # or None if not known yet
        p_value = self.db_mic1

        # Generate the control using QP
        velocity, x_current, jqt_value = self.qp_tracker.generate_control(
            self.current_position,  # Tello current position
            x_ref,                  # Next reference point
            dx_ref,                 # Next reference velocity
            q_mic1,                 # Mic1 position
            current_t,              # Current time
            p_value,                # dB mic1
            self.db_history,        # Fine-grained history
            self.reference_velocities_times
        )
        print(f"x_current: {x_current}")
        print(f"x_ref: {x_ref}")
        print(f"u_ref: {dx_ref}")
        # Publish the calculated velocity
        control_msg = Float32MultiArray(data=[velocity[0], velocity[1]])
        print(f"control: {control_msg}")
        self.control_publisher.publish(control_msg)

        # Publish Jqt_value
        jqt_msg = Float32MultiArray(data=[jqt_value])
        self.jqt_publisher.publish(jqt_msg)

        # Log to CSV
        self.record_data_to_csv(current_t)

    def record_data_to_csv(self, current_t: float):
        with self.lock:
            # If you only want one line for each control step, do it here
            self.csv_writer.writerow([
                round(current_t, 3),
                list(self.mic_position) if self.mic_position is not None else None,
                list(self.mic_position2) if self.mic_position2 is not None else None,
                list(self.current_position),
                self.distance1,
                self.distance2,
                self.db_mic1,
                self.db_mic2
            ])

def main(args=None):
    rclpy.init(args=args)
    tello_controller = TelloControllerNode()
    rclpy.spin(tello_controller)
    tello_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()























# #CODE FOR LIVE SOUND CONSTRAINT


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Float32MultiArray, Bool
# import numpy as np
# import time
# import csv
# import threading
# from my_robot_controller.QP_for_tracking_node import QPTrajectoryTracker

# class TelloControllerNode(Node):
#     def __init__(self):
#         super().__init__('tello_controller_node')
#         self.sound_sample_count = 0  # Add counter for sound samples
#         # Initialize the trajectory tracker
#         self.qp_tracker = QPTrajectoryTracker()

#         # Define trajectory points and reference velocities with timings
#         self.trajectory_points = [
#             [0.695, 2.266], [-0.50, 5.20]]
#         # self.trajectory_points = [
#         #     [0.695, 2.266], [0.760, 8.203]]
#         self.reference_velocities_times = [120.0, 10.2]
#         self.reference_velocities = [
#             (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
#             (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
#             (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
#             (-0.0512, 0.0859), (-0.0498, 0.0867)
#         ]



#         # self.trajectory_points = [
#         #     [-0.3, 2.3], [-0.13, 2.79], [0.098, 3.23], [0.2312, 3.567],
#         #     [0.4584, 4.084], [0.742, 4.92], [0.78, 5.36], [0.78, 6.05],
#         #     [0.468, 6.62], [0.294, 6.94], [-0.211, 7.79], [-0.3, 7.95]
#         # ]
#         # self.reference_velocities_times = [5.2, 10.2, 13.8, 19.4, 28.5, 32.9, 39.8, 46.3, 50, 59.8, 61.6]
#         # self.reference_velocities = [
#         #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
#         #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
#         #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
#         #     (-0.0512, 0.0859), (-0.0498, 0.0867)
#         # ]

#         # Initialize trajectory and target settings
#         self.current_trajectory_index = 0
#         self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         self.movement_start_time = None
#         self.db_history = []  # Store (timestamp, dB) tuples
#         self.ready_to_publish = False  # Flag to check if ready signal is received

#         # Initialize position and sound data    
#         self.current_position = None
#         self.mic_position = None  # Mic1 positionn
#         self.mic_position2 = None  # Mic2 position
#         self.distance1 = None
#         self.distance2 = None
#         self.db_mic1 = None
#         self.db_mic2 = None

#         # Subscribe to position, sound, distance data, and ready signal
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', self.pose_callback, 10)
#         self.create_subscription(Float32MultiArray, 'sound_data', self.sound_callback, 10)
#         self.create_subscription(Float32MultiArray, 'distance_data', self.distance_callback, 10)
#         self.create_subscription(Bool, 'tello_ready', self.ready_callback, 10)

#         # Publisher for control commands
#         self.control_publisher = self.create_publisher(Float32MultiArray, 'tello_control', 10)
#         self.stop_publisher = self.create_publisher(Float32MultiArray, 'tello_stop', 10)
#         self.jqt_publisher = self.create_publisher(Float32MultiArray, 'Jqt_value', 10)  # New publisher for Jqt_value_mic1
#         self.debug_data_publisher = self.create_publisher(Float32MultiArray, 'debug_data', 10)

#         # CSV setup for logging data
#         self.sound_data_folder = '/home/idris/ros2_ws/drone_data/'
#         self.sound_log_file = open(self.sound_data_folder + 'Idris_Tracking_real_audio_distance_log.csv', 'w', newline='')
#         self.csv_writer = csv.writer(self.sound_log_file)
#         self.csv_writer.writerow(['Timestamp', 'Mic1 Pos', 'Mic2 Pos', 'Tello Pos', 'Distance1', 'Distance2', 'Mic1 Sound (dB)', 'Mic2 Sound (dB)'])

#         self.lock = threading.Lock()
#         self.global_start_time = time.time()

#     def ready_callback(self, msg):
#         # Start publishing control commands only after receiving the ready signal
#         if msg.data:
#             self.get_logger().info("Received ready signal from TelloMovementNode.")
#             self.ready_to_publish = True
#             self.global_start_time = time.time()
#             # Start control command timer now that Tello is ready
#             self.control_timer = self.create_timer(0.1, self.publish_control_command)
#             #self.control_timer = self.create_timer(1.0, self.publish_control_command)

#     def pose_callback(self, msg):
#         self.current_position = np.array([msg.pose.position.x, -msg.pose.position.y])

#     def sound_callback(self, msg):
#         self.db_mic1 = msg.data[0]
#         self.db_mic2 = msg.data[1]
#         timestamp = time.time() - self.global_start_time
#         self.db_history.append((timestamp, self.db_mic1))
        
  

#     def distance_callback(self, msg):
#         self.mic_position = np.array(msg.data[3:6])
#         #self.mic_position = np.array(msg.data[:3])
#         self.mic_position2 = np.array(msg.data[6:9])
#         self.distance1 = msg.data[9]
#         self.distance2 = msg.data[10]
        

#     def publish_control_command(self):
#         if not self.ready_to_publish or self.current_position is None:
#             return

#         current_time = time.time() - self.global_start_time
#         if self.movement_start_time is None:
#             self.movement_start_time = current_time

#         time_elapsed = current_time - self.movement_start_time
#         distance_to_goal = np.linalg.norm(self.current_position - self.goal)

#         time_threshold = self.reference_velocities_times[self.current_trajectory_index] if self.current_trajectory_index == 0 \
#             else self.reference_velocities_times[self.current_trajectory_index] - self.reference_velocities_times[self.current_trajectory_index - 1]

#         if time_elapsed >= time_threshold or distance_to_goal < 0.1:
#             if self.current_trajectory_index < len(self.trajectory_points) - 2:
#                 self.current_trajectory_index += 1
#                 self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#                 self.movement_start_time = current_time
#                 print(f"Switching to next goal: {self.goal}")
#             else:
#                 self.get_logger().info('Final goal reached')
#                 self.control_timer.cancel()
#                 self.sound_log_file.close()
#                 stop_msg = Float32MultiArray(data=[])
#                 self.stop_publisher.publish(stop_msg)
#                 return

#         x_ref = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         dx_ref = np.array(self.reference_velocities[self.current_trajectory_index])
#         p_value = self.db_mic1
#         q_mic1 = self.mic_position

#         velocity, x_current, jqt_value, grad_v_flat, grad_p_T_flat = self.qp_tracker.generate_control(
#             self.current_position, x_ref, dx_ref, q_mic1, current_time, p_value, 
#             self.db_history, self.reference_velocities_times
#         )

#         # Publish the calculated velocities
#         control_msg = Float32MultiArray(data=[velocity[0], velocity[1]])
#         self.control_publisher.publish(control_msg)

#         # Publish gradient and velocity debug data
#         debug_data = Float32MultiArray(
#             data=[velocity[0], velocity[1], grad_v_flat[0], grad_v_flat[1], grad_p_T_flat[0], grad_p_T_flat[1], x_current[0], x_current[1]]
#         )
#         print (f"gradient_data: {debug_data}")
#         self.debug_data_publisher.publish(debug_data)

#         self.record_data_to_csv(current_time)

#         # Publish Jqt_value_mic1
#         print(f"Publishing Jqt_value_mic1: {jqt_value}")
#         print(f"Tello_position: {self.current_position}")
#         jqt_msg = Float32MultiArray(data=[jqt_value])
#         self.jqt_publisher.publish(jqt_msg)

#     def record_data_to_csv(self, time_elapsed):
#         with self.lock:
#             self.csv_writer.writerow([
#                 round(time_elapsed, 1),
#                 list(self.mic_position) if self.mic_position is not None else None,
#                 list(self.mic_position2) if self.mic_position2 is not None else None,
#                 list(self.current_position),
#                 self.distance1,
#                 self.distance2,
#                 self.db_mic1,
#                 self.db_mic2
#             ])

# def main(args=None):
#     rclpy.init(args=args)
#     tello_controller = TelloControllerNode()
#     rclpy.spin(tello_controller)
#     tello_controller.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

















# #DRONE_TRACKING_RC_WITH_SOUND _CONSTRAINT


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Float32MultiArray, Bool
# import numpy as np
# import time
# import csv
# import threading
# from my_robot_controller.QP_for_tracking_node import QPTrajectoryTracker

# class TelloControllerNode(Node):
#     def __init__(self):
#         super().__init__('tello_controller_node')
#         self.sound_sample_count = 0  # Add counter for sound samples
#         # Initialize the trajectory tracker
#         self.qp_tracker = QPTrajectoryTracker()

#         # Define trajectory points and reference velocities with timings
#         # self.trajectory_points = [
#         #     [0.695, 2.266], [0.760, 8.203]]
#         # self.reference_velocities_times = [120.0, 10.2]
#         # self.reference_velocities = [
#         #     (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
#         #     (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
#         #     (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
#         #     (-0.0512, 0.0859), (-0.0498, 0.0867)
#         # ]


#         self.trajectory_points = [
#             [-0.3, 2.3], [-0.18, 2.60], [0.13, 3.36], [0.44, 4.2], [0.7, 4.85],
#             [0.84, 5.47], [0.89, 5.59], [1.0, 5.88], [1.1, 6.02],
#             [1.09, 6.72], [0.99, 6.85], [0.62, 7.17], [0.48, 7.30], [-0.17, 7.86], [-0.3, 7.95]
#         ]
        

#         self.reference_velocities_times = [3.2, 11.4, 20.4, 27.3, 33.7, 35.0, 38.1, 39.9, 46.8, 48.5, 53.4, 55.4, 64.0, 65.5]
     
#     #     self.reference_velocities = [(0.0375, 0.0938),(0.0378, 0.0927),(0.0344, 0.0933),(0.0377, 0.0942),(0.0219, 0.0969),
#     # (0.0385, 0.0923),(0.0355, 0.0936),(0.0556, 0.0778),(-0.0015, 0.1014),(-0.0588, 0.0765),(-0.0755, 0.0653),
#     # (-0.0700, 0.0650),(-0.0756, 0.0651),(-0.0867, 0.0600)]

#         self.reference_velocities = [(0.0375*1.4, 0.0938),(0.0378, 0.0927*1.5),(0.0344, 0.0933*1.5),(0.0377, 0.0942*2.5),(0.0219, 0.0969*3.5),
#     (0.0385, 0.0923*4.5),(0.0355, 0.0936*4.5),(0.0556, 0.0778*4.5),(-0.0015, 0.1014*3.5),(-0.0588, 0.0765*3.5),(-0.0755*0.2, 0.0653*0.8),
#     (-0.0700*0.7, 0.0650*2.0),(-0.0756*0.2, 0.0651*1.0),(-0.0867, 0.0600*3.0)]


#         # Initialize trajectory and target settings
#         self.current_trajectory_index = 0
#         self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         self.movement_start_time = None
#         self.db_history = []  # Store (timestamp, dB) tuples
#         self.ready_to_publish = False  # Flag to check if ready signal is received

#         # Initialize position and sound data    
#         self.current_position = None
#         self.mic_position = None  # Mic1 positionn
#         self.mic_position2 = None  # Mic2 position
#         self.distance1 = None
#         self.distance2 = None
#         self.db_mic1 = None
#         self.db_mic2 = None

#         # Subscribe to position, sound, distance data, and ready signal
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', self.pose_callback, 10)
#         self.create_subscription(Float32MultiArray, 'sound_data', self.sound_callback, 10)
#         self.create_subscription(Float32MultiArray, 'distance_data', self.distance_callback, 10)
#         self.create_subscription(Bool, 'tello_ready', self.ready_callback, 10)

#         # Publisher for control commands
#         self.control_publisher = self.create_publisher(Float32MultiArray, 'tello_control', 10)
#         self.stop_publisher = self.create_publisher(Float32MultiArray, 'tello_stop', 10)
#         self.jqt_publisher = self.create_publisher(Float32MultiArray, 'Jqt_value', 10)  # New publisher for Jqt_value_mic1
#         self.debug_data_publisher = self.create_publisher(Float32MultiArray, 'debug_data', 10)

#         # CSV setup for logging data
#         self.sound_data_folder = '/home/idris/ros2_ws/drone_data/'
#         self.sound_log_file = open(self.sound_data_folder + 'Idris_Tracking_real_audio_distance_log.csv', 'w', newline='')
#         self.csv_writer = csv.writer(self.sound_log_file)
#         self.csv_writer.writerow(['Timestamp', 'Mic1 Pos', 'Mic2 Pos', 'Tello Pos', 'Distance1', 'Distance2', 'Mic1 Sound (dB)', 'Mic2 Sound (dB)'])

#         self.lock = threading.Lock()
#         self.global_start_time = time.time()

#     def ready_callback(self, msg):
#         # Start publishing control commands only after receiving the ready signal
#         if msg.data:
#             self.get_logger().info("Received ready signal from TelloMovementNode.")
#             self.ready_to_publish = True
#             self.global_start_time = time.time()
#             # Start control command timer now that Tello is ready
#             self.control_timer = self.create_timer(0.1, self.publish_control_command)
#             #self.control_timer = self.create_timer(1.0, self.publish_control_command)

#     def pose_callback(self, msg):
#         self.current_position = np.array([msg.pose.position.x, -msg.pose.position.y])

#     def sound_callback(self, msg):
#         self.db_mic1 = msg.data[0]
#         self.db_mic2 = msg.data[1]
#         timestamp = time.time() - self.global_start_time
#         self.db_history.append((timestamp, self.db_mic1))
        
  

#     def distance_callback(self, msg):
#         self.mic_position = np.array(msg.data[3:6])
#         #self.mic_position = np.array(msg.data[:3])
#         self.mic_position2 = np.array(msg.data[6:9])
#         self.distance1 = msg.data[9]
#         self.distance2 = msg.data[10]
        

#     def publish_control_command(self):
#         if not self.ready_to_publish or self.current_position is None:
#             return

#         current_time = time.time() - self.global_start_time
#         if self.movement_start_time is None:
#             self.movement_start_time = current_time

#         time_elapsed = current_time - self.movement_start_time
#         distance_to_goal = np.linalg.norm(self.current_position - self.goal)

#         time_threshold = self.reference_velocities_times[self.current_trajectory_index] if self.current_trajectory_index == 0 \
#             else self.reference_velocities_times[self.current_trajectory_index] - self.reference_velocities_times[self.current_trajectory_index - 1]

#         if time_elapsed >= time_threshold or distance_to_goal < 0.1:
#             if self.current_trajectory_index < len(self.trajectory_points) - 2:
#                 self.current_trajectory_index += 1
#                 self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#                 self.movement_start_time = current_time
#                 print(f"Switching to next goal: {self.goal}")
#             else:
#                 self.get_logger().info('Final goal reached')
#                 self.control_timer.cancel()
#                 self.sound_log_file.close()
#                 stop_msg = Float32MultiArray(data=[])
#                 self.stop_publisher.publish(stop_msg)
#                 return

#         x_ref = np.array(self.trajectory_points[self.current_trajectory_index + 1])
#         dx_ref = np.array(self.reference_velocities[self.current_trajectory_index])
#         p_value = self.db_mic1
#         q_mic1 = self.mic_position

#         velocity, x_current, jqt_value, grad_v_flat, grad_p_T_flat = self.qp_tracker.generate_control(
#             self.current_position, x_ref, dx_ref, q_mic1, current_time, p_value, 
#             self.db_history, self.reference_velocities_times
#         )

#         # Publish the calculated velocities
#         control_msg = Float32MultiArray(data=[velocity[0], velocity[1]])
#         self.control_publisher.publish(control_msg)

#         # Publish gradient and velocity debug data
#         debug_data = Float32MultiArray(
#             data=[velocity[0], velocity[1], grad_v_flat[0], grad_v_flat[1], grad_p_T_flat[0], grad_p_T_flat[1], x_current[0], x_current[1]]
#         )
#         print (f"gradient_data: {debug_data}")
#         self.debug_data_publisher.publish(debug_data)

#         self.record_data_to_csv(current_time)

#         # Publish Jqt_value_mic1
#         print(f"Publishing Jqt_value_mic1: {jqt_value}")
#         print(f"Tello_position: {self.current_position}")
#         jqt_msg = Float32MultiArray(data=[jqt_value])
#         self.jqt_publisher.publish(jqt_msg)

#     def record_data_to_csv(self, time_elapsed):
#         with self.lock:
#             self.csv_writer.writerow([
#                 round(time_elapsed, 1),
#                 list(self.mic_position) if self.mic_position is not None else None,
#                 list(self.mic_position2) if self.mic_position2 is not None else None,
#                 list(self.current_position),
#                 self.distance1,
#                 self.distance2,
#                 self.db_mic1,
#                 self.db_mic2
#             ])

# def main(args=None):
#     rclpy.init(args=args)
#     tello_controller = TelloControllerNode()
#     rclpy.spin(tello_controller)
#     tello_controller.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()