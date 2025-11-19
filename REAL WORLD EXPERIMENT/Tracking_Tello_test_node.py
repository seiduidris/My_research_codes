#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from djitellopy import Tello
import numpy as np
import time
import csv
import threading
from std_msgs.msg import Float32MultiArray
from my_robot_controller.QP_for_tracking_node import QPTrajectoryTracker

class TelloControllerNode(Node):
    def __init__(self):
        super().__init__('tello_controller_node')
        
        # Initialize the drone
        self.tello = Tello()
        self.tello.connect()
        self.tello.takeoff()
        time.sleep(3)
        self.tello.move_up(70)
        time.sleep(5)

        self.global_start_time = time.time()

        # Initialize the trajectory tracker
        self.qp_tracker = QPTrajectoryTracker()

        # Define trajectory points and reference velocities with timings
        self.trajectory_points = [
            [-0.3, 2.3], [-0.13, 2.79], [0.098, 3.23], [0.2312, 3.567],
            [0.4584, 4.084], [0.742, 4.92], [0.78, 5.36], [0.78, 6.05],
            [0.468, 6.62], [0.294, 6.94], [-0.211, 7.79], [-0.3, 7.95]
        ]
        self.reference_velocities_times = [5.2, 10.2, 13.8, 19.4, 28.5, 32.9, 39.8, 46.3, 50, 59.8, 61.6]
        # self.reference_velocities = [
        #     (0.0319*165, 0.0948*90), (0.0465*155, 0.0885*100), (0.0373*145, 0.0928*60),
        #     (0.0402*165, 0.0916*85), (0.0312*165, 0.0927*85), (0.0101*450, 0.0995*100),
        #     (-0.0007*5714, 0.1000*110), (-0.0476*120, 0.0879*125), (-0.0479*135, 0.0878*95),
        #     (-0.0512*135, 0.0859*140), (-0.0498*130, 0.0867*140)
        # ]

        self.reference_velocities = [
            (0.0319, 0.0948), (0.0465, 0.0885), (0.0373, 0.0928),
            (0.0402, 0.0916), (0.0312, 0.0927), (0.0101, 0.0995),
            (-0.0007, 0.1000), (-0.0476, 0.0879), (-0.0479, 0.0878),
            (-0.0512, 0.0859), (-0.0498, 0.0867)
        ]


        # Initialize trajectory and target settings
        self.current_trajectory_index = 0
        self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
        self.movement_start_time = None
        self.db_history = []  # Store (timestamp, dB) tuples

        # Initialize position and sound data
        self.current_position = None
        self.mic_position = None  # Mic1 position
        self.mic_position2 = None  # Mic2 position
        self.distance1 = None
        self.distance2 = None
        self.db_mic1 = None
        self.db_mic2 = None

        # Subscribe to position, sound, and distance data topics
        self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', self.pose_callback, 10)
        self.create_subscription(Float32MultiArray, 'sound_data', self.sound_callback, 10)
        self.create_subscription(Float32MultiArray, 'distance_data', self.distance_callback, 10)

        # Timer for control commands
        self.control_timer = self.create_timer(0.1, self.send_control_command)

        # CSV setup for logging data
        self.sound_data_folder = '/home/idris/ros2_ws/drone_data/'
        self.sound_log_file = open(self.sound_data_folder + 'Tracking_real_audio_distance_log.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.sound_log_file)
        self.csv_writer.writerow(['Timestamp', 'Mic1 Pos', 'Mic2 Pos', 'Tello Pos', 'Distance1', 'Distance2', 'Mic1 Sound (dB)', 'Mic2 Sound (dB)'])

        self.lock = threading.Lock()

    def pose_callback(self, msg):
        # Update the current drone position
        self.current_position = np.array([msg.pose.position.x, -msg.pose.position.y])

    def sound_callback(self, msg):
        # Update the sound levels and store history for db calculations
        self.db_mic1, self.db_mic2 = msg.data
        timestamp = time.time() - self.global_start_time
        self.db_history.append((timestamp, self.db_mic1))  # Storing only mic1 dB values with timestamps

    def distance_callback(self, msg):
        # Update microphone positions and distances
        self.mic_position = np.array(msg.data[:3])
        self.mic_position2 = np.array(msg.data[3:6])
        self.distance1 = msg.data[6]
        self.distance2 = msg.data[7]

    def send_control_command(self):
        if self.current_position is None:
            return
        
        current_time = time.time() - self.global_start_time

        if self.movement_start_time is None:
            self.movement_start_time = current_time

        time_elapsed = current_time - self.movement_start_time
        distance_to_goal = np.linalg.norm(self.current_position - self.goal)
        
        time_threshold = self.reference_velocities_times[self.current_trajectory_index] if self.current_trajectory_index == 0 \
            else self.reference_velocities_times[self.current_trajectory_index] - self.reference_velocities_times[self.current_trajectory_index - 1]

        if time_elapsed >= time_threshold or distance_to_goal < 0.1:
            if self.current_trajectory_index < len(self.trajectory_points) - 2:
                self.current_trajectory_index += 1
                self.goal = np.array(self.trajectory_points[self.current_trajectory_index + 1])
                self.movement_start_time = current_time
                print(f"Switching to next goal: {self.goal}")
            else:
                self.get_logger().info('Final goal reached, landing...')
                self.tello.land()
                self.control_timer.cancel()
                self.sound_log_file.close()
                return

        x_ref = np.array(self.trajectory_points[self.current_trajectory_index + 1])
        dx_ref = np.array(self.reference_velocities[self.current_trajectory_index])
        p_value = self.db_mic1
        q_mic1 = self.mic_position

        velocity, x_current = self.qp_tracker.generate_control(
            self.current_position, x_ref, dx_ref, q_mic1, current_time, p_value, 
            self.db_history, self.reference_velocities_times
        )
        # cmd_msg=Twist()    Tron edit
        # cmd_msg.linear.x=velocity[0]  Tron edit
        # cmd_msg.linear.y=velocity[1]  Tron edit
        # self.cmd_pub.publish(cmd_msg) Tron edit
        self.tello.send_rc_control(-int(velocity[0]), int(velocity[1]), 0, 0)
        self.record_data_to_csv(current_time)

    def record_data_to_csv(self, time_elapsed):
        """ Record drone's position, sound data, and other relevant data to CSV """
        with self.lock:
            self.csv_writer.writerow([
                round(time_elapsed, 1),
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
