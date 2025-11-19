#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from djitellopy import Tello
import time

class TelloMovementNode(Node):
    def __init__(self):
        super().__init__('tello_movement_node')
        
        # Initialize the Tello drone
        self.tello = Tello()
        self.tello.connect()
        self.tello.takeoff()
        time.sleep(3)
        self.tello.move_up(70)
        time.sleep(5)

        # Publisher to notify that Tello is ready for control commands
        self.ready_publisher = self.create_publisher(Bool, 'tello_ready', 10)
        self.publish_ready_signal()

        # Subscribe to control commands from TelloControllerNode
        self.create_subscription(Float32MultiArray, 'tello_control', self.control_callback, 10)
        
        # Subscribe to stop signal from TelloControllerNode
        self.create_subscription(Float32MultiArray, 'tello_stop', self.stop_callback, 10)

    def publish_ready_signal(self):
        # Publish a ready signal to indicate the drone is ready for commands.
        ready_msg = Bool()
        ready_msg.data = True
        self.ready_publisher.publish(ready_msg)
        self.get_logger().info("Published ready signal for control commands.")

    def control_callback(self, msg):
        # Extract velocities from the control message and send them to the Tello drone
        velocity_x = int(msg.data[0])
        velocity_y = int(msg.data[1])
        
        # Send control command to the drone
        self.tello.send_rc_control(-velocity_x, velocity_y, 0, 0)

    def stop_callback(self, msg):
        # Land the drone upon receiving the stop signal
        self.tello.land()
        self.get_logger().info("Landing Tello drone on final goal reached.")

    def on_shutdown(self):
        # Land the drone safely when shutting down
        self.tello.land()
        self.get_logger().info("Landing Tello drone.")

def main(args=None):
    rclpy.init(args=args)
    tello_movement = TelloMovementNode()
    try:
        rclpy.spin(tello_movement)
    except KeyboardInterrupt:
        tello_movement.on_shutdown()
    finally:
        tello_movement.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
















