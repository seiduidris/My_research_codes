#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np

class DistanceCalculator(Node):
    def __init__(self):
        super().__init__('distance_calculator')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'distance_data', 10)
        self.tello_pose_sub = self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', self.tello_pose_callback, 10)
        self.mic_pose_sub = self.create_subscription(PoseStamped, '/vrpn_client_node/Microphone1/pose', self.mic_pose_callback, 10)
        self.mic_pose_sub2 = self.create_subscription(PoseStamped, '/vrpn_client_node/Microphone2/pose', self.mic_pose_callback2, 10)

        self.tello_position = None
        self.mic_position = None
        self.mic_position2 = None

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        if self.tello_position is not None and self.mic_position is not None and self.mic_position2 is not None:
            distance1 = np.linalg.norm(self.tello_position - self.mic_position)
            distance2 = np.linalg.norm(self.tello_position - self.mic_position2)
            msg = Float32MultiArray()
            msg.data = [*self.tello_position, *self.mic_position, *self.mic_position2, distance1, distance2]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published Distance 1: {distance1:.2f} meters, Distance 2: {distance2:.2f} meters')
            self.get_logger().info(f'Published msg.data: {msg.data}')

    def tello_pose_callback(self, msg):
        self.tello_position = np.array([msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z])
        print(f"Tello Position: {self.tello_position}")

    def mic_pose_callback(self, msg):
        self.mic_position = np.array([msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z])
        print(f"Mic1 Position: {self.mic_position}")

    def mic_pose_callback2(self, msg):
        self.mic_position2 = np.array([msg.pose.position.x, -msg.pose.position.y, msg.pose.position.z])
        print(f"Mic2 Position: {self.mic_position2}") 

def main(args=None):
    rclpy.init(args=args)
    node = DistanceCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# import numpy as np

# class MicrophonePosePrinter(Node):
#     def __init__(self):
#         super().__init__('microphone_pose_printer')
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Microphone1/pose', self.mic1_callback, 10)
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Microphone2/pose', self.mic2_callback, 10)

#     def mic1_callback(self, msg):
#         mic1_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
#         self.get_logger().info(f'Microphone 1 Position: {mic1_position}')

#     def mic2_callback(self, msg):
#         mic2_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
#         self.get_logger().info(f'Microphone 2 Position: {mic2_position}')

# def main(args=None):
#     rclpy.init(args=args)
#     node = MicrophonePosePrinter()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# import numpy as np

# class TelloPosePrinter(Node):
#     def __init__(self):
#         super().__init__('tello_pose_printer')
#         self.create_subscription(PoseStamped, '/vrpn_client_node/Tello/pose', self.tello_callback, 10)

#     def tello_callback(self, msg):
#         tello_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
#         self.get_logger().info(f'Tello Position: {tello_position}')

# def main(args=None):
#     rclpy.init(args=args)
#     node = TelloPosePrinter()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
