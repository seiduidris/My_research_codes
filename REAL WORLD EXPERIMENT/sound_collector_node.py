#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import sounddevice as sd
import numpy as np
import threading

class SoundCollector(Node):
    def __init__(self):
        super().__init__('sound_collector')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'sound_data', 10)

        self.duration = 0.1  # Still record for 0.1s (more stable)
        self.sample_rate = 44100
        self.reference_amplitude = 1.0

        self.device_index_mic1 = 0
        self.device_index_mic2 = 1

        # Store last computed dB values
        self.db_mic1 = 0.0
        self.db_mic2 = 0.0
        self.lock = threading.Lock()

        # Start the audio collection thread
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()

        # Publish at a faster rate (every 0.01s)
        self.timer = self.create_timer(0.01, self.timer_callback)

    def audio_loop(self):
        """ Continuously record audio every 0.1s and update dB values. """
        while rclpy.ok():
            try:
                # 1) Record from mic1
                audio_data_mic1 = sd.rec(int(self.duration * self.sample_rate),
                                         samplerate=self.sample_rate,
                                         channels=1,
                                         dtype='float32',
                                         device=self.device_index_mic1)
                sd.wait()
                rms_amplitude1 = np.sqrt(np.mean(np.square(audio_data_mic1)))
                db1 = 20 * np.log10(rms_amplitude1 / self.reference_amplitude) + 60

                # 2) Record from mic2
                audio_data_mic2 = sd.rec(int(self.duration * self.sample_rate),
                                         samplerate=self.sample_rate,
                                         channels=1,
                                         dtype='float32',
                                         device=self.device_index_mic2)
                sd.wait()
                rms_amplitude2 = np.sqrt(np.mean(np.square(audio_data_mic2)))
                db2 = 20 * np.log10(rms_amplitude2 / self.reference_amplitude) + 60

                # Store the values in a thread-safe manner
                with self.lock:
                    self.db_mic1 = db1
                    self.db_mic2 = db2

            except Exception as e:
                self.get_logger().error(f"Failed to capture audio: {str(e)}")

    def timer_callback(self):
        """ Publishes the last stored dB values at a faster rate (0.01s). """
        with self.lock:
            db1 = self.db_mic1
            db2 = self.db_mic2

        msg = Float32MultiArray()
        msg.data = [float(db1), float(db2)]
        self.publisher_.publish(msg)

        self.get_logger().info(f"Sound Level 1: {db1:.4f} dB, Sound Level 2: {db2:.4f} dB")

def main(args=None):
    rclpy.init(args=args)
    node = SoundCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()











# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray
# import sounddevice as sd

# import numpy as np
# import threading
# print(sd.query_devices())

# class SoundCollector(Node):
#     def __init__(self):
#         super().__init__('sound_collector')
#         self.publisher_ = self.create_publisher(Float32MultiArray, 'sound_data', 10)
#         self.duration = 0.1  # seconds
#         self.sample_rate = 44100
#         self.reference_amplitude = 1.0
#         # self.device_index_mic1 = 2
#         # self.device_index_mic2 = 10
#         self.device_index_mic1 = 0
#         self.device_index_mic2 = 1

#         self.timer = self.create_timer(0.1, self.timer_callback)
#         self.lock = threading.Lock()

#     def timer_callback(self):
#         # Create and start threads for recording from each microphonee
#         mic1_thread = threading.Thread(target=self.record_mic, args=(self.device_index_mic1, 'mic1'))
#         mic2_thread = threading.Thread(target=self.record_mic, args=(self.device_index_mic2, 'mic2'))

#         mic1_thread.start()
#         mic2_thread.start()

#         mic1_thread.join()
#         mic2_thread.join()

#         msg = Float32MultiArray()
#         msg.data = [float(self.db_mic1), float(self.db_mic2)]
#         self.publisher_.publish(msg)
#         self.get_logger().info(f'Sound Level 1: {self.db_mic1:.4f} dB, Sound Level 2: {self.db_mic2:.4f} dB')

#     def record_mic(self, device_index, mic_label):
#         try:
#             audio_data = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='float32', device=device_index)
#             sd.wait()
#             rms_amplitude = np.sqrt(np.mean(np.square(audio_data)))
#             db = 20 * np.log10(rms_amplitude / self.reference_amplitude) + 60
#             with self.lock:
#                 if mic_label == 'mic1':
#                     self.db_mic1 = db
#                 elif mic_label == 'mic2':
#                     self.db_mic2 = db
#         except Exception as e:
#             self.get_logger().error(f"Failed to capture audio on {mic_label}: {str(e)}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = SoundCollector()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()





















# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray
# import sounddevice as sd
# import numpy as np
# import threading

# class SoundCollector(Node):
#     def __init__(self):
#         super().__init__('sound_collector')
#         self.publisher_ = self.create_publisher(Float32MultiArray, 'sound_data', 10)
#         self.sample_rate = 44100
#         self.duration = 0.1  # seconds
#         self.reference_amplitude = 1.0
#         # Use your desired device indices
#         self.device_index_mic1 = 0
#         self.device_index_mic2 = 1

#         # Initialize dB readings
#         self.db_mic1 = 0.0
#         self.db_mic2 = 0.0

#         # Lock for thread safety
#         self.lock = threading.Lock()

#         # Set block size corresponding to the duration
#         self.blocksize = int(self.duration * self.sample_rate)

#         # Open persistent input streams with callbacks for each microphone
#         self.stream1 = sd.InputStream(samplerate=self.sample_rate,
#                                       device=self.device_index_mic1,
#                                       channels=1,
#                                       dtype='float32',
#                                       blocksize=self.blocksize,
#                                       callback=self.callback_mic1)
#         self.stream2 = sd.InputStream(samplerate=self.sample_rate,
#                                       device=self.device_index_mic2,
#                                       channels=1,
#                                       dtype='float32',
#                                       blocksize=self.blocksize,
#                                       callback=self.callback_mic2)
#         try:
#             self.stream1.start()
#             self.stream2.start()
#         except Exception as e:
#             self.get_logger().error(f"Error starting streams: {e}")

#         # Create a timer to publish the dB levels
#         self.timer = self.create_timer(0.01, self.timer_callback)

#     def callback_mic1(self, indata, frames, time, status):
#         if status:
#             self.get_logger().error(f"Mic1 status: {status}")
#         # Compute RMS and dB value
#         rms_amplitude = np.sqrt(np.mean(np.square(indata)))
#         # dB calculation: offset added as in your original code (+60+25)
#         db = 20 * np.log10(rms_amplitude / self.reference_amplitude) + 60 -12
#         with self.lock:
#             self.db_mic1 = db

#     def callback_mic2(self, indata, frames, time, status):
#         if status:
#             self.get_logger().error(f"Mic2 status: {status}")
#         rms_amplitude = np.sqrt(np.mean(np.square(indata)))
#         db = 20 * np.log10(rms_amplitude / self.reference_amplitude) + 60 -12
#         with self.lock:
#             self.db_mic2 = db

#     def timer_callback(self):
#         with self.lock:
#             db1 = self.db_mic1
#             db2 = self.db_mic2
#         msg = Float32MultiArray()
#         msg.data = [float(db1), float(db2)]
#         self.publisher_.publish(msg)
#         self.get_logger().info(f'Sound Level 1: {db1:.4f} dB, Sound Level 2: {db2:.4f} dB')

#     def destroy_node(self):
#         try:
#             self.stream1.stop()
#             self.stream1.close()
#             self.stream2.stop()
#             self.stream2.close()
#         except Exception as e:
#             self.get_logger().error(f"Error stopping streams: {e}")
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = SoundCollector()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
