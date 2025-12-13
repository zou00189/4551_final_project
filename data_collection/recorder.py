# this file serving as a recorder node should record: obstacle(states); object(states, velocity); and vehicle (states, velocity)
# by subscribing to 
# /box/pose
# /obstacle/odometry
# /vehicle_blue/odometry
# and then it should write the data into a .a file.

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import csv
import time
import pathlib
from datetime import datetime

OUTPUT_ROOT = pathlib.Path("episodes")
SAMPLING_RATE_HZ = 5.0
DURATION_SEC = 90.0

class DataRecorder(Node):
    def __init__(self):
        super().__init__('data_recorder')

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = OUTPUT_ROOT / f"episode_{run_ts}.csv"
        
        self.csv_fields = [
            "timestamp",
            "vehicle_x", "vehicle_y", "vehicle_z", 
            "vehicle_qx", "vehicle_qy", "vehicle_qz", "vehicle_qw",
            "vehicle_vx", "vehicle_vy", "vehicle_wz",
            "obstacle_x", "obstacle_y", "obstacle_z",
            "obstacle_qx", "obstacle_qy", "obstacle_qz", "obstacle_qw",
            "obstacle_vx", "obstacle_vy", "obstacle_wz",
            "box_x", "box_y", "box_z",
            "box_qx", "box_qy", "box_qz", "box_qw"
        ]
        
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fields)
        self.writer.writeheader()
        
        self.get_logger().info(f"Recording to {self.csv_path} at {SAMPLING_RATE_HZ} Hz")

        self.veh_pose = None
        self.veh_odom = None
        self.obs_odom = None
        self.box_pose = None

        self.create_subscription(PoseStamped, '/vehicle_blue/pose', self.veh_cb, 10)
        self.create_subscription(Odometry, '/vehicle_blue/odometry', self.veh_odom_cb, 10)
        self.create_subscription(Odometry, '/obstacle/odometry', self.obs_cb, 10)
        self.create_subscription(PoseStamped, '/box/pose', self.box_cb, 10)

        self.start_time = time.time()
        self.timer = self.create_timer(1.0 / SAMPLING_RATE_HZ, self.timer_callback)

    def veh_cb(self, msg):
        self.veh_pose = msg

    def obs_cb(self, msg):
        self.obs_odom = msg

    def veh_odom_cb(self, msg):
        self.veh_odom = msg

    def box_cb(self, msg):
        self.box_pose = msg

    def timer_callback(self):
        now = time.time()
        
        if (now - self.start_time) > DURATION_SEC:
            self.get_logger().info("Duration limit reached. Stopping recorder.")
            self.csv_file.close()
            self.destroy_node()
            exit()


        missing = []
        if self.veh_pose is None:
            missing.append("/vehicle_blue/pose (PoseStamped)")
        if self.veh_odom is None:
            missing.append("/vehicle_blue/odometry (Odometry)")
        if self.obs_odom is None:
            missing.append("/obstacle/odometry (Odometry)")
        if self.box_pose is None:
            missing.append("/box/pose (PoseStamped)")

        if missing:
            self.get_logger().warn(f"Waiting for topics: {', '.join(missing)}", throttle_duration_sec=2.0)
            return

        current_time_sec = self.box_pose.header.stamp.sec + self.box_pose.header.stamp.nanosec * 1e-9

        row = {
            "timestamp": current_time_sec,
            
            "vehicle_x": self.veh_pose.pose.position.x,
            "vehicle_y": self.veh_pose.pose.position.y,
            "vehicle_z": self.veh_pose.pose.position.z,
            "vehicle_qx": self.veh_pose.pose.orientation.x,
            "vehicle_qy": self.veh_pose.pose.orientation.y,
            "vehicle_qz": self.veh_pose.pose.orientation.z,
            "vehicle_qw": self.veh_pose.pose.orientation.w,
            "vehicle_vx": self.veh_odom.twist.twist.linear.x, 
            "vehicle_vy": self.veh_odom.twist.twist.linear.y,
            "vehicle_wz": self.veh_odom.twist.twist.angular.z,

            "obstacle_x": self.obs_odom.pose.pose.position.x,
            "obstacle_y": self.obs_odom.pose.pose.position.y,
            "obstacle_z": self.obs_odom.pose.pose.position.z,
            "obstacle_qx": self.obs_odom.pose.pose.orientation.x,
            "obstacle_qy": self.obs_odom.pose.pose.orientation.y,
            "obstacle_qz": self.obs_odom.pose.pose.orientation.z,
            "obstacle_qw": self.obs_odom.pose.pose.orientation.w,
            "obstacle_vx": self.obs_odom.twist.twist.linear.x,
            "obstacle_vy": self.obs_odom.twist.twist.linear.y,
            "obstacle_wz": self.obs_odom.twist.twist.angular.z,

            "box_x": self.box_pose.pose.position.x,
            "box_y": self.box_pose.pose.position.y,
            "box_z": self.box_pose.pose.position.z,
            "box_qx": self.box_pose.pose.orientation.x,
            "box_qy": self.box_pose.pose.orientation.y,
            "box_qz": self.box_pose.pose.orientation.z,
            "box_qw": self.box_pose.pose.orientation.w,
        }
        
        self.writer.writerow(row)
        self.csv_file.flush()

def main(args=None):
    rclpy.init(args=args)
    recorder = DataRecorder()
    try:
        rclpy.spin(recorder)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(recorder, 'csv_file') and not recorder.csv_file.closed:
            recorder.csv_file.close()
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
