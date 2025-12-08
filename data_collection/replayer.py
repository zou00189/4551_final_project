# this file should replicate the motion recorded for the vehicle_blue

# 1. publish the velocity command.

# 2. publish the state command. But may also need a control node. 

# Latency problem?

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import csv
import time
import pathlib
import sys

# --- Configuration ---
EPISODE_ROOT = pathlib.Path("episodes")
REPLAY_RATE_HZ = 20.0

class VehicleReplayer(Node):
    def __init__(self):
        super().__init__('vehicle_replayer')

        # 1. Find the most recent CSV file
        self.csv_path = self.get_latest_recording()
        if not self.csv_path:
            self.get_logger().error("No CSV files found in 'episodes/' directory.")
            sys.exit(1)
        
        self.get_logger().info(f"Loading replay data from: {self.csv_path}")

        # 2. Load Data into Memory
        self.replay_data = []
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.replay_data.append(row)
        except Exception as e:
            self.get_logger().error(f"Failed to read CSV: {e}")
            sys.exit(1)

        self.get_logger().info(f"Loaded {len(self.replay_data)} samples.")

        # 3. Setup Publisher (To Control the Vehicle)
        # We publish to cmd_vel to drive the robot
        self.publisher_ = self.create_publisher(Twist, '/vehicle_blue/cmd_vel', 10)

        # 4. Setup Timer
        self.index = 0
        self.timer = self.create_timer(1.0 / REPLAY_RATE_HZ, self.timer_callback)
        self.get_logger().info("Replay starting in 3 seconds...")
        time.sleep(3.0) # Give Gazebo a moment to settle

    def get_latest_recording(self):
        """Helper to find the most recently created CSV file."""
        if not EPISODE_ROOT.exists():
            return None
        csv_files = list(EPISODE_ROOT.glob("episode_*.csv"))
        if not csv_files:
            return None
        # Sort by modification time, newest last
        return max(csv_files, key=lambda p: p.stat().st_mtime)

    def timer_callback(self):
        # Stop if we have reached the end of the file
        if self.index >= len(self.replay_data):
            self.get_logger().info("Replay finished.")
            # Send a zero stop command before quitting
            stop_msg = Twist()
            self.publisher_.publish(stop_msg)
            self.destroy_node()
            sys.exit(0)

        # Get current row
        row = self.replay_data[self.index]

        # Create Twist Message
        # The recorder saved 'Odom' Twist (Body Frame), which is exactly what cmd_vel expects
        msg = Twist()
        
        try:
            # Linear X (Forward/Backward)
            msg.linear.x = float(row['vehicle_vx'])
            
            # Linear Y (Strafing - important for Mecanum drives)
            msg.linear.y = float(row['vehicle_vy'])
            
            # Angular Z (Turning)
            msg.angular.z = float(row['vehicle_wz'])
            
            # Publish
            self.publisher_.publish(msg)
            
            # Optional: Log progress every second (every 20 steps)
            if self.index % 20 == 0:
                ts = row['timestamp']
                self.get_logger().info(f"Replaying time: {ts}s | vx={msg.linear.x:.3f}")

        except ValueError as e:
            self.get_logger().warn(f"Error parsing row {self.index}: {e}")

        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    replayer = VehicleReplayer()
    try:
        rclpy.spin(replayer)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        # Send stop command on Ctrl+C
        stop_msg = Twist()
        replayer.publisher_.publish(stop_msg)
        pass
    finally:
        replayer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    