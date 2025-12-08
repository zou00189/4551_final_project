#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class PatrollingObstacle(Node):
    def __init__(self):
        super().__init__('patrolling_obstacle')

        self.cmd_vel_topic = '/obstacle/cmd_vel'
        self.odom_topic = '/obstacle/odometry'

        self.speed = 2.0
        self.max_distance = 10.0   # meters

        # Publisher
        self.publisher_ = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Subscriber
        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.1, self.timer_callback)

        self.start_x = None
        self.start_y = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.move_direction = 1.0
        self.odom_received = False

    def odom_callback(self, msg):
        if self.start_x is None:
            # First odom: Initialize reference point
            self.start_x = msg.pose.pose.position.x
            self.start_y = msg.pose.pose.position.y

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.odom_received = True

    def timer_callback(self):
        if not self.odom_received:
            return

        # Compute traveled distance relative to start
        dx = self.current_x - self.start_x
        dy = self.current_y - self.start_y
        # dist = math.sqrt(dx*dx + dy*dy)
        dist = math.sqrt(dx*dx)

        # Reverse if exceeding ±10 m
        if dist >= self.max_distance:
            self.move_direction *= -1
            # Reset start position to current so we measure new 10m range
            self.start_x = self.current_x
            self.start_y = self.current_y

        twist = Twist()
        twist.linear.x = self.speed * self.move_direction
        twist.angular.z = 0.0
        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = PatrollingObstacle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
