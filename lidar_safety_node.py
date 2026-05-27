#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


RAW_DRIVE_TOPIC = '/raw_drive'
SAFE_DRIVE_TOPIC = '/drive'
SCAN_TOPIC = '/scan'

MAX_STEERING_ANGLE = math.radians(24.0)
SIDE_CLEARANCE = 0.33
CRITICAL_SIDE_CLEARANCE = 0.20
FRONT_SLOW_DISTANCE = 1.00
FRONT_STOP_DISTANCE = 0.45
MIN_MOVING_SPEED = 0.35
MAX_SAFE_SPEED_NEAR_WALL = 0.75
WALL_CORRECTION_GAIN = 0.22
MAX_VALID_CLEARANCE = 5.0


class LidarSafetyNode(Node):
    """Filter pure pursuit drive commands using LiDAR clearance checks."""

    def __init__(self):
        super().__init__('lidar_safety_node')

        self.latest_scan = None
        self.scan_sub = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            qos_profile_sensor_data)
        self.raw_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            RAW_DRIVE_TOPIC,
            self.raw_drive_callback,
            10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, SAFE_DRIVE_TOPIC, 10)

        self.get_logger().info(
            f'LiDAR safety enabled: {RAW_DRIVE_TOPIC} + {SCAN_TOPIC} -> {SAFE_DRIVE_TOPIC}')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def raw_drive_callback(self, msg):
        safe_msg = AckermannDriveStamped()
        safe_msg.header = msg.header
        safe_msg.drive = msg.drive

        if self.latest_scan is None:
            safe_msg.drive.speed = 0.0
            self.drive_pub.publish(safe_msg)
            self.get_logger().warn('No /scan yet; holding vehicle stopped', throttle_duration_sec=1.0)
            return

        front = self.get_sector_min(-12.0, 12.0)
        front_left = self.get_sector_min(15.0, 55.0)
        front_right = self.get_sector_min(-55.0, -15.0)
        left = self.get_sector_min(55.0, 95.0)
        right = self.get_sector_min(-95.0, -55.0)
        front = self.normalize_clearance(front)
        front_left = self.normalize_clearance(front_left)
        front_right = self.normalize_clearance(front_right)
        left = self.normalize_clearance(left)
        right = self.normalize_clearance(right)

        steering = float(msg.drive.steering_angle)
        speed = float(msg.drive.speed)
        speed_sign = -1.0 if speed < 0.0 else 1.0
        speed_abs = abs(speed)

        correction = self.get_wall_correction(left, right, front_left, front_right)
        steering = float(np.clip(steering + correction, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))

        speed_abs = self.apply_speed_limits(speed_abs, front, left, right, front_left, front_right)
        safe_msg.drive.steering_angle = steering
        safe_msg.drive.speed = speed_sign * speed_abs if speed_abs > 0.0 else 0.0

        self.drive_pub.publish(safe_msg)
        self.log_safety_state(front, left, right, front_left, front_right, correction, safe_msg)

    def get_wall_correction(self, left, right, front_left, front_right):
        correction = 0.0

        if left < SIDE_CLEARANCE or right < SIDE_CLEARANCE:
            correction += WALL_CORRECTION_GAIN * (left - right)

        if front_left < SIDE_CLEARANCE:
            correction -= WALL_CORRECTION_GAIN * (SIDE_CLEARANCE - front_left)
        if front_right < SIDE_CLEARANCE:
            correction += WALL_CORRECTION_GAIN * (SIDE_CLEARANCE - front_right)

        return correction

    def normalize_clearance(self, distance):
        if not math.isfinite(distance):
            return MAX_VALID_CLEARANCE
        return max(0.0, min(distance, MAX_VALID_CLEARANCE))

    def apply_speed_limits(self, speed_abs, front, left, right, front_left, front_right):
        min_side = min(left, right, front_left, front_right)

        if front < FRONT_STOP_DISTANCE or min_side < CRITICAL_SIDE_CLEARANCE:
            return 0.0

        if front < FRONT_SLOW_DISTANCE:
            speed_abs = min(speed_abs, MAX_SAFE_SPEED_NEAR_WALL)

        if min_side < SIDE_CLEARANCE:
            speed_abs = min(speed_abs, MAX_SAFE_SPEED_NEAR_WALL)

        if speed_abs < MIN_MOVING_SPEED:
            return 0.0

        return speed_abs

    def get_sector_min(self, start_deg, end_deg):
        values = self.get_sector_ranges(start_deg, end_deg)
        if values.size == 0:
            return float('inf')
        return float(np.percentile(values, 10))

    def get_sector_ranges(self, start_deg, end_deg):
        scan = self.latest_scan
        start = math.radians(start_deg)
        end = math.radians(end_deg)
        if start > end:
            start, end = end, start

        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
        valid = np.isfinite(ranges)
        valid &= ranges >= max(scan.range_min, 0.02)
        valid &= ranges <= scan.range_max
        valid &= angles >= start
        valid &= angles <= end
        return ranges[valid]

    def log_safety_state(self, front, left, right, front_left, front_right, correction, msg):
        self.get_logger().info(
            'front={:.2f} left={:.2f} right={:.2f} front_left={:.2f} front_right={:.2f} '
            'corr={:.3f} steering={:.1f}deg speed={:.2f}'.format(
                front,
                left,
                right,
                front_left,
                front_right,
                correction,
                math.degrees(msg.drive.steering_angle),
                msg.drive.speed),
            throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
