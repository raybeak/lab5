#!/usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformException, TransformListener
from tf_transformations import euler_from_quaternion
import math

import csv
from pathlib import Path
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

LOOKAHEAD_DISTANCE = 0.80
MIN_LOOKAHEAD_DISTANCE = 0.65
MAX_LOOKAHEAD_DISTANCE = 1.25
HEADING_THRESHOLD = np.radians(45)  # Define threshold in radians (±45 degrees here)
WHEELBASE = 0.33
MAX_STEERING_ANGLE = np.radians(24.0)
MIN_SPEED = 0.75
MAX_SPEED = 2.00
BASE_SPEED = 1.50
SPEED_TEST_SCALE = 1.25
DRIVE_SPEED_SIGN = -1.0

WAYPOINTS_FILENAME = 'waypoints.csv'
WAYPOINTS_INTERVAL = 1
MAP_FRAME = 'map'
BASE_FRAME = 'base_link'


PACKAGE_NAME = 'lab5'


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Topics for publishing and subscribing
        self.drive_topic = '/raw_drive'
        self.waypoints_topic = '/waypoints'
        self.waypoints_marker_topic = '/waypoints_marker'
        self.target_marker_topic = '/target_marker'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publisher for drive command
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        # Publisher for visualization markers
        self.waypoints_pub = self.create_publisher(MarkerArray, self.waypoints_topic, marker_qos)
        self.waypoints_marker_pub = self.create_publisher(Marker, self.waypoints_marker_topic, marker_qos)
        self.target_marker_pub = self.create_publisher(Marker, self.target_marker_topic, marker_qos)

        # Load waypoints from CSV file
        waypoint_filepath = self.resolve_waypoint_filepath()
        self.load_waypoints(waypoint_filepath, WAYPOINTS_INTERVAL)
        
        # Initialize variables
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading_angle = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.last_target_index = 0
        self.create_timer(0.05, self.control_callback)
        self.create_timer(1.0, self.publish_markers)

    def resolve_waypoint_filepath(self):
        """Find the waypoint CSV in the map frame."""
        self.declare_parameter('waypoints_file', '')
        param_path = self.get_parameter('waypoints_file').value
        if param_path:
            return param_path

        cwd = Path.cwd()
        candidates = [
            cwd / WAYPOINTS_FILENAME,
            cwd / PACKAGE_NAME / WAYPOINTS_FILENAME,
            cwd / 'f1tenth_labs' / PACKAGE_NAME / WAYPOINTS_FILENAME,
            Path(__file__).resolve().parent.parent / WAYPOINTS_FILENAME,
        ]

        try:
            candidates.append(Path(get_package_share_directory(PACKAGE_NAME)) / WAYPOINTS_FILENAME)
        except (PackageNotFoundError, Exception):
            pass

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(candidates[-1])

    def load_waypoints(self, filename, interval=1):
        """Loads waypoints from a CSV file with a specified interval."""
        self.waypoints_x = []
        self.waypoints_y = []
        self.waypoints_speed = []
        
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i % interval == 0:  
                    x, y, heading, speed = map(float, row)

                    self.waypoints_x.append(x)
                    self.waypoints_y.append(y)
                    self.waypoints_speed.append(float(np.clip(speed, MIN_SPEED, MAX_SPEED)))
        self.waypoints_curvature = self.calculate_waypoint_curvatures()
        self.get_logger().info(
            "Loaded {} waypoints from {} with interval: {}".format(
                len(self.waypoints_x), filename, interval))

    def calculate_waypoint_curvatures(self):
        """Estimate local path curvature for adaptive lookahead."""
        waypoint_count = len(self.waypoints_x)
        if waypoint_count < 3:
            return [0.0] * waypoint_count

        points = np.column_stack((self.waypoints_x, self.waypoints_y))
        curvatures = []
        for i in range(waypoint_count):
            p_prev = points[(i - 1) % waypoint_count]
            p = points[i]
            p_next = points[(i + 1) % waypoint_count]

            a = np.linalg.norm(p - p_prev)
            b = np.linalg.norm(p_next - p)
            c = np.linalg.norm(p_next - p_prev)
            v1 = p - p_prev
            v2 = p_next - p_prev
            area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])

            if a * b * c < 1e-6:
                curvatures.append(0.0)
            else:
                curvatures.append(float(2.0 * area2 / (a * b * c)))

        return curvatures
        
    def control_callback(self):
        """Callback to process vehicle pose and publish control commands."""
        try:
            transform = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                BASE_FRAME,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(
                f"No TF {MAP_FRAME} -> {BASE_FRAME}: {ex}",
                throttle_duration_sec=1.0)
            self.publish_drive(0.0, 0.0)
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.pos_x = translation.x
        self.pos_y = translation.y
        self.heading_angle = euler_from_quaternion([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        ])[2]

        if not self.waypoints_x:
            self.get_logger().warn("No waypoints loaded")
            self.publish_drive(0.0, 0.0)
            return

        waypoints = np.column_stack((self.waypoints_x, self.waypoints_y))
        deltas = waypoints - np.array([self.pos_x, self.pos_y])
        distances = np.linalg.norm(deltas, axis=1)
        closest_index = int(np.argmin(distances))
        lookahead_distance = self.get_adaptive_lookahead(closest_index)
        target_index = self.find_lookahead_index(waypoints, closest_index, lookahead_distance)
        self.last_target_index = target_index

        self.target_x = self.waypoints_x[target_index]
        self.target_y = self.waypoints_y[target_index]

        dx = self.target_x - self.pos_x
        dy = self.target_y - self.pos_y
        local_x = math.cos(self.heading_angle) * dx + math.sin(self.heading_angle) * dy
        local_y = -math.sin(self.heading_angle) * dx + math.cos(self.heading_angle) * dy
        actual_lookahead_distance = max(math.hypot(local_x, local_y), 1e-6)

        curvature = 2.0 * local_y / (actual_lookahead_distance ** 2)
        steering_angle = math.atan(WHEELBASE * curvature)
        steering_angle = float(np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))
        speed = self.get_target_speed(target_index, steering_angle)
                
        self.publish_drive(speed, steering_angle)
        self.publish_markers()

    def get_adaptive_lookahead(self, closest_index):
        """Use longer lookahead on straights and shorter lookahead in tight corners."""
        speed = self.waypoints_speed[closest_index] if self.waypoints_speed else BASE_SPEED
        curvature = abs(self.waypoints_curvature[closest_index]) if self.waypoints_curvature else 0.0
        speed_ratio = (speed - MIN_SPEED) / max(MAX_SPEED - MIN_SPEED, 1e-6)
        lookahead = MIN_LOOKAHEAD_DISTANCE + speed_ratio * (MAX_LOOKAHEAD_DISTANCE - MIN_LOOKAHEAD_DISTANCE)
        lookahead /= (1.0 + 0.20 * curvature)
        return float(np.clip(lookahead, MIN_LOOKAHEAD_DISTANCE, MAX_LOOKAHEAD_DISTANCE))

    def get_target_speed(self, target_index, steering_angle):
        """Follow the waypoint speed profile, with a final steering safety cap."""
        if self.waypoints_speed:
            speed = self.waypoints_speed[target_index]
        else:
            speed = BASE_SPEED

        speed *= SPEED_TEST_SCALE

        steering_abs = abs(steering_angle)
        if steering_abs > np.radians(18.0):
            speed = min(speed, 1.05)
        elif steering_abs > np.radians(12.0):
            speed = min(speed, 1.30)

        return float(np.clip(speed, MIN_SPEED, MAX_SPEED))

    def find_lookahead_index(self, waypoints, closest_index, lookahead_distance):
        """Pick a waypoint ahead on the recorded path, not just any map point."""
        waypoint_count = len(waypoints)
        if waypoint_count <= 1:
            return closest_index

        for offset in range(1, waypoint_count + 1):
            index = (closest_index + offset) % waypoint_count
            dx = waypoints[index, 0] - self.pos_x
            dy = waypoints[index, 1] - self.pos_y
            local_x = math.cos(self.heading_angle) * dx + math.sin(self.heading_angle) * dy
            local_y = -math.sin(self.heading_angle) * dx + math.cos(self.heading_angle) * dy
            distance = math.hypot(local_x, local_y)
            angle_error = math.atan2(local_y, local_x)

            if (
                distance >= lookahead_distance
                and local_x > 0.0
                and abs(angle_error) <= HEADING_THRESHOLD
            ):
                return index

        fallback_offset = max(1, int(lookahead_distance / 0.1))
        return (closest_index + fallback_offset) % waypoint_count

    def publish_drive(self, speed, steering_angle):
        """Publishes the drive command with speed and steering_angle."""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = DRIVE_SPEED_SIGN * speed
        self.drive_pub.publish(drive_msg) 
        self.get_logger().info(f"Steering Angle: {steering_angle*180/np.pi}, Speed: {drive_msg.drive.speed}")
        

    def publish_markers(self):
        """Publishes markers for visualization in RViz."""
        # Publish marker for waypoints
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit_waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.points = [Point(x=x, y=y, z=0.0) for x, y in zip(self.waypoints_x, self.waypoints_y)]
        self.waypoints_marker_pub.publish(marker)

        # Publish marker for target waypoint
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.header.stamp = self.get_clock().now().to_msg()
        target_marker.ns = "pure_pursuit_target"
        target_marker.id = 1
        target_marker.type = Marker.POINTS
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.color.a = 1.0
        target_marker.color.r = 1.0
        target_marker.points = [Point(x=self.target_x, y=self.target_y, z=0.0)]
        self.target_marker_pub.publish(target_marker)

        marker_array = MarkerArray()
        marker_array.markers = [marker, target_marker]
        self.waypoints_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
