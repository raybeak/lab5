#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from time import gmtime, strftime
from numpy import linalg as LA
from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
# 통신 규격(QoS) 불일치를 해결하기 위한 모듈 추가
from rclpy.qos import qos_profile_sensor_data
import os

class AmclWaypointsLogger(Node):
    def __init__(self):
        super().__init__('amcl_waypoints_logger')
        self.file = open(os.path.join(os.getcwd(), strftime('wp_amcl-%Y-%m-%d-%H-%M-%S.csv', gmtime())), 'w')
        
        self.current_speed = 0.0
        self.latest_pose = None
        
        # Odom 구독 (속도용) - QoS를 sensor_data(Best Effort)로 유연하게 설정
        self.odom_sub = self.create_subscription(
            Odometry, 
            '/odom', 
            self.odom_callback, 
            qos_profile_sensor_data)
        
        # AMCL Pose 구독 (위치용) - QoS를 sensor_data로 설정하여 통신 거부 해결!
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 
            '/amcl_pose', 
            self.pose_callback, 
            qos_profile_sensor_data)
        
        self.timer = self.create_timer(0.1, self.save_waypoint_callback)
        
        atexit.register(self.shutdown)
        self.get_logger().info('Saving AMCL-based waypoints at 10Hz...')

    def odom_callback(self, msg):
        self.current_speed = LA.norm([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ], 2)

    def pose_callback(self, msg):
        # 이제 데이터를 정상적으로 수신하여 위치를 업데이트합니다.
        self.latest_pose = msg.pose.pose

    def save_waypoint_callback(self):
        if self.latest_pose is None:
            self.get_logger().warn(
                'Waiting for /amcl_pose... (Did you set 2D Pose Estimate in RViz?)', 
                throttle_duration_sec=2.0)
            return

        x = self.latest_pose.position.x
        y = self.latest_pose.position.y
        orientation = self.latest_pose.orientation
        quaternion = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        euler = euler_from_quaternion(quaternion)
        
        self.file.write('%f, %f, %f, %f\n' % (x, y, euler[2], self.current_speed))
        self.file.flush()

    def shutdown(self):
        if not self.file.closed:
            self.file.close()
        self.get_logger().info('Goodbye. Waypoints saved successfully.')

def main(args=None):
    rclpy.init(args=args)
    logger_node = AmclWaypointsLogger()

    try:
        rclpy.spin(logger_node)
    except KeyboardInterrupt:
        pass
    finally:
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()