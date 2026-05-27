#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class GapFollowNode(Node):
    def __init__(self):
        super().__init__('gap_follow_node')
        
        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('bubble_radius', 0.2)  # 안전 버블 반경 (미터)
        self.declare_parameter('car_width', 0.3)  # 차량 폭
        self.declare_parameter('disparity_threshold', 0.20)  # Disparity 감지 임계값
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('min_speed', 2.5)
        self.declare_parameter('brake_distance', 1.5)  # 급제동 거리
        self.declare_parameter('fov_degrees', 180.0)  # 🔥 FOV 설정 (210도)
        
        scan_topic = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.bubble_radius = self.get_parameter('bubble_radius').value
        self.car_width = self.get_parameter('car_width').value
        self.disparity_threshold = self.get_parameter('disparity_threshold').value
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        self.brake_distance = self.get_parameter('brake_distance').value
        self.fov_degrees = self.get_parameter('fov_degrees').value
        
        # Subscribers and Publishers
        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            10
        )
        
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )
        
        self.get_logger().info(f'Gap Follow Node Initialized - FOV: {self.fov_degrees}°, Bubble: {self.bubble_radius}m')
    
    def preprocess_lidar(self, ranges):
        """
        LiDAR 데이터 전처리
        - inf, nan 값 처리
        - 최소/최대 거리 클리핑
        """
        proc_ranges = np.array(ranges)
        
        # inf, nan 값을 0으로 변환
        proc_ranges[np.isinf(proc_ranges)] = 0.0
        proc_ranges[np.isnan(proc_ranges)] = 0.0
        
        # 최대 거리 제한 (10m)
        proc_ranges = np.clip(proc_ranges, 0.0, 10.0)
        
        return proc_ranges
    
    def find_disparities(self, ranges):
        """
        급격한 거리 변화(Disparity) 찾기
        """
        disparities = []
        
        for i in range(len(ranges) - 1):
            # 인접한 두 스캔 포인트 간 거리 차이 계산
            diff = abs(ranges[i] - ranges[i + 1])
            
            # Disparity 임계값을 넘으면 기록
            if diff > self.disparity_threshold:
                # 더 가까운 포인트의 인덱스 저장
                closer_idx = i if ranges[i] < ranges[i + 1] else i + 1
                disparities.append(closer_idx)
        
        return disparities
    
    def apply_disparity_extender(self, ranges, disparities, angle_increment):
        """
        🔥 수정: Disparity Extender 적용 - 실제 angle_increment 사용
        """
        processed_ranges = ranges.copy()
        
        for disp_idx in disparities:
            # 장애물까지의 거리
            obstacle_distance = ranges[disp_idx]
            
            if obstacle_distance < 0.1:  # 너무 가까운 경우 스킵
                continue
            
            # 안전 버블의 각도 범위 계산
            bubble_angle = np.arcsin(min(1.0, self.bubble_radius / obstacle_distance))
            
            # 🔥 수정: 실제 angle_increment 사용!
            bubble_indices = int(bubble_angle / angle_increment)
            
            # 양쪽으로 버블 적용
            start_idx = max(0, disp_idx - bubble_indices)
            end_idx = min(len(ranges), disp_idx + bubble_indices + 1)
            
            # 버블 영역을 0으로 설정 (통과 불가)
            processed_ranges[start_idx:end_idx] = 0.0
        
        return processed_ranges
    
    def find_gaps(self, ranges):
        """
        통과 가능한 갭 찾기
        """
        gaps = []
        in_gap = False
        gap_start = 0
        
        # 차량이 통과 가능한 최소 거리
        min_gap_distance = self.car_width * 1.5
        
        for i, distance in enumerate(ranges):
            if distance > min_gap_distance:
                if not in_gap:
                    # 새로운 갭 시작
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    # 갭 종료
                    in_gap = False
                    gap_end = i - 1
                    gaps.append((gap_start, gap_end))
        
        # 마지막 갭 처리
        if in_gap:
            gaps.append((gap_start, len(ranges) - 1))
        
        return gaps
    
    def select_best_gap(self, ranges, gaps):
        """
        가장 좋은 갭 선택
        - 가장 깊은(먼) 갭 선택
        """
        if not gaps:
            return None
        
        best_gap = None
        max_depth = 0
        
        for gap_start, gap_end in gaps:
            # 갭의 최대 깊이 계산
            gap_ranges = ranges[gap_start:gap_end + 1]
            gap_depth = np.max(gap_ranges)
            
            if gap_depth > max_depth:
                max_depth = gap_depth
                best_gap = (gap_start, gap_end)
        
        return best_gap
    
    def find_target_point(self, ranges, best_gap):
        """
        타겟 포인트 찾기
        - 갭 내에서 가장 먼 지점
        """
        if best_gap is None:
            return None
        
        gap_start, gap_end = best_gap
        gap_ranges = ranges[gap_start:gap_end + 1]
        
        # 갭 내에서 가장 먼 지점의 인덱스
        max_idx = np.argmax(gap_ranges)
        target_idx = gap_start + max_idx
        
        return target_idx
    
    def calculate_steering_angle(self, target_idx, angle_min, angle_increment):
        """
        🔥 수정: 조향각 계산 - 실제 LiDAR 각도 사용
        """
        # 실제 타겟 각도 계산
        target_angle = angle_min + target_idx * angle_increment * 1.15
        
        # 조향각 제한 (±60도)
        steering_angle = np.clip(target_angle, -1.047, 1.047)
        
        return steering_angle
    
    def calculate_speed(self, steering_angle, min_distance):
        """
        조향각과 전방 거리에 따른 속도 계산
        """
        # 급제동 필요 시
        if min_distance < self.brake_distance:
            return self.min_speed
        
        # 조향각에 따른 속도 조절
        angle_abs = abs(steering_angle)
        
        if angle_abs < 0.15:  # ~10도 미만: 직선 구간
            speed = self.max_speed 
        elif angle_abs < 0.35:  # ~20도 미만: 완만한 코너
            speed = self.max_speed * 0.2
        elif angle_abs < 0.55:  # ~20도 미만: 완만한 코너
            speed = self.max_speed * 0.35
        elif angle_abs < 0.75:  # ~20도 미만: 완만한 코너
            speed = self.max_speed * 0.55        
        else:  # 급커브
            speed = self.min_speed * 0.5
        
        # 전방 거리에 따른 속도 보정
        distance_factor = min(1.0, min_distance / 4.0)
        speed *= distance_factor
        
        return max(self.min_speed * 0.5, speed)
    
    def scan_callback(self, scan_msg):
        """
        LiDAR 스캔 콜백
        """
        # 🔥 angle 정보 가져오기
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        # 1. 전처리
        ranges = self.preprocess_lidar(scan_msg.ranges)
        
        # 🔥 추가: FOV 제한 (180도)
        fov_rad = np.deg2rad(self.fov_degrees)
        half_fov = fov_rad / 2.0
        
        # FOV 범위 계산
        fov_start_angle = -half_fov
        fov_end_angle = +half_fov
        
        i_min = int((fov_start_angle - angle_min) / angle_increment)
        i_max = int((fov_end_angle - angle_min) / angle_increment)
        
        i_min = max(0, i_min)
        i_max = min(len(ranges) - 1, i_max)
        
        # FOV 밖은 0으로 설정
        fov_ranges = np.zeros_like(ranges)
        fov_ranges[i_min:i_max+1] = ranges[i_min:i_max+1]
        
        # 2. Disparity 찾기
        disparities = self.find_disparities(fov_ranges)
        
        # 3. 🔥 Disparity Extender 적용 (angle_increment 전달!)
        processed_ranges = self.apply_disparity_extender(fov_ranges, disparities, angle_increment)
        
        # 4. 갭 찾기
        gaps = self.find_gaps(processed_ranges)
        
        # 5. 최적 갭 선택
        best_gap = self.select_best_gap(processed_ranges, gaps)
        
        # 6. 타겟 포인트 결정
        target_idx = self.find_target_point(processed_ranges, best_gap)
        
        # 7. 제어 명령 계산
        if target_idx is not None:
            # 🔥 실제 각도 정보 전달
            steering_angle = self.calculate_steering_angle(
                target_idx,
                angle_min,
                angle_increment
            )
            
            # 🔥 수정: 전방 최소 거리 계산 (min 사용!)
            center_idx = len(ranges) // 2
            front_range = int(len(ranges) * 120 / 360)
            front_ranges = processed_ranges[
                center_idx - front_range:center_idx + front_range
            ]
            min_distance = np.max(front_ranges[front_ranges > 0.1]) if np.any(front_ranges > 0.1) else 0.5
            
            speed = self.calculate_speed(steering_angle, min_distance)
        else:
            # 갭이 없으면 천천히 회전하며 탐색
            self.get_logger().warn('No valid gap found! Searching...')
            steering_angle = 0.3
            speed = self.min_speed * 0.5
        
        # 8. 제어 명령 발행
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = -speed
        
        self.drive_pub.publish(drive_msg)
        
        # 디버그 정보
        # self.get_logger().info(
        #     f'Steering: {np.rad2deg(steering_angle):.1f}°, '
        #     f'Speed: {speed:.2f} m/s, '
        #     f'Gaps: {len(gaps)}, '
        #     f'Disparities: {len(disparities)}'
        # )

def main(args=None):
    rclpy.init(args=args)
    node = GapFollowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()