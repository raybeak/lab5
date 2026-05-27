#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker


class GapFollow(Node):
    def __init__(self):
        super().__init__('gap_follow_node')

        # Topics for publishing and subscribing
        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'
        self.best_point_marker_topic = '/best_point_marker'
        self.bubble_marker_topic = '/bubble_point_marker'

        # Create subscribers and publishers
        self.lidar_sub = self.create_subscription(
            LaserScan, self.lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, self.drive_topic, 10)
        self.best_point_marker_pub = self.create_publisher(
            Marker, self.best_point_marker_topic, 10)
        self.bubble_marker_pub = self.create_publisher(
            Marker, self.bubble_marker_topic, 10)

        # 센서 및 안전 파라미터
        self.robot_radius = 0.0
        self.safety_distance = 0.35  
        self.disparity_thresh = 0.20  
        self.fov_deg = 220.0  
        self.min_gap_len = 5  


        self.lookahead_tight = 0.80   
        self.lookahead_wide = 0.60    
        self.prev_speed = 1.95      
        self.prev_steering = 0.0
        self.prev_ratio = 0.65
        self.max_accel = 3.25      
        self.max_decel = 4.55       
        self.emergency_brake_dist = 0.7
        self.caution_dist = 1.56   

        self.get_logger().info("🚗 30% Speed Boost Gap Follower Initialized.")

    # 라이다 데이터 전처리
    def preprocess_scan(self, ranges, min_r=0.1, max_r=10.0, smooth_win=4):
       
        r = np.array(ranges, dtype=np.float32)
        r[~np.isfinite(r)] = max_r  # 무한값을 최대거리로
        r = np.clip(r, min_r, max_r)  # 범위 제한
        
        # Moving average 필터로 노이즈 제거
        if smooth_win > 1:
            kernel = np.ones(smooth_win) / smooth_win
            r = np.convolve(r, kernel, mode='same')
        return r

    # 안전 버블 적용 (사용하지 않지만 유지)
    def apply_safety_bubble(self, ranges):

        safe_ranges = np.copy(ranges)
        too_close = safe_ranges < self.safety_distance
        safe_ranges[too_close] = 0.0
        return safe_ranges

    # Disparity 탐지
    def detect_disparities(self, ranges, thresh):
       
        disps = []
        for i in range(len(ranges) - 1):
            diff = ranges[i + 1] - ranges[i]
            if abs(diff) >= thresh:
                # 어느 쪽이 더 가까운지 판단
                if ranges[i] < ranges[i + 1]:
                    i_close, i_far, direction = i, i + 1, +1
                else:
                    i_close, i_far, direction = i + 1, i, -1
                disps.append((i_close, i_far, direction))
        return disps

    # Disparity 확장
    def extend_disparity(self, vranges, i_close, i_far, direction, angle_inc):
    
        r_close = vranges[i_close]  # 가까운 쪽 거리
        dtheta = angle_inc  # 각도 간격
        per_sample_span = max(r_close * dtheta, 1e-6)  # 한 샘플당 각도 범위
        n_cover = int(np.ceil(self.safety_distance / per_sample_span))  # 보정할 범위 계산
        
        j = i_far
        for _ in range(n_cover):
            if 0 <= j < len(vranges):
                if vranges[j] > r_close or vranges[j] <= 0.0:
                    vranges[j] = r_close
                j += direction

    # Virtual Range 생성
    def build_virtual_ranges(self, ranges, angle_inc):
        vranges = np.copy(ranges)
        disps = self.detect_disparities(vranges, self.disparity_thresh)
        for (i_close, i_far, direction) in disps:
            self.extend_disparity(vranges, i_close, i_far, direction, angle_inc)
        return vranges, disps

    # 갭 찾기
    def find_gaps(self, ranges):

        gaps = []
        i = 0
        N = len(ranges)
        while i < N:
            if ranges[i] > 0:  # 감지된 부분 시작
                s = i
                # 연속된 > 0 찾기
                while i + 1 < N and ranges[i + 1] > 0:
                    i += 1
                e = i
                # 최소 길이 이상만 갭으로 인정
                if e - s >= self.min_gap_len:
                    gaps.append((s, e))
            i += 1
        return gaps
    
    # 최적 갭 선택
    def select_best_gap(self, ranges, gaps):
        if not gaps:
            return None
        
        # 가장 넓은 갭을 선택 (거리 아니라 폭!)
        widths = [e - s for (s, e) in gaps]
        max_width = np.max(widths)
        candidates = [g for g, w in zip(gaps, widths) if w == max_width]
        
        # 같은 폭이 여러 개면 그 중 가장 먼 갭
        if len(candidates) == 1:
            return candidates[0]
    
        scores = [np.mean(ranges[s:e]) for (s, e) in candidates]
        return candidates[int(np.argmax(scores))]

    # 갭 내 목표점 선택 
    def choose_target(self, ranges, gap):
      
        s, e = gap
        gap_width = e - s
        
        # 갭 너비에 따라 목표점 선택 전략 결정
        # 좁을수록 중앙을 더 선호
        if gap_width <= 10:  # 매우 좁은 길
            target_ratio = 0.92
        elif gap_width <= 15:  # 좁은 길
            target_ratio = 0.85
        elif gap_width <= 20:  # 중간
            target_ratio = 0.75
        elif gap_width <= 30:  # 넓은 길 진입
            target_ratio = 0.65
        else:  # 매우 넓은 길
            target_ratio = 0.55
        
        # Ratio 평활화: 급격한 변화 방지 (코너 진입 부드럽게)
        alpha_ratio = 0.25
        ratio = alpha_ratio * target_ratio + (1 - alpha_ratio) * self.prev_ratio
        self.prev_ratio = ratio
        
        # 목표점 결정
        mid = (s + e) // 2  # 갭의 중앙
        farthest = s + int(np.argmax(ranges[s:e + 1]))  # 갭 내 가장 먼 점
        target = int(ratio * mid + (1 - ratio) * farthest)  # 가중치 조합
        
        return target, ranges[target]

    # 속도 제어 (동적 속도 조정)
    def compute_speed(self, front_min, steer_angle, dt=0.05):
        
        # 전방 거리에 따른 속도 결정
        if front_min < self.emergency_brake_dist:
            target_speed = 0.78    
        elif front_min < self.caution_dist:
            target_speed = 2.34   
        else:
            # 일반 주행: 거리에 비례해서 속도 조절
            base_speed = min(front_min * 1.3, 5.85) 
            # 회전할 때는 속도 감속
            steer_penalty = abs(steer_angle) * 2.0
            target_speed = max(base_speed - steer_penalty, 1.56) 

        # 가속/감속 제한 
        speed_diff = target_speed - self.prev_speed
        max_change = (self.max_accel if speed_diff > 0 else self.max_decel) * dt
        speed_diff = np.clip(speed_diff, -max_change, max_change)
        new_speed = self.prev_speed + speed_diff
        self.prev_speed = new_speed
        
        return float(np.clip(new_speed, 0.78, 5.85))  # 0.6 × 1.3, 4.5 × 1.3


    def publish_best_point_marker(self, best_point_idx, best_point_distance, angle_min, angle_increment):
        """RViz에 목표점을 녹색 공으로 표시"""
        best_point_angle = angle_min + best_point_idx * angle_increment
        x = best_point_distance * np.cos(best_point_angle)
        y = best_point_distance * np.sin(best_point_angle)

        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.g = 1.0
        self.best_point_marker_pub.publish(marker)
 
    def publish_closest_bubble_marker(self, bubble_idx, bubble_distance, bubble_radius, angle_min, angle_increment):
        """RViz에 Disparity 확장 범위를 파란색 공으로 표시"""
        bubble_point_angle = angle_min + bubble_idx * angle_increment
        x = bubble_distance * np.cos(bubble_point_angle)
        y = bubble_distance * np.sin(bubble_point_angle)

        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bubble_point"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bubble_radius
        marker.scale.y = bubble_radius
        marker.scale.z = bubble_radius
        marker.color.a = 0.5
        marker.color.b = 1.0
        self.bubble_marker_pub.publish(marker)


    def scan_callback(self, data):
        
        # 1. 라이다 데이터 처리 
        ranges = np.array(data.ranges, dtype=np.float32)
        angle_min = data.angle_min
        angle_increment = data.angle_increment

        # 데이터 전처리 (노이즈 제거)
        proc_ranges = self.preprocess_scan(ranges)
        
        # Virtual range 생성 (Disparity 확장)
        virtual_ranges, disps = self.build_virtual_ranges(proc_ranges, angle_increment)

        # 2. FOV(Field of View) 제한
        # 뒤쪽 데이터는 무시하고 앞쪽 270도만 사용
        fov = np.deg2rad(self.fov_deg)
        half_fov = fov / 2.0
        
        fov_start_angle = -half_fov  # -135도
        fov_end_angle = +half_fov    # +135도
        
        # 각도를 인덱스로 변환
        i_min = int((fov_start_angle - angle_min) / angle_increment)
        i_max = int((fov_end_angle - angle_min) / angle_increment)
        
        i_min = max(0, i_min)
        i_max = min(len(virtual_ranges) - 1, i_max)
        
        # FOV 범위만 사용
        vr = np.zeros_like(virtual_ranges)
        vr[i_min:i_max+1] = virtual_ranges[i_min:i_max+1]

        # 3. 갭 찾기 및 선택 
        gaps = self.find_gaps(vr)
        if gaps:
            best_gap = self.select_best_gap(vr, gaps)
            gap_width = best_gap[1] - best_gap[0]
            
            # 좁은 갭이면 safety_distance 줄여서 통과 가능하게
            if gap_width < 15:  # 좁은 갭
                self.safety_distance = 0.20  # 보정 범위 줄임
            else:  # 넓은 갭
                self.safety_distance = 0.25  # 일반적인 보정 범위
            
            best_point, best_point_distance = self.choose_target(vr, best_gap)
        else:
            # 갭이 없으면 가장 먼 점으로
            valid_idx = np.where(vr > 0)[0]
            if len(valid_idx) > 0:
                best_point = valid_idx[np.argmax(vr[valid_idx])]
            else:
                best_point = len(vr) // 2
            best_point_distance = float(vr[best_point])

        # 4. 스티어링 계산 및 평활화
        # 목표점을 각도로 변환
        raw_steering = float(np.clip(angle_min + best_point * angle_increment, -0.48, 0.48))
        
        # 1단계: 저주파 필터 (LPF) - 노이즈 제거
        alpha = 0.08
        smoothed = alpha * raw_steering + (1 - alpha) * self.prev_steering
        
        # 2단계: 변화량 제한 - 너무 급하게 꺾이는 것 방지
        max_steering_change = 0.08
        steering_diff = smoothed - self.prev_steering
        steering_diff = np.clip(steering_diff, -max_steering_change, max_steering_change)
        steering_angle = self.prev_steering + steering_diff
        
        self.prev_steering = steering_angle

        # 5. 정면 최소 거리 계산 (안전성 체크)
        center_deg = 0.0
        span_deg = 30.0  # 정면 30도 범위
        i0 = int((np.deg2rad(center_deg - span_deg/2) - angle_min) / angle_increment)
        i1 = int((np.deg2rad(center_deg + span_deg/2) - angle_min) / angle_increment)
        front_min = float(np.min(vr[i0:i1])) if i1 > i0 else 1.0

        # 6. 속도 계산
        speed = self.compute_speed(front_min, steering_angle)
        
        # 7. 제어 명령 송신
        self.reactive_control(steering_angle, speed)

        # 8. 마커 시각화
        self.publish_best_point_marker(best_point, best_point_distance, angle_min, angle_increment)
        if disps:
            i_close, i_far, direction = disps[0]
            bubble_index = i_close
            bubble_distance = vr[bubble_index]
            bubble_radius = 0.3
            self.publish_closest_bubble_marker(
                bubble_index, bubble_distance, bubble_radius, angle_min, angle_increment)

        # 9. 디버그 로그
        self.get_logger().info(
            f"[FTG] gaps={len(gaps) if gaps else 0}, idx={best_point}/{len(vr)}, "
            f"dist={best_point_distance:.2f}m, steer={np.degrees(steering_angle):.1f}°, "
            f"speed={speed:.2f}m/s, front={front_min:.2f}m"
        )

    # 제어 명령 송신 
    def reactive_control(self, steering_angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = -speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    reactive_node = GapFollow()
    rclpy.spin(reactive_node)
    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()