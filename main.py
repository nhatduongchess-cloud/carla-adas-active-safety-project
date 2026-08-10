"""Main execution script for CARLA Autonomous Driving ADAS Pipeline."""

import sys
import os

# ==============================================================================
# BẮT BUỘC PHẢI NẰM Ở ĐẦU FILE: Tự động nhận diện thư mục chứa code
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- SAU KHI SỬA ĐƯỜNG DẪN MỚI ĐƯỢC GỌI IMPORT ---
import time
import queue
import weakref
from typing import Optional

import numpy as np
import cv2
import carla

# Các module của dự án
from midas_depth import DepthEstimator
from object_tracking import EnsembleVehicleTracker
from lidar_processor import LidarProcessor
from sensor_fusion import SensorFusion
from active_safety import ActiveSafetySystem
from lane_detection import LaneDetector
from planner import TransFuserPlanner 
from local_planner import LocalPlanner
from lateral_controller import LateralController
from vehicle_controller import VehicleController 
from scenario_manager import ScenarioManager
from data_logger import TelemetryLogger
from dashboard import draw_dashboard
from bev import draw_bev
from visualization import Visualizer
from collisions import CollisionTracker
from counter import ObjectCounter


class CarlaSyncManager:
    """Context manager to handle CARLA synchronous mode and cleanup."""
    def __init__(self, world: carla.World, fps: int = 20) -> None:
        self.world = world
        self.original_settings = world.get_settings()
        self.sync_settings = world.get_settings()
        self.sync_settings.synchronous_mode = True
        self.sync_settings.fixed_delta_seconds = 1.0 / fps

    def __enter__(self):
        self.world.apply_settings(self.sync_settings)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.world.apply_settings(self.original_settings)
        print("[System] Restored CARLA to asynchronous mode.")


def process_rgb_image(carla_image: carla.Image) -> np.ndarray:
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    return array[:, :, :3] 


def process_lidar_data(carla_lidar: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(carla_lidar.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


def main() -> None:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    actor_list = []

    print("[System] Initializing ADAS pipeline modules...")
    depth_estimator = DepthEstimator()
    object_tracker = EnsembleVehicleTracker(device='cuda')
    lidar_processor = LidarProcessor(eps=0.6, min_samples=5)
    lane_detector = LaneDetector()
    active_safety = ActiveSafetySystem(img_width=1280)
    
    lateral_control = LateralController(k_gain=2.5)
    vehicle_control = VehicleController()
    local_planner = LocalPlanner()
    telemetry = TelemetryLogger()

    scenario_manager = ScenarioManager(world)
    spawn_points = world.get_map().get_spawn_points()
    ego_transform = spawn_points[0]
    
    ego_bp = blueprint_library.find('vehicle.tesla.model3')
    ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
    actor_list.append(ego_vehicle)
    
    scenario_manager.setup_lead_vehicle_braking_scenario(spawn_points[1])

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
    actor_list.append(camera_sensor)

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '50.0')
    lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    actor_list.append(lidar_sensor)

    image_queue = queue.Queue()
    lidar_queue = queue.Queue()
    camera_sensor.listen(image_queue.put)
    lidar_sensor.listen(lidar_queue.put)

    print("[System] Pipeline initialized. Entering synchronous loop.")

    try:
        with CarlaSyncManager(world, fps=20):
            frame_count = 0
            while True:
                world.tick()
                raw_image = image_queue.get()
                raw_lidar = lidar_queue.get()

                rgb_frame = process_rgb_image(raw_image)
                point_cloud = process_lidar_data(raw_lidar)

                depth_map = depth_estimator.predict(rgb_frame)
                
                annotated_frame, detections, metrics = object_tracker.process(
                    rgb_frame, depth_map, frame_count
                )
                
                lidar_clusters = lidar_processor.extract_obstacles(point_cloud)
                fused_detections = SensorFusion.fuse_data(detections, lidar_clusters)
                
                lane_frame = lane_detector.process(annotated_frame)

                current_speed_m_s = ego_vehicle.get_velocity().length()
                current_speed_kmh = current_speed_m_s * 3.6
                ego_transform = ego_vehicle.get_transform()
                
                current_location = (ego_transform.location.x, ego_transform.location.y)
                target_waypoints = local_planner.generate_trajectory(current_location, current_speed_kmh)

                is_emergency = active_safety.evaluate_and_control(ego_vehicle, fused_detections, lane_frame)

                if not is_emergency:
                    cross_track_error = 0.0  
                    target_heading = ego_transform.rotation.yaw
                    
                    steering_cmd = lateral_control.compute_steering_angle(
                        np.radians(ego_transform.rotation.yaw), cross_track_error, current_speed_m_s
                    )
                    
                    target_speed = 30.0
                    throttle_cmd = 0.5 if current_speed_kmh < target_speed else 0.0
                    brake_cmd = 0.0 if current_speed_kmh < target_speed else 0.2
                    
                    final_control = vehicle_control.create_control_command(
                        target_throttle=throttle_cmd,
                        target_brake=brake_cmd,
                        raw_steering=steering_cmd
                    )
                    ego_vehicle.apply_control(final_control)

                metrics['speed'] = current_speed_kmh
                metrics['fsm_state'] = "EMERGENCY" if is_emergency else "LANE_FOLLOWING"
                
                telemetry.log(ego_vehicle, len(fused_detections))
                
                hud_frame = Visualizer.draw_hud(lane_frame, metrics)
                
                cv2.imshow("ADAS Portfolio - Main Dashboard", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[System] User requested shutdown.")
                    break
                
                frame_count += 1

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    except Exception as e:
        print(f"\n[System] Fatal Error encountered: {e}")
    finally:
        print("[System] Initiating resource cleanup...")
        telemetry.close()
        cv2.destroyAllWindows()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("[System] Cleanup complete. Pipeline terminated safely.")


if __name__ == '__main__':
    main()
    