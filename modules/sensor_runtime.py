"""CARLA sensor ownership, buffering and frame acquisition.

``SensorRig`` is the only runtime component that knows how camera, LiDAR,
radar and collision actors are created and read.  The orchestrator consumes a
single ``SensorReadout`` contract and does not manipulate sensor queues.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import carla
import numpy as np

from collision_sensor import CollisionSensor
from sensor_setup import (configure_camera_blueprint,
                          configure_lidar_blueprint,
                          configure_radar_blueprint)
from sensor_sync import (LatestFrameReader, OptionalFrameReader,
                         SensorSyncStats, put_latest,
                         retrieve_exact_frame, warmup_sensor_streams)


def decode_rgb(carla_image: carla.Image) -> np.ndarray:
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    return array[:, :, :3].copy()


def decode_lidar(carla_lidar: carla.LidarMeasurement) -> np.ndarray:
    points = np.frombuffer(carla_lidar.raw_data, dtype=np.dtype("f4"))
    return np.reshape(points, (int(points.shape[0] / 4), 4)).copy()


@dataclass(frozen=True)
class SensorReadout:
    frame_id: int
    capture_timestamp: float
    rgb: np.ndarray
    point_cloud: np.ndarray
    radar_measurement: Any
    lidar_timestamp: float
    camera_fresh: bool
    camera_frame_id: Optional[int]
    camera_sensor_timestamp: Optional[float]
    camera_wall_timestamp: Optional[float]
    radar_received_monotonic_s: Optional[float] = None


@dataclass(frozen=True)
class _ReceivedRadar:
    frame: int
    measurement: Any
    received_monotonic_s: float


class SensorRig:
    """Own the CARLA sensor actors and expose bounded, mode-aware reads."""

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle,
                 cfg: Any, stable_async: bool) -> None:
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.cfg = cfg
        self.stable_async = stable_async

        self.camera = None
        self.lidar = None
        self.radar = None
        self.collision: Optional[CollisionSensor] = None
        self._collision_actor = None
        self.sync_stats = SensorSyncStats()
        self.radar_sync_stats = SensorSyncStats()

        self._image_queue: queue.Queue = queue.Queue(maxsize=4)
        self._lidar_queue: queue.Queue = queue.Queue(maxsize=4)
        self._radar_queue: queue.Queue = queue.Queue(maxsize=4)
        self._camera_latest = LatestFrameReader(
            self._image_queue, "camera", self.sync_stats)
        self._lidar_latest = LatestFrameReader(
            self._lidar_queue, "lidar", self.sync_stats)
        self._radar_latest = LatestFrameReader(
            self._radar_queue, "radar", self.radar_sync_stats)
        self._radar_exact = OptionalFrameReader(
            self._radar_queue, "radar", self.radar_sync_stats)

        self._rgb: Optional[np.ndarray] = None
        self._camera_frame_id: Optional[int] = None
        self._camera_sensor_timestamp: Optional[float] = None
        self._camera_wall_timestamp: Optional[float] = None
        self._stopped = False

    @property
    def actors(self) -> List[carla.Actor]:
        actors = [self.camera, self.lidar, self.radar]
        if self.collision is not None:
            actors.append(self.collision.sensor)
        elif self._collision_actor is not None:
            actors.append(self._collision_actor)
        return [actor for actor in actors if actor is not None]

    def spawn(self) -> None:
        blueprints = self.world.get_blueprint_library()
        camera_bp = configure_camera_blueprint(
            blueprints.find('sensor.camera.rgb'), self.cfg)
        lidar_bp = configure_lidar_blueprint(
            blueprints.find('sensor.lidar.ray_cast'), self.cfg)
        radar_bp = (configure_radar_blueprint(
            blueprints.find('sensor.other.radar'), self.cfg)
                    if self.cfg.ENABLE_RADAR else None)

        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=self.cfg.CAM_X, y=self.cfg.CAM_Y, z=self.cfg.CAM_Z),
                carla.Rotation(roll=self.cfg.CAM_ROLL, pitch=self.cfg.CAM_PITCH,
                               yaw=self.cfg.CAM_YAW)),
            attach_to=self.ego_vehicle)
        self.lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(
                carla.Location(x=self.cfg.LIDAR_X, y=self.cfg.LIDAR_Y,
                               z=self.cfg.LIDAR_Z),
                carla.Rotation(roll=self.cfg.LIDAR_ROLL,
                               pitch=self.cfg.LIDAR_PITCH,
                               yaw=self.cfg.LIDAR_YAW)),
            attach_to=self.ego_vehicle)
        if radar_bp is not None:
            self.radar = self.world.spawn_actor(
                radar_bp,
                carla.Transform(
                    carla.Location(x=self.cfg.RADAR_X, y=self.cfg.RADAR_Y,
                                   z=self.cfg.RADAR_Z),
                    carla.Rotation(roll=self.cfg.RADAR_ROLL,
                                   pitch=self.cfg.RADAR_PITCH,
                                   yaw=self.cfg.RADAR_YAW)),
                attach_to=self.ego_vehicle)

        self.collision = CollisionSensor(
            self.world, self.ego_vehicle, fps=self.cfg.FPS,
            on_spawn=lambda actor: setattr(self, '_collision_actor', actor))
        self.camera.listen(lambda data: put_latest(self._image_queue, data))
        self.lidar.listen(lambda data: put_latest(self._lidar_queue, data))
        if self.radar is not None:
            self.radar.listen(self._queue_radar)

    def _queue_radar(self, data: Any) -> None:
        # Keep the arrival clock attached to THIS queued sample (including any
        # future-frame stash); never substitute the loop's pre-read timestamp.
        put_latest(self._radar_queue, _ReceivedRadar(
            int(data.frame), data, time.monotonic()))

    def warmup(self, wall_timestamp: float) -> None:
        if self.stable_async:
            camera = self._camera_latest.get_latest(timeout=10.0)
            lidar = self._lidar_latest.get_latest(timeout=10.0)
            if camera is None or lidar is None:
                raise TimeoutError("async-stable sensor warm-up thiếu camera hoặc LiDAR")
            self._rgb = decode_rgb(camera)
            self._camera_frame_id = int(camera.frame)
            self._camera_sensor_timestamp = float(camera.timestamp)
            self._camera_wall_timestamp = wall_timestamp
            print(f"[System] Async sensors ready: camera={camera.frame}, "
                  f"lidar={lidar.frame}.")
            return

        warmup_sensor_streams(
            self.world,
            {"camera": self._image_queue, "lidar": self._lidar_queue})

    def read(self, capture_timestamp: float) -> SensorReadout:
        if self.stable_async:
            lidar_data = self._lidar_latest.get_latest(timeout=2.0)
            if lidar_data is None:
                raise TimeoutError("async-stable timeout chờ LiDAR geometry")
            frame_id = int(lidar_data.frame)
            camera_data = self._camera_latest.get_at_or_before(frame_id)
            camera_fresh = bool(
                camera_data is not None
                and int(camera_data.frame) != self._camera_frame_id)
            if camera_fresh:
                self._accept_camera(camera_data, capture_timestamp)
            radar_data = (self._radar_latest.get_at_or_before(
                frame_id, self.cfg.RADAR_OPTIONAL_TIMEOUT_S)
                          if self.cfg.ENABLE_RADAR else None)
        else:
            frame_id = int(self.world.tick())
            camera_data = retrieve_exact_frame(
                self._image_queue, frame_id, sensor_name="camera",
                stats=self.sync_stats)
            lidar_data = retrieve_exact_frame(
                self._lidar_queue, frame_id, sensor_name="lidar",
                stats=self.sync_stats)
            radar_data = (self._radar_exact.get(
                frame_id, self.cfg.RADAR_OPTIONAL_TIMEOUT_S)
                          if self.cfg.ENABLE_RADAR else None)
            camera_fresh = True
            self._accept_camera(camera_data, capture_timestamp)

        if self.sync_stats is not None and self._camera_frame_id is not None:
            self.sync_stats.record_pairing_skew(frame_id - int(self._camera_frame_id))
        if self._rgb is None:
            raise RuntimeError("camera frame unavailable after sensor warm-up")
        return SensorReadout(
            frame_id=frame_id,
            capture_timestamp=capture_timestamp,
            rgb=self._rgb,
            point_cloud=decode_lidar(lidar_data),
            radar_measurement=radar_data.measurement if radar_data is not None else None,
            lidar_timestamp=float(lidar_data.timestamp),
            camera_fresh=camera_fresh,
            camera_frame_id=self._camera_frame_id,
            camera_sensor_timestamp=self._camera_sensor_timestamp,
            camera_wall_timestamp=self._camera_wall_timestamp,
            radar_received_monotonic_s=(radar_data.received_monotonic_s
                                        if radar_data is not None else None),
        )

    def _accept_camera(self, camera_data: carla.Image,
                       wall_timestamp: float) -> None:
        self._rgb = decode_rgb(camera_data)
        self._camera_frame_id = int(camera_data.frame)
        self._camera_sensor_timestamp = float(camera_data.timestamp)
        self._camera_wall_timestamp = wall_timestamp

    def stop(self) -> None:
        """Stop callbacks before the orchestrator destroys the actors."""
        if self._stopped:
            return
        self._stopped = True
        for sensor in (self.camera, self.lidar, self.radar):
            try:
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
            except Exception as exc:
                print(f"[Sensors] Cảnh báo không thể dừng sensor: {exc}")
        if self.collision is not None:
            try:
                self.collision.stop()
            except Exception as exc:
                print(f"[Sensors] Cảnh báo không thể dừng collision sensor: {exc}")

    def destroy(self, client: carla.Client) -> None:
        """Stop callbacks and destroy every sensor owned by this rig.

        Keeping this operation here makes partial-spawn failures safe too:
        ``chinh.py`` does not need to know which sensor actors were created.
        """
        self.stop()
        actors = self.actors
        if actors:
            try:
                client.apply_batch([carla.command.DestroyActor(actor)
                                    for actor in actors])
            except Exception as exc:
                print(f"[Sensors] Cảnh báo destroy sensor actors: {exc}")
