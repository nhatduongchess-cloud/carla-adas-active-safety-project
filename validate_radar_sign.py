"""Controlled CARLA check for radar range and closing-speed sign.

The generated PASS token is required before radial velocity is allowed into
tracker TTC. Radar range remains available without the token.
"""

import argparse
import json
import math
import os
from pathlib import Path
import queue
import statistics
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (CURRENT_DIR, os.path.join(CURRENT_DIR, "modules")):
    if path not in sys.path:
        sys.path.insert(0, path)

import carla
import config as cfg
from ego_control import spawn_ego_safe
from sensor_setup import configure_radar_blueprint
from sensor_sync import OptionalFrameReader, SensorSyncStats


def speed_along(transform, signed_speed):
    forward = transform.get_forward_vector()
    return carla.Vector3D(forward.x * signed_speed, forward.y * signed_speed,
                          forward.z * signed_speed)


def ray_actor_intersection(actor, sensor_transform, azimuth, altitude):
    sensor_matrix = np.asarray(sensor_transform.get_matrix(), dtype=float)
    ray_sensor = np.array([math.cos(altitude) * math.cos(azimuth),
                           math.cos(altitude) * math.sin(azimuth),
                           math.sin(altitude)], dtype=float)
    ray_world = sensor_matrix[:3, :3] @ ray_sensor
    origin_world = sensor_matrix[:3, 3]
    inverse = np.asarray(actor.get_transform().get_inverse_matrix(), dtype=float)
    origin = (inverse @ np.append(origin_world, 1.0))[:3]
    direction = inverse[:3, :3] @ ray_world
    box = actor.bounding_box
    lower = np.array([box.location.x - box.extent.x,
                      box.location.y - box.extent.y,
                      box.location.z - box.extent.z])
    upper = np.array([box.location.x + box.extent.x,
                      box.location.y + box.extent.y,
                      box.location.z + box.extent.z])
    near, far = -math.inf, math.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-8:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return None
            continue
        a = (lower[axis] - origin[axis]) / direction[axis]
        b = (upper[axis] - origin[axis]) / direction[axis]
        near, far = max(near, min(a, b)), min(far, max(a, b))
        if far < near:
            return None
    if far < max(0.0, near):
        return None
    return max(0.0, near), ray_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=cfg.RADAR_SIGN_VALIDATION_PATH)
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.FIXED_DELTA
    world.apply_settings(settings)
    actors, radar = [], None
    try:
        blueprints = world.get_blueprint_library()
        spawns = world.get_map().get_spawn_points()
        ego, _ = spawn_ego_safe(world, blueprints.find("vehicle.tesla.model3"),
                                spawns, preferred_index=args.spawn)
        actors.append(ego)
        ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        ego_transform = ego.get_transform()
        ego_waypoint = world.get_map().get_waypoint(ego_transform.location)
        next_waypoints = ego_waypoint.next(22.0) if ego_waypoint is not None else []
        if not next_waypoints:
            raise RuntimeError("spawn point has no waypoint 22 m ahead")
        target_transform = next_waypoints[0].transform
        target_transform.location.z += 0.3
        candidates = blueprints.filter("vehicle.*")
        target = None
        for blueprint in candidates:
            if blueprint.id == "vehicle.tesla.model3":
                continue
            target = world.try_spawn_actor(blueprint, target_transform)
            if target is not None:
                break
        if target is None:
            raise RuntimeError("cannot spawn controlled radar target 22 m ahead")
        actors.append(target)
        radar_bp = configure_radar_blueprint(blueprints.find("sensor.other.radar"), cfg)
        radar = world.spawn_actor(radar_bp, carla.Transform(
            carla.Location(x=cfg.RADAR_X, y=cfg.RADAR_Y, z=cfg.RADAR_Z)), attach_to=ego)
        actors.append(radar)
        radar_queue, stats = queue.Queue(), SensorSyncStats()
        radar.listen(radar_queue.put)
        reader = OptionalFrameReader(radar_queue, "radar-calibration", stats)
        for _ in range(8):
            reader.get(world.tick(), 0.1)

        phases = [("stationary", 0.0, 0.0),
                  ("approaching", -5.0, 5.0),
                  ("departing", 5.0, -5.0)]
        results = []
        for name, target_speed, expected_closing in phases:
            target.set_target_velocity(speed_along(ego_transform, target_speed))
            velocities, expected_velocities, range_errors = [], [], []
            raw_points, nearest_candidates, observed_ranges = 0, [], []
            for _ in range(max(5, args.ticks)):
                target.set_target_velocity(speed_along(ego_transform, target_speed))
                frame_id = world.tick()
                measurement = reader.get(frame_id, 0.1)
                if measurement is None:
                    continue
                rows = np.frombuffer(measurement.raw_data, np.float32).reshape((-1, 4))
                raw_points += len(rows)
                observed_ranges.extend(float(row[3]) for row in rows)
                candidates_for_target = []
                for row in rows:
                    hit = ray_actor_intersection(
                        target, radar.get_transform(), float(row[1]), float(row[2]))
                    if hit is None:
                        continue
                    expected_depth, ray_world = hit
                    if abs(float(row[3]) - expected_depth) <= 1.5:
                        candidates_for_target.append((row, expected_depth, ray_world))
                if not candidates_for_target:
                    continue
                chosen, expected_depth, ray_world = min(
                    candidates_for_target,
                    key=lambda item: abs(float(item[0][3]) - item[1]))
                nearest_candidates.append((abs(float(chosen[3]) - expected_depth),
                                           float(chosen[1]), float(chosen[3]),
                                           float(chosen[0])))
                velocities.append(cfg.RADAR_VELOCITY_SIGN * float(chosen[0]))
                target_velocity = target.get_velocity()
                ego_velocity = ego.get_velocity()
                relative = np.array([target_velocity.x - ego_velocity.x,
                                     target_velocity.y - ego_velocity.y,
                                     target_velocity.z - ego_velocity.z])
                expected_velocities.append(float(-relative @ ray_world))
                range_errors.append(abs(float(chosen[3]) - expected_depth))
            measured = statistics.median(velocities) if velocities else None
            measured_gt = (statistics.median(expected_velocities)
                           if expected_velocities else None)
            range_mae = statistics.mean(range_errors) if range_errors else None
            results.append({"phase": name, "expected_closing_ms": expected_closing,
                            "ground_truth_closing_ms": measured_gt,
                            "measured_closing_ms": measured, "range_mae_m": range_mae,
                            "samples": len(velocities), "raw_points": raw_points,
                            "observed_range_minmax_m": ([min(observed_ranges), max(observed_ranges)]
                                                         if observed_ranges else None),
                            "nearest_raw_sample": (min(nearest_candidates)
                                                   if nearest_candidates else None)})

        valid = all(result["samples"] >= 5 and result["measured_closing_ms"] is not None
                    for result in results)
        if valid:
            speed_mae = statistics.mean(abs(result["measured_closing_ms"]
                                             - result["ground_truth_closing_ms"])
                                        for result in results)
            range_mae = statistics.mean(result["range_mae_m"] for result in results)
            signs_ok = (abs(results[0]["measured_closing_ms"]) <= 1.0
                        and results[1]["measured_closing_ms"] > 0.0
                        and results[2]["measured_closing_ms"] < 0.0)
        else:
            speed_mae, range_mae, signs_ok = None, None, False
        report = {"pass": bool(valid and signs_ok and speed_mae <= 1.0 and range_mae <= 0.5),
                  "velocity_sign": cfg.RADAR_VELOCITY_SIGN,
                  "closing_speed_mae_ms": speed_mae, "range_mae_m": range_mae,
                  "criteria": {"closing_speed_mae_ms": "<= 1.0",
                               "range_mae_m": "<= 0.5", "phase_signs": "0,+,-"},
                  "phases": results, "sync": stats.summary()}
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["pass"] else 1)
    finally:
        if radar is not None and radar.is_alive:
            radar.stop()
        if actors:
            client.apply_batch([carla.command.DestroyActor(actor) for actor in actors])
        world.apply_settings(original)


if __name__ == "__main__":
    main()
