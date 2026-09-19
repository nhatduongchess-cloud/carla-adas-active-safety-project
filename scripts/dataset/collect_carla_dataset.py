"""Collect synchronized CARLA perception data into an out-of-Git dataset.

The command performs no downloads. Use ``--dry-run`` to inspect the resolved
plan before starting a potentially long 20,000-frame collection.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from modules.dataset_labels import (POLICY, project_box, require_matching_policy,
                                   semantic_ids, visible_annotation)
from modules.sensor_sync import put_latest

# Windows consoles may default to cp1252. A logging character must never abort
# collection or, more importantly, skip restoration of CARLA synchronous mode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


TAXONOMY = ["Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck",
            "TrafficLight", "StopSign"]
TARGET_VEHICLE_BASE_TYPES = {
    "Car": "car",
    "Bicycle": "bicycle",
    "Motorcycle": "motorcycle",
    "Bus": "bus",
    "Truck": "truck",
}
TARGETABLE_CLASSES = set(TARGET_VEHICLE_BASE_TYPES) | {"TrafficLight", "StopSign"}


def parse_target_classes(value):
    """Parse a stable, de-duplicated taxonomy subset from the CLI."""
    canonical = {name.lower(): name for name in TAXONOMY}
    result = []
    for raw_name in str(value or "").split(","):
        raw_name = raw_name.strip()
        if not raw_name:
            continue
        class_name = canonical.get(raw_name.lower())
        if class_name is None:
            raise ValueError(
                f"unknown targeted class {raw_name!r}; expected one of {TAXONOMY}")
        if class_name not in TARGETABLE_CLASSES:
            raise ValueError(
                f"targeted spawning is not supported for {class_name}; "
                f"supported={sorted(TARGETABLE_CLASSES)}")
        if class_name not in result:
            result.append(class_name)
    return result


def traffic_light_spawn_score(spawn_transform, light_location):
    """Prefer a spawn 8-65 m before a traffic light and facing toward it."""
    origin = spawn_transform.location
    forward = spawn_transform.get_forward_vector()
    dx = float(light_location.x - origin.x)
    dy = float(light_location.y - origin.y)
    longitudinal = dx * float(forward.x) + dy * float(forward.y)
    lateral = abs(-dx * float(forward.y) + dy * float(forward.x))
    if not 8.0 <= longitudinal <= 65.0 or lateral > 14.0:
        return None
    return lateral + 0.15 * abs(longitudinal - 30.0)


def prioritize_traffic_light_spawns(world, spawn_points):
    """Return spawn points with valid traffic-light approaches first."""
    lights = list(world.get_actors().filter("traffic.traffic_light*"))
    scored = []
    for index, transform in enumerate(spawn_points):
        candidates = [traffic_light_spawn_score(transform, light.get_location())
                      for light in lights]
        candidates = [score for score in candidates if score is not None]
        if candidates:
            scored.append((min(candidates), index, transform))
    scored.sort(key=lambda row: (row[0], row[1]))
    preferred_indices = {index for _, index, _ in scored}
    ordered = [transform for _, _, transform in scored]
    ordered.extend(transform for index, transform in enumerate(spawn_points)
                   if index not in preferred_indices)
    return ordered, len(scored)


def prioritize_static_stop_sign_spawns(stop_signs, spawn_points):
    """Return spawn points that approach a visible static stop sign first.

    ``traffic.stop`` actors are invisible trigger volumes and are deliberately
    not accepted here.  The caller supplies only visible map environment
    objects from ``CityObjectLabel.TrafficSigns``.
    """
    scored = []
    for index, transform in enumerate(spawn_points):
        candidates = [
            traffic_light_spawn_score(transform, sign.transform.location)
            for sign in stop_signs
        ]
        candidates = [score for score in candidates if score is not None]
        if candidates:
            scored.append((min(candidates), index, transform))
    scored.sort(key=lambda row: (row[0], row[1]))
    preferred_indices = {index for _, index, _ in scored}
    ordered = [transform for _, _, transform in scored]
    ordered.extend(transform for index, transform in enumerate(spawn_points)
                   if index not in preferred_indices)
    return ordered, len(scored)


def canonical_town(town):
    """Return a safe CARLA map basename while preserving the optional suffix."""
    name = str(town).replace("\\", "/").rsplit("/", 1)[-1]
    if not re.fullmatch(r"Town(?:0[1-9]|10HD)(?:_Opt)?", name, re.IGNORECASE):
        raise ValueError(f"unsupported or unsafe CARLA town name: {town!r}")
    stem, suffix = (name[:-4], "_Opt") if name.lower().endswith("_opt") else (name, "")
    stem = "Town10HD" if stem.lower() == "town10hd" else f"Town{int(stem[4:]):02d}"
    return stem + suffix


def town_split(town):
    town = canonical_town(town).lower().removesuffix("_opt")
    if town == "town10hd":
        return "test"
    # The installed CARLA package exposes Town01-05 and Town10HD. Keep Town05
    # fully outside training so consecutive frames cannot leak across splits.
    if town in {"town05", "town06", "town07", "town08", "town09"}:
        return "validation"
    return "train"


def actor_class(type_id, attributes=None):
    name = str(type_id).lower()
    base_type = str((attributes or {}).get("base_type", "")).lower()
    if name.startswith("walker.pedestrian"):
        return "Person"
    if base_type == "bicycle":
        return "Bicycle"
    if base_type == "motorcycle":
        return "Motorcycle"
    if base_type == "bus":
        return "Bus"
    if base_type == "truck":
        return "Truck"
    if any(token in name for token in ("bike", "bicycle", "diamondback", "omafiets")):
        return "Bicycle"
    if "motorcycle" in name or any(token in name for token in ("harley", "kawasaki", "vespa")):
        return "Motorcycle"
    if "bus" in name:
        return "Bus"
    if any(token in name for token in ("truck", "carlacola", "firetruck", "ambulance")):
        return "Truck"
    if name.startswith("vehicle."):
        return "Car"
    if "traffic_light" in name:
        return "TrafficLight"
    if "stop" in name and "sign" in name:
        return "StopSign"
    return None


def sample_stem(seed, index):
    if int(index) < 0:
        raise ValueError("sample index must be non-negative")
    return f"seed{int(seed):010d}_{int(index):06d}"


def existing_sample_ids(path):
    ids = set()
    if not path.is_file():
        return ids
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not sample_id:
                raise ValueError(
                    f"legacy/non-resumable annotation at {path}:{line_number}")
            if sample_id in ids:
                raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
            ids.add(sample_id)
    return ids


def available_free_gb(path):
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def _two_wheel_blueprints(library):
    return [bp for bp in library.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 2]


def blueprint_base_type(blueprint):
    return (blueprint.get_attribute("base_type").as_str().lower()
            if blueprint.has_attribute("base_type") else "")


def _nearby_vehicle_transforms(world, reference_vehicle):
    import carla

    if reference_vehicle is None:
        return []
    carla_map = world.get_map()
    start = carla_map.get_waypoint(
        reference_vehicle.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    transforms = []
    for distance_m in (14.0, 20.0, 27.0, 35.0, 44.0, 54.0, 66.0):
        ahead = start.next(distance_m)
        if not ahead:
            continue
        center = ahead[0]
        candidates = [center, center.get_left_lane(), center.get_right_lane()]
        for waypoint in candidates:
            if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
                continue
            transform = waypoint.transform
            transform.location.z += 0.25
            transforms.append(transform)
    return transforms


def _spawn_balanced_vehicles(world, traffic_manager, desired_types, rng, role_name,
                             reference_vehicle=None, on_spawn=None):
    if not desired_types:
        return [], []
    library = world.get_blueprint_library()
    grouped = {}
    for blueprint in library.filter("vehicle.*"):
        grouped.setdefault(blueprint_base_type(blueprint), []).append(blueprint)
    nearby = _nearby_vehicle_transforms(world, reference_vehicle)
    spawn_points = nearby + list(world.get_map().get_spawn_points())
    # Preserve near-to-far priority; randomize only the global fallback points.
    fallback_start = len(nearby)
    fallback = spawn_points[fallback_start:]
    rng.shuffle(fallback)
    spawn_points[fallback_start:] = fallback
    remaining, spawned = list(desired_types), []
    for transform in spawn_points:
        if not remaining:
            break
        base_type = remaining[0]
        candidates = grouped.get(base_type, [])
        if not candidates:
            print(f"[collect] no blueprint for requested base_type={base_type}")
            remaining.pop(0)
            continue
        blueprint = rng.choice(candidates)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        spawned.append(actor)
        if on_spawn is not None:
            on_spawn(actor)
        actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.auto_lane_change(actor, False)
        traffic_manager.vehicle_percentage_speed_difference(actor, rng.randint(5, 30))
        remaining.pop(0)
    return spawned, remaining


def spawn_two_wheelers(world, traffic_manager, count, rng, reference_vehicle=None,
                       on_spawn=None):
    desired = ["bicycle" if index % 2 == 0 else "motorcycle"
               for index in range(int(count))]
    spawned, remaining = _spawn_balanced_vehicles(
        world, traffic_manager, desired, rng, "dataset_two_wheeler",
        reference_vehicle=reference_vehicle, on_spawn=on_spawn)
    print(f"[collect] spawned {len(spawned)}/{count} balanced two-wheelers"
          f" (unfilled={remaining})")
    return spawned


def spawn_heavy_vehicles(world, traffic_manager, count, rng, reference_vehicle=None,
                         on_spawn=None):
    desired = ["bus" if index % 2 == 0 else "truck"
               for index in range(int(count))]
    spawned, remaining = _spawn_balanced_vehicles(
        world, traffic_manager, desired, rng, "dataset_heavy_vehicle",
        reference_vehicle=reference_vehicle, on_spawn=on_spawn)
    print(f"[collect] spawned {len(spawned)}/{count} balanced heavy vehicles"
          f" (unfilled={remaining})")
    return spawned


def spawn_target_vehicles(world, traffic_manager, class_names, instances_per_class,
                          rng, reference_vehicle=None, on_spawn=None):
    desired = []
    for _ in range(int(instances_per_class)):
        desired.extend(TARGET_VEHICLE_BASE_TYPES[class_name]
                       for class_name in class_names
                       if class_name in TARGET_VEHICLE_BASE_TYPES)
    spawned, remaining = _spawn_balanced_vehicles(
        world, traffic_manager, desired, rng, "dataset_target",
        reference_vehicle=reference_vehicle, on_spawn=on_spawn)
    print(f"[collect] spawned {len(spawned)}/{len(desired)} targeted vehicles"
          f" (unfilled={remaining})")
    return spawned


def spawn_walkers(world, count, rng, reference_vehicle=None, on_spawn=None):
    if int(count) == 0:
        return [], []
    import carla

    library = world.get_blueprint_library()
    walker_blueprints = list(library.filter("walker.pedestrian.*"))
    walkers = []
    nearby = []
    if reference_vehicle is not None:
        waypoint = world.get_map().get_waypoint(reference_vehicle.get_location())
        for index, distance_m in enumerate(
                (10.0, 16.0, 23.0, 31.0, 40.0, 50.0, 61.0, 73.0)):
            ahead = waypoint.next(distance_m)
            if not ahead:
                continue
            lane_waypoint = ahead[0]
            right = lane_waypoint.transform.get_right_vector()
            side = -1.0 if index % 2 == 0 else 1.0
            offset = side * (lane_waypoint.lane_width * 0.5 + 0.6)
            center = lane_waypoint.transform.location
            location = carla.Location(
                x=center.x + right.x * offset,
                y=center.y + right.y * offset,
                z=center.z + 0.5)
            direction = carla.Vector3D(-side * right.x, -side * right.y, 0.0)
            nearby.append((location, direction))
    for _ in range(int(count) * 4):
        if len(walkers) >= int(count):
            break
        if nearby:
            location, direction = nearby.pop(0)
        else:
            location = world.get_random_location_from_navigation()
            if location is None:
                continue
            location.z += 0.5
            angle = rng.uniform(-math.pi, math.pi)
            direction = carla.Vector3D(math.cos(angle), math.sin(angle), 0.0)
        blueprint = rng.choice(walker_blueprints)
        walker = world.try_spawn_actor(
            blueprint, carla.Transform(location))
        if walker is None:
            continue
        walkers.append(walker)
        if on_spawn is not None:
            on_spawn(walker)
        # controller.ai.walker can terminate the Python 3.12 client process on
        # this CARLA/Windows build. Direct WalkerControl keeps VRUs moving and
        # avoids another actor whose lifecycle could leave sync mode locked.
        control = carla.WalkerControl()
        control.direction = direction
        control.speed = rng.uniform(0.8, 1.8)
        control.jump = False
        walker.apply_control(control)
    print(f"[collect] spawned {len(walkers)}/{count} walkers")
    return walkers, []


def bgra(image):
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4)


def depth_m(image):
    array = bgra(image).astype(np.float32)
    normalized = (array[:, :, 2] + array[:, :, 1] * 256.0
                  + array[:, :, 0] * 65536.0) / 16777215.0
    return normalized * 1000.0


def project_carla_box(box, parent_transform, camera, width, height, fov):
    import carla
    box_transform = carla.Transform(box.location, box.rotation)
    matrix = np.asarray(box_transform.get_matrix())
    if parent_transform is not None:
        matrix = np.asarray(parent_transform.get_matrix()) @ matrix
    return project_box(matrix, [box.extent.x, box.extent.y, box.extent.z],
                       camera.get_transform().get_matrix(), width, height, fov)


def project_actor(actor, camera, width, height, fov):
    annotation = project_carla_box(actor.bounding_box, actor.get_transform(),
                                   camera, width, height, fov)
    if annotation is None:
        return None
    velocity = actor.get_velocity()
    annotation.update({"velocity_world_ms": [velocity.x, velocity.y, velocity.z],
                       "actor_id": int(actor.id), "type_id": actor.type_id})
    return annotation


def project_environment_object(obj, camera, width, height, fov):
    # Environment-object boxes are already in world coordinates.
    annotation = project_carla_box(obj.bounding_box, None, camera, width, height, fov)
    if annotation is None:
        return None
    annotation.update({"velocity_world_ms": [0.0, 0.0, 0.0],
                       "actor_id": int(obj.id), "type_id": obj.name})
    return annotation


def actor_camera_coordinates(actor, camera):
    inverse = np.asarray(camera.get_transform().get_inverse_matrix(), dtype=float)
    location = actor.get_location()
    local = inverse @ np.array([location.x, location.y, location.z, 1.0])
    return [round(float(value), 3) for value in local[:3]]


def annotation_rejection_reason(annotation, depth_values_m, max_distance_m=80.0):
    # Legacy policy retained only for auditing/sanitizing old data. V3 capture
    # uses visible_annotation with aligned semantic and oriented-box evidence.
    distance = float(annotation.get("distance_m", float("inf")))
    if not math.isfinite(distance) or distance <= 0.0 or distance > max_distance_m:
        return "outside_distance_range"
    if (annotation.get("class") == "StopSign"
            and str(annotation.get("type_id", "")).startswith("traffic.stop")):
        return "invisible_stop_trigger"
    box = annotation.get("bbox_xyxy", [])
    if len(box) != 4:
        return "invalid_bbox"
    height, width = depth_values_m.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(float(box[0])))))
    y1 = max(0, min(height - 1, int(math.floor(float(box[1])))))
    x2 = max(0, min(width, int(math.ceil(float(box[2])))))
    y2 = max(0, min(height, int(math.ceil(float(box[3])))))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return "invalid_bbox"
    crop = np.asarray(depth_values_m[y1:y2, x1:x2], dtype=np.float32)
    tolerance = max(3.0, distance * 0.15)
    matching = np.isfinite(crop) & (np.abs(crop - distance) <= tolerance)
    required = max(2, int(math.ceil(crop.size * 0.005)))
    if int(matching.sum()) < required:
        return "depth_occluded"
    return None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True,
                        help="dataset root outside Git (15-20 GB budget)")
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--town", default="Town02")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--npcs", type=int, default=15)
    parser.add_argument("--walkers", type=int, default=8)
    parser.add_argument("--two-wheelers", type=int, default=4)
    parser.add_argument("--heavy-vehicles", type=int, default=2)
    parser.add_argument("--targeted-classes", default="",
                        help="comma-separated taxonomy classes for targeted capture")
    parser.add_argument("--target-instances-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument(
        "--ego-control",
        choices=("custom", "stationary", "experimental-autopilot"),
        default="custom",
        help="explicit ego owner; autopilot remains an unqualified experiment")
    parser.add_argument("--target-speed-kmh", type=float, default=20.0)
    parser.add_argument("--radar-policy", choices=("required", "optional"),
                        default="required")
    parser.add_argument("--diagnose-visibility", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        canonical_town(args.town)
        parse_target_classes(args.targeted_classes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if min(args.npcs, args.walkers, args.two_wheelers, args.heavy_vehicles,
           args.target_instances_per_class) < 0:
        parser.error("actor counts must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if (not math.isfinite(args.min_free_gb) or args.min_free_gb < 0.0):
        parser.error("--min-free-gb must be finite and non-negative")
    if (not math.isfinite(args.target_speed_kmh)
            or not 0.0 < args.target_speed_kmh <= 100.0):
        parser.error("--target-speed-kmh must be finite and in (0, 100]")
    import config as cfg
    from modules.weather_config import load_weather_profiles
    profiles = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)
    if args.weather not in profiles or not re.fullmatch(r'[A-Za-z0-9_-]+', args.weather):
        parser.error(f'unknown/unsafe weather {args.weather!r}; available={sorted(profiles)}')
    return args


def enqueue_sensor(sensor_queue, data, drop_counts, sensor_name):
    """Bound callback memory and preserve an explicit overflow count."""
    if put_latest(sensor_queue, data):
        drop_counts[sensor_name] += 1


def persist_capture_files(root, stem, arrays, jpeg_quality, cv2_module):
    """Commit one sample's six modalities or remove that partial commit."""
    extensions = {
        "rgb": ".jpg", "semantic": ".png", "road_line": ".png",
        "depth": ".png", "lidar": ".npz", "radar": ".npz",
    }
    missing = set(extensions) - set(arrays)
    if missing:
        raise ValueError(f"missing capture arrays: {sorted(missing)}")
    finals = {name: Path(root) / name / f"{stem}{extension}"
              for name, extension in extensions.items()}
    occupied = [str(path) for path in finals.values() if path.exists()]
    if occupied:
        raise FileExistsError(f"capture output already exists: {occupied}")

    temporary, promoted = {}, []
    try:
        for name, final in finals.items():
            descriptor, filename = tempfile.mkstemp(
                prefix=f"{stem}.partial.", suffix=final.suffix,
                dir=str(final.parent))
            os.close(descriptor)
            temporary[name] = Path(filename)

        image_options = [cv2_module.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        if not cv2_module.imwrite(str(temporary["rgb"]), arrays["rgb"],
                                  image_options):
            raise IOError(f"failed to write RGB sample {stem}")
        for name in ("semantic", "road_line", "depth"):
            if not cv2_module.imwrite(str(temporary[name]), arrays[name]):
                raise IOError(f"failed to write {name} sample {stem}")
        np.savez_compressed(temporary["lidar"], points=arrays["lidar"])
        np.savez_compressed(temporary["radar"], points=arrays["radar"])
        empty = [str(path) for path in temporary.values()
                 if not path.is_file() or path.stat().st_size <= 0]
        if empty:
            raise IOError(f"empty capture output for {stem}: {empty}")
        for name in extensions:
            os.replace(temporary[name], finals[name])
            promoted.append(finals[name])
        return {name: str(path) for name, path in finals.items()}
    except BaseException:
        for path in list(temporary.values()) + promoted:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _public_snapshot(value, names):
    return {name: getattr(value, name) for name in names
            if hasattr(value, name)}


def settings_snapshot(settings):
    return _public_snapshot(settings, (
        "synchronous_mode", "no_rendering_mode", "fixed_delta_seconds",
        "substepping", "max_substep_delta_time", "max_substeps",
        "max_culling_distance", "deterministic_ragdolls",
        "tile_stream_distance", "actor_active_distance", "spectator_as_ego"))


def weather_snapshot(weather):
    return _public_snapshot(weather, (
        "cloudiness", "precipitation", "precipitation_deposits",
        "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
        "fog_density", "fog_distance", "fog_falloff", "wetness",
        "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm"))


def sensor_manifest(cfg, radar_policy):
    camera = {
        "pose_m_deg": [cfg.CAM_X, cfg.CAM_Y, cfg.CAM_Z,
                       cfg.CAM_ROLL, cfg.CAM_PITCH, cfg.CAM_YAW],
        "sensor_tick_s": cfg.CAMERA_SENSOR_TICK_S,
        "width": cfg.CAM_WIDTH, "height": cfg.CAM_HEIGHT,
        "fov_deg": cfg.CAM_FOV,
        "postprocess": bool(cfg.CAMERA_POSTPROCESS),
    }
    return {
        "rgb": {**camera, "blueprint": "sensor.camera.rgb",
                "encoding": "jpeg_bgr", "required": True},
        "semantic": {**camera,
                     "blueprint": "sensor.camera.semantic_segmentation",
                     "encoding": "png_uint8_city_object_tag", "required": True},
        "depth": {**camera, "blueprint": "sensor.camera.depth",
                  "encoding": "png_uint16_centimetres", "required": True},
        "lidar": {
            "blueprint": "sensor.lidar.ray_cast", "required": True,
            "pose_m_deg": [cfg.LIDAR_X, cfg.LIDAR_Y, cfg.LIDAR_Z,
                           cfg.LIDAR_ROLL, cfg.LIDAR_PITCH, cfg.LIDAR_YAW],
            "sensor_tick_s": cfg.LIDAR_SENSOR_TICK_S,
            "range_m": cfg.LIDAR_RANGE_M, "channels": cfg.LIDAR_CHANNELS,
            "points_per_second": cfg.LIDAR_POINTS_PER_SECOND,
            "rotation_frequency_hz": cfg.FPS,
            "encoding": "npz_float32_x_y_z_intensity",
        },
        "radar": {
            "blueprint": "sensor.other.radar",
            "required": radar_policy == "required", "policy": radar_policy,
            "pose_m_deg": [cfg.RADAR_X, cfg.RADAR_Y, cfg.RADAR_Z,
                           cfg.RADAR_ROLL, cfg.RADAR_PITCH, cfg.RADAR_YAW],
            "sensor_tick_s": cfg.RADAR_SENSOR_TICK_S,
            "range_m": cfg.RADAR_RANGE_M,
            "horizontal_fov_deg": cfg.RADAR_HORIZONTAL_FOV,
            "vertical_fov_deg": cfg.RADAR_VERTICAL_FOV,
            "points_per_second": cfg.RADAR_POINTS_PER_SECOND,
            "encoding": "npz_float32_velocity_azimuth_altitude_depth",
        },
        "road_line": {"derived_from": "semantic", "required": True,
                      "encoding": "png_uint8_binary"},
    }


def canonical_json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_contract(plan, cfg):
    """Return the capture-affecting contract used to authorize exact resume."""
    from modules.weather_config import load_weather_profiles, weather_kwargs
    weather = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)[plan['weather']]
    return {
        "schema_version": 4,
        "frames": int(plan["frames"]),
        "town": plan["town"],
        "weather": plan["weather"],
        "weather_parameters": weather_kwargs(weather),
        "split": plan["split"],
        "seed": int(plan["seed"]),
        "label_policy": plan["label_policy"],
        "actors": {
            name: int(plan[name]) for name in (
                "npcs", "walkers", "two_wheelers", "heavy_vehicles")
        },
        "targeted_classes": list(plan["targeted_classes"]),
        "target_instances_per_class": int(plan["target_instances_per_class"]),
        "jpeg_quality": int(plan["jpeg_quality"]),
        "ego_control": plan["ego_control"],
        "target_speed_kmh": float(plan["target_speed_kmh"]),
        "radar_policy": plan["radar_policy"],
        "fps": int(cfg.FPS),
        "sensors": sensor_manifest(cfg, plan["radar_policy"]),
    }


def expected_sample_paths(root, stem):
    root = Path(root)
    return {
        "rgb": root / "rgb" / f"{stem}.jpg",
        "semantic": root / "semantic" / f"{stem}.png",
        "road_line": root / "road_line" / f"{stem}.png",
        "depth": root / "depth" / f"{stem}.png",
        "lidar": root / "lidar" / f"{stem}.npz",
        "radar": root / "radar" / f"{stem}.npz",
    }


def validate_resume_state(labels_path, split_root, job_hash):
    """Fail closed on a mixed contract, a partial sample, or an orphan file."""
    from scripts.dataset.validate_dataset import MODALITIES, _read_declared_file

    labels_path, split_root = Path(labels_path), Path(split_root)
    records, ids = [], set()
    if labels_path.is_file():
        with open(labels_path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                stem = record.get("sample_id")
                if not stem or Path(stem).name != stem:
                    raise ValueError(
                        f"unsafe/non-resumable sample at {labels_path}:{line_number}")
                if stem in ids:
                    raise ValueError(f"duplicate sample_id {stem!r} in {labels_path}")
                if record.get("job_hash") != job_hash:
                    raise ValueError(
                        f"resume contract mismatch at {labels_path}:{line_number}")
                paths = expected_sample_paths(split_root, stem)
                missing = [str(path) for path in paths.values()
                           if not path.is_file() or path.stat().st_size <= 0]
                if missing:
                    raise ValueError(
                        f"partial resumable sample {stem!r}; missing/empty={missing}")
                for name, path in paths.items():
                    try:
                        _read_declared_file(path, MODALITIES[name][1])
                    except Exception as exc:
                        raise ValueError(f'unreadable resumable modality: {path}') from exc
                ids.add(stem)
                records.append(record)

    modality_stems = set()
    for name, extension in (("rgb", ".jpg"), ("semantic", ".png"),
                            ("road_line", ".png"), ("depth", ".png"),
                            ("lidar", ".npz"), ("radar", ".npz")):
        folder = split_root / name
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() == extension:
                modality_stems.add(path.stem)
            elif path.is_file():
                raise ValueError(f"unexpected partial/output file blocks resume: {path}")
    orphaned = sorted(modality_stems - ids)
    if orphaned:
        raise ValueError(f"orphan modality files block resume: {orphaned[:10]}")
    return records


def transform_metadata(transform):
    location, rotation = transform.location, transform.rotation
    return {
        "location_m": [float(location.x), float(location.y), float(location.z)],
        "rotation_deg": [float(rotation.roll), float(rotation.pitch),
                         float(rotation.yaw)],
        "matrix": transform.get_matrix(),
    }


def append_sample_record(labels, record, committed_paths):
    """Append and fsync one label, rolling back its files on any failure."""
    offset = labels.tell()
    try:
        labels.write(json.dumps(record, ensure_ascii=False) + "\n")
        labels.flush()
        os.fsync(labels.fileno())
    except BaseException as primary:
        rollback_errors = []
        try:
            labels.seek(offset)
            labels.truncate()
            labels.flush()
            os.fsync(labels.fileno())
        except Exception as exc:
            rollback_errors.append(f"label rollback: {exc}")
        for path in committed_paths.values():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"file rollback {path}: {exc}")
        if rollback_errors and hasattr(primary, "add_note"):
            primary.add_note("; ".join(rollback_errors))
        raise


def _snapshots_match(actual, expected, tolerance=1e-6):
    if set(actual) != set(expected):
        return False
    for name, wanted in expected.items():
        observed = actual[name]
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            if observed is None or wanted is None:
                if observed != wanted:
                    return False
            elif (not math.isfinite(float(observed))
                  or not math.isfinite(float(wanted))
                  or abs(float(observed) - float(wanted)) > tolerance):
                return False
        elif observed != wanted:
            return False
    return True


def cleanup_capture(*, client, world, session, traffic_manager,
                    traffic_manager_touched, actors, sensors,
                    original_settings, original_weather,
                    world_settings_touched, weather_touched, carla_module,
                    timeout_s=10.0):
    """Attempt every cleanup stage and return independently checked evidence."""
    report = {"verified": False, "errors": []}

    def failure(stage, exc):
        report["errors"].append({"stage": stage, "error": str(exc)})

    stopped, stop_failed = [], []
    for name, sensor in list(sensors.items()):
        try:
            if getattr(sensor, "is_alive", True):
                sensor.stop()
            listening = getattr(sensor, "is_listening", False)
            listening = listening() if callable(listening) else listening
            if bool(listening):
                raise RuntimeError("sensor callback is still listening")
            stopped.append(name)
        except Exception as exc:
            stop_failed.append(name)
            failure(f"sensor_stop:{name}", exc)
    report["sensor_stop"] = {
        "attempted": list(sensors), "stopped": stopped,
        "failed": stop_failed, "verified": not stop_failed,
    }

    unique = []
    seen_ids = set()
    for actor in reversed(list(actors)):
        actor_id = int(actor.id)
        if actor_id not in seen_ids:
            seen_ids.add(actor_id)
            unique.append(actor)
    destroy_errors = []
    if unique:
        try:
            commands = [carla_module.command.DestroyActor(int(actor.id))
                        for actor in unique]
            responses = client.apply_batch_sync(
                commands, bool(session is not None and session.active))
            if len(responses) != len(commands):
                raise RuntimeError(
                    f"destroy response count {len(responses)} != {len(commands)}")
            for actor, response in zip(unique, responses):
                if response.has_error():
                    error = {"actor_id": int(actor.id),
                             "error": str(response.error)}
                    destroy_errors.append(error)
                    failure("actor_destroy_response", RuntimeError(str(error)))
        except Exception as exc:
            failure("actor_destroy", exc)
            destroy_errors.append({"actor_id": None, "error": str(exc)})
    report["actor_destroy"] = {
        "requested_ids": [int(actor.id) for actor in unique],
        "response_errors": destroy_errors,
        "verified": not destroy_errors,
    }

    tm_error = None
    if traffic_manager_touched:
        if traffic_manager is None:
            tm_error = "Traffic Manager handle is missing"
            failure("traffic_manager_async", RuntimeError(tm_error))
        else:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception as exc:
                tm_error = str(exc)
                failure("traffic_manager_async", exc)
    report["traffic_manager_async"] = {
        "required": bool(traffic_manager_touched),
        "acknowledged": not traffic_manager_touched or tm_error is None,
        "error": tm_error,
    }

    weather_error = None
    if weather_touched:
        if world is None or original_weather is None:
            weather_error = "world/original weather is missing"
            failure("weather_restore", RuntimeError(weather_error))
        else:
            try:
                world.set_weather(original_weather)
            except Exception as exc:
                weather_error = str(exc)
                failure("weather_restore", exc)
    report["weather_restore"] = {
        "required": bool(weather_touched),
        "acknowledged": not weather_touched or weather_error is None,
        "error": weather_error,
    }

    settings_error = None
    if world_settings_touched:
        if world is None or original_settings is None:
            settings_error = "world/original settings are missing"
            failure("settings_restore", RuntimeError(settings_error))
        else:
            try:
                if session is not None and session.active:
                    session.close()
                else:
                    world.apply_settings(original_settings, float(timeout_s))
            except Exception as exc:
                settings_error = str(exc)
                failure("settings_restore", exc)
    report["settings_restore"] = {
        "required": bool(world_settings_touched),
        "acknowledged": not world_settings_touched or settings_error is None,
        "error": settings_error,
    }

    async_progress = {"required": bool(world_settings_touched),
                      "verified": not world_settings_touched}
    if world_settings_touched:
        if world is None:
            exc = RuntimeError("world is missing for asynchronous progress check")
            async_progress["error"] = str(exc)
            failure("async_progress", exc)
        else:
            try:
                before = world.get_snapshot()
                after = world.wait_for_tick(float(timeout_s))
                async_progress.update({
                    "before_frame": int(before.frame),
                    "after_frame": int(after.frame),
                    "before_time_s": float(before.timestamp.elapsed_seconds),
                    "after_time_s": float(after.timestamp.elapsed_seconds),
                })
                async_progress["verified"] = (
                    int(after.frame) > int(before.frame)
                    and float(after.timestamp.elapsed_seconds)
                    > float(before.timestamp.elapsed_seconds))
                if not async_progress["verified"]:
                    raise RuntimeError("world did not resume asynchronous progress")
            except Exception as exc:
                async_progress["verified"] = False
                async_progress["error"] = str(exc)
                failure("async_progress", exc)
    report["async_progress"] = async_progress

    state_check = {"settings": not world_settings_touched,
                   "weather": not weather_touched,
                   "surviving_actor_ids": [], "verified": False,
                   "errors": []}
    if world_settings_touched:
        try:
            if world is None:
                raise RuntimeError("world is missing for settings verification")
            state_check["settings"] = _snapshots_match(
                settings_snapshot(world.get_settings()),
                settings_snapshot(original_settings))
            if not state_check["settings"]:
                raise RuntimeError("world settings do not match the original snapshot")
        except Exception as exc:
            state_check["settings"] = False
            state_check["errors"].append({"stage": "settings", "error": str(exc)})
            failure("settings_verification", exc)
    if weather_touched:
        try:
            if world is None:
                raise RuntimeError("world is missing for weather verification")
            state_check["weather"] = _snapshots_match(
                weather_snapshot(world.get_weather()),
                weather_snapshot(original_weather))
            if not state_check["weather"]:
                raise RuntimeError("weather does not match the original snapshot")
        except Exception as exc:
            state_check["weather"] = False
            state_check["errors"].append({"stage": "weather", "error": str(exc)})
            failure("weather_verification", exc)
    if seen_ids:
        try:
            if world is None:
                raise RuntimeError("world is missing for actor verification")
            state_check["surviving_actor_ids"] = [
                int(actor.id) for actor in world.get_actors(list(seen_ids))]
            if state_check["surviving_actor_ids"]:
                raise RuntimeError(
                    f"owned actors survived: {state_check['surviving_actor_ids']}")
        except Exception as exc:
            state_check["errors"].append({"stage": "actors", "error": str(exc)})
            failure("actor_verification", exc)
    state_check["verified"] = (
        state_check["settings"] and state_check["weather"]
        and not state_check["surviving_actor_ids"]
        and not state_check["errors"])
    report["state_verification"] = state_check

    report["verified"] = (
        report["sensor_stop"]["verified"]
        and report["actor_destroy"]["verified"]
        and report["traffic_manager_async"]["acknowledged"]
        and report["weather_restore"]["acknowledged"]
        and report["settings_restore"]["acknowledged"]
        and report["async_progress"]["verified"]
        and report["state_verification"]["verified"])
    return report


class _ConfigOverride:
    def __init__(self, base, **values):
        self._base = base
        self._values = values

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        return getattr(self._base, name)


def main(argv=None):
    args = parse_args(argv)
    import config as cfg

    root = Path(args.output).resolve()
    town_name = canonical_town(args.town)
    targeted_classes = parse_target_classes(args.targeted_classes)
    free_gb = available_free_gb(root)
    plan = {
        "output": str(root), "frames": args.frames, "town": town_name,
        "label_policy": POLICY, "split": town_split(town_name),
        "weather": args.weather, "npcs": args.npcs,
        "walkers": args.walkers, "two_wheelers": args.two_wheelers,
        "heavy_vehicles": args.heavy_vehicles, "seed": args.seed,
        "targeted_classes": targeted_classes,
        "target_instances_per_class": args.target_instances_per_class,
        "jpeg_quality": args.jpeg_quality,
        "free_gb_before": round(free_gb, 2), "taxonomy": TAXONOMY,
        "estimated_budget_gb": "15-20 for full multi-town collection",
        "ego_control": args.ego_control,
        "target_speed_kmh": args.target_speed_kmh,
        "radar_policy": args.radar_policy,
    }
    contract = capture_contract(plan, cfg)
    job_hash = canonical_json_hash(contract)
    plan.update({"job_hash": job_hash, "capture_contract": contract})
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    if free_gb < args.min_free_gb:
        raise SystemExit(
            f"only {free_gb:.2f} GiB free; require {args.min_free_gb:.2f} GiB")

    import cv2
    import carla
    from modules.active_safety import SafetyDecision
    from modules.ego_control import EgoController, spawn_ego_safe
    from modules.sensor_setup import (configure_camera_blueprint,
                                      configure_lidar_blueprint,
                                      configure_radar_blueprint)
    from modules.sensor_sync import (OptionalFrameReader, SensorFrameError,
                                     SensorSyncStats, retrieve_exact_frame,
                                     warmup_sensor_streams)
    from modules.simulation_guard import (SynchronousWorldSession,
                                          select_probe_world)
    from modules.traffic_spawner import TrafficSpawner
    from modules.weather_config import load_weather_profiles, weather_kwargs

    profiles = load_weather_profiles(cfg.WEATHER_CONFIG_PATH)
    profile = profiles.get(args.weather)
    if profile is None:
        raise ValueError(
            f"unknown weather {args.weather}; available={sorted(profiles)}")

    split_root = root / plan["split"] / town_name / args.weather
    labels_path = split_root / "annotations.jsonl"
    require_matching_policy(labels_path)
    for folder in ("rgb", "road_line", "semantic", "depth", "lidar", "radar"):
        (split_root / folder).mkdir(parents=True, exist_ok=True)
    existing_records = validate_resume_state(labels_path, split_root, job_hash)
    existing_ids = {record["sample_id"] for record in existing_records}
    expected_ids = {sample_stem(args.seed, index) for index in range(args.frames)}
    unexpected = sorted(existing_ids - expected_ids)
    if unexpected:
        raise ValueError(f"resume contains samples outside this job: {unexpected[:10]}")
    remaining = [index for index in range(args.frames)
                 if sample_stem(args.seed, index) not in existing_ids]
    canonical_manifest_path = split_root / "manifest.json"
    if not remaining:
        if not canonical_manifest_path.is_file():
            raise RuntimeError("all sample files exist but completed manifest is missing")
        completed = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        if (completed.get("status") != "completed"
                or completed.get("job_hash") != job_hash
                or completed.get("total_job_samples") != args.frames
                or not completed.get("cleanup", {}).get("verified", False)):
            raise RuntimeError("existing completed manifest does not authorize resume skip")
        print(f"[collect] verified completed job: {len(existing_ids)} samples")
        return 0
    if canonical_manifest_path.is_file():
        stale = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        if stale.get("status") == "completed":
            raise RuntimeError("completed manifest conflicts with incomplete sample set")
        if stale.get("job_hash") not in (None, job_hash):
            raise RuntimeError("existing manifest belongs to another capture contract")
    print(f"[collect] resume: existing={len(existing_ids)}, remaining={len(remaining)}")

    attempt_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}"
    attempt_path = split_root / "attempts" / f"{attempt_id}.json"
    attempt = {
        "schema_version": 4, "attempt_id": attempt_id, "status": "running",
        "started_unix_s": time.time(), "job_hash": job_hash,
        "plan": plan, "existing_before": len(existing_ids),
        "capture_progress": {"added_frames": 0,
                             "total_job_samples": len(existing_ids)},
    }
    write_json_atomic(attempt_path, attempt)

    client = world = session = traffic_manager = None
    original_settings = original_weather = None
    world_settings_touched = weather_touched = traffic_manager_touched = False
    actors, sensors, streams = [], {}, {}
    drop_counts = Counter()
    primary_error, primary_traceback = None, None
    capture_result = None
    sensor_stats, sensor_counts = {}, {}
    added_frames = 0

    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(30.0)
        initial_world = client.get_world()
        if initial_world.get_settings().synchronous_mode:
            raise RuntimeError(
                "another/stale synchronous client owns the world; refusing collection")
        world, map_reloaded, map_evidence = select_probe_world(
            client, initial_world, town_name, timeout_s=30.0,
            client_timeout_s=30.0)
        attempt["map_selection"] = {
            "reloaded": bool(map_reloaded), **map_evidence,
            "selected_map": world.get_map().name,
            "episode_id": int(world.id),
        }
        print(f"[collect] {'loaded' if map_reloaded else 'reusing'} "
              f"map {world.get_map().name}")
        original_settings = world.get_settings()
        original_weather = world.get_weather()
        if original_settings.synchronous_mode:
            raise RuntimeError("selected world unexpectedly became synchronous")

        session = SynchronousWorldSession(world, cfg.FPS, timeout_s=10.0)
        world_settings_touched = True
        session.__enter__()
        weather_touched = True
        world.set_weather(carla.WeatherParameters(**weather_kwargs(profile)))

        blueprints = world.get_blueprint_library()
        ego_spawns = world.get_map().get_spawn_points()
        traffic_light_spawn_candidates = 0
        static_stop_signs = [
            obj for obj in world.get_environment_objects(
                carla.CityObjectLabel.TrafficSigns)
            if "stop" in str(obj.name).lower()
        ]
        stop_sign_spawn_candidates = 0
        if "TrafficLight" in targeted_classes:
            ego_spawns, traffic_light_spawn_candidates = (
                prioritize_traffic_light_spawns(world, ego_spawns))
            print("[collect] traffic-light approach spawn candidates="
                  f"{traffic_light_spawn_candidates}")
        if "StopSign" in targeted_classes:
            ego_spawns, stop_sign_spawn_candidates = (
                prioritize_static_stop_sign_spawns(static_stop_signs, ego_spawns))
            print("[collect] visible stop signs="
                  f"{len(static_stop_signs)}, approach spawn candidates="
                  f"{stop_sign_spawn_candidates}")
            if not static_stop_signs or not stop_sign_spawn_candidates:
                raise RuntimeError(
                    f"map {town_name} has no targetable visible stop-sign approach")

        ego, _ = spawn_ego_safe(
            world, blueprints.find("vehicle.tesla.model3"), ego_spawns)
        actors.append(ego)
        stabilization_frame = world.tick(10.0)
        print(f"[collect] ego transform stabilized at frame={stabilization_frame}")

        traffic_manager = client.get_trafficmanager(cfg.TM_PORT)
        traffic_manager_touched = True
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        ego_controller = None
        control_status = {"mode": args.ego_control}
        if args.ego_control == "custom":
            ego_controller = EgoController(
                ego, traffic_manager,
                _ConfigOverride(cfg, EGO_CONTROL_MODE="custom"), world=world)
        elif args.ego_control == "stationary":
            ego.set_autopilot(False)
            ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            control_status = {"mode": "stationary", "target_speed_kmh": 0.0}
        else:
            ego.set_autopilot(True, traffic_manager.get_port())
            control_status = {"mode": "experimental-autopilot",
                              "qualification": "unqualified"}

        def advance_ego_control(frame):
            nonlocal control_status
            if ego_controller is None:
                return control_status
            ego_controller.path_context()
            control_status = ego_controller.apply(
                SafetyDecision(),
                {"override": False, "state": "L3_ACTIVE", "hazard": False,
                 "target_decel_ms2": 0.0},
                "NORMAL", int(frame), target_speed_kmh=args.target_speed_kmh)
            if control_status.get("mode") != "custom":
                raise RuntimeError(
                    f"custom ego control left its authorized mode: {control_status}")
            return control_status

        advance_ego_control(stabilization_frame)
        spawner = TrafficSpawner(
            world, traffic_manager, seed=args.seed, strict=True,
            on_spawn=actors.append)
        spawned_npcs = spawner.spawn_traffic(args.npcs) if args.npcs else 0
        if spawned_npcs != args.npcs:
            raise RuntimeError(
                f"ordinary NPC shortfall: requested={args.npcs}, got={spawned_npcs}")

        rng = random.Random(args.seed)
        targeted_actors = []
        targeted_actors.extend(spawn_target_vehicles(
            world, traffic_manager, targeted_classes,
            args.target_instances_per_class, rng, reference_vehicle=ego,
            on_spawn=actors.append))
        expected_target_vehicles = sum(
            name in TARGET_VEHICLE_BASE_TYPES for name in targeted_classes
        ) * args.target_instances_per_class
        if len(targeted_actors) != expected_target_vehicles:
            raise RuntimeError(
                "targeted vehicle shortfall: "
                f"requested={expected_target_vehicles}, got={len(targeted_actors)}")
        heavy = spawn_heavy_vehicles(
            world, traffic_manager, args.heavy_vehicles, rng,
            reference_vehicle=ego, on_spawn=actors.append)
        if len(heavy) != args.heavy_vehicles:
            raise RuntimeError(
                f"heavy vehicle shortfall: requested={args.heavy_vehicles}, "
                f"got={len(heavy)}")
        targeted_actors.extend(heavy)
        two_wheelers = spawn_two_wheelers(
            world, traffic_manager, args.two_wheelers, rng,
            reference_vehicle=ego, on_spawn=actors.append)
        if len(two_wheelers) != args.two_wheelers:
            raise RuntimeError(
                f"two-wheeler shortfall: requested={args.two_wheelers}, "
                f"got={len(two_wheelers)}")
        targeted_actors.extend(two_wheelers)
        walkers, _ = spawn_walkers(
            world, args.walkers, rng, reference_vehicle=ego,
            on_spawn=actors.append)
        if len(walkers) != args.walkers:
            raise RuntimeError(
                f"walker shortfall: requested={args.walkers}, got={len(walkers)}")
        targeted_actors.extend(walkers)

        camera_transform = carla.Transform(
            carla.Location(x=cfg.CAM_X, y=cfg.CAM_Y, z=cfg.CAM_Z),
            carla.Rotation(roll=cfg.CAM_ROLL, pitch=cfg.CAM_PITCH,
                           yaw=cfg.CAM_YAW))
        lidar_transform = carla.Transform(
            carla.Location(x=cfg.LIDAR_X, y=cfg.LIDAR_Y, z=cfg.LIDAR_Z),
            carla.Rotation(roll=cfg.LIDAR_ROLL, pitch=cfg.LIDAR_PITCH,
                           yaw=cfg.LIDAR_YAW))
        radar_transform = carla.Transform(
            carla.Location(x=cfg.RADAR_X, y=cfg.RADAR_Y, z=cfg.RADAR_Z),
            carla.Rotation(roll=cfg.RADAR_ROLL, pitch=cfg.RADAR_PITCH,
                           yaw=cfg.RADAR_YAW))
        specs = {
            "rgb": (configure_camera_blueprint(
                blueprints.find("sensor.camera.rgb"), cfg), camera_transform),
            "semantic": (configure_camera_blueprint(
                blueprints.find("sensor.camera.semantic_segmentation"), cfg),
                camera_transform),
            "depth": (configure_camera_blueprint(
                blueprints.find("sensor.camera.depth"), cfg), camera_transform),
            "lidar": (configure_lidar_blueprint(
                blueprints.find("sensor.lidar.ray_cast"), cfg), lidar_transform),
            "radar": (configure_radar_blueprint(
                blueprints.find("sensor.other.radar"), cfg), radar_transform),
        }
        for name, (blueprint, transform) in specs.items():
            print(f"[collect] spawning sensor={name}", flush=True)
            sensor = world.spawn_actor(blueprint, transform, attach_to=ego)
            actors.append(sensor)
            sensors[name] = sensor
            sensor_queue = queue.Queue(maxsize=8)
            streams[name] = sensor_queue
            sensor.listen(lambda data, q=sensor_queue, n=name:
                          enqueue_sensor(q, data, drop_counts, n))

        print("[collect] warming up aligned sensor streams", flush=True)
        required_warmup = ["rgb", "semantic", "depth", "lidar"]
        if args.radar_policy == "required":
            required_warmup.append("radar")
        last_frame = warmup_sensor_streams(
            world, {name: streams[name] for name in required_warmup},
            tick_timeout=10.0)

        if args.diagnose_visibility:
            for actor in targeted_actors:
                print("[collect][visibility]", actor.id, actor.type_id,
                      actor_camera_coordinates(actor, sensors["rgb"]),
                      "bbox=", project_actor(actor, sensors["rgb"], cfg.CAM_WIDTH,
                                             cfg.CAM_HEIGHT, cfg.CAM_FOV))
            for stop_sign in static_stop_signs:
                print("[collect][visibility]", stop_sign.id, stop_sign.name,
                      "bbox=", project_environment_object(
                          stop_sign, sensors["rgb"], cfg.CAM_WIDTH,
                          cfg.CAM_HEIGHT, cfg.CAM_FOV))

        captured_class_counts, rejected_counts = Counter(), Counter()
        class_tags = semantic_ids(carla)
        road_line_tag = int(carla.CityObjectLabel.RoadLines)
        light_boxes = {}
        sensor_stats = {name: SensorSyncStats() for name in specs}
        sensor_counts = {
            name: {"expected": 0, "received": 0, "missing": 0, "invalid": 0}
            for name in specs
        }
        attempt['sensor_counts'] = sensor_counts
        radar_reader = (OptionalFrameReader(
            streams["radar"], "radar", sensor_stats["radar"])
            if args.radar_policy == "optional" else None)
        start_snapshot = world.get_snapshot()
        start_time = float(start_snapshot.timestamp.elapsed_seconds)
        previous_time = start_time
        if not math.isfinite(start_time) or start_time < 0:
            raise ValueError('invalid initial world timestamp')
        capture_episode = int(world.id)
        label_mode = "r+" if labels_path.is_file() else "w+"
        with open(labels_path, label_mode, encoding="utf-8", newline="\n") as labels:
            labels.seek(0, os.SEEK_END)
            for index in remaining:
                advance_ego_control(last_frame)
                frame_id = world.tick(10.0)
                if int(frame_id) != int(last_frame) + 1:
                    raise RuntimeError(
                        f'nonconsecutive capture frame: {last_frame} -> {frame_id}')
                last_frame = frame_id
                for counts in sensor_counts.values():
                    counts['expected'] += 1
                sample_data = {}
                for name in ("rgb", "semantic", "depth", "lidar"):
                    try:
                        sample_data[name] = retrieve_exact_frame(
                            streams[name], frame_id, timeout=2.0,
                            sensor_name=name, stats=sensor_stats[name])
                        sensor_counts[name]["received"] += 1
                    except TimeoutError:
                        sensor_counts[name]["missing"] += 1
                        raise
                    except SensorFrameError:
                        sensor_counts[name]["invalid"] += 1
                        raise
                if args.radar_policy == "required":
                    try:
                        radar = retrieve_exact_frame(
                            streams["radar"], frame_id, timeout=2.0,
                            sensor_name="radar", stats=sensor_stats["radar"])
                        sensor_counts["radar"]["received"] += 1
                    except TimeoutError:
                        sensor_counts["radar"]["missing"] += 1
                        raise
                    except SensorFrameError:
                        sensor_counts["radar"]["invalid"] += 1
                        raise
                else:
                    radar = radar_reader.get(frame_id, cfg.RADAR_OPTIONAL_TIMEOUT_S)
                    sensor_counts["radar"][
                        "received" if radar is not None else "missing"] += 1
                sample_data["radar"] = radar

                rgb_image = sample_data["rgb"]
                semantic = sample_data["semantic"]
                depth = sample_data["depth"]
                lidar = sample_data["lidar"]
                stem = sample_stem(args.seed, index)
                rgb_array = bgra(rgb_image)[:, :, :3]
                semantic_tags = bgra(semantic)[:, :, 2]
                road_line = np.where(
                    semantic_tags == road_line_tag, 255, 0).astype(np.uint8)
                depth_values_m = depth_m(depth)
                depth_cm = np.clip(
                    depth_values_m * 100.0, 0, 65535).astype(np.uint16)
                lidar_rows = np.frombuffer(
                    lidar.raw_data, np.float32).reshape((-1, 4))
                radar_rows = (np.frombuffer(
                    radar.raw_data, np.float32).reshape((-1, 4))
                    if radar is not None else np.empty((0, 4), np.float32))
                for name, rows in (('lidar', lidar_rows), ('radar', radar_rows)):
                    if not np.isfinite(rows).all():
                        sensor_counts[name]['invalid'] += 1
                        raise ValueError(f'{name}: nonfinite range data')

                objects, rejected_objects = [], []
                for actor in world.get_actors():
                    cls = actor_class(actor.type_id, actor.attributes)
                    if cls is None or actor.id == ego.id:
                        continue
                    if cls == "TrafficLight":
                        if actor.id not in light_boxes:
                            light_boxes[actor.id] = actor.get_light_boxes()
                        candidates = []
                        for head_index, head in enumerate(light_boxes[actor.id]):
                            annotation = project_carla_box(
                                head, None, sensors["rgb"], cfg.CAM_WIDTH,
                                cfg.CAM_HEIGHT, cfg.CAM_FOV)
                            if annotation:
                                annotation.update({
                                    "actor_id": int(actor.id),
                                    "type_id": actor.type_id,
                                    "velocity_world_ms": [0.0, 0.0, 0.0],
                                    "light_head_index": head_index,
                                    "traffic_light_state": str(
                                        actor.get_state()).rsplit(".", 1)[-1].lower(),
                                })
                                candidates.append(annotation)
                    else:
                        annotation = project_actor(
                            actor, sensors["rgb"], cfg.CAM_WIDTH,
                            cfg.CAM_HEIGHT, cfg.CAM_FOV)
                        candidates = [annotation] if annotation else []
                    for annotation in candidates:
                        annotation["class"] = cls
                        visible, reason = visible_annotation(
                            annotation, depth_values_m, semantic_tags, class_tags)
                        if visible is not None:
                            objects.append(visible)
                            captured_class_counts[cls] += 1
                        else:
                            rejected_counts[f"{cls}:{reason}"] += 1
                            rejected_objects.append(
                                {**annotation, "rejection_reason": reason})
                for stop_sign in static_stop_signs:
                    annotation = project_environment_object(
                        stop_sign, sensors["rgb"], cfg.CAM_WIDTH,
                        cfg.CAM_HEIGHT, cfg.CAM_FOV)
                    if annotation:
                        annotation["class"] = "StopSign"
                        visible, reason = visible_annotation(
                            annotation, depth_values_m, semantic_tags, class_tags)
                        if visible is not None:
                            objects.append(visible)
                            captured_class_counts["StopSign"] += 1
                        else:
                            rejected_counts[f"StopSign:{reason}"] += 1
                            rejected_objects.append(
                                {**annotation, "rejection_reason": reason})

                world_snapshot = world.get_snapshot()
                if int(world_snapshot.frame) != int(frame_id):
                    raise RuntimeError(
                        f"world snapshot mismatch: tick={frame_id}, "
                        f"snapshot={world_snapshot.frame}")
                world_time = float(world_snapshot.timestamp.elapsed_seconds)
                if (not math.isfinite(world_time) or world_time <= previous_time
                        or abs(world_time - previous_time - 1.0 / cfg.FPS) > 1e-5):
                    raise ValueError('invalid or discontinuous world timestamp')
                if int(world.id) != capture_episode:
                    raise RuntimeError('capture episode changed')
                for name, data in sample_data.items():
                    if data is not None and (
                            not math.isfinite(float(data.timestamp))
                            or abs(float(data.timestamp) - world_time) > 1e-5):
                        sensor_counts[name]['invalid'] += 1
                        raise ValueError(f'{name}: invalid or misaligned sensor timestamp')
                previous_time = world_time
                sensor_frames = {
                    name: (int(data.frame) if data is not None else None)
                    for name, data in sample_data.items()
                }
                sensor_timestamps = {
                    name: (float(data.timestamp) if data is not None else None)
                    for name, data in sample_data.items()
                }
                actual_transforms = {
                    name: transform_metadata(sensor.get_transform())
                    for name, sensor in sensors.items()
                }
                ego_location = ego.get_location()
                record = {
                    "schema_version": 4, "sample_id": stem,
                    "attempt_id": attempt_id,
                    "capture_index": index, "job_hash": job_hash,
                    "episode_id": int(world.id), "frame_id": int(frame_id),
                    "world_timestamp_s": float(
                        world_snapshot.timestamp.elapsed_seconds),
                    "timestamp": float(rgb_image.timestamp),
                    "sensor_frames": sensor_frames,
                    "sensor_timestamps_s": sensor_timestamps,
                    "sensor_available": {
                        name: data is not None for name, data in sample_data.items()},
                    "town": town_name, "weather": args.weather,
                    "split": plan["split"], "label_policy": POLICY,
                    "sensor_calibration": contract["sensors"],
                    "sensor_transforms_world": actual_transforms,
                    "camera_to_world": actual_transforms["rgb"]["matrix"],
                    "camera_fov": cfg.CAM_FOV,
                    "ego_location": [float(ego_location.x),
                                     float(ego_location.y),
                                     float(ego_location.z)],
                    "ego_control_owner": args.ego_control,
                    "ego_control_status": control_status,
                    "image_size": [cfg.CAM_WIDTH, cfg.CAM_HEIGHT],
                    "objects": objects,
                    "rejected_objects": rejected_objects,
                    "radar_available": radar is not None,
                }
                arrays = {
                    "rgb": rgb_array, "semantic": semantic_tags,
                    "road_line": road_line, "depth": depth_cm,
                    "lidar": lidar_rows, "radar": radar_rows,
                }
                try:
                    committed = persist_capture_files(
                        split_root, stem, arrays, args.jpeg_quality, cv2)
                    append_sample_record(labels, record, committed)
                except Exception:
                    attempt.setdefault("counters", {})["write_errors"] = (
                        attempt.get("counters", {}).get("write_errors", 0) + 1)
                    raise
                added_frames += 1
                attempt['capture_progress'].update(
                    last_frame_id=int(frame_id), last_timestamp_s=world_time)
                if added_frames == 1 or added_frames % 100 == 0:
                    attempt["capture_progress"].update({
                        "added_frames": added_frames,
                        "total_job_samples": len(existing_ids) + added_frames,
                        "last_frame_id": int(frame_id),
                    })
                    write_json_atomic(attempt_path, attempt)
                    print(f"[collect] {len(existing_ids) + added_frames}/"
                          f"{args.frames} frame={frame_id}")

        capture_result = {
            **plan, "schema_version": 4,
            "sensor_sync": {name: stats.summary()
                            for name, stats in sensor_stats.items()},
            "sensor_counts": sensor_counts,
            "sensor_queue_drops": {name: int(drop_counts[name])
                                   for name in specs},
            "semantic_class_ids": class_tags,
            "road_line_tag": road_line_tag,
            "label_review_status": "pilot_unreviewed_not_training_ready",
            "training_ready": False,
            "rejected_annotations": dict(sorted(rejected_counts.items())),
            "existing_before": len(existing_ids),
            "added_frames": added_frames,
            "total_job_samples": len(existing_ids) + added_frames,
            "capture_start_time_s": start_time,
            "capture_end_time_s": previous_time,
            "capture_duration_s": previous_time - start_time,
            "duration_convention": "pre-first-tick to last captured tick; excludes warmup",
            "traffic_light_spawn_candidates": traffic_light_spawn_candidates,
            "visible_static_stop_signs": len(static_stop_signs),
            "stop_sign_spawn_candidates": stop_sign_spawn_candidates,
            "captured_class_counts": dict(sorted(captured_class_counts.items())),
        }
        if capture_result["total_job_samples"] != args.frames:
            raise RuntimeError(
                "capture ended without the exact requested sample count")
    except BaseException as exc:
        primary_error, primary_traceback = exc, exc.__traceback__
    finally:
        cleanup = cleanup_capture(
            client=client, world=world, session=session,
            traffic_manager=traffic_manager,
            traffic_manager_touched=traffic_manager_touched,
            actors=actors, sensors=sensors,
            original_settings=original_settings,
            original_weather=original_weather,
            world_settings_touched=world_settings_touched,
            weather_touched=weather_touched, carla_module=carla,
            timeout_s=10.0)

        attempt["ended_unix_s"] = time.time()
        attempt['capture_progress'].update({
            'added_frames': added_frames,
            'total_job_samples': len(existing_ids) + added_frames})
        attempt['sensor_counts'] = sensor_counts
        attempt['sensor_sync'] = {name: stats.summary()
                                  for name, stats in sensor_stats.items()}
        attempt['sensor_queue_drops'] = dict(drop_counts)
        attempt["cleanup"] = cleanup
        if capture_result is not None:
            attempt["capture_result"] = capture_result
        if primary_error is None and cleanup["verified"]:
            completed_manifest = {
                **capture_result, "status": "completed", "cleanup": cleanup,
                "attempt_id": attempt_id,
            }
            try:
                attempt["status"] = "completed"
                # The canonical manifest is the final commit marker. Its presence
                # must never precede successful durable attempt/cleanup evidence.
                write_json_atomic(attempt_path, attempt)
                write_json_atomic(canonical_manifest_path, completed_manifest)
            except BaseException as exc:
                primary_error, primary_traceback = exc, exc.__traceback__
        elif primary_error is None:
            primary_error = RuntimeError(
                f"capture cleanup was not verified: {cleanup['errors']}")
            primary_traceback = primary_error.__traceback__
        if primary_error is not None:
            attempt["status"] = "failed"
            attempt["error"] = {
                "type": type(primary_error).__name__,
                "message": str(primary_error),
                "traceback": "".join(traceback.format_exception(
                    type(primary_error), primary_error, primary_traceback)),
            }
            if cleanup["errors"] and hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    "cleanup evidence: " + json.dumps(
                        cleanup["errors"], ensure_ascii=False))
        if primary_error is not None:
            try:
                write_json_atomic(attempt_path, attempt)
            except BaseException as attempt_error:
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        f"could not finalize attempt manifest: {attempt_error}")

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    print(f"[collect] completed and cleanup verified: {canonical_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
