"""Minimal real-CARLA RGB sensor smoke test.

This intentionally avoids neural models, Traffic Manager, LiDAR, and radar so
GPU-render failures can be isolated from the rest of the ADAS stack.
"""

import argparse
import queue
import sys

import carla

import config as cfg
from modules.sensor_setup import configure_camera_blueprint
from modules.sensor_sync import retrieve_exact_frame, warmup_sensor_streams
from modules.simulation_guard import (SynchronousWorldSession,
                                      get_or_load_world)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def spawn_test_vehicle(world):
    blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
    for transform in world.get_map().get_spawn_points():
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor
    raise RuntimeError("CARLA has no free spawn point for the camera smoke test")


def run(host, port, town, frames, timeout_s, width, height, postprocess,
        spawn_async, fps, async_only, sensor_tick):
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    world, reloaded = get_or_load_world(client, town)
    print(f"[camera-smoke] map={world.get_map().name} reloaded={reloaded}", flush=True)

    session = SynchronousWorldSession(world, fps, timeout_s=timeout_s)
    actors = []
    camera = None
    image_queue = queue.Queue()
    captured = 0
    try:
        camera_bp = configure_camera_blueprint(
            world.get_blueprint_library().find("sensor.camera.rgb"), cfg)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        if camera_bp.has_attribute("sensor_tick"):
            camera_bp.set_attribute("sensor_tick", str(sensor_tick))
        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute(
                "enable_postprocess_effects", "True" if postprocess else "False")
        print(
            f"[camera-smoke] config={width}x{height} postprocess={postprocess} "
            f"sensor_tick={sensor_tick}",
            flush=True)
        spawn_async = bool(spawn_async or async_only)
        if not spawn_async:
            session.__enter__()
        vehicle = spawn_test_vehicle(world)
        actors.append(vehicle)
        if not spawn_async:
            parent_frame = world.tick(timeout_s)
            print(f"[camera-smoke] parent_ready_frame={parent_frame}", flush=True)
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(
                carla.Location(x=cfg.CAM_X, y=cfg.CAM_Y, z=cfg.CAM_Z),
                carla.Rotation(roll=cfg.CAM_ROLL, pitch=cfg.CAM_PITCH,
                               yaw=cfg.CAM_YAW)),
            attach_to=vehicle)
        actors.append(camera)
        camera.listen(image_queue.put)

        if spawn_async:
            async_image = image_queue.get(timeout=timeout_s)
            print(
                f"[camera-smoke] async_image_frame={async_image.frame}",
                flush=True)
            captured += 1
            if async_only:
                while captured < frames:
                    image = image_queue.get(timeout=timeout_s)
                    captured += 1
                print(
                    f"[camera-smoke] PASS async captured={captured}/{frames} "
                    f"last_frame={image.frame if frames > 1 else async_image.frame}",
                    flush=True)
                return True
            session.__enter__()

        warmup_frame = warmup_sensor_streams(
            world, {"camera": image_queue}, attempts=20, timeout=0.5)
        print(f"[camera-smoke] warmup_frame={warmup_frame}", flush=True)

        max_world_ticks = max(frames, int(frames * max(1.0, sensor_tick * fps) * 2))
        world_ticks = 0
        while captured < frames and world_ticks < max_world_ticks:
            frame = world.tick(timeout_s)
            world_ticks += 1
            if sensor_tick > 0.0:
                try:
                    image = image_queue.get(timeout=min(0.5, timeout_s))
                except queue.Empty:
                    continue
                if image.frame > frame:
                    raise RuntimeError(
                        f"camera returned future frame {image.frame} > world {frame}")
            else:
                image = retrieve_exact_frame(
                    image_queue, frame, timeout=timeout_s, sensor_name="camera")
            captured += 1
            print(
                f"[camera-smoke] frame={frame} image_frame={image.frame} "
                f"image={image.width}x{image.height}", flush=True)
        if captured != frames:
            raise RuntimeError(
                f"camera produced only {captured}/{frames} frames over "
                f"{world_ticks} world ticks")
        print(f"[camera-smoke] PASS captured={captured}/{frames}", flush=True)
        return True
    finally:
        if camera is not None and camera.is_alive:
            camera.stop()
        if actors:
            client.apply_batch([carla.command.DestroyActor(actor) for actor in actors])
            try:
                if session.active:
                    world.tick(timeout_s)
                else:
                    world.wait_for_tick(timeout_s)
            except Exception as exc:
                print(f"[camera-smoke] cleanup tick warning: {exc}", flush=True)
        try:
            session.close()
        except Exception as exc:
            print(f"[camera-smoke] async restore warning: {exc}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Minimal CARLA RGB camera test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town02")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--spawn-async", action="store_true")
    parser.add_argument("--async-only", action="store_true")
    parser.add_argument("--fps", type=int, default=cfg.FPS)
    parser.add_argument("--sensor-tick", type=float, default=0.0)
    args = parser.parse_args()
    if args.frames <= 0 or args.fps <= 0 or args.sensor_tick < 0.0:
        parser.error("--frames/--fps must be positive and --sensor-tick >= 0")
    try:
        success = run(
            args.host, args.port, args.town, args.frames, args.timeout,
            args.width, args.height, args.postprocess, args.spawn_async,
            args.fps, args.async_only, args.sensor_tick)
    except Exception as exc:
        print(f"[camera-smoke] FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
