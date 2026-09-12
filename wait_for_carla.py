"""Bounded CARLA readiness probe used by the Windows launcher."""

import argparse
import sys
import time


def wait_until_ready(client_factory, host, port, timeout_s, poll_s=2.0):
    deadline = time.monotonic() + float(timeout_s)
    attempts = 0
    last_error = "not attempted"
    while time.monotonic() < deadline:
        attempts += 1
        remaining = max(1.0, deadline - time.monotonic())
        try:
            client = client_factory(host, int(port))
            client.set_timeout(min(5.0, remaining))
            server_version = client.get_server_version()
            world = client.get_world()
            settings = world.get_settings()
            if settings.synchronous_mode:
                raise RuntimeError(
                    "world is already synchronous; another/stale client may own ticks")
            snapshot = world.wait_for_tick(min(5.0, remaining))
            return {
                "attempts": attempts,
                "server_version": server_version,
                "map": world.get_map().name,
                "frame": int(snapshot.frame),
            }
        except Exception as error:
            last_error = str(error)
            time.sleep(min(float(poll_s), max(0.0, deadline - time.monotonic())))
    raise TimeoutError(
        f"CARLA world was not ready after {attempts} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Wait until CARLA world advances in async mode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        import carla
        result = wait_until_ready(
            carla.Client, args.host, args.port, args.timeout)
    except Exception as error:
        print(f"[CARLA] NOT READY: {error}", file=sys.stderr)
        return 1
    print(
        f"[CARLA] READY server={result['server_version']} "
        f"map={result['map']} frame={result['frame']} "
        f"attempts={result['attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
