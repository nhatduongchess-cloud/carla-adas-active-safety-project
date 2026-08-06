import carla
import math
import weakref


class CollisionSensor:
    def __init__(self, parent_actor, metrics_ref):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.metrics = metrics_ref
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        if intensity > 300:
            self.metrics['collision_risk'] = True

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


class LaneInvasionSensor:
    def __init__(self, parent_actor, metrics_ref):
        self.sensor = None
        self.metrics = metrics_ref
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(
                bp, carla.Transform(), attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(str(x.type).split()[-1]
                         for x in event.crossed_lane_markings)
        self.metrics['lane_status'] = f"Crossed: {', '.join(lane_types)}"

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()


class IMUSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        self.accelerometer = (sensor_data.accelerometer.x,
                              sensor_data.accelerometer.y, sensor_data.accelerometer.z)
        self.compass = math.degrees(sensor_data.compass)

    def destroy(self):
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
