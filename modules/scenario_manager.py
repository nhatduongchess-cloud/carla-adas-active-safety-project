"""Scenario manager module defining autonomous driving test cases directly via CARLA API."""

import carla


class ScenarioManager:
    """Manages spawning and configuration of safety test scenarios in CARLA."""

    def __init__(self, world: carla.World) -> None:
        self.world: carla.World = world
        self.blueprint_library = world.get_blueprint_library()

    def setup_pedestrian_crossing_scenario(self, spawn_location: carla.Transform) -> carla.Actor:
        """Scenario 1: Sudden pedestrian crossing."""
        walker_bp = self.blueprint_library.find('walker.pedestrian.0001')
        walker = self.world.spawn_actor(walker_bp, spawn_location)
        return walker

    def setup_lead_vehicle_braking_scenario(self, spawn_location: carla.Transform) -> carla.Vehicle:
        """Scenario 2: Lead vehicle sudden braking."""
        vehicle_bp = self.blueprint_library.find('vehicle.audi.etron')
        lead_vehicle = self.world.spawn_actor(vehicle_bp, spawn_location)
        lead_vehicle.set_autopilot(True)
        return lead_vehicle
