"""Cấu hình sensor dùng chung để runtime và validation không bị lệch."""


def configure_camera_blueprint(blueprint, cfg):
    blueprint.set_attribute('image_size_x', str(cfg.CAM_WIDTH))
    blueprint.set_attribute('image_size_y', str(cfg.CAM_HEIGHT))
    blueprint.set_attribute('fov', str(cfg.CAM_FOV))
    if blueprint.has_attribute('sensor_tick'):
        blueprint.set_attribute(
            'sensor_tick', str(getattr(cfg, 'CAMERA_SENSOR_TICK_S', 0.0)))
    if blueprint.has_attribute('enable_postprocess_effects'):
        blueprint.set_attribute(
            'enable_postprocess_effects',
            'True' if getattr(cfg, 'CAMERA_POSTPROCESS', True) else 'False')
    return blueprint


def configure_lidar_blueprint(blueprint, cfg):
    blueprint.set_attribute('range', str(getattr(cfg, 'LIDAR_RANGE_M', 60.0)))
    blueprint.set_attribute('rotation_frequency', str(cfg.FPS))
    blueprint.set_attribute('points_per_second', str(cfg.LIDAR_POINTS_PER_SECOND))
    blueprint.set_attribute('channels', str(getattr(cfg, 'LIDAR_CHANNELS', 32)))
    if blueprint.has_attribute('sensor_tick'):
        blueprint.set_attribute(
            'sensor_tick', str(getattr(cfg, 'LIDAR_SENSOR_TICK_S', 0.0)))
    return blueprint


def configure_radar_blueprint(blueprint, cfg):
    """Configure the front radar as an optional every-tick safety sensor."""
    blueprint.set_attribute('horizontal_fov', str(getattr(cfg, 'RADAR_HORIZONTAL_FOV', 35.0)))
    blueprint.set_attribute('vertical_fov', str(getattr(cfg, 'RADAR_VERTICAL_FOV', 10.0)))
    blueprint.set_attribute('range', str(getattr(cfg, 'RADAR_RANGE_M', 80.0)))
    blueprint.set_attribute('points_per_second', str(getattr(cfg, 'RADAR_POINTS_PER_SECOND', 4000)))
    if blueprint.has_attribute('sensor_tick'):
        blueprint.set_attribute(
            'sensor_tick', str(getattr(cfg, 'RADAR_SENSOR_TICK_S', 0.0)))
    return blueprint
