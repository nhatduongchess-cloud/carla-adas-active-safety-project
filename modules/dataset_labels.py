"""Offline-testable label geometry; never used by runtime safety/control.

V3 requires aligned raw semantic/depth images and stores visible (not amodal)
boxes. Semantic identity plus 3-D support is not an instance-ID guarantee.
"""
import itertools
import math

import numpy as np

POLICY = 'geometry-semantic-v3'


def require_training_review(record, manifest):
    """A structural integrity PASS must not promote an unreviewed V3 pilot."""
    if record.get('label_policy') == POLICY:
        if (manifest.get('label_policy') != POLICY
                or manifest.get('label_review_status') != 'approved'):
            raise ValueError('V3 pilot labels require a documented visual review before YOLO build')


def semantic_ids(carla):
    """Resolve the installed CARLA enum, not version-specific numeric tags."""
    names = {'Person': 'Pedestrians', 'StopSign': 'TrafficSigns',
             'TrafficLight': 'TrafficLight', 'Car': 'Car', 'Bus': 'Bus',
             'Truck': 'Truck', 'Bicycle': 'Bicycle', 'Motorcycle': 'Motorcycle'}
    return {cls: int(getattr(carla.CityObjectLabel, name)) for cls, name in names.items()}


def project_box(box_to_world, extent, camera_to_world, width, height, fov):
    """Project an oriented box. Reject near-plane intersections conservatively."""
    box_to_world = np.asarray(box_to_world, dtype=float)
    camera_to_world = np.asarray(camera_to_world, dtype=float)
    extent = np.asarray(extent, dtype=float)
    if (box_to_world.shape != (4, 4) or camera_to_world.shape != (4, 4)
            or extent.shape != (3,) or not np.isfinite(extent).all()
            or (extent <= 0).any() or not np.isfinite(box_to_world).all()
            or not np.isfinite(camera_to_world).all()
            or width < 4 or height < 4 or not 0 < fov < 180):
        return None
    local_vertices = np.asarray(list(itertools.product((-1, 1), repeat=3))) * extent
    vertices = np.column_stack((local_vertices, np.ones(8)))
    box_to_camera = np.linalg.inv(camera_to_world) @ box_to_world
    camera_vertices = (box_to_camera @ vertices.T).T[:, :3]
    if (camera_vertices[:, 0] <= 0.2).any():
        return None
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    u = focal * camera_vertices[:, 1] / camera_vertices[:, 0] + width / 2
    v = -focal * camera_vertices[:, 2] / camera_vertices[:, 0] + height / 2
    box = [max(0., u.min()), max(0., v.min()),
           min(width - 1., u.max()), min(height - 1., v.max())]
    if box[2] - box[0] < 3 or box[3] - box[1] < 3:
        return None
    return {'bbox_xyxy': [round(float(x), 2) for x in box],
            'distance_m': round(float(np.linalg.norm(box_to_camera[:3, 3])), 3),
            'camera_to_box': np.linalg.inv(box_to_camera).tolist(),
            'box_extent_m': extent.tolist(), 'camera_fov': float(fov)}


def visible_annotation(annotation, depth_m, tags, class_tags):
    """Return (visible label, rejection reason), preserving the input annotation.

    CARLA depth is radial distance in metres. Backproject each candidate pixel
    into the oriented box; a nearby occluder must not count as target evidence.
    Thresholds are conservative pilot defaults, not validated dataset gates.
    """
    depth = np.asarray(depth_m)
    tags = np.asarray(tags)
    if depth.ndim != 2 or tags.shape != depth.shape:
        raise ValueError('aligned 2-D depth/semantic images required')
    cls = annotation.get('class')
    if cls not in class_tags:
        return None, 'unknown_semantic_class'
    if cls == 'StopSign' and str(annotation.get('type_id', '')).startswith('traffic.stop'):
        return None, 'invisible_stop_trigger'
    distance = float(annotation.get('distance_m', float('nan')))
    if not math.isfinite(distance) or not 0 < distance <= 80:
        return None, 'outside_distance_range'
    box = np.asarray(annotation.get('bbox_xyxy', []), dtype=float)
    matrix = np.asarray(annotation.get('camera_to_box', []), dtype=float)
    extent = np.asarray(annotation.get('box_extent_m', []), dtype=float)
    fov = float(annotation.get('camera_fov', float('nan')))
    h, w = depth.shape
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not (0 <= box[0] < box[2] <= w - 1 and 0 <= box[1] < box[3] <= h - 1)
            or matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or extent.shape != (3,) or not np.isfinite(extent).all()
            or (extent <= 0).any() or not 0 < fov < 180):
        return None, 'invalid_geometry'
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    yy, xx = np.mgrid[y1:y2+1, x1:x2+1]
    focal = w / (2 * math.tan(math.radians(fov) / 2))
    rays = np.stack((np.ones_like(xx), (xx - w/2)/focal, -(yy - h/2)/focal), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    measured = depth[y1:y2+1, x1:x2+1]
    points = rays * measured[..., None]
    local = points @ matrix[:3, :3].T + matrix[:3, 3]
    # Small metric allowance for quantization/mesh-vs-box discrepancy, not 15% range.
    support = ((np.abs(local) <= extent + 0.25).all(axis=-1)
               & np.isfinite(measured) & (measured > 0)
               & (tags[y1:y2+1, x1:x2+1] == class_tags[cls]))
    pixels = int(support.sum())
    fraction = pixels / support.size
    if pixels < 4 or fraction < 0.10:
        return None, 'insufficient_visible_support'
    ys, xs = np.where(support)
    visible = [int(xs.min()+x1), int(ys.min()+y1), int(xs.max()+x1), int(ys.max()+y1)]
    if visible[2]-visible[0] < 3 or visible[3]-visible[1] < 3:
        return None, 'visible_box_too_small'
    result = dict(annotation)
    result.update({'projected_bbox_xyxy': box.tolist(), 'bbox_xyxy': visible,
                   'label_policy': POLICY, 'visible_support_pixels': pixels,
                   'visible_support_fraction': round(fraction, 5)})
    return result, None


def require_matching_policy(labels_path):
    """Never append new-policy annotations to an existing legacy collection."""
    import json
    if labels_path.is_file():
        with labels_path.open(encoding='utf-8') as source:
            for line in source:
                if line.strip() and json.loads(line).get('label_policy') != POLICY:
                    raise ValueError('Existing dataset has a different label policy; use a NEW output root')
