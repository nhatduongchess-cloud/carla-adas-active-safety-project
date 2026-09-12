"""LiDAR data processing module using DBSCAN clustering for obstacle detection."""

from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN


class LidarProcessor:
    """Processes raw LiDAR point clouds to extract spatial obstacle clusters."""

    def __init__(self, eps: float = 0.6, min_samples: int = 5, max_points: int = 3000,
                 rear_range_m: float = 15.0, sparse_corridor_half_m: float = 2.5) -> None:
        """Initializes the DBSCAN clustering parameters.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            max_points: Hạ mẫu ngẫu nhiên xuống tối đa ngần này điểm trước khi chạy
                DBSCAN. DBSCAN trên toàn bộ đám mây (chục nghìn điểm) là nút thắt CPU
                gây lag; giới hạn ~3000 điểm giữ cụm vẫn tốt mà nhanh hơn nhiều lần.
        """
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        self.min_samples = int(min_samples)
        self.max_points = max_points
        self.rear_range_m = rear_range_m
        self.sparse_corridor_half = sparse_corridor_half_m
        # Nhánh safety riêng cho vật nhỏ/thưa như cone; không thay DBSCAN chính.
        self.safety_clusterer = DBSCAN(eps=0.40, min_samples=2)
        self._rng = np.random.default_rng(42)

    def extract_obstacles(self, raw_point_cloud: np.ndarray,
                          include_sparse: bool = True) -> List[Dict[str, Any]]:
        """Clusters raw point cloud data to filter out ground plane and identify obstacles.

        Args:
            raw_point_cloud: Numpy array of shape (N, 4) containing [x, y, z, intensity].

        Returns:
            A list of detected obstacles with their 3D centroid positions.
        ```
        """
        if raw_point_cloud is None or len(raw_point_cloud) == 0:
            return []

        # Filter out ground points (retaining points slightly above the road surface)
        height_filtered_points = raw_point_cloud[raw_point_cloud[:, 2] > -1.5]

        # Crop region of interest ahead of the ego vehicle
        forward_filtered_points = height_filtered_points[
            (height_filtered_points[:, 0] > -self.rear_range_m) &
            (height_filtered_points[:, 0] < 30.0) &
            (np.abs(height_filtered_points[:, 1]) < 10.0)
        ]

        if len(forward_filtered_points) == 0:
            return []

        # Hạ mẫu ngẫu nhiên để DBSCAN chạy nhanh (giảm lag) khi đám mây quá dày.
        if len(forward_filtered_points) > self.max_points:
            idx = self._rng.choice(len(forward_filtered_points), self.max_points, replace=False)
            forward_filtered_points = forward_filtered_points[idx]

        spatial_coordinates = forward_filtered_points[:, :3]
        cluster_labels = self.clusterer.fit_predict(spatial_coordinates)

        unique_labels = set(cluster_labels)
        detected_obstacles: List[Dict[str, Any]] = []

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points identified by DBSCAN

            cluster_mask = (cluster_labels == label)
            cluster_points = spatial_coordinates[cluster_mask]

            centroid_x = float(np.mean(cluster_points[:, 0]))
            centroid_y = float(np.mean(cluster_points[:, 1]))
            centroid_z = float(np.mean(cluster_points[:, 2]))
            cluster_size = int(len(cluster_points))
            extent = np.ptp(cluster_points, axis=0)
            covariance = np.cov(cluster_points.T) if cluster_size > 1 else np.eye(3) * 0.25

            detected_obstacles.append({
                "centroid": [centroid_x, centroid_y, centroid_z],
                "point_count": cluster_size,
                "source": "dbscan",
                "extent": extent.astype(float).tolist(),
                "covariance": covariance.astype(float).tolist(),
            })

        # Safety micro-cluster lấy từ point cloud TRƯỚC khi random downsample để
        # vật nhỏ trong hành lang (cone/debris) không biến mất do lấy mẫu.
        sparse = height_filtered_points[
            (height_filtered_points[:, 0] > 0.0) &
            (height_filtered_points[:, 0] < 30.0) &
            (np.abs(height_filtered_points[:, 1]) < self.sparse_corridor_half) &
            (height_filtered_points[:, 2] < 2.5)
        ][:, :3]
        if len(sparse) > 2500:
            # Voxel representative deterministic; tránh random làm mất vật nhỏ.
            vox = np.floor(sparse / np.array([0.12, 0.12, 0.12])).astype(np.int32)
            _, keep = np.unique(vox, axis=0, return_index=True)
            sparse = sparse[np.sort(keep)[:2500]]
        if include_sparse and len(sparse) >= 2:
            safety_labels = self.safety_clusterer.fit_predict(sparse)
            for label in set(safety_labels):
                if label == -1:
                    continue
                pts = sparse[safety_labels == label]
                center = np.mean(pts, axis=0)
                # Không lặp cluster chính đã đủ mạnh.
                duplicate = any(
                    np.linalg.norm(center - np.asarray(obs["centroid"])) < 0.8
                    for obs in detected_obstacles)
                if not duplicate:
                    detected_obstacles.append({
                        "centroid": center.astype(float).tolist(),
                        "point_count": int(len(pts)),
                        "source": "sparse_safety",
                        "safety_critical": True,
                        "extent": np.ptp(pts, axis=0).astype(float).tolist(),
                        "covariance": (np.cov(pts.T) if len(pts) > 1
                                       else np.eye(3) * 0.25).astype(float).tolist(),
                    })

        return detected_obstacles

    def extract_safety_obstacles(self, raw_point_cloud: np.ndarray,
                                 max_obstacles: int = 256) -> List[Dict[str, Any]]:
        """Fast current-frame occupancy path used by the 40 Hz safety loop.

        This avoids DBSCAN and random sampling. Occupied 3-D voxels preserve the
        nearest metric return needed by AEB; full object clustering remains on
        the 20 Hz perception worker for camera association/tracking.
        """
        if raw_point_cloud is None or len(raw_point_cloud) == 0:
            return []
        points = np.asarray(raw_point_cloud, dtype=np.float32)
        points = points[
            (points[:, 2] > -1.5) &
            (points[:, 2] < 3.0) &
            (points[:, 0] > -self.rear_range_m) & (points[:, 0] < 30.0) &
            (np.abs(points[:, 1]) < 10.0)
        ][:, :3]
        if len(points) < 2:
            return []
        voxel_size = np.array([0.55, 0.45, 0.55], dtype=np.float32)
        voxels = np.floor(points / voxel_size).astype(np.int32)
        # Encode bounded voxel coordinates into one int64 key. np.unique on a
        # 1-D vector is materially faster than axis=0 on the 40 Hz path.
        keys = ((voxels[:, 0].astype(np.int64) + 64) * 128
                + (voxels[:, 1].astype(np.int64) + 32)) * 64 \
               + (voxels[:, 2].astype(np.int64) + 8)
        unique, inverse, counts = np.unique(
            keys, return_inverse=True, return_counts=True)
        occupied = np.flatnonzero(counts >= 2)
        sums = [np.bincount(inverse, weights=points[:, axis], minlength=len(unique))
                for axis in range(3)]
        obstacles = []
        for label in occupied:
            count = int(counts[label])
            center = np.array([sums[axis][label] / count for axis in range(3)])
            obstacles.append({
                "centroid": center.astype(float).tolist(),
                "point_count": count,
                "source": "safety_voxel",
                "safety_critical": bool(count >= 2 and abs(float(center[1]))
                                        < self.sparse_corridor_half),
                "extent": voxel_size.astype(float).tolist(),
                "covariance": np.diag((voxel_size / 2.0) ** 2).astype(float).tolist(),
            })
        obstacles.sort(key=lambda item: (
            max(0.0, float(item["centroid"][0])),
            abs(float(item["centroid"][1])), -int(item["point_count"])))
        return obstacles[:max(1, int(max_obstacles))]

    def extract_perception_obstacles(self, raw_point_cloud: np.ndarray,
                                     max_obstacles: int = 128) -> List[Dict[str, Any]]:
        """Fast deterministic voxel connected-components for 20 Hz fusion.

        DBSCAN remains available through :meth:`extract_obstacles` for offline
        ablation. Runtime fusion uses this bounded implementation to avoid CPU
        latency spikes while preserving object-level centroids and extents.
        """
        if raw_point_cloud is None or len(raw_point_cloud) == 0:
            return []
        points = np.asarray(raw_point_cloud, dtype=np.float32)
        points = points[
            (points[:, 2] > -1.5) & (points[:, 2] < 4.0) &
            (points[:, 0] > -self.rear_range_m) & (points[:, 0] < 30.0) &
            (np.abs(points[:, 1]) < 10.0)
        ][:, :3]
        if len(points) < self.min_samples:
            return []

        voxel_size = np.array([0.60, 0.55, 0.60], dtype=np.float32)
        voxels = np.floor(points / voxel_size).astype(np.int32)
        keys = ((voxels[:, 0].astype(np.int64) + 64) * 128
                + (voxels[:, 1].astype(np.int64) + 32)) * 64 \
               + (voxels[:, 2].astype(np.int64) + 8)
        _keys, first, inverse, counts = np.unique(
            keys, return_index=True, return_inverse=True, return_counts=True)
        unique_voxels = voxels[first]
        sums = np.column_stack([
            np.bincount(inverse, weights=points[:, axis], minlength=len(_keys))
            for axis in range(3)])
        cell_centers = sums / counts[:, None]
        lookup = {tuple(coord): index for index, coord in enumerate(unique_voxels)}
        visited = np.zeros(len(unique_voxels), dtype=bool)
        obstacles: List[Dict[str, Any]] = []

        for start in range(len(unique_voxels)):
            if visited[start]:
                continue
            stack, component = [start], []
            visited[start] = True
            while stack:
                index = stack.pop()
                component.append(index)
                vx, vy, vz = unique_voxels[index]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == dy == dz == 0:
                                continue
                            neighbor = lookup.get((int(vx + dx), int(vy + dy),
                                                   int(vz + dz)))
                            if neighbor is not None and not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)
            component = np.asarray(component, dtype=np.int32)
            total_points = int(np.sum(counts[component]))
            if total_points < self.min_samples:
                continue
            center = np.sum(
                cell_centers[component] * counts[component, None], axis=0) / total_points
            cell_min = np.min(unique_voxels[component], axis=0) * voxel_size
            cell_max = (np.max(unique_voxels[component], axis=0) + 1) * voxel_size
            extent = cell_max - cell_min
            obstacles.append({
                "centroid": center.astype(float).tolist(),
                "point_count": total_points,
                "source": "voxel_components",
                "extent": extent.astype(float).tolist(),
                "covariance": np.diag(np.maximum(0.15, extent / 4.0) ** 2)
                                .astype(float).tolist(),
            })

        obstacles.sort(key=lambda item: (
            max(0.0, float(item["centroid"][0])),
            abs(float(item["centroid"][1])), -int(item["point_count"])))
        return obstacles[:max(1, int(max_obstacles))]
