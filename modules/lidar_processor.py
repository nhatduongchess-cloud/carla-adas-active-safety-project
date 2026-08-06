"""LiDAR data processing module using DBSCAN clustering for obstacle detection."""

from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import DBSCAN


class LidarProcessor:
    """Processes raw LiDAR point clouds to extract spatial obstacle clusters."""

    def __init__(self, eps: float = 0.6, min_samples: int = 5, max_points: int = 3000) -> None:
        """Initializes the DBSCAN clustering parameters.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
            max_points: Hạ mẫu ngẫu nhiên xuống tối đa ngần này điểm trước khi chạy
                DBSCAN. DBSCAN trên toàn bộ đám mây (chục nghìn điểm) là nút thắt CPU
                gây lag; giới hạn ~3000 điểm giữ cụm vẫn tốt mà nhanh hơn nhiều lần.
        """
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        self.max_points = max_points
        self._rng = np.random.default_rng(42)

    def extract_obstacles(self, raw_point_cloud: np.ndarray) -> List[Dict[str, Any]]:
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
            (height_filtered_points[:, 0] > 0.0) &
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

            detected_obstacles.append({
                "centroid": [centroid_x, centroid_y, centroid_z],
                "point_count": cluster_size
            })

        return detected_obstacles
