class ObjectCounter:
    """Quản lý đếm đối tượng hiện tại và tập hợp ID duy nhất tích lũy."""

    def __init__(self):
        self.unique_vehicles = set()
        self.unique_pedestrians = set()

    def update(self, track_id, class_name):
        if track_id is not None:
            if class_name in ["car", "truck", "bus", "motorcycle"]:
                self.unique_vehicles.add(track_id)
            elif class_name == "person":
                self.unique_pedestrians.add(track_id)
