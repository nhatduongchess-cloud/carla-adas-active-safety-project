"""Offline regression tests for dataset label visibility and projection."""
import unittest
import numpy as np
from modules.dataset_labels import (POLICY, project_box, visible_annotation,
                                   semantic_ids, require_matching_policy, require_training_review)


class LabelTests(unittest.TestCase):
    def candidate(self, cls='Person'):
        transform = np.eye(4)
        transform[0, 3] = 10
        ann = project_box(transform, [1, 1, 1], np.eye(4), 100, 60, 90)
        return dict(ann, **{'class': cls, 'type_id': 'walker.test'})

    def depth(self):
        yy, xx = np.mgrid[:60, :100]
        # Plane at camera-forward 9 m, encoded as radial distance.
        return 9 * np.sqrt(1 + ((xx-50)/50)**2 + ((yy-30)/50)**2)

    def test_visible_surface_not_actor_center(self):
        ann = self.candidate()
        out, reason = visible_annotation(ann, self.depth(), np.full((60, 100), 12), {'Person': 12})
        self.assertIsNone(reason)
        self.assertGreater(out['visible_support_pixels'], 4)
        self.assertNotIn('label_policy', ann)

    def test_bus_pixels_cannot_label_person(self):
        out, reason = visible_annotation(self.candidate(), self.depth(), np.full((60, 100), 16), {'Person': 12})
        self.assertIsNone(out)
        self.assertEqual(reason, 'insufficient_visible_support')

    def test_same_class_occluder_rejected(self):
        out, _ = visible_annotation(self.candidate(), self.depth()/2, np.full((60, 100), 12), {'Person': 12})
        self.assertIsNone(out)

    def test_sparse_support_rejected(self):
        tags = np.zeros((60, 100), dtype=int)
        tags[30, 50:52] = 12
        self.assertIsNone(visible_annotation(self.candidate(), self.depth(), tags, {'Person': 12})[0])

    def test_camera_inside_or_crossing_box_rejected(self):
        self.assertIsNone(project_box(np.eye(4), [1, 1, 1], np.eye(4), 100, 60, 90))

    def test_world_box_not_double_transformed(self):
        world = np.eye(4); world[0, 3] = 110
        camera = np.eye(4); camera[0, 3] = 100
        self.assertAlmostEqual(project_box(world, [1, 1, 1], camera, 100, 60, 90)['distance_m'], 10)

    def test_invalid_geometry_and_sensor_alignment(self):
        ann = self.candidate(); ann['bbox_xyxy'][0] = float('nan')
        self.assertEqual(visible_annotation(ann, self.depth(), np.zeros((60, 100)), {'Person': 12})[1], 'invalid_geometry')
        with self.assertRaises(ValueError):
            visible_annotation(self.candidate(), self.depth(), np.zeros((1, 1)), {'Person': 12})

    def test_semantic_enum_not_hardcoded(self):
        from types import SimpleNamespace
        names = ['Pedestrians', 'TrafficSigns', 'TrafficLight', 'Car', 'Bus', 'Truck', 'Bicycle', 'Motorcycle']
        carla = SimpleNamespace(CityObjectLabel=SimpleNamespace(**dict(zip(names, range(40, 48)))))
        self.assertEqual(semantic_ids(carla)['Person'], 40)

    def test_resume_policy_rejects_legacy_and_mixed_records(self):
        from io import StringIO
        from unittest.mock import Mock
        import json
        path = Mock(); path.is_file.return_value = True
        for records in ([{}], [{'label_policy': POLICY}, {}]):
            path.open.return_value = StringIO('\n'.join(json.dumps(r) for r in records))
            with self.assertRaises(ValueError):
                require_matching_policy(path)
        path.open.return_value = StringIO(json.dumps({'label_policy': POLICY}))
        require_matching_policy(path)

    def test_invalid_depth_and_stop_trigger_rejected(self):
        self.assertIsNone(visible_annotation(self.candidate(), np.full((60, 100), np.nan),
                                             np.full((60, 100), 12), {'Person': 12})[0])
        ann = self.candidate('StopSign'); ann['type_id'] = 'traffic.stop'
        self.assertEqual(visible_annotation(ann, self.depth(), np.full((60, 100), 8),
                                            {'StopSign': 8})[1], 'invisible_stop_trigger')

    def test_unreviewed_pilot_cannot_enter_training_build(self):
        record = {'label_policy': POLICY}
        for manifest in ({}, {'label_policy': POLICY, 'label_review_status': 'pilot_unreviewed_not_training_ready'}):
            with self.assertRaises(ValueError):
                require_training_review(record, manifest)
        require_training_review(record, {'label_policy': POLICY, 'label_review_status': 'approved'})
        require_training_review({}, {})  # old dataset remains readable, not retroactively promoted

    def test_offline_pilot_review(self):
        import json
        import tempfile
        from pathlib import Path
        from PIL import Image
        from contextlib import redirect_stdout
        from io import StringIO
        from review_label_pilot import review
        with tempfile.TemporaryDirectory(prefix='adas-label-review-') as folder:
            root = Path(folder)/'source'
            (root/'rgb').mkdir(parents=True); (root/'road_line').mkdir()
            Image.new('RGB', (100, 60), 'white').save(root/'rgb/sample.jpg')
            Image.new('L', (100, 60), 255).save(root/'road_line/sample.png')
            record = {'sample_id': 'sample', 'frame_id': 42,
                      'objects': [{'class': 'Person', 'bbox_xyxy': [10, 10, 20, 30]}]}
            (root/'annotations.jsonl').write_text(json.dumps(record), encoding='utf-8')
            with redirect_stdout(StringIO()):
                review(root, Path(folder)/'report')
            result = json.loads((Path(folder)/'report/review.json').read_text())
            self.assertEqual(result['kept'], {'Person': 1})
            self.assertEqual(result['road_line_nonzero_pixels'], [6000])
            self.assertFalse(result['training_ready'])
            with Image.open(result['overlays'][0]['overlay']) as image:
                self.assertEqual(image.size, (100, 60))


if __name__ == '__main__':
    unittest.main()
