"""Small, offline regression tests for the read-only training-data audit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.dataset.audit_training_dataset import audit, group_name


class DatasetAuditTests(unittest.TestCase):
    def fixture(self, parent):
        root = parent / 'dataset'
        counts = {}
        for split in ('train', 'val', 'test'):
            images = root / split / 'images'
            labels = root / split / 'labels'
            images.mkdir(parents=True)
            labels.mkdir()
            Image.new('RGB', (100, 50), 'white').save(images / 'sample.png')
            (labels / 'sample.txt').write_text('7 0.5 0.5 0.1 0.2\n', encoding='utf-8')
            counts[split] = {'StopSign': 1}
        (root / 'dataset_manifest.json').write_text(
            json.dumps({'class_counts_by_split': counts}), encoding='utf-8')
        return root

    def run_audit(self, root, output):
        with contextlib.redirect_stdout(io.StringIO()):
            audit(root, output)
        return json.loads((output / 'audit.json').read_text(encoding='utf-8'))

    def test_counts_geometry_leakage_and_input_preservation(self):
        with tempfile.TemporaryDirectory(prefix='adas-audit-test-') as folder:
            parent = Path(folder)
            root = self.fixture(parent)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result = self.run_audit(root, parent / 'report')
            self.assertEqual(result['issues'], [])
            self.assertEqual(len(result['cross_split_exact_leakage']), 1)
            self.assertEqual(result['splits']['train']['classes']['StopSign']['median_wh_at_640'], [64., 64.])
            self.assertTrue(all(s['manifest_counts_match'] for s in result['splits'].values()))
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_invalid_missing_orphan_and_corrupt_inputs(self):
        with tempfile.TemporaryDirectory(prefix='adas-audit-test-') as folder:
            parent = Path(folder)
            root = self.fixture(parent)
            (root / 'train/labels/sample.txt').write_text('7 nan 0.5 0.1 0.2\n', encoding='utf-8')
            (root / 'val/labels/sample.txt').rename(root / 'val/labels/orphan.txt')
            (root / 'test/images/sample.png').write_bytes(b'not an image')
            result = self.run_audit(root, parent / 'report')
            self.assertEqual({i['type'] for i in result['issues']},
                             {'invalid_label', 'missing_label', 'orphan_label', 'image_decode'})

    def test_reject_dataset_output_and_existing_report(self):
        with tempfile.TemporaryDirectory(prefix='adas-audit-test-') as folder:
            parent = Path(folder)
            root = self.fixture(parent)
            for output in (root, root / 'audit'):
                with self.assertRaises(ValueError):
                    audit(root, output)
            with self.assertRaises(FileExistsError):
                audit(root, parent)

    def test_capture_groups_preserve_town_and_weather(self):
        self.assertEqual(group_name('carla1_test_Town10HD_Opt_heavy_rain_seed00042_000007'),
                         'carla1_test_Town10HD_Opt_heavy_rain')


if __name__ == '__main__':
    unittest.main()
