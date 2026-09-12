"""Pure CPU synthetic tests; no files, server, downloads or training."""
import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from modules.image_quality import image_features, NearDuplicateIndex


class QualityTests(unittest.TestCase):
    def image(self):
        rng = np.random.default_rng(42)
        return Image.fromarray(rng.integers(0, 256, size=(128, 256, 3), dtype=np.uint8))

    def test_identical_pixels_different_png_compression(self):
        image = self.image()
        decoded = []
        for level in (0, 9):
            buffer = BytesIO(); image.save(buffer, format='PNG', compress_level=level)
            buffer.seek(0)
            with Image.open(buffer) as other:
                decoded.append(image_features(other))
        self.assertEqual(decoded[0], decoded[1])

    def test_blur_reduces_detail_score_not_auto_rejection(self):
        image = self.image()
        clear = image_features(image)
        blurred = image_features(image.filter(ImageFilter.GaussianBlur(3)))
        self.assertGreater(clear['laplacian_variance_at_max640'], blurred['laplacian_variance_at_max640'])
        self.assertNotIn('approved', blurred)

    def test_resize_preserves_size_provenance(self):
        result = image_features(self.image())
        self.assertEqual((result['width'], result['height']), (256, 128))

    def test_near_hamming_and_aspect_constraints(self):
        index = NearDuplicateIndex()
        self.assertEqual(index.add('0000000000000000', 2), (0, 0, True))
        self.assertEqual(index.add('0001000100010000', 2), (0, 3, False))
        self.assertEqual(index.add('0001000100010001', 2), (1, 0, True))
        self.assertEqual(index.add('0000000000000000', 1), (2, 0, True))

    def test_banding_matches_exhaustive_representative_search(self):
        rng = np.random.default_rng(12)
        index = NearDuplicateIndex()
        for _ in range(100):
            number = int(rng.integers(0, 2**63))
            index.add(f'{number:016x}', 1.5)
        for value, ratio in list(index.representatives):
            query = value ^ (1 << 6) ^ (1 << 28) ^ (1 << 44)
            expected = next(i for i, (old, _) in enumerate(index.representatives) if (old ^ query).bit_count() <= 3)
            self.assertEqual(index.add(f'{query:016x}', ratio)[0], expected)

    def test_invalid_hash_parameters(self):
        with self.assertRaises(ValueError):
            NearDuplicateIndex(4)
        index = NearDuplicateIndex()
        for value, ratio in (('-1', 1), ('10000000000000000', 1), ('0', float('nan'))):
            with self.assertRaises(ValueError):
                index.add(value, ratio)


if __name__ == '__main__':
    unittest.main()
