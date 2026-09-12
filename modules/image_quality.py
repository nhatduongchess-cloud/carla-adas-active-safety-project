"""Deterministic image-quality triage, not an automatic label approval gate."""
import hashlib

import cv2
import numpy as np
from PIL import ImageOps


def image_features(image):
    rgb = ImageOps.exif_transpose(image).convert('RGB')
    width, height = rgb.size
    pixels = np.asarray(rgb)
    digest = hashlib.sha256(f'{width}x{height}:RGB:'.encode() + pixels.tobytes()).hexdigest()
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    # Never upsample and pretend to recover detail. Record original dimensions.
    scale = min(1., 640. / max(width, height))
    normalized = cv2.resize(gray, (max(1, round(width*scale)), max(1, round(height*scale))),
                            interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(thumb)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    bits[0] = False  # exclude global brightness/DC term
    phash = sum(int(bit) << i for i, bit in enumerate(bits))
    return {'width': width, 'height': height, 'pixel_sha256': digest,
            'phash': f'{phash:016x}', 'laplacian_variance_at_max640': round(sharpness, 4),
            'gray_mean': round(float(gray.mean()), 4),
            'dark_pixel_fraction': round(float((gray < 5).mean()), 5),
            'bright_pixel_fraction': round(float((gray > 250).mean()), 5)}


class NearDuplicateIndex:
    """Group against representatives using <=3/64 pHash bit differences.

    Four 16-bit bands guarantee at least one exact band for this radius. This
    is representative clustering, NOT an exhaustive all-pairs leakage audit.
    Perceptual collisions require visual review and must never auto-delete data.
    """
    def __init__(self, radius=3):
        if not 0 <= radius <= 3:
            raise ValueError('This index supports Hamming radius 0..3 only')
        self.radius = radius
        self.bands = [{} for _ in range(4)]
        self.representatives = []

    def add(self, signature, aspect_ratio):
        value = int(signature, 16)
        if not 0 <= value < 2**64 or not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError('Expected 64-bit signature and positive finite aspect ratio')
        candidates = set()
        for band in range(4):
            candidates.update(self.bands[band].get((value >> (16*band)) & 65535, []))
        for index in sorted(candidates):
            old, ratio = self.representatives[index]
            distance = (old ^ value).bit_count()
            if distance <= self.radius and abs(ratio/aspect_ratio - 1) <= 0.02:
                return index, distance, False
        index = len(self.representatives)
        self.representatives.append((value, aspect_ratio))
        for band in range(4):
            key = (value >> (16*band)) & 65535
            self.bands[band].setdefault(key, []).append(index)
        return index, 0, True
