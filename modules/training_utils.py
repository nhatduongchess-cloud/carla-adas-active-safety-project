"""Small dependency-free helpers shared by training and self-tests."""

from pathlib import Path
import random


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_image_paths(split_source):
    """Resolve an Ultralytics split directory/list into unique image paths."""
    sources = split_source if isinstance(split_source, (list, tuple)) else [split_source]
    images = []
    for source in sources:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir():
            images.extend(
                path.resolve() for path in source_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            continue
        if source_path.is_file() and source_path.suffix.lower() == ".txt":
            for raw_line in source_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                image_path = Path(line).expanduser()
                if not image_path.is_absolute():
                    image_path = source_path.parent / image_path
                image_path = image_path.resolve()
                if image_path.suffix.lower() in IMAGE_SUFFIXES and image_path.is_file():
                    images.append(image_path)
            continue
        if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source_path)
            continue
        raise FileNotFoundError(f"unsupported or missing image split source: {source_path}")
    unique = sorted(set(images), key=lambda path: path.as_posix().lower())
    if not unique:
        raise ValueError("image split resolved to zero images")
    return unique


def deterministic_subset(items, fraction, seed):
    """Return a stable, sorted sample while always retaining at least one item."""
    ordered = sorted(items, key=lambda item: str(item).lower())
    if not ordered:
        raise ValueError("cannot sample an empty collection")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1.0:
        return ordered
    sample_size = max(1, min(len(ordered), int(len(ordered) * fraction + 0.5)))
    selected = random.Random(seed).sample(ordered, sample_size)
    return sorted(selected, key=lambda item: str(item).lower())
