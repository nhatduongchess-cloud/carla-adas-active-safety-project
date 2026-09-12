"""Read-only audit of materialized YOLO data; write evidence outside the dataset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw

from build_yolo_adas_dataset import TAXONOMY, parse_adas_yolo


def group_name(stem):
    parts = stem.split('_')
    if stem.startswith('carla'):
        return '_'.join(parts[:-1]).split('_seed')[0]
    return parts[0]


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('audit output must be outside the dataset')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / 'dataset_manifest.json').read_text(encoding='utf-8'))
    issues, splits, digest_index, examples = [], {}, defaultdict(list), defaultdict(list)
    per_group = defaultdict(Counter)
    annotation_patterns = defaultdict(Counter)
    sample_index = {}
    for split in ('train', 'val', 'test'):
        counts, image_counts, sizes = Counter(), Counter(), Counter()
        wh = defaultdict(list)
        previous, adjacent = {}, Counter()
        dup_rows, empty = 0, 0
        labels = {p.stem: p for p in (root / split / 'labels').glob('*.txt')}
        images = sorted(p for p in (root / split / 'images').iterdir() if p.is_file())
        image_stems = set()
        for i, path in enumerate(images):
            if path.stem in image_stems:
                issues.append({'type': 'ambiguous_image_stem', 'path': str(path)})
            image_stems.add(path.stem)
            label = labels.get(path.stem)
            if label is None:
                issues.append({'type': 'missing_label', 'path': str(path)})
                continue
            rows = []
            text = label.read_text(encoding='utf-8')
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(parse_adas_yolo(line, label))
                except ValueError as exc:
                    issues.append({'type': 'invalid_label', 'path': str(label),
                                   'line': line_no, 'error': str(exc)})
            dup_rows += len(rows) - len(set(rows))
            if not rows:
                empty += 1
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    small = np.asarray(image.convert('L').resize((64, 36)), dtype=np.float32)
            except Exception as exc:
                issues.append({'type': 'image_decode', 'path': str(path), 'error': str(exc)})
                continue
            sizes[f'{width}x{height}'] += 1
            digest_index[hashlib.sha256(path.read_bytes()).hexdigest()].append((split, path.name))
            group = group_name(path.stem)
            per_group[(split, group)]['images'] += 1
            annotation_patterns[(split, group)][tuple(rows)] += 1
            if path.stem.startswith('carla') and group in previous:
                adjacent['pairs'] += 1
                # Low-resolution whole-scene similarity is a diagnostic proxy,
                # not proof of duplication or of absent object motion.
                if float(np.abs(small - previous[group]).mean()) <= 1.5:
                    adjacent['mae_le_1_5'] += 1
            previous[group] = small
            present = set()
            for cls, box in rows:
                name = TAXONOMY[cls]
                counts[name] += 1
                present.add(name)
                scale = 640.0 / max(width, height)
                wh[name].append((box[2] * width * scale, box[3] * height * scale))
                per_group[(split, group)][name] += 1
            for name in present:
                image_counts[name] += 1
                examples[(split, name)].append(path)
            sample_index[str(path)] = rows
            if i % 5000 == 0:
                print(f'{split}: {i}/{len(images)}', flush=True)
        for stem in labels.keys() - image_stems:
            issues.append({'type': 'orphan_label', 'path': str(labels[stem])})
        stats = {}
        for name in TAXONOMY:
            arr = np.asarray(wh[name], dtype=float).reshape(-1, 2)
            stats[name] = {'boxes': counts[name], 'images': image_counts[name]}
            if len(arr):
                stats[name].update({
                    'median_wh_at_640': np.median(arr, axis=0).round(2).tolist(),
                    'min_side_lt_8px_pct': round(float((arr.min(axis=1) < 8).mean() * 100), 2),
                    'area_lt_32_squared_pct': round(float((arr.prod(axis=1) < 1024).mean() * 100), 2),
                })
        splits[split] = {'images': len(images), 'empty_labels': empty,
                         'duplicate_label_rows': dup_rows, 'image_sizes': dict(sizes),
                         'classes': stats, 'adjacent_scene_similarity': dict(adjacent),
                         'manifest_counts_match': dict(counts) == manifest['class_counts_by_split'][split]}
    duplicates = [v for v in digest_index.values() if len(v) > 1]
    leakage = [v for v in duplicates if len({x[0] for x in v}) > 1]
    groups = [{ 'split': split, 'group': group, **counts,
                'unique_annotation_patterns': len(annotation_patterns[(split, group)]),
                'most_common_annotation_pattern_frames': annotation_patterns[(split, group)].most_common(1)[0][1]}
              for (split, group), counts in sorted(per_group.items())]
    panels = []
    rng = random.Random(42)
    for name in ('StopSign', 'TrafficLight', 'Person', 'Motorcycle'):
        panel = Image.new('RGB', (1500, 930), 'white')
        draw = ImageDraw.Draw(panel)
        selected = []
        for row_index, split in enumerate(('train', 'val', 'test')):
            candidates = examples[(split, name)]
            # Select from distinct capture/source groups where available.
            buckets = defaultdict(list)
            for p in candidates:
                buckets[group_name(p.stem)].append(p)
            group_keys = sorted(buckets)
            rng.shuffle(group_keys)
            choices = [rng.choice(buckets[key]) for key in group_keys[:3]]
            if len(choices) < 3:
                choices += rng.sample([p for p in candidates if p not in choices], min(3-len(choices), len(candidates)-len(choices)))
            for col, path in enumerate(choices):
                with Image.open(path) as source:
                    frame = source.convert('RGB')
                width, height = frame.size
                fd = ImageDraw.Draw(frame)
                for cls, (x, y, w, h) in sample_index[str(path)]:
                    if TAXONOMY[cls] == name:
                        fd.rectangle(((x-w/2)*width, (y-h/2)*height,
                                      (x+w/2)*width, (y+h/2)*height), outline='red', width=2)
                frame.thumbnail((496, 276))
                x0, y0 = col*500, row_index*310
                panel.paste(frame, (x0, y0+30))
                draw.text((x0+4, y0+2), f'{split} {name} {len(selected)+1}', fill='black')
                selected.append(str(path))
        panel_path = output / f'samples_{name}.jpg'
        panel.save(panel_path, quality=90)
        panels.append({'path': str(panel_path), 'samples': selected})
    report = {'dataset': str(root), 'scope': 'full materialized YOLO dataset; read-only',
              'splits': splits, 'groups': groups, 'issues': issues,
              'exact_duplicate_groups': len(duplicates), 'cross_split_exact_leakage': leakage,
              'duplicate_examples': duplicates[:10], 'panels': panels,
              'similarity_definition': 'adjacent filenames in same source/town/weather: mean absolute grayscale difference <=1.5 at 64x36; proxy only'}
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('panels', 'groups', 'duplicate_examples')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.dataset, args.output)
