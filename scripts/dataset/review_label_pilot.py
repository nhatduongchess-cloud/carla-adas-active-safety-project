"""Generate reproducible overlays and counters from a small V3 capture."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def review(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError('report must be outside the dataset')
    output.mkdir(parents=True, exist_ok=False)
    kept, rejected = Counter(), Counter()
    frames, selected, road_pixels = 0, [], []
    for annotations in sorted(root.rglob('annotations.jsonl')):
        rows = [json.loads(line) for line in annotations.read_text(encoding='utf-8').splitlines() if line.strip()]
        frames += len(rows)
        for row in rows:
            kept.update(a['class'] for a in row['objects'])
            rejected.update(a['class']+':'+a['rejection_reason'] for a in row.get('rejected_objects', []))
            with Image.open(annotations.parent / 'road_line' / (row['sample_id']+'.png')) as mask:
                road_pixels.append(int(np.count_nonzero(np.asarray(mask))))
        for index in sorted({0, len(rows)//2, len(rows)-1}) if rows else []:
            row = rows[index]
            path = annotations.parent / 'rgb' / (row['sample_id']+'.jpg')
            with Image.open(path) as image:
                canvas = image.convert('RGB')
            draw = ImageDraw.Draw(canvas)
            for ann in row.get('rejected_objects', []):
                draw.rectangle(ann['bbox_xyxy'], outline='orange', width=1)
            for ann in row['objects']:
                box = ann['bbox_xyxy']
                draw.rectangle(box, outline='lime', width=2)
                draw.text((box[0], max(0, box[1]-12)), ann['class'], fill='lime')
            dest = output / f'overlay_{len(selected):02d}.jpg'
            canvas.save(dest, quality=92)
            selected.append({'source': str(path), 'overlay': str(dest), 'frame_id': row['frame_id']})
    report = {'dataset': str(root), 'frames': frames, 'kept': dict(kept),
              'rejected': dict(rejected), 'road_line_nonzero_pixels': road_pixels,
              'overlays': selected, 'legend': 'green=kept, orange=rejected projected candidate',
              'training_ready': False}
    (output / 'review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review(args.dataset, args.output)
