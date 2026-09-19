"""Read-only full-dataset quality/near-duplicate scan. Never exports training data."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.dataset.audit_training_dataset import group_name
from modules.image_quality import image_features, NearDuplicateIndex


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError('Output must be outside dataset; source is read-only')
    inputs = []
    for split in ('train', 'val', 'test'):
        folder = root / split / 'images'
        if not folder.is_dir():
            raise ValueError(f'Missing image split: {folder}')
        inputs.extend((split, p) for p in sorted(folder.iterdir()) if p.is_file())
    if not inputs:
        raise ValueError('Dataset has no images')
    output.mkdir(parents=True, exist_ok=False)
    rows, issues, clusters = [], [], defaultdict(list)
    pixels, index = defaultdict(list), NearDuplicateIndex()
    by_group = defaultdict(list)
    with (output / 'images.jsonl').open('x', encoding='utf-8') as stream:
        for sequence, (split, path) in enumerate(inputs):
            try:
                with Image.open(path) as image:
                    image.load()
                    features = image_features(image)
            except Exception as exc:
                issues.append({'path': str(path), 'error': str(exc)})
                continue
            group, distance, _ = index.add(features['phash'], features['width']/features['height'])
            row = {'path': str(path), 'split': split, 'source_group': group_name(path.stem),
                   **features, 'near_group': group, 'distance_to_representative': distance}
            stream.write(json.dumps(row) + '\n')
            row_id = len(rows)
            rows.append(row)
            clusters[group].append(row_id)
            pixels[features['pixel_sha256']].append(row_id)
            by_group[(split, row['source_group'])].append(row_id)
            if sequence % 2000 == 0:
                stream.flush()
                print(f'[quality] {sequence}/{len(inputs)} scanned', flush=True)
    exact_groups = [ids for ids in pixels.values() if len(ids) > 1]
    near_groups = [ids for ids in clusters.values() if len(ids) > 1]
    cross = [ids for ids in near_groups if len({rows[i]['split'] for i in ids}) > 1]
    group_stats = []
    for (split, group), ids in sorted(by_group.items()):
        scores = np.asarray([rows[i]['laplacian_variance_at_max640'] for i in ids])
        group_stats.append({'split': split, 'source_group': group, 'images': len(ids),
                            'sharpness_p10_p50_p90': np.percentile(scores, [10, 50, 90]).round(3).tolist(),
                            'below_30_triage_only': int((scores < 30).sum()),
                            'lowest_score_samples': [rows[i]['path'] for i in sorted(ids, key=lambda i: rows[i]['laplacian_variance_at_max640'])[:3]]})
    # A bounded visual review queue, including endpoints of large clusters.
    pairs = []
    for ids in sorted(cross, key=len, reverse=True)[:6] + sorted(near_groups, key=len, reverse=True)[:6]:
        left = ids[0]
        right = next((i for i in reversed(ids) if rows[i]['split'] != rows[left]['split']), ids[-1])
        if (left, right) not in pairs:
            pairs.append((left, right))
    panels = []
    for page, start in enumerate(range(0, len(pairs), 3)):
        panel = Image.new('RGB', (1000, 3*320), 'white')
        draw = ImageDraw.Draw(panel)
        selected = []
        for rownum, pair in enumerate(pairs[start:start+3]):
            for col, i in enumerate(pair):
                record = rows[i]
                with Image.open(record['path']) as source:
                    frame = source.convert('RGB')
                frame.thumbnail((490, 280))
                x, y = col*500, rownum*320
                panel.paste(frame, (x, y+30))
                draw.text((x+3, y+3), f"{record['split']} near_group={record['near_group']} sharp={record['laplacian_variance_at_max640']:.1f}", fill='black')
                selected.append(record['path'])
        dest = output / f'near_review_{page:02d}.jpg'
        panel.save(dest, quality=92)
        panels.append({'path': str(dest), 'sources': selected})
    report = {'dataset': str(root), 'images_scanned': len(rows), 'files_expected': len(inputs),
              'by_split': dict(Counter(r['split'] for r in rows)), 'decode_issues': issues,
              'decoded_pixel_exact_groups': len(exact_groups),
              'decoded_pixel_exact_excess_images': sum(len(ids)-1 for ids in exact_groups),
              'decoded_pixel_cross_split_groups': sum(len({rows[i]['split'] for i in ids}) > 1 for ids in exact_groups),
              'near_duplicate_candidate_groups': len(near_groups),
              'near_duplicate_candidate_excess_images': sum(len(ids)-1 for ids in near_groups),
              'cross_split_near_candidate_groups': len(cross),
              'method': '64-bit pHash, representative groups radius<=3 and aspect ratio tolerance 2%; triage, not proof of duplication',
              'sharpness_method': 'Laplacian variance, downsample only to max640; <30 is review flag, NOT rejection/acceptance',
              'group_stats': group_stats, 'panels': panels, 'training_ready': False,
              'blockers': ['visual_label_review_required', 'near_duplicate_review_required',
                           'per_class_scale_weather_route_coverage_required', 'capture_V3_not_simulator_validated']}
    (output / 'groups.json').write_text(json.dumps({
        'exact': [[rows[i]['path'] for i in ids] for ids in exact_groups],
        'near_candidates': [[rows[i]['path'] for i in ids] for ids in near_groups]}, indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('group_stats', 'panels')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.dataset, args.output)
