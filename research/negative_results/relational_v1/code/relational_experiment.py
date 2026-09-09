"""Cache-only EXP2 exploration; reuse historical accuracy without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .metrics import softmax
from .relational import projected_cka, relational_score, exact_cosine_relational_score
from .stats import pearsonr, kendall_tau_b


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tasks', default='2,5,10,20,50,100')
    parser.add_argument('--draws', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260906)
    parser.add_argument('--exact-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = list(range(2,101)) if args.tasks == 'all' else [int(k) for k in args.tasks.split(',')]
    cache = args.root / 'results/intermediate/exp2_logits/published'
    reference_root = cache / 'ResNet34/ImageNet'
    reference_manifest = load(reference_root / 'manifest.json')
    reference = np.load(reference_root / 'features.npy')
    reference_labels = np.load(reference_root / 'labels.npy')
    report = {'status': 'exploratory', 'reference': 'ResNet34/ImageNet',
              'reference_warning': 'Reference model itself excluded from primary comparisons; shared extraction order assumed from cache pipeline.',
              'draws': args.draws, 'seed': args.seed, 'records': [], 'correlations': [],
              'accuracy_policy': 'Each legacy accuracy branch retained separately; all new metrics use same valid keys within branch.',
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ['relational.py', 'relational_experiment.py']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(args.output)
    for model in ['ResNet18', 'ResNet34']:
        for source in ['CIFAR10', 'ImageNet']:
            root = cache / model / source
            manifest = load(root / 'manifest.json')
            labels = np.load(root / 'labels.npy')
            if not np.array_equal(labels, reference_labels) or manifest['task_classes'] != reference_manifest['task_classes']:
                raise ValueError('Reference/candidate cache alignment mismatch')
            logits = np.load(root / 'prediction_logits.npy', mmap_mode='r')
            features = np.load(root / 'features.npy', mmap_mode='r')
            group = []
            for k in tasks:
                # Labels are used only to reconstruct the existing predefined benchmark subset.
                classes = manifest['task_classes'][f'{k:03d}']
                ix = np.flatnonzero(np.isin(labels, classes))
                p = softmax(logits[ix])
                tick = perf_counter()
                scores = exact_cosine_relational_score(p, reference[ix]) if args.exact_only else relational_score(p, reference[ix], draws=args.draws, seed=args.seed)
                elapsed = perf_counter() - tick
                h = -(p * np.log(np.clip(p, 1e-300, None))).sum(1).mean()
                marginal = p.mean(0)
                scores.update({'negative_entropy': float(-h),
                               'mutual_information': float(-(marginal * np.log(np.clip(marginal, 1e-300, None))).sum() - h),
                               'projected_cka32': projected_cka(features[ix], reference[ix]),
                               'minus_class_count': -k, 'inverse_class_count': 1/k})
                row = {'model': model, 'source': source, 'classes': k, 'n': len(ix),
                       'reference_self': model == 'ResNet34' and source == 'ImageNet',
                       'prediction_rule': manifest['prediction_rule'], 'rpt_seconds': elapsed, 'scores': scores}
                group.append(row); report['records'].append(row); save()
                print(model, source, k, scores.get('rpt', scores.get('rpt_cosine_exact')), elapsed, flush=True)
            for strategy in ['Finetune', 'Retrain']:
                for accuracy_metric in ['LEEP', 'GLEEP']:
                    path = args.root / 'historical/EXP2/result' / strategy / model / source / 'test_ACC' / (accuracy_metric + '_ACC.json')
                    if not path.exists():
                        continue
                    accuracy = {str(int(key)): value for key, value in load(path).items()}
                    # Legacy files normally use task keys such as "2"; fail loudly if none align.
                    paired = [row for row in group if str(row['classes']) in accuracy]
                    if not paired:
                        raise ValueError(f'No accuracy keys align: {path}, examples={list(accuracy)[:3]}')
                    metric_names = ['rpt_cosine_exact', 'relation_cosine_raw'] if args.exact_only else ['rpt', 'relation_raw']
                    for metric in metric_names + ['negative_entropy', 'mutual_information', 'projected_cka32', 'minus_class_count', 'inverse_class_count']:
                        valid = [row for row in paired if row['scores'][metric] is not None]
                        if len(valid) < 3:
                            continue
                        x = [row['scores'][metric] for row in valid]
                        y = [accuracy[str(row['classes'])] for row in valid]
                        report['correlations'].append({'model': model, 'source': source, 'strategy': strategy,
                            'accuracy_metric': accuracy_metric, 'metric': metric, 'pairs': len(valid),
                            'pearson': pearsonr(x,y), 'kendall': kendall_tau_b(x,y), 'accuracy_path': str(path)})
            save()
    report['status'] = 'complete_exploratory'; save()


if __name__ == '__main__':
    main()
