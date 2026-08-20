"""
Summarize inference results produced by infer.py.

For each option folder (random/heur/agent) present, reports over the evaluated
ids: how many produced a result, mean/median best IOU, the exact-reconstruction
rate (IOU >= threshold), mean search time to the best solution, and mean
sequence length.

Usage:
    python summarize_results.py                     # all ids found in the option folders
    python summarize_results.py --ids_file ../../test_ids.txt
"""

import argparse
import glob
import os


def read_float(path):
    with open(path) as f:
        return float(f.read().strip())


def collect(option, ids=None):
    rows = []
    for seq_dir in sorted(glob.glob(os.path.join(option, '*'))):
        seq_id = os.path.basename(seq_dir)
        if seq_id.startswith('.'):
            continue
        if ids is not None and seq_id not in ids:
            continue
        iou_file = os.path.join(seq_dir, 'IOU.txt')
        if not os.path.isfile(iou_file):
            continue
        rows.append({
            'seq_id': seq_id,
            'iou': read_float(iou_file),
            'time': read_float(os.path.join(seq_dir, 'time.txt')),
            'steps': len(glob.glob(os.path.join(seq_dir, 'step_*_extrusion.stp'))),
        })
    return rows


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids_file', default='', type=str,
                        help='restrict the summary to ids listed in this file')
    parser.add_argument('--iou_threshold', default=0.999, type=float,
                        help='IOU at or above this counts as exact reconstruction')
    args = parser.parse_args()

    ids = None
    attempted = None
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = set(line.strip() for line in f if line.strip())

    print('%-8s %9s %10s %9s %9s %11s %10s %9s' % (
        'option', 'attempted', 'solved', 'meanIOU', 'medIOU',
        'exact(IOU=1)', 'meanTime', 'meanSteps'))
    for option in ('random', 'heur', 'agent'):
        if not os.path.isdir(option):
            continue
        markers = os.path.join(option, '.attempted')
        if os.path.isdir(markers):
            marked = set(m for m in os.listdir(markers))
            attempted = len(marked if ids is None else marked & ids)
        else:
            attempted = '-'
        rows = collect(option, ids)
        ious = [r['iou'] for r in rows]
        exact = [r for r in rows if r['iou'] >= args.iou_threshold]
        print('%-8s %9s %10d %9.4f %9.4f %11s %9.1fs %9.2f' % (
            option, attempted, len(rows), mean(ious), median(ious),
            '%d (%.0f%%)' % (len(exact), 100 * len(exact) / len(rows)) if rows else '0',
            mean([r['time'] for r in rows]),
            mean([r['steps'] for r in rows])))


if __name__ == '__main__':
    main()
