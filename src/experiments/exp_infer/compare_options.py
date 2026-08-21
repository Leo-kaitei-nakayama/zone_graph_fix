"""
Compare inference options head-to-head, stratified by ground-truth sequence
length. The aggregate table in summarize_results.py is dominated by 1-2 step
designs where the volumetric heuristic is near-optimal; the learned agent's
value, if any, shows on the longer sequences.

Usage:
    python compare_options.py --data_path ../../../data/fusion_processed --ids_file ../../test_ids.txt
"""

import argparse
import glob
import os


def read_float(path):
    with open(path) as f:
        return float(f.read().strip())


def gt_length(data_path, seq_id):
    steps = [d for d in glob.glob(os.path.join(data_path, seq_id, '*')) if os.path.basename(d).isdigit()]
    return len(steps)


def collect(option, ids):
    results = {}
    for seq_dir in glob.glob(os.path.join(option, '*')):
        seq_id = os.path.basename(seq_dir)
        if seq_id.startswith('.') or (ids is not None and seq_id not in ids):
            continue
        iou_file = os.path.join(seq_dir, 'IOU.txt')
        if not os.path.isfile(iou_file):
            continue
        results[seq_id] = {
            'iou': read_float(iou_file),
            'time': read_float(os.path.join(seq_dir, 'time.txt')),
            'steps': len(glob.glob(os.path.join(seq_dir, 'step_*_extrusion.stp'))),
        }
    return results


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def bucket_of(length):
    if length <= 2:
        return '1-2'
    if length <= 4:
        return '3-4'
    return '5+'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../../data/fusion_processed', type=str)
    parser.add_argument('--ids_file', default='', type=str)
    parser.add_argument('--options', default='random,heur,agent', type=str)
    parser.add_argument('--iou_threshold', default=0.999, type=float)
    args = parser.parse_args()

    ids = None
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = set(line.strip() for line in f if line.strip())

    options = [o for o in args.options.split(',') if os.path.isdir(o)]
    results = {o: collect(o, ids) for o in options}

    all_solved = set()
    for o in options:
        all_solved |= set(results[o])
    lengths = {s: gt_length(args.data_path, s) for s in all_solved}

    # per-bucket table
    buckets = ('1-2', '3-4', '5+')
    print('== by ground-truth sequence length ==')
    header = '%-7s %6s' % ('bucket', 'n_gt')
    for o in options:
        header += ' | %s: %6s %8s %8s %7s' % (o, 'solved', 'exact%', 'time', 'steps')
    print(header)
    for b in buckets:
        seq_in_bucket = [s for s in all_solved if bucket_of(lengths[s]) == b]
        row = '%-7s %6d' % (b, len(seq_in_bucket))
        for o in options:
            rows = [results[o][s] for s in seq_in_bucket if s in results[o]]
            exact = [r for r in rows if r['iou'] >= args.iou_threshold]
            pct = 100.0 * len(exact) / len(rows) if rows else 0.0
            row += ' | %s  %6d %7.0f%% %7.1fs %7.2f' % (
                ' ' * len(o), len(rows), pct, mean([r['time'] for r in rows]), mean([r['steps'] for r in rows]))
        print(row)

    # head-to-head on sequences both solved exactly
    if 'agent' in results and 'heur' in results:
        print()
        print('== agent vs heur, sequences both reconstruct exactly ==')
        both = [s for s in results['agent'] if s in results['heur']
                and results['agent'][s]['iou'] >= args.iou_threshold
                and results['heur'][s]['iou'] >= args.iou_threshold]
        for b in buckets:
            bs = [s for s in both if bucket_of(lengths[s]) == b]
            if not bs:
                print('%-4s: none' % b)
                continue
            agent_faster = sum(1 for s in bs if results['agent'][s]['time'] < results['heur'][s]['time'])
            agent_shorter = sum(1 for s in bs if results['agent'][s]['steps'] < results['heur'][s]['steps'])
            ties = sum(1 for s in bs if results['agent'][s]['steps'] == results['heur'][s]['steps'])
            print('%-4s: n=%d  agent faster: %d (%.0f%%)  agent fewer steps: %d, equal: %d, heur fewer: %d' % (
                b, len(bs), agent_faster, 100.0 * agent_faster / len(bs),
                agent_shorter, ties, len(bs) - agent_shorter - ties))

        only_agent = [s for s in results['agent'] if results['agent'][s]['iou'] >= args.iou_threshold
                      and (s not in results['heur'] or results['heur'][s]['iou'] < args.iou_threshold)]
        only_heur = [s for s in results['heur'] if results['heur'][s]['iou'] >= args.iou_threshold
                     and (s not in results['agent'] or results['agent'][s]['iou'] < args.iou_threshold)]
        print('exact only with agent: %d   exact only with heur: %d' % (len(only_agent), len(only_heur)))


if __name__ == '__main__':
    main()
