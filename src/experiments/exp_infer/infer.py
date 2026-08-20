import sys
from os.path import dirname, realpath
sys.path.append('../..')

import numpy as np
import os
import argparse
import numpy as np

from dataset import *
from search import *
from utils.file_utils import *
import time
import shutil
import multiprocessing

# Agent is imported lazily below: it pulls in torch and DGL, which the random
# and heuristic options do not need.

# from utils.vis_utils import *

def infer(seq_id, sort_option, data_path, max_time, max_step):

    if sort_option not in ('random', 'heur', 'agent'):
        raise ValueError("invalid --option %r, expected one of random/heur/agent" % sort_option)

    folder = str(sort_option)
    agent = None
    if sort_option == 'agent':
        from agent import Agent

        agent = Agent('../../train_output')
        # evaluate with the checkpoint that had the best validation loss
        agent.load_best_weights()
        agent.eval()

    if not os.path.exists(folder):
        os.makedirs(folder)

    print('infer----------------------------------------', seq_id)

    data_mgr = DataManager()
    start_time = time.time()
    sequence_length = len(list(Path(os.path.join(data_path, seq_id)).glob('*')))
    gt_seq, error_type = data_mgr.load_raw_sequence(os.path.join(data_path, seq_id), 0, sequence_length)
     
    if len(gt_seq) == 0:
        return

    start_zone_graph = gt_seq[0][0]
    expand_width = 15

    probablistical = False
    use_concurrent = False
    
    best_sol=SearchSolution()
    dfs_best_recon(start_zone_graph, max_step, max_time, expand_width, sort_option, best_sol, start_time, os.path.join(folder, seq_id), agent)

def marker_path(folder, seq_id):
    return os.path.join(folder, '.attempted', seq_id)


def mark_done(folder, seq_id):
    with open(marker_path(folder, seq_id), 'w') as f:
        f.write('done\n')


def infer_all(sort_option, data_path, max_time, max_step, ids_file=None, num_workers=1):
    """
    Run the search over sequences, num_workers at a time, each in its own
    process with a max_time + 100 second deadline.

    ids_file restricts the run to the ids listed in that file (e.g. the
    test_ids.txt written by train_preprocess.py) - without it, every sequence
    in data_path is evaluated. Attempted sequences are recorded under
    <option>/.attempted, so an interrupted run resumes when re-run.

    num_workers defaults to 1 because search time is part of what the paper
    measures: concurrent searches contend for cores and inflate per-sequence
    times. Raise it for faster-but-approximate timing.
    """
    available = set(s for s in os.listdir(data_path) if not s.startswith('.'))
    if ids_file:
        listed = read_file_to_list(ids_file)
        all_ids = [s for s in listed if s in available]
        print(len(all_ids), 'of', len(listed), 'listed ids present in', data_path)
    else:
        all_ids = sorted(available)

    folder = str(sort_option)
    os.makedirs(os.path.join(folder, '.attempted'), exist_ok=True)

    pending = [s for s in all_ids if not os.path.exists(marker_path(folder, s))]
    skipped = len(all_ids) - len(pending)
    if skipped > 0:
        print('resume: skipping', skipped, 'already attempted sequences')
    total = len(pending)
    print('evaluating', total, 'sequences with', num_workers, 'workers')

    running = []  # (process, deadline, seq_id)
    done_count = 0
    while pending or running:
        while pending and len(running) < num_workers:
            seq_id = pending.pop(0)
            worker = multiprocessing.Process(target=infer, name="infer", args=(seq_id, sort_option, data_path, max_time, max_step, ))
            worker.start()
            running.append((worker, time.time() + max_time + 100, seq_id))

        time.sleep(1)

        still_running = []
        for worker, deadline, seq_id in running:
            if not worker.is_alive():
                worker.join()
            elif time.time() > deadline:
                print('sequence', seq_id, 'timed out, killing worker')
                worker.terminate()
                worker.join()
            else:
                still_running.append((worker, deadline, seq_id))
                continue
            mark_done(folder, seq_id)
            done_count += 1
            print('progress:', done_count, '/', total, 'sequences attempted')
        running = still_running

    print('evaluation complete !')


processed_data_folder = "../processed_files/processed_data/"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='heur', type=str, help='infer option')
    parser.add_argument('--data_path', default='../../../data/fusion_processed', type=str)
    parser.add_argument('--max_time', default=300, type=float)
    parser.add_argument('--max_step', default=15, type=int)
    parser.add_argument('--ids_file', default='', type=str,
                        help='only evaluate ids listed in this file (e.g. ../../test_ids.txt)')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='sequences evaluated concurrently (1 = faithful timing)')
    args = parser.parse_args()
    infer_all(args.option, args.data_path, args.max_time, args.max_step, args.ids_file or None, args.num_workers)

    


