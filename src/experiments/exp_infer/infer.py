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
        agent.load_weights()
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

def infer_all(sort_option, data_path, max_time, max_step):

    all_ids = os.listdir(data_path)

    for seq_id in all_ids:

        worker_process = multiprocessing.Process(target=infer, name="infer", args=(seq_id, sort_option, data_path, max_time, max_step, ))
        worker_process.start()
        worker_process.join(max_time+100)
        
        if worker_process.is_alive():
            print ("process_single_data is running... let's kill it...")
            worker_process.terminate()
            worker_process.join()


processed_data_folder = "../processed_files/processed_data/"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', default='heur', type=str, help='infer option')
    parser.add_argument('--data_path', default='../../../data/fusion_processed', type=str)
    parser.add_argument('--max_time', default=300, type=float)
    parser.add_argument('--max_step', default=15, type=int)
    args = parser.parse_args()
    infer_all(args.option, args.data_path, args.max_time, args.max_step)

    


