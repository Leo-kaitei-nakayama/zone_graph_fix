# Zone Graph Reproduction Report

Reproduction of *Inferring CAD Modeling Sequences Using Zone Graphs* (Xu et
al., CVPR 2021) on the Fusion360 Gallery reconstruction subset (r1.0.0,
8,625 designs), using this repository's fixed pipeline.

## Verdict

- **Reproduced:** the core claim. The zone-graph representation plus guided
  search reconstructs essentially every representable design: 93% of loadable
  test designs reconstruct exactly (IOU >= 0.999) within a 300 s budget,
  median IOU 1.0. Blind random search degrades sharply with sequence length
  (89% / 32% / 6% exact for 1-2 / 3-4 / 5+ step designs), while guided search
  holds up - the framework, not luck, does the work.
- **Not reproduced:** the learned agent outperforming the hand-written
  heuristic. In this run the heuristic wins every difficulty bucket and the
  gap widens on long sequences (5+ steps: 57% vs 27% exact). See
  "Deviations" for suspected causes.

## Pipeline funnel

| stage | count | notes |
|---|---|---|
| raw designs | 8,625 | Fusion360 reconstruction subset |
| converted (`dataset_fusion.py`) | 8,625 (100%) | 0 hangs / 0 crashes with per-design workers |
| training preprocessing attempted | 8,623 | |
| fully processed for training | 4,638 (54%) | training requires replaying the exact GT sequence |
| rejected: zone graph build | 1,075 | `single_zone`, `target_mismatch`, ... |
| rejected: GT extrusion != zones | 649 | `extrusion_mismatch` |
| rejected: GT sequence not re-discoverable | 800 | proposal generator can't replay it |
| rejected: other (silent `extrusion_outside`, crashes, timeouts) | ~1,460 | |

Split: 85% train / 5% validate / 10% test over the 8,623 converted ids
(the split is made before filtering; filtered ids fail fast at load).

## Training

9 epochs, batch 32, lr 1e-4, CUDA (torch 2.2.1+cu121, dgl 2.1.0+cu121).
Focal-style loss fell 0.5 -> ~0.02. Epoch-end validation (rank-sum of the GT
extrusion among scored proposals, 100 sequences) was noisy - 347, 317, 381,
369, 419, 340, 441, 452, 353 - best at epoch 2; that checkpoint was used for
evaluation.

## Evaluation (test split, 862 ids, --max_time 300 --max_step 15)

Aggregate (denominator: sequences producing a solution; ~560 of 862 test ids
are loadable at all given the zone-graph validity filters):

| option | attempted | solved | mean IOU | median IOU | exact (IOU=1) | mean time | mean steps |
|---|---|---|---|---|---|---|---|
| random | 862 | 427 | 0.9669 | 1.0 | 78% | 34.8 s | 5.86 |
| heur | 862 | 559 | 0.9940 | 1.0 | **93%** | **10.2 s** | **1.54** |
| agent | 862 | 558 | 0.9862 | 1.0 | 88% | 13.6 s | 2.18 |

Stratified by ground-truth sequence length (exact-reconstruction rate):

| GT length | n | random | heur | agent |
|---|---|---|---|---|
| 1-2 | 460 | 89% | 97% | 95% |
| 3-4 | 76 | 32% | 80% | 68% |
| 5+ | 24 | 6% | 57% | 27% |

Head-to-head on designs both heur and agent reconstruct exactly: mostly step
ties on easy cases; heur finds shorter sequences more often in the 3-4 bucket
(24 vs 2, 22 ties). 32 designs reconstruct exactly only with heur, 7 only
with the agent.

## Deviations from the paper, and suspected causes for the agent gap

1. **Training budget.** 9 epochs, one negative sample per positive, and model
   selection by a noisy 100-sequence rank-sum that bottomed at epoch 2. The
   agent is likely under-trained relative to the paper's.
2. **Inference cost inside a fixed budget.** The agent encodes the full zone
   graph once per candidate extrusion at every expansion; that wall-clock
   comes out of the same 300 s the heuristic spends purely on search.
3. **Geometry kernel drift.** FreeCAD 1.1.x / OpenCascade 7.x (2026) vs the
   FreeCAD 0.18-era kernel of 2021 changes boolean-op behavior, zone counts,
   and which designs pass the validity filters - denominators are not
   directly comparable to the published tables.
4. **Released-code note.** `get_extrusion_heur_score` computes a volumetric
   IOU term and then explicitly zeroes it; the heuristic as shipped is pure
   zone-count matching. It is nonetheless very strong, especially on the
   short sequences that dominate the dataset.

## Reproducing this report

```
conda env create -f src/environment.yml     # FreeCAD + pinned torch/DGL
cd src
python dataset_fusion.py --fusion_path <raw .step dir> --extrusion_path <extrude tools dir> --output_path ../data/fusion_processed
python train_preprocess.py --data_path ../data/fusion_processed --output_path processed_data
python train.py --data_path processed_data --output_path train_output --validate_limit 100
cd experiments/exp_infer
for opt in agent heur random; do
  python infer.py --option $opt --data_path ../../../data/fusion_processed \
      --ids_file ../../test_ids.txt --max_time 300 --max_step 15 &
done
python summarize_results.py --ids_file ../../test_ids.txt
python compare_options.py --data_path ../../../data/fusion_processed --ids_file ../../test_ids.txt
```

All stages are parallel and resumable; see `src/README.md`.
