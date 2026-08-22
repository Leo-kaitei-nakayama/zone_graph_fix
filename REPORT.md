# Zone Graph Reproduction Report

Reproduction of *Inferring CAD Modeling Sequences Using Zone Graphs* (Xu et
al., CVPR 2021, arXiv 2104.03900) on the Fusion360 Gallery reconstruction
subset (r1.0.0, 8,625 designs), using this repository's fixed pipeline.

## Verdict

Checked against the paper's actual claims (main text + supplement), this run
**reproduces three of the paper's four measurable claim groups** - including
one that is easy to misread as a failure - and documents one claim that does
not reproduce with the released training recipe (the network's Fig 6 ranking
quality; details below):

1. **Dataset funnel matches.** Paper: zone graphs constructible for 6,900 /
   8,625 models (80%); GT modeling sequence representable for 5,175 / 8,625
   (60%). This run: ~65% of test ids loadable for search; 54% survive the
   stricter training filter (which additionally requires GT replay through the
   proposal generator within timeouts). A few points of drift over five years
   of OpenCascade changes.
2. **Reconstruction matches.** Guided search reconstructs 93% (heur) / 88%
   (agent) of loadable test designs exactly within 300 s, median IOU 1.0;
   random guidance degrades sharply with sequence length (89/32/6% exact for
   1-2/3-4/5+ steps) and does not converge - the paper's Figure 8 behavior.
3. **Heuristic-vs-network ordering matches the paper.** The paper states it
   directly (Sec 6.2): "Heuristic guidance is faster than the network because
   (a) it does not incur the cost of network evaluation, and (b) by
   construction it tries to use the fewest possible modeling operations" -
   and its Table 1 bolds Ours Heur over Ours Net (0.50% vs 0.72% median
   error, 197 s vs 242 s). Our tables show the same ordering for the same
   reasons, including heur finding shorter sequences.

The network's advantages claimed by the paper are elsewhere: a ~2x better
ranking of the GT extrusion (Fig 6), and CAD designers preferring its
sequences in a perceptual study (Table 2; requires human raters, out of
scope here).

**The one claim that did not reproduce: Fig 6's ranking quality.** Measured
with `src/rank_eval.py` (150 test sequences, 238 ranked steps):

| method | paper (Fig 6) | 9-epoch checkpoint | 40-epoch checkpoint |
|---|---|---|---|
| random | 0.486 | 0.374 | - |
| heur | 0.070 | **0.059** | - |
| network | **0.036** | 0.107 | 0.150 |

The heuristic matches the paper closely; the network does not, and more
training makes it worse, not better. In the 40-epoch run the validation
rank-sum wandered noisily (best at epoch 28), and the final six epochs
produced bit-identical rank-sums - the signature of the network collapsing
to constant (or NaN) outputs. Conclusion: the released training recipe
(fixed lr 1e-4, one negative per positive, focal loss, no early-stopping
criterion on ranking) does not reach the paper's reported ranking quality on
this data, and destabilizes when run longer. Whatever schedule produced the
paper's 0.036 is not in the released code or the paper text. This also
explains the wider-than-expected agent-vs-heur gap on 5+ step designs in the
search evaluation: search quality is downstream of ranking quality.

## Pipeline funnel

| stage | count | notes |
|---|---|---|
| raw designs | 8,625 | Fusion360 reconstruction subset |
| converted (`dataset_fusion.py`) | 8,625 (100%) | 0 hangs / 0 crashes with per-design workers |
| training preprocessing attempted | 8,623 | |
| fully processed for training | 4,638 (54%) | paper's comparable figure: 5,175 (60%) |
| rejected: zone graph build | 1,075 | `single_zone`, `target_mismatch`, ... |
| rejected: GT extrusion != zones | 649 | `extrusion_mismatch` |
| rejected: GT sequence not re-discoverable | 800 | matches paper's "sequences not captured by our extrusion proposals" |
| rejected: other (silent `extrusion_outside`, crashes, timeouts) | ~1,460 | |

Split: 85% train / 5% validate / 10% test over the 8,623 converted ids (the
split is made before filtering; filtered ids fail fast at load). The paper
used 3,000 training and 440 test sequences from its 5,175 representable ones,
so this run trains on more data (~4,600) and tests on more (862 ids, ~560
loadable).

## Training

Exactly the released recipe: 9 epochs, batch 32, Adam lr 1e-4, focal-style
loss, one Monte-Carlo-selected negative per positive - the released
implementation of the paper's ternary positive/negative/neutral labeling
scheme (Sec 5.2). **Neither the paper nor its supplement states an epoch
count or optimizer settings; `hyperparameters.py` (9 epochs) is the only
documented configuration and is what this run used.** Loss fell 0.5 -> ~0.02
on CUDA (torch 2.2.1+cu121, dgl 2.1.0+cu121). Epoch-end validation (rank-sum
over 100 sequences) was noisy - best at epoch 2; that checkpoint was used for
evaluation. The network architecture matches the supplement's Tables 3-5
(PointNet encoder, 3 message-passing rounds, max-pool, MLP).

## Evaluation (test split, 862 ids, --max_time 300 --max_step 15)

Aggregate (denominator: sequences producing a solution; ~560 of 862 test ids
are loadable given the zone-graph validity filters). Random's row was
captured at 645/862 attempted:

| option | attempted | solved | mean IOU | median IOU | exact (IOU=1) | mean time | mean steps |
|---|---|---|---|---|---|---|---|
| random | 645 | 427 | 0.9669 | 1.0 | 78% | 34.8 s | 5.86 |
| heur | 862 | 559 | 0.9940 | 1.0 | **93%** | **10.2 s** | **1.54** |
| agent | 862 | 558 | 0.9862 | 1.0 | 88% | 13.6 s | 2.18 |

Stratified by ground-truth sequence length (exact-reconstruction rate):

| GT length | n | random | heur | agent |
|---|---|---|---|---|
| 1-2 | 460 | 89% | 97% | 95% |
| 3-4 | 76 | 32% | 80% | 68% |
| 5+ | 24 | 6% | 57% | 27% |

Head-to-head on designs both heur and agent reconstruct exactly: mostly step
ties on easy cases; heur finds shorter sequences more often (consistent with
the paper's observation that the heuristic minimizes operation count by
construction). 32 designs reconstruct exactly only with heur, 7 only with the
agent.

## Differences from the paper's setup, and the remaining agent gap

1. **Search width.** The paper uses k = 5 (and k = 15 with decay only for the
   harder InverseCSG shapes); the released `infer.py` ships expand_width = 15
   with decay, which is what this run used. A narrower beam leans harder on
   ranking quality and is the paper's actual operating point.
2. **Heuristic tie-breaking.** The paper's heuristic breaks zone-count ties
   with volumetric IoU; the released `get_extrusion_heur_score` computes that
   term and then zeroes it. Our heur is therefore slightly weaker than the
   paper's - and still fastest, strengthening rather than weakening the
   reproduction of the paper's heur-vs-net ordering.
3. **Network evaluation cost** inside the fixed budget - confirmed by the
   paper itself as the reason heur is faster.
4. **Geometry kernel drift.** FreeCAD 1.1.x / OpenCascade 7.x (2026) vs the
   2021-era kernel shifts zone counts and filter outcomes; denominators are
   close (54-65% vs 60-80%) but not identical.
5. **Training recipe limitations (measured).** A 40-epoch retrain with a
   300-sequence validation (--epochs 40 --validate_limit 300 --seed 0) was
   run to test whether the 9-epoch recipe was simply stopped early. It was
   not: ranking quality degraded (0.150 vs 0.107) and training collapsed in
   the final epochs (constant validation rank-sums). Closing the gap to the
   paper's 0.036 would need actual training research - e.g. lower or decayed
   learning rate, gradient clipping, more/neutral-labeled negatives per
   positive, early stopping on the relative-rank metric itself - which is
   beyond faithful reproduction of the released recipe. For baseline use,
   the 9-epoch checkpoint is the better network baseline, and heur remains
   this system's strongest configuration (as in the paper's own Table 1).

## Reproducing this report

```
conda env create -f src/environment.yml     # FreeCAD + pinned torch/DGL
cd src
python dataset_fusion.py --fusion_path <raw .step dir> --extrusion_path <extrude tools dir> --output_path ../data/fusion_processed
python train_preprocess.py --data_path ../data/fusion_processed --output_path processed_data
python train.py --data_path processed_data --output_path train_output --validate_limit 100
python rank_eval.py --data_path processed_data --ids_file test_ids.txt --train_output train_output
cd experiments/exp_infer
for opt in agent heur random; do
  python infer.py --option $opt --data_path ../../../data/fusion_processed \
      --ids_file ../../test_ids.txt --max_time 300 --max_step 15 &
done
python summarize_results.py --ids_file ../../test_ids.txt
python compare_options.py --data_path ../../../data/fusion_processed --ids_file ../../test_ids.txt
```

All stages are parallel and resumable; see `src/README.md`.
