# ZoneGraphs

### Required Packages
- [Freecad](https://www.freecadweb.org/) 
- [Pytorch](https://pytorch.org/)
- [dgl](https://www.dgl.ai/)
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) (for visualization only)
- [Trimesh]
- [networkx]
- joblib, numpy, matplotlib

FreeCAD is not installable from pip: it has to be built (or installed via conda/apt) against the
*same* Python interpreter that runs this code, otherwise `import FreeCAD` will fail.

### FreeCAD path setup

Please download and compile FreeCAD and put absolute path to your FreeCAD lib in setup.py,
or set it in the environment instead:

```
export FREECAD_LIB_PATH=/path/to/freecad/lib
python -c "import FreeCAD; print(FreeCAD.Version())"   # should print, not raise
```

### Dataset

We use the [Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) for training. Sepecifically, we use the reconstruction subset of the Fusion dataset. The download links for the data we use are:
- [Main reconstruction subset](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0.zip)
- [GT extrusion set (extrude tools )](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0_extrude_tools.zip)

### Preprocess Fusion360 Raw data to reconstruction sequence data (each step contains current shape, target shape and extrusion shape)

Downloaded the above fusion and extrude_tool 
```
python dataset_fusion.py 

--fusion_path "path to your fusion reconstruction data folder"

--extrusion_path "path to your fusion GT extrusion tool data folder"

--output_path "path to your output processed fusion data folder"
```
### Generating training data
```
python train_preprocess.py

--data_path "path to your processed fusion data folder"

--output_path "path to your output processed data for training"
```

This does two things:

1. It splits the sequence ids into `train_ids.txt` / `validate_ids.txt` / `test_ids.txt`.
   **These are written into the current working directory**, so run the script from `src/`.
2. It builds the zone graphs and writes one folder per sequence under `--output_path`:

```
<output_path>/<seq_id>/gt/<step>_g.joblib       zone graph of the step
<output_path>/<seq_id>/gt/<step>_e.joblib       ground truth extrusion of the step
<output_path>/<seq_id>/train/<step>_1_g.joblib  positive (zone graph, extrusion) pair
<output_path>/<seq_id>/train/<step>_1_e.joblib
<output_path>/<seq_id>/train/<step>_0_g.joblib  negative (zone graph, extrusion) pair
<output_path>/<seq_id>/train/<step>_0_e.joblib
```

Sequences that fail to build a zone graph, fail the simulation check, or hit the per-sequence
timeout are skipped, so `--output_path` normally holds fewer sequences than `--data_path`.
That is expected; `train.py` skips ids that have no folder.

### Training the extrusion scoring network
```
python train.py

--data_path "path to the training data (the --output_path of train_preprocess.py)"

--output_path "folder to write network weights and loss curves to"

--train_ids "path to train_ids.txt"        (default: ./train_ids.txt)

--validate_ids "path to validate_ids.txt"  (default: ./validate_ids.txt)
```

Run it from `src/`, so the default id file paths resolve:

```
cd src
python train.py --data_path processed_data --output_path train_output
```

Notes:

- `--data_path` must be the *training* data folder produced by `train_preprocess.py`,
  not the raw fusion folder. If no ids in `train_ids.txt` have data under it, training now
  stops with an explicit error instead of silently doing nothing.
- `--output_path` is **deleted and recreated** at the start of every run.
- It writes `zone_encoder.pkl` / `decision_maker.pkl` (latest) and
  `best_zone_encoder.pkl` / `best_decision_maker.pkl` (lowest validation rank sum),
  plus `trainloss.txt` and `validationloss.txt`.
- It uses the GPU when `torch.cuda.is_available()` and falls back to the CPU otherwise.
- `infer.py --option agent` loads weights from `../../train_output`, so either train into
  that path or adjust the path in `experiments/exp_infer/infer.py`.

### Testing/Infering reconstruction sequences
```
cd experiments/exp_infer

python infer.py

--option "the option for ranking the proposed extrusions: random/heur/agent"

--data_path "path to your processed fusion data folder"

--max_time "time limit for the search to terminate"

--max_step "maximum sequence length"
```




