# ZoneGraphs

Reference implementation of *Inferring CAD Modeling Sequences Using Zone Graphs*
(CVPR 2021).

## Installation

```
conda env create -f environment.yml
conda activate zonegraph
```

That installs FreeCAD, PyTorch and DGL along with the rest of the dependencies.
See the comments in `environment.yml` for why PyTorch comes from pip rather than
conda-forge, and why its version is pinned.

### FreeCAD path setup

Every module imports `setup` before it imports `FreeCAD`; `setup.py` locates the
FreeCAD python bindings and puts them on `sys.path`. It checks, in order:

1. the `FREECAD_LIB_PATH` environment variable,
2. `$CONDA_PREFIX/lib` (where the conda-forge FreeCAD package installs them),
3. a few standard system locations.

If you build FreeCAD yourself, point it at the directory holding `FreeCAD.so`:

```
export FREECAD_LIB_PATH=/path/to/freecad/lib
```

### Optional: visualization

`utils/vis_utils.py` renders shapes with [mayavi](https://docs.enthought.com/mayavi/mayavi/).
It is not imported by the training or inference paths, so mayavi is not in
`environment.yml`; install it separately if you want the rendering helpers.

## Quick start on synthetic data

The full Fusion360 dataset is a large download. To exercise the pipeline right
away, `make_demo_data.py` builds a handful of small extrude sequences with
FreeCAD in exactly the layout the rest of the code expects:

```
python make_demo_data.py --output_path ../data/demo

cd experiments/exp_infer
python infer.py --option heur --data_path ../../../data/demo --max_time 60 --max_step 8
```

Each reconstructed sequence is written to `<option>/<sequence_id>/` along with
`IOU.txt` and `time.txt`.

## Dataset

We use the [Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)
for training. Specifically, we use the reconstruction subset. The download links
for the data we use are:
- [Main reconstruction subset](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0.zip)
- [GT extrusion set (extrude tools)](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/reconstruction/r1.0.0/r1.0.0_extrude_tools.zip)

### Preprocess Fusion360 raw data into reconstruction sequence data

Each step contains the current shape, target shape and extrusion shape.

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

This also writes `train_ids.txt`, `validate_ids.txt` and `test_ids.txt`, which
`train.py` reads from the current directory.

### Training

```
python train.py

--data_path "path to your processed training data"

--output_path "folder to write checkpoints and loss curves to"

--batch_size "number of graphs per optimizer step (default 32)"
```

### Testing/Inferring reconstruction sequences

```
cd experiments/exp_infer

python infer.py

--option "the option for ranking the proposed extrusions: random/heur/agent"

--data_path "path to your processed fusion data folder"

--max_time "time limit for the search to terminate"

--max_step "maximum sequence length"
```

The `agent` option loads the checkpoints written by `train.py` from
`../../train_output`.

## Notes on the data layout

`dataset_fusion.py` and `make_demo_data.py` both produce this layout, which is
what `DataManager.load_raw_sequence` reads:

```
<data_path>/<sequence_id>/<step_index>/target_shape.stp    final target shape
                                      /current_shape.stp   shape before this step (absent at step 0)
                                      /extrusion.stp       extrusion volume applied at this step
                                      /bool_type.txt       'addition' or 'subtraction'
```

Step folders must be named `0`, `1`, ... with no gaps. Shapes are normalised so
the bounding box diagonal of the final target is 1.

A sequence is skipped, with a reason, when the zone graph cannot represent it —
for example when the target partitions into a single zone, or when a ground
truth extrusion does not decompose exactly into zones (`extrusion_mismatch`).
This is expected for some inputs and is not an error.
