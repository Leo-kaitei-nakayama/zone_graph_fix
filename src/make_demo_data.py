"""
Generate a small synthetic dataset in the same layout as the preprocessed
Fusion360 data, so the pipeline can be exercised without downloading the full
Fusion360 Gallery reconstruction subset.

Layout produced (identical to what ``dataset_fusion.py`` writes):

    <output_path>/<sequence_id>/<step_index>/target_shape.stp   final target shape
                                            /current_shape.stp  shape before this step (absent at step 0)
                                            /extrusion.stp      extrusion volume applied at this step
                                            /bool_type.txt      'addition' or 'subtraction'

Every shape is normalised so that the bounding box diagonal of the final target
is 1, matching the normalisation ``dataset_fusion.py`` applies.

Usage:
    python make_demo_data.py --output_path ../data/demo
"""

import argparse
import os

from setup import *

import FreeCAD
import Part
from FreeCAD import Base

# Each sequence is a list of (box_min, box_max, bool_type) extrusion steps.
# bool_type: 0 = addition, 1 = subtraction.
DEMO_SEQUENCES = {
    # An L shaped solid built from two additive extrusions.
    "demo_l_shape": [
        ((0, 0, 0), (20, 20, 5), 0),
        ((0, 0, 5), (10, 10, 15), 0),
    ],
    # A block with one corner cut away.
    "demo_notch": [
        ((0, 0, 0), (20, 20, 10), 0),
        ((10, 10, 5), (20, 20, 10), 1),
    ],
    # A three step sequence: base plate, a tower on it, then a corner cut away.
    "demo_bracket": [
        ((0, 0, 0), (24, 16, 4), 0),
        ((0, 0, 4), (10, 16, 14), 0),
        ((16, 8, 0), (24, 16, 4), 1),
    ],
    # A block with a corner cut by a tool that massively over-extrudes past
    # the bounding box - standard CAD practice ("cut all the way through with
    # margin"). Exercises the tool-clipping path in load_raw_sequence.
    "demo_overcut": [
        ((0, 0, 0), (20, 20, 10), 0),
        ((10, 10, -30), (50, 50, 40), 1),
    ],
}


def make_box(box_min, box_max):
    return Part.makeBox(
        box_max[0] - box_min[0],
        box_max[1] - box_min[1],
        box_max[2] - box_min[2],
        Base.Vector(*box_min),
    )


def apply_steps(steps):
    """Replay the extrusions and return (shape_before_each_step, extrusions, target)."""
    current = None
    shapes_before = []
    extrusions = []

    for box_min, box_max, bool_type in steps:
        extrusion = make_box(box_min, box_max)
        shapes_before.append(current)
        extrusions.append((extrusion, bool_type))

        if current is None:
            current = extrusion.copy()
        elif bool_type == 0:
            current = current.fuse(extrusion)
        else:
            current = current.cut(extrusion)
        current = current.removeSplitter()

    return shapes_before, extrusions, current


def normalization_scale(shape):
    bbox = shape.BoundBox
    diagonal = (bbox.XLength ** 2 + bbox.YLength ** 2 + bbox.ZLength ** 2) ** 0.5
    return 1.0 / diagonal


def write_sequence(seq_id, steps, output_path):
    shapes_before, extrusions, target = apply_steps(steps)
    scale = normalization_scale(target)

    target_scaled = target.copy()
    target_scaled.scale(scale, Base.Vector(0, 0, 0))

    seq_path = os.path.join(output_path, seq_id)
    for step_index, (extrusion, bool_type) in enumerate(extrusions):
        step_path = os.path.join(seq_path, str(step_index))
        os.makedirs(step_path, exist_ok=True)

        target_scaled.exportStep(os.path.join(step_path, "target_shape.stp"))

        current = shapes_before[step_index]
        if current is not None:
            current_scaled = current.copy()
            current_scaled.scale(scale, Base.Vector(0, 0, 0))
            current_scaled.exportStep(os.path.join(step_path, "current_shape.stp"))

        extrusion_scaled = extrusion.copy()
        extrusion_scaled.scale(scale, Base.Vector(0, 0, 0))
        extrusion_scaled.exportStep(os.path.join(step_path, "extrusion.stp"))

        with open(os.path.join(step_path, "bool_type.txt"), "w") as f:
            f.write("addition" if bool_type == 0 else "subtraction")

    print("wrote", len(extrusions), "steps to", seq_path)


def main():
    parser = argparse.ArgumentParser(description="generate a synthetic ZoneGraph dataset")
    parser.add_argument("--output_path", default="../data/demo", type=str)
    parser.add_argument(
        "--sequences",
        default="",
        type=str,
        help="comma separated subset of " + ",".join(DEMO_SEQUENCES),
    )
    args = parser.parse_args()

    wanted = [s for s in args.sequences.split(",") if s] or list(DEMO_SEQUENCES)

    os.makedirs(args.output_path, exist_ok=True)
    for seq_id in wanted:
        write_sequence(seq_id, DEMO_SEQUENCES[seq_id], args.output_path)


if __name__ == "__main__":
    main()
