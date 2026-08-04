"""
Setup functions for the running environment

FreeCAD ships its Python bindings (``FreeCAD.so`` / ``FreeCAD.pyd``) in a
directory that is usually *not* on ``sys.path``, so every module in this project
imports ``setup`` before it imports ``FreeCAD``.

The lookup order is:

1. the ``FREECAD_LIB_PATH`` environment variable,
2. ``$CONDA_PREFIX/lib`` (how ``conda install -c conda-forge freecad`` lays it out),
3. a handful of well known system locations.

Set ``FREECAD_LIB_PATH`` explicitly if your FreeCAD lives somewhere else.
"""

import os
import sys

# Set this (or the FREECAD_LIB_PATH environment variable) to the absolute path
# of the directory holding FreeCAD.so if auto-detection does not find it.
FREECAD_LIB_PATH = os.environ.get("FREECAD_LIB_PATH", "")

_CANDIDATE_DIRS = [
    os.path.join(sys.prefix, "lib"),
    os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib"),
    "/usr/lib/freecad/lib",
    "/usr/lib/freecad-python3/lib",
    "/usr/local/lib/freecad/lib",
    "/opt/freecad/lib",
]


def _looks_like_freecad_lib(path):
    if not path or not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if name == "FreeCAD.so" or name.startswith("FreeCAD.cpython"):
            return True
    return False


def find_freecad_lib_path():
    """Return the directory containing the FreeCAD python bindings, or None."""
    candidates = [FREECAD_LIB_PATH] + _CANDIDATE_DIRS
    for path in candidates:
        if _looks_like_freecad_lib(path):
            return path
    return None


def setup_freecad():
    """Put the FreeCAD bindings on sys.path and import them.

    ``Part`` segfaults if it is imported before ``FreeCAD`` has initialised, so
    this always imports ``FreeCAD`` itself rather than leaving that to the
    caller. Returns the path that was added to ``sys.path``, or None if FreeCAD
    was already importable.
    """
    global FREECAD_LIB_PATH
    try:
        # Already importable (e.g. installed as a normal package) - nothing to do.
        import FreeCAD  # noqa: F401

        return None
    except ImportError:
        pass

    path = find_freecad_lib_path()
    if path is None:
        raise ImportError(
            "Could not locate the FreeCAD python bindings.\n"
            "Install FreeCAD (e.g. `conda install -c conda-forge freecad`) and, if\n"
            "needed, point FREECAD_LIB_PATH at the directory containing FreeCAD.so:\n"
            "    export FREECAD_LIB_PATH=/path/to/freecad/lib\n"
            "Searched: " + ", ".join(p for p in [FREECAD_LIB_PATH] + _CANDIDATE_DIRS if p)
        )

    if path not in sys.path:
        sys.path.append(path)

    import FreeCAD  # noqa: F401  (must happen before any `import Part`)

    FREECAD_LIB_PATH = path
    return path


setup_freecad()
