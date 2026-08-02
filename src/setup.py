"""
Setup functions for the running environment

"""

import os
import sys

# Absolute path to the directory holding FreeCAD.so / FreeCAD.pyd.
# Set the FREECAD_LIB_PATH environment variable to override it without editing this file.
FREECAD_LIB_PATH = os.environ.get("FREECAD_LIB_PATH", "/home/zhangkaicheng/miniconda3/envs/zonegraphs/lib")

sys.path.append(FREECAD_LIB_PATH)

