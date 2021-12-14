#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# Batch inversion processing
#
# Notebook for the extraction of isovolumes and sections from a list on inversions.
#
# Fournier & Naylor 2020


import numpy as np
import shutil
from geoh5py.workspace import Workspace
from discretize import TreeMesh
from geoh5py.objects import Octree, BlockModel
from geoh5py.data import FloatData
# from geoapps.inversion import get_inversion_output
from GeoToolkit.Mag import Simulator
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator, griddata
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from geoapps.utils import octree_2_treemesh, rotate_xy # , find_value
from shutil import copyfile
import os
from tqdm import tqdm
import qrcode
import pyvista
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def compute_bounds(bounds, factor):
    # Define a region of interest
    bounds = list(bounds) # COPY IT
    delta = np.array([np.abs(bounds[1] - bounds[0]),
                      np.abs(bounds[3] - bounds[2]),
                      np.abs(bounds[5] - bounds[4])])
    cushion = delta * factor
    bounds[::2] += cushion
    bounds[1::2] -= cushion
    return bounds
