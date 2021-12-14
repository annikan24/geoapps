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





def get_2d_profile(x, y, data, a, b, npts,
                  coordinate_system='local'):
    """
    Plot the data and line profile inside the spcified limits
    """
    def linefun(x1, x2, y1, y2, nx, tol=1e-3):
        dx = x2-x1
        dy = y2-y1

        if np.abs(dx) <= tol:
            y = np.linspace(y1, y2, nx)
            x = np.ones_like(y)*x1
        elif np.abs(dy) <= tol:
            x = np.linspace(x1, x2, nx)
            y = np.ones_like(x)*y1
        else:
            x = np.linspace(x1, x2, nx)
            slope = (y2-y1)/(x2-x1)
            y = slope*(x-x1)+y1
        return x, y

    xLine, yLine = linefun(a[0], b[0], a[1], b[1], npts)

    ind = (xLine > x.min()) * (xLine < x.max()) * (yLine > y.min()) * (yLine < y.max())

    xLine = xLine[ind]
    yLine = yLine[ind]

    distance = np.sqrt((xLine-a[0])**2.+(yLine-a[1])**2.)
    if coordinate_system == 'xProfile':
        distance += a[0]
    elif coordinate_system == 'yProfile':
        distance += a[1]

    if not isinstance(data, list):
        data = [data]

    profiles = []
    for ii, d in enumerate(data):
        if d.ndim == 1:
            dline = griddata(np.c_[x, y], d, (xLine, yLine), method='linear')

        else:
            F = RegularGridInterpolator((x, y), d.T)
            dline = F(np.c_[xLine, yLine])

        # Check for nan
        profiles += [dline]

    return xLine, yLine, profiles