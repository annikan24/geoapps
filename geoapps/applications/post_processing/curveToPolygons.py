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
from geoapps.inversion import get_inversion_output
from GeoToolkit.Mag import Simulator
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator, griddata
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from geoapps.utils import octree_2_treemesh, find_value, rotate_xy
from shutil import copyfile
import os
from tqdm import tqdm
import qrcode
import pyvista
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

get_ipython().system('pip install pyvista')
get_ipython().run_line_magic('matplotlib', 'inline')
pyvista.set_plot_theme('document')
pyvista.rcParams['use_panel'] = False


root_path = r"."
dsep = os.path.sep
data_file = dsep.join([root_path, "Mag_Albers.geoh5"])
inv_path = dsep.join([root_path, "Inversion"])
workspace = Workspace(data_file)

mag_inv_sites = workspace.get_entity("CAMP_Inversion_site_specs_Dec2020")[0]

geology = workspace.get_entity("CaMP_geounit_classes_no_subunits")[0]

# um_sites = mag_inv_sites.get_data('UmSite')[0].values.tolist()
eastings = mag_inv_sites.vertices[:, 0].tolist()
northings = mag_inv_sites.vertices[:, 1].tolist()

anomaly_inv = mag_inv_sites.get_data("ID")[0].values.tolist()

lower_bound = 0.02
upper_bound = 1.0


def curve_to_polygons(curve, ax, zorder=10, linewidth=1):
    """
    The a curve object and returns matplotlib.patches.Polygon objects.
    One object for each parts of the curve.
    """

    polygons = []
    for part in curve.unique_parts:
        if not any(curve.parts == part):
            print("empty part", part)

        ax.add_patch(Polygon(curve.vertices[curve.parts == part, :2], fill=False, linewidth=linewidth, zorder=zorder))

    return ax


axs = plt.subplot()
axs = curve_to_polygons(geology, axs)
axs.set_xlim(geology.vertices[:, 0].min(), geology.vertices[:, 0].max())
axs.set_ylim(geology.vertices[:, 1].min(), geology.vertices[:, 1].max())
# axs.set_xlim(ax1.get_xlim())
# axs.set_ylim(ax1.get_ylim())