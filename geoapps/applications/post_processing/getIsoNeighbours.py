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


def get_iso_neighbours(indices, mesh, model, lower_bound, upper):
    keeper = []
    if type(indices) == int:
        indices = [indices]
    while indices:
        neighbours = []
        for ind in indices:
            neighbours += mesh[ind].neighbors

        neighbours = np.unique(np.hstack(neighbours)).tolist()
        indices = [
            nn for nn in neighbours if (
                    (model[nn] > lower_bound) and
                    (model[nn] < upper) and
                    (nn not in keeper)
            )
        ]

        keeper += indices
    return keeper