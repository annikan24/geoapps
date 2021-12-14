import gc
import geoh5py
import numpy as np
import re
import json
from shutil import copyfile
from geoh5py.workspace import Workspace
import os
from geoapps.drivers import base_inversion
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.spatial import Delaunay
from geoh5py.objects import Surface, Points, Grid2D
from geoapps.utils import octree_2_treemesh, rotate_xy # , find_value
import geopandas as gpd
import pyIGRF
from pyproj import Transformer

# directories
root_path = r"C:\Users\inversion\Documents\2021_YK_CaMP\geoh5_JSON\First_Run_26_11_21"
dsep = os.path.sep
data_file_orig = dsep.join([root_path, "Mag_Albers_YK_NRCan.geoh5"])
data_file = dsep.join([root_path, "Inversion_0.0.geoh5"])
invs_to_plot = os.listdir(root_path)
workspace_orig = Workspace(data_file_orig)
workspace = Workspace(data_file)
norm_runs = ["L2-L2", "L0-L2", "L0-L1"]

mesh = workspace.get_entity("Octree_Mesh")[0]
tree = octree_2_treemesh(mesh)  # conversion of geoh5 to discretize

survey = workspace.get_entity(f"Data")[0]
data = survey.get_data(f"Observed_tmi")[0]
survey_pts = survey.vertices[:, 1].max()


print(survey_pts)

# inversions = workspace.get_entity("Iteration_25__amplitude_s")[0].to_list()
#
# polygons = workspace_orig.get_entity("UM_poly_extract_May_2020")[0]
#
# for inv_id in range(22):
#
#     axs = plt.subplot(3,1,1)
#     axs = curve_to_polygons(polygons, axs)
#     axs.set_xlim(polygons.vertices[:,0].min(), polygons.vertices[:,0].max())
#     axs.set_ylim(polygons.vertices[:,1].min(), polygons.vertices[:,1].max())
#
#     axs2 = plt.subplot(3,1,2)
#     axs2.set_xlim(inversion_L1.vertices[:,0].min(), polygons.vertices[:,0].max())
#     axs2.set_ylim(polygons.vertices[:,1].min(), polygons.vertices[:,1].max())
