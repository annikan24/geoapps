
#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

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
import geopandas as gpd
from computeBounds import compute_bounds
from curveToPolygons import curve_to_polygons
from get2DProfile import get_2d_profile
from getIsoNeighbours import get_iso_neighbours
from plotVtjki import plot_vtki

pyvista.set_plot_theme('document')
# pyvista.global_theme['use_panel'] = False


# Batch inversion processing
# 
# Notebook for the extraction of isovolumes and sections from a list on inversions.
# 
# Fournier & Naylor 2020


root_path = r"C:\Users\inversion\Documents\2021_YK_CaMP"
dsep = os.path.sep
data_file = dsep.join([root_path, "Mag_Albers_YK_NRCan.geoh5"])
arc_path = r"C:\Users\inversion\Documents\2021_YK_CaMP\Arc_July2021"
inv_path = dsep.join([root_path, "geoh5_JSON"])
workspace = Workspace(data_file)

# old_data_file = dsep.join([root_path, "Mag_Albers.geoh5"])
# old_workspace = Workspace(old_data_file)
# old_geology = old_workspace.get_entity("CaMP_geounit_classes_no_subunits")[0]
# print(type(old_geology))
# um_sites = mag_inv_sites.get_data('UmSite')[0].values.tolist()
# anomaly_inv = mag_inv_sites.get_data("ID")[0].values.tolist()
# print('UMsites', um_sites,'anomaly_inv', anomaly_inv)

geology = workspace.get_entity("UM_poly_extract_May_2020")[0]

localities = [9]
loc_file = gpd.read_file(arc_path + r'\Yukon_UM_localities.shp')
loc_file = loc_file[loc_file['Locality'].isin(localities)]



boxes = [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7]
box_file = gpd.read_file(arc_path + '\Yukon_inversion_sites.shp')
box_file = box_file[box_file['Inv_ID'].isin(boxes)]
eastings, northings = box_file.geometry.centroid.x, box_file.geometry.centroid.y
box_num = box_file.shape[0]
box_ind = [list(box_file.geometry.exterior.iloc[row_id].coords) for row_id in range(box_num)]

anomaly_inv = box_ind.copy()

lower_bound = 0.02
upper_bound = 1.0


#
axs = plt.subplot()
axs = curve_to_polygons(geology, axs)
axs.set_xlim(geology.vertices[:,0].min(), geology.vertices[:,0].max())
axs.set_ylim(geology.vertices[:,1].min(), geology.vertices[:,1].max())
# axs.set_xlim(ax1.get_xlim())
# axs.set_ylim(ax1.get_ylim())

links = ["https://www.dropbox.com/sh/3ogqy7ickxekjon/AAD0EBJh8tZ02GBR8-0V_wmFa?dl=0"
]

# TODO: separate lines 110-200

results = {}
cmaps = [
    'GnBu', 'winter', 'summer',
]
colours = [
    "#0065A2", "#4DD091", "#FFA23A"
]
depth = 1000
data_contours = np.sort(np.unique(np.arange(-500, 5000, 100).tolist() + np.arange(-200, 200, 25).tolist()))
topo_contours = np.arange(0, 2500, 100)
norm_runs = ["L2-L2", "L0-L2", "L0-L1"]
plot = True


url = "https://rockcloud.eos.ubc.ca:8443/index.php/s/f7oHXGHyDnGM7Kj/download"

# def convertGitHubURL(url):
#     url = url.replace("https://github.com", "https://rawgit.com")
#     url = url.replace("raw/", "")
#     return url
# def generateViewerURL(dataURL):
#     viewerURL = "http://viewer.pvgeo.org/"
#     return viewerURL + '%s%s' % ("?fileURL=", dataURL)
# generateViewerURL(convertGitHubURL(url))

# pvm.export.getVTKjsURL('github', 'https://github.com/OpenGeoVis/PVGeo/raw/docs/ripple.vtkjs')

pyvista.get_vtkjs_url(url)


B = np.random.randn(3, 8, 16)
A = np.random.randn(8, 16)

mask = A > 0

new_rgb = [100, 120, 200]
for ii in range(3):
    B[ii, :, :][mask] = new_rgb[ii]

B[ii, :, :][mask]


total_vol = 0
total_vol_scaled = 0
mins = 0
maxs = 0
print()

f = open("summary.txt","w+")
f.write("Inversion, 500, 1000, 2000, 4000, Full\n")
for file in results.keys():

    for ii, (run, summaries) in enumerate(results[file].items()):
        volumes = []
        for level in [500, 1000, 2000, 4000, "all"]:
            values = []
            volume = 0
            for anom, result in summaries.items():
    #             values.append(result['values'])
                volume += result['volumes'][level]
            volumes.append(volume*1e-9)
        f.write(file + norm_runs[run] + ", " + ', '.join(f"{e:.2f}" for e in volumes) + "\n")
#         if len(values) == 0:
#             continue
f.close()
#         volumes += [volume]

#         print(f"{norm_runs[ii]}, {volume*1e-9:.1f}, {volume_scaled*1e-9:.1f}, 0")

#     print(f"{file}, {np.min(volumes)*1e-9:.1f}, {np.max(volumes)*1e-9:.1f}, {np.median(volumes)*1e-9:.1f}, {np.std(volumes)*1e-9:.1f}")
#     total_vol += np.median(volumes)
#     mins += np.min(volumes)
#     maxs += np.max(volumes)
#     total_vol_scaled += np.median(volume_scaled)
# print(f"Total, {total_vol*1e-9:.1f}, {mins*1e-9:.1f}, {maxs*1e-9:.1f}")#" $km^3$: Scaled {np.median(total_vol_scaled)*1e-9:.1f} $km^3$")
# print(f"Normalized Total, {total_vol*1e-9:.1f}, {mins*1e-9:.1f}, {maxs*1e-9:.1f}, {total_vol_scaled*1e-9:.1f}")#" $km^3$: Scaled {np.median(total_vol_scaled)*1e-9:.1f} $km^3$")


# ax
#
#
# [1, 3, 4] * 2
#
#
#
# # results[file][ii][anom_id] = {
# #                 'volumes': {
# #                     "all": blob_vol
# #                 },
# #                 'mass': mass,
# #                 'values': vals[indices],
# #                 'indices': indices,
# #                 'seed_local': cell_location,
# #                 'centroid': centroid_utm,
# #                 'centroid_index': centroid_index,
# #             }
#
# #             for level in [500, 1000, 2000, 4000]:
# #                 level_ind = np.where(
# #                     (topo_interp(tree.gridCC[indices, :2]) - tree.gridCC[indices, 2]) < level
# #                 )[0]
# #                 results[file][ii][anom_id][level] = sum(tree.vol[indices][level_ind])
#
#
