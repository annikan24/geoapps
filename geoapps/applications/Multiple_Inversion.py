#!/usr/bin/env python
# coding: utf-8
import gc
import geoh5py

import numpy as np
import re
import json
from shutil import copyfile

from geoh5py.workspace import Workspace
import os
from geoapps.drivers import base_inversion
# from geoapps.inversion import get_inversion_output
import matplotlib.pyplot as plts
import xarray as xr
from matplotlib.patches import Rectangle

from scipy.spatial import Delaunay
from geoh5py.objects import Surface, Points, Grid2D
import geopandas as gpd
import pyIGRF
from pyproj import Transformer
from geoapps.utils import calculate_2D_trend
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.drivers.magnetic_vector_inversion import MagneticVectorDriver

"""
Multiple_inversion.py

Latest update: November 14th, 2021 (DF)

Runs from geoapps v0.6.0 
"""
def rotXY(xyz, center, angle):
    """
    Rotate xyz coordinates about the z axis by angle, using center as center point of data

    :param xyz: array of floats, shape(nx, ) | shape(ny, nx)
        X- Y- and Z-Coordinates of points composing grid
    :param center: array of floats, shape(ny, ) | shape(ny, nx)
        X- and Y-Coordinates of center point in area
    :param angle: array of floats, shape(ny, )
        Angle to rotate grid about center

    """

    R = np.r_[np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
              np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)]]

    xyz[:, 0] -= center[0]  # easting
    xyz[:, 1] -= center[1]  # northing

    xy_rot = np.dot(R, xyz[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], xyz[:, 2:]]


def get_bounded_data(x, y, values, center, angle, width, height, resolution=1, ndv=-999):
    """
    Get a 2D grid of data with coordinates and extract within bounds

    :param x: array of floats, shape(nx, ) | shape(ny, nx)
        X-Coordinates of grid
    :param y: array of floats, shape(ny, ) | shape(ny, nx)
        Y-Coordinates of grid
    :param values: array of floats, shape(ny, nx)
        Grid values to be extracted

    """
    ew_rot = rotXY(np.c_[center].reshape((1, 2)), center, angle)
    data_lim_x = [ew_rot[0, 0] - width / 2, ew_rot[0, 0] + width / 2]
    data_lim_y = [ew_rot[0, 1] - height / 2, ew_rot[0, 1] + height / 2]

    assert x.ndim == y.ndim, "X and Y locations must have the same shape"

    if x.ndim == 1:
        x, y = np.meshgrid(x, y)

    # lowest reso possible for grid

    dwnS = int(np.ceil(resolution / np.min(x[0, 1:] - x[0, :-1])))

    xx = x[::dwnS, ::dwnS].flatten()  # every "dwnS" cells indexing
    yy = y[::dwnS, ::dwnS].flatten()
    xy = rotXY(np.c_[xx, yy], center, angle)

    # return T/F if cells are within desired bounds
    ind = (
            (xy[:, 0] > data_lim_x[0]) *
            (xy[:, 0] < data_lim_x[1]) *
            (xy[:, 1] > data_lim_y[0]) *
            (xy[:, 1] < data_lim_y[1])
    )

    # checks if any of the cells returned true and then adds them to the list

    if np.any(ind):
        d = np.asarray(values[::dwnS, ::dwnS])
        d = d.flatten()[ind]

        # remove no-data-values
        temp = np.c_[xy[ind, :], d]
        ga_null = (d > 1e-38) * (d < 2e-38)
        ind = (d != ndv) * (ga_null == False)

        xy_vals = rotXY(temp[ind, :], center, -angle)

        return xy_vals

    return []


def get_topo(center, angle, zarr_files, resolution, width=71000, height=300000, ndv=-999):
    """
    Get a 3D grid of data from zarr files representing topography

    :param zarr_files:
        List of .zarr files of topography in zarr directory
    :returns:
        Data for topography within bounds width and height

    """

    # Adapt the resolution on the window extent
    dtm = []

    for file in zarr_files:
        if f'HR' in file:
            topo = xr.open_zarr(dsep.join([zarr_dir, file]))
        else:
            topo = xr.open_zarr(dsep.join([zarr_dir, file]))

        xx = topo['x'].values
        yy = topo['y'].values
        values = topo['topo']

        topo = get_bounded_data(xx, yy, values, center, angle, width, height, resolution=resolution, ndv=ndv)

        if np.any(topo):
            topo = topo[topo[:, 2] > -999, :]
            dtm.append(topo)

    return np.vstack(dtm)  # X, Y, Z array of topo data


# directories
root_path = r"C:\Users\inversion\Documents\2021_YK_CaMP"
dsep = os.path.sep
data_file = dsep.join([root_path, "Mag_Albers_YK_NRCan.geoh5"])
inv_path = dsep.join([root_path, "geoh5_JSON"])

# closes the file after reading
input_template = dsep.join([root_path, "Inversion_.json"])
zarr_dir = dsep.join([root_path, "Topo\\topo_3579_nts"])
zarr_files = os.listdir(zarr_dir)
workspace = Workspace(data_file)

# inversion settings
resolution = 500
cell_size = 100
drape = [0, 0, 305]
depth_core = 10000
# object_names = ['Mag_105','Mag_106','Mag_116','Mag_115','Mag_non_TMI']
channel_names = ['105A_TMI', '105B_TMI', '105C_TMI', '105D_TMI', '105D_TMI', '105E_TMI',
                 '10F5_TMI', '105G_TMI', '105H_TMI', '105K_TMI', '10L5_TMI',
                 '106C_TMI', '106D_TMI',
                 '115A_TMI', '115B_TMI', '115F_TMI', '115G_TMI', '115I_TMI',
                 '115J_TMI', '115K_TMI', '115N_TMI', '115O_TMI', '115P_TMI',
                 '116B_TMI', '116C_TMI',
                ]

known_grids = {grid.uid: grid.name for grid in workspace.objects if isinstance(grid, Grid2D)}

# uncertainties = [0, 30]

# for easy copy and paste
# get arc file info
arc_path = r".\Arc_July2021"

# polygons for localities, get heights, widths, id's
loc_file = gpd.read_file(arc_path + '\Yukon_UM_localities.shp')

# choose which localities you want to look at inversions sites in:
localities = [9, 13]
loc_file = loc_file[loc_file['Locality'].isin(localities)]
loc_num = loc_file.shape[0]
loc_widths = list(np.zeros(loc_num))
loc_heights = list(np.zeros(loc_num))
loc_ind = [list(loc_file.geometry.exterior[row_id].coords) for row_id in range(loc_num - 1)]

# "eastings" and "northings" from below
loc_east, loc_north = loc_file.geometry.centroid.x, loc_file.geometry.centroid.y

# polygons for inversion sites
# get heights, widths, id's, azimuths, eastings, northings, removes angles too small

poly_file = gpd.read_file(arc_path + '\Yukon_inversion_sites.shp')

# choose which inversions sites you want to invert:
polygons = ['9.1']#, '9.4', '9.5', '9.6', '13.1', '13.3']
poly_file = poly_file[poly_file['Inv_ID'].isin(polygons)]
poly_num = poly_file.shape[0]
poly_ind = [list(poly_file.geometry.exterior.iloc[row_id].coords) for row_id in range(poly_num)]
invs = range(poly_num)
widths = list(np.zeros(poly_num))
heights = list(np.zeros(poly_num))
poly_vert_diff = list(np.zeros(poly_num))
azimuths = list(np.zeros(poly_num)) 

for pol in range(poly_num):
    # 5th point == 1st point of rectangle, so can delete it
    poly_ind[pol].pop()    
    
    # widths, heights, and angle calculation
    widths[pol] = np.abs(poly_ind[pol][0][0] - poly_ind[pol][1][0])
    heights[pol] = np.abs(poly_ind[pol][0][1] - poly_ind[pol][2][1])
    poly_vert_diff[pol] = poly_ind[pol][0][1] - poly_ind[pol][1][1] 
    azimuths[pol] = np.tan(poly_vert_diff[pol]/widths[pol])

    if len(poly_ind[pol]) != 4:
        raise ValueError('The rectangle must have four sides')

# remove any angles less than 5
for ang, pol in zip(azimuths, range(len(azimuths))): 
    if ang == 0:
        azimuths[pol] = 0

eastings, northings = poly_file.geometry.centroid.x, poly_file.geometry.centroid.y
''' 
    Find inducing field parameters for all sites 
    :param: D: declination (+ve east)
    :param: I: inclination (+ve down)
    :param: H: horizontal intensity
    :param: X: north component
    :param: Y: east component
    :param: Z: vertical component (+ve down)
    :param: F: total intensity unit: degree or nT 
'''

inc = list(np.zeros(loc_num))
dec = list(np.zeros(loc_num))
amp = list(np.zeros(loc_num))

date = 2015.00
alt = .305

for easting, northing, loc in zip(loc_east, loc_north, range(loc_num)):
    x1, y1 = easting, northing
    transformer = Transformer.from_crs("epsg:3579", "epsg:4326")
    lat, lon = transformer.transform(x1, y1)
    D, I, H, X, Y, Z, F = pyIGRF.igrf_value(lat, lon, alt, date)
    dec[loc] = D
    inc[loc] = I
    amp[loc] = F

# Compute inducing field parameters for all sites

A = np.c_[np.ones(loc_num), loc_east, loc_north]

c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, inc))
inclins = (eastings * a + northings * b + c).tolist()

c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, dec))
declins = (eastings * a + northings * b + c).tolist()

c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, amp))
intensities = (eastings * a + northings * b + c).tolist()

# plotting the locations of inversions for sanity check

plt.figure()
axs = plt.subplot(3, 1, 1)
im = axs.scatter(eastings, northings, 10, c=inclins)
axs.set_aspect('equal')
plt.colorbar(im)

axs = plt.subplot(3, 1, 2)
im = axs.scatter(eastings, northings, 10, c=declins)
axs.set_aspect('equal')
plt.colorbar(im)

axs = plt.subplot(3, 1, 3)
im = axs.scatter(eastings, northings, 10, c=intensities)
axs.set_aspect('equal')
plt.colorbar(im)

plt.show()

# loop through inversion sites to create JSON files for each, which will be used to run inversions later

norms_names = [
    'L0_L2', 'L0_L1', 'L0_L0'
] #'L1_L1', 'L0_L0'

for easting, northing, inv, width, height, azimuth, inclin, declin, intensity in zip(
    eastings, northings, invs, widths, heights, azimuths, inclins, declins, intensities
): 
    
    geoh5_file_name = dsep.join([inv_path, f"Inversion_{inv:.1f}.geoh5" ])
    
    # Create a new workspace for the inversion
    tile_ws = Workspace(geoh5_file_name)
    topo = get_topo([easting, northing], azimuth, zarr_files, resolution, width=width*4.0, height=height*4.0)
    tri2D = Delaunay(topo[:, :2])
    topo_obj = Surface.create(tile_ws, vertices=topo, name="topo", cells=tri2D.simplices)
    
    # Extract data
    data = []
    workspace = Workspace(data_file)
    for name in channel_names:  # Your data has the same name as the grid

        if name in known_grids.values():
            grid = [obj for obj in workspace.objects if obj.name == name][0]
        else:
            continue

        values = grid.get_data(name)[0].values.reshape((grid.v_count, grid.u_count))

        x = grid.centroids[:, 0].reshape(values.shape)
        y = grid.centroids[:, 1].reshape(values.shape)

        xyz_d = get_bounded_data(x, y, values, [easting, northing], azimuth, width, height, resolution=resolution, ndv=0)

        if np.any(xyz_d):
            data.append(xyz_d)
        del x, y, grid

    del workspace  # The grid data lookup loads up memory ... temporarely free up
    gc.collect()
    data = np.vstack(data)
    data = np.unique(data, axis = 0)
    data_trend, _ = calculate_2D_trend(
        np.c_[data[:, :2], np.zeros(data.shape[0])], data[:, 2], 1, "all"
    )

    data[:, 2] -= data_trend
    survey = Points.create(tile_ws, vertices=np.c_[data[:, :2], np.zeros(data.shape[0])], name=f"Data_Tile{inv:.1f}")
    data_obj = survey.add_data({"TMI": {"values": data[:, 2],}})
    
    norm_runs = [
#         [2,2,2,2],
        [0,2,2,2],
#         [1,1,1,1],
        [0,1,1,1],
        [0,0,0,0]
    ]
    
    for ii, norms in enumerate(norm_runs):
        
        # with open(input_template) as f:
        #     inv_input = json.load(f)
        
        params = MagneticVectorParams()

        json_file_name = f"Inversion_{inv:.1f}_{norms_names[ii]}"
        
        # Restart the inversion with previous beta at end of Cartesian
#         if norms_names[ii] != "L2_L2":
#             tile_ws = Workspace(geoh5_file_name)
#             inv_num = tile_ws.get_entity(f"Inversion_{inv:.1f}_L2_L2")[0]

#             mesh = [child for child in inv_num.children if child.name=="Mesh"][0]

#             count = -1
#             for model in mesh.children:

#                 if re.match("Iteration_\d_model", model.name) is not None:
#                     if re.match("\w+_s", model.name) is not None:
#                         continue
#                     count += 1

#             inv_log = get_inversion_output(geoh5_file_name, f"Inversion_{inv:.1f}_L2_L2") 
#             inv_input["initial_beta"] = inv_log['beta'][count]
        
        # if f"Inversion_{inv:.1f}_{norms_names[ii]}" in list(tile_ws.list_groups_name.values()):
        #     continue
        
        params.out_group = f"Inversion_{inv:.1f}_{norms_names[ii]}"
        params.workspace = tile_ws
        params.geoh5 = tile_ws
        params.inducing_field_strength = intensity
        params.inducing_field_inclination = inclin
        params.inducing_field_declination = declin
        params.u_cell_size = cell_size
        params.v_cell_size = cell_size
        params.w_cell_size = cell_size
        params.depth_core = depth_core
        params.horizontal_padding = depth_core
        params.vertical_padding = 0
        params.octree_levels_topo = [0, 0, 4, 4]
        params.octree_levels_obs = [6, 6, 6]
        params.s_norm, params.x_norm, params.y_norm, params.z_norm = norms
        params.resolution = resolution
        params.starting_model = 1e-4
        params.window_center_x, params.window_center_y = easting, northing
        params.window_width, params.window_height = float(width), float(height)
        params.window_azimuth = azimuth
        params.initial_beta_ratio = 100.
        params.data_object = survey.uid
        params.tmi_channel_bool = True
        params.tmi_channel = data_obj.uid
        params.tmi_uncertainty = float(np.percentile(np.abs(data[:, 2]), 20))
        params.z_from_topo = True
        params.receivers_offset_x = drape[0]
        params.receivers_offset_y = drape[1]
        params.receivers_offset_z = drape[2]
        params.topography_object = topo_obj.uid
        params.workpath = inv_path
        params.write_input_file(
            name=json_file_name,
            path=inv_path
        )

        driver = MagneticVectorDriver(params)
        driver.run()


list(tile_ws.list_groups_name.values())

tile_ws.list_groups_name, f"Inversion_{inv}_{norms_names[0]}"









# # old mag sites information - when grabbing from GA project

# inv_sites = workspace.get_entity("CAMP_Inversion_site_specs_Dec2020")[0]
# invs = inv_sites.get_data("ID")[0].values.tolist()
# eastings = inv_sites.get_data("Center_X")[0].values.tolist()
# northings = inv_sites.get_data("Center_Y")[0].values.tolist()
# widths = inv_sites.get_data("Width")[0].values.tolist()
# heights = inv_sites.get_data("height")[0].values.tolist()
# azimuths =  inv_sites.get_data("Az")[0].values.tolist()


# # Compute inducing field parameters for all sites
# points = workspace.get_entity("Inversion_sites")[0]
# inc = points.get_data("Inclination")[0].values.tolist()
# dec = points.get_data("Declination")[0].values.tolist()
# amp = points.get_data("Intensity")[0].values.tolist()

# A = np.c_[np.ones(points.n_vertices), points.vertices[:, 0], points.vertices[:, 1]]

# c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, inc))
# inclins = (inv_sites.vertices[:, 0] * a + inv_sites.vertices[:, 1] * b + c).tolist()

# c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, dec))
# declins = (inv_sites.vertices[:, 0] * a + inv_sites.vertices[:, 1] * b + c).tolist()

# c, a, b = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, amp))
# intensities = (inv_sites.vertices[:, 0] * a + inv_sites.vertices[:, 1] * b + c).tolist()

# # invs = [20.0]
# # eastings = [eastings[-1]]
# # northings = [northings[-1]]
# # widths = [widths[-1]]
# # heights = [heights[-1]]
# # azimuths = [azimuths[-1]]
# # inclins = [inclins[-1]]
# # declins = [declins[-1]]
# # intensities = [intensities[-1]]

# invs = [20.0]
# eastings = [847433.0]
# northings = [1536839.0]
# widths = [15000.0]
# heights = [25000.0]
# azimuths = [-45.0]
# inclins = [76.07417297363281]
# declins = [19.906850814819336]
# intensities = [55300.5]

