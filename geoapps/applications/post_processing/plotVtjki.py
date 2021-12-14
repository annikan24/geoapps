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




def plot_vtki(
        mesh=None, model=None, interactive=False,
        topo=None, survey=None,
        clim=None, cmap=None, screenshot=False,
        scalar_name='Units', figure_name='figure',
        plotter=None
):
    # Instantiate plotting window
    if plotter is None:
        plotter = pyvista.Plotter(window_size=[800, 400])  # notebook=interactive, window_size=[800, 400])

    if survey is not None:
        survey_points = pyvista.PolyData(survey['vertices'].copy())
        survey_points.cell_arrays[scalar_name] = survey['values']

        plotter.add_mesh(
            survey_points,
            render_points_as_spheres=True,
            point_size=5,
            scalars=scalar_name,
            show_scalar_bar=False,
            cmap='Spectral_r',
            #         clim=[-100,100]
        )

    if topo is not None:
        # Show input surface topography
        surface = pyvista.PolyData(topo.copy()).delaunay_2d()
        #         surface.cell_arrays["Z"] = surface.points[:, 2]

        #         contours = surface.contour()
        plotter.add_mesh(surface, cmap='gray', opacity=1.)
    #         plotter.add_mesh(contours, color="k", line_width=1)
    #     survey_points.translate([-val for val in centroid])
    #     survey_points.rotate_z(-rotation)
    #     survey_points.translate(centroid)

    if mesh is not None and model is not None:
        #         bounds = [survey[:, 0].min(), survey[:, 0].max(),
        #              survey[:, 1].min(), survey[:, 1].max(),
        #              survey[:, 2].min()-5000, survey[:, 2].max()
        #             ]
        #         centroid = np.mean(xyz_local, axis=0)
        # Convert TreeMesh to VTK
        mesh_model = mesh.to_vtk()
        #     mesh_model.clip_box(bounds)

        # Add data values to mesh cells
        mesh_model.cell_arrays[scalar_name] = model
        #         mesh_model.cell_arrays['Active'] = actv
        #         mesh_model.active_scalars_name = scalar_name

        threshed_values = mesh_model.threshold(lower_bound, scalars=scalar_name)

        # Remove the inactive cells
        #     threshed = mesh_model.threshold(0.5, scalars='Active')
        #     threshed.clip_box(bounds, invert=False)

        #### Plotting Routine ####
        # Plotting paramaters for data on mesh
        d_params = dict(
            show_edges=False,
            cmap=cmap,
            scalars=scalar_name,
        )
        plotter.add_mesh(mesh_model.outline(), color='k')

        #     plotter.add_mesh_slice(mesh_model)
        #     plotter.add_mesh_threshold(mesh_model)
        plotter.add_mesh(threshed_values, opacity=.25, clim=[0, lower_bound], show_scalar_bar=False, **d_params)

    # Show axes labels
    plotter.show_grid(all_edges=False, )

    # Add a bounding box of original mesh to see total extent
    # Clip volume in half
    #     plotter.add_mesh(threshed.clip('x', np.r_[600000, 6000000, origin[2]] ), **d_params)

    # Add all the slices
    #     slices = threshed.slice_orthogonal(x=centroid[0], y=centroid[1], z=centroid[2]-2000).clip_box(bounds, invert=False)#.clip('z', np.r_[bounds[0], bounds[2], bounds[4]], invert=False)
    #     slices.clip_box(bounds, False)
    #     plotter.add_mesh(slices, name='slices', clim=[0, 0.1], **d_params)

    #     for xx in np.unique(xyz[:,0]).tolist():
    #         plotter.add_mesh(threshed.slice('x', np.r_[xx, origin[1], origin[2]]), name='slice %i'%xx, **d_params)

    #     origin[1] = 5118959

    #     single_slice = threshed.slice('y', np.r_[origin[0], 5118959, origin[2]])
    #     plotter.add_mesh(single_slice, name='single_slice', **d_params)

    #     plotter.export_vtkjs(figure_name)
    #     plotter.camera_position = cpos
    #     plotter.show(auto_close=False)

    #     plotter.close()
    return plotter


links = [
    "https://www.dropbox.com/s/urxmzx7ag51nq8w/Inversion_0.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/r6knllfj0tbw6y6/Inversion_1.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/efmdvprttme80b1/Inversion_2.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/ct4bgti1qptevmd/Inversion_3.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/nmqeakt7snre2s6/Inversion_4.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/bgu16ynlrzqool1/Inversion_5.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/e5bxw0cx473qjme/Inversion_6.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/0takc0k0x72opst/Inversion_7.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/ya7skklztulrwfq/Inversion_8.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/neck7m5ork39yql/Inversion_9.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/s0et7edfsovlgrg/Inversion_10.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/jnkyjzxvxmwsfvy/Inversion_11.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/1w54cayjxuae8c4/Inversion_12.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/d7ta27b73l2gz46/Inversion_13.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/hjf3w5ngjdpwifa/Inversion_14.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/tdho9cd9gm9arqp/Inversion_15.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/0w9p00rvgs982fn/Inversion_16.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/n1edqjclz4tr1ru/Inversion_17.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/9wxzbvf1pmwfplc/Inversion_18.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/23482mwksqc09zk/Inversion_19.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/lgggnq4ziexkmmi/Inversion_20.0.vtkjs?dl=0",
    "https://www.dropbox.com/s/vuycwebzpyozzc7/Inversion_21.0.vtkjs?dl=0",
]