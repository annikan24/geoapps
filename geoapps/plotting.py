import re
import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import (
    Dropdown,
    ColorPicker,
    SelectMultiple,
    Text,
    IntSlider,
    Checkbox,
    FloatSlider,
    FloatText,
    VBox,
    HBox,
    ToggleButton,
    ToggleButtons,
    interactive_output,
    FloatLogSlider,
    Label,
    Layout,
    RadioButtons,
)
from geoapps.base import Widget
from geoapps.utils import filter_xy, rotate_xy, format_labels
from geoapps.selection import ObjectDataSelection


class PlotSelection2D(Widget):
    """
    Application for selecting data in 2D plan map view
    """

    def __init__(self, h5file, **kwargs):

        self.workspace = Workspace(h5file)

        self.selection = ObjectDataSelection(h5file)
        self.objects = self.selection.objects

        self.center_x = FloatSlider(
            min=-100, max=100, steps=10, description="Easting", continuous_update=False,
        )
        self.center_y = FloatSlider(
            min=-100,
            max=100,
            steps=10,
            description="Northing",
            continuous_update=False,
            orientation="vertical",
        )
        self.azimuth = FloatSlider(
            min=-90,
            max=90,
            value=0,
            steps=5,
            description="Orientation",
            continuous_update=False,
        )
        self.width_x = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Width",
            continuous_update=False,
        )
        self.width_y = FloatSlider(
            min=0,
            max=100,
            steps=10,
            value=1000,
            description="Height",
            continuous_update=False,
            orientation="vertical",
        )
        self.resolution = FloatText(
            description="Grid Resolution (m)", style={"description_width": "initial"},
        )
        self.data_count = Label("Data Count: 0", tooltip="Keep <1500 for speed")
        self.zoom_extent = ToggleButton(
            value=True,
            description="Zoom on selection",
            tooltip="Keep plot extent on selection",
            icon="check",
        )

        if "contours" in kwargs.keys():
            contours = kwargs["contours"]
        else:
            contours = ""

        self.contours = widgets.Text(
            value=contours,
            description="Contours",
            disabled=False,
            continuous_update=False,
        )

        def set_bounding_box(_):
            self.set_bounding_box()

        self.selection.objects.observe(set_bounding_box)
        self.pause = False

        def plot_selection(
            data_name,
            resolution,
            center_x,
            center_y,
            width_x,
            width_y,
            azimuth,
            zoom_extent,
            contours,
        ):

            self.plot_selection(
                data_name,
                resolution,
                center_x,
                center_y,
                width_x,
                width_y,
                azimuth,
                zoom_extent,
                contours,
            )

        self.window_plot = widgets.interactive_output(
            plot_selection,
            {
                "data_name": self.selection.data,
                "resolution": self.resolution,
                "center_x": self.center_x,
                "center_y": self.center_y,
                "width_x": self.width_x,
                "width_y": self.width_y,
                "azimuth": self.azimuth,
                "zoom_extent": self.zoom_extent,
                "contours": self.contours,
            },
        )
        self.widget = VBox(
            [
                VBox([self.selection.widget, self.resolution, self.data_count,]),
                HBox(
                    [
                        self.center_y,
                        self.width_y,
                        VBox(
                            [
                                self.width_x,
                                self.center_x,
                                self.window_plot,
                                self.azimuth,
                                self.zoom_extent,
                            ]
                        ),
                    ],
                    layout=Layout(align_items="center"),
                ),
            ]
        )
        self.figure = None
        self.axis = None
        self.indices = None

        super().__init__(**kwargs)

    def plot_selection(
        self,
        data_name,
        resolution,
        center_x,
        center_y,
        width_x,
        width_y,
        azimuth,
        zoom_extent,
        contours,
    ):

        if self.pause:
            return

        # Parse the contours string
        if contours != "":
            vals = re.split(",", contours)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = np.asarray(re.split(":", val), dtype="int")
                    if len(param) == 2:
                        cntrs += [np.arange(param[0], param[1])]
                    else:
                        cntrs += [np.arange(param[0], param[1], param[2])]
                else:
                    cntrs += [np.float(val)]
            contours = np.unique(np.sort(np.hstack(cntrs)))
        else:
            contours = None

        entity, data_obj = self.selection.get_selected_entities()

        if isinstance(entity, (Grid2D, Surface, Points, Curve)):

            self.figure = plt.figure(figsize=(10, 10))

            self.axis = plt.subplot()
            corners = np.r_[
                np.c_[-1.0, -1.0],
                np.c_[-1.0, 1.0],
                np.c_[1.0, 1.0],
                np.c_[1.0, -1.0],
                np.c_[-1.0, -1.0],
            ]
            corners[:, 0] *= width_x / 2
            corners[:, 1] *= width_y / 2
            corners = rotate_xy(corners, [0, 0], -azimuth)
            self.axis.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, "k")

            _, _, ind_filter, _, contour_set = plot_plan_data_selection(
                entity,
                data_obj,
                **{
                    "ax": self.axis,
                    "resolution": resolution,
                    "window": {
                        "center": [center_x, center_y],
                        "size": [width_x, width_y],
                        "azimuth": azimuth,
                    },
                    "zoom_extent": zoom_extent,
                    "resize": True,
                    "contours": contours,
                },
            )
            self.indices = ind_filter
            self.contours.contour_set = contour_set
            self.data_count.value = f"Data Count: {ind_filter.sum()}"

    def set_bounding_box(self):
        # Fetch vertices in the project
        lim_x = [1e8, -1e8]
        lim_y = [1e8, -1e8]

        obj, _ = self.selection.get_selected_entities()
        if isinstance(obj, Grid2D):
            lim_x[0], lim_x[1] = obj.centroids[:, 0].min(), obj.centroids[:, 0].max()
            lim_y[0], lim_y[1] = obj.centroids[:, 1].min(), obj.centroids[:, 1].max()
        elif isinstance(obj, (Points, Curve, Surface)):
            lim_x[0], lim_x[1] = obj.vertices[:, 0].min(), obj.vertices[:, 0].max()
            lim_y[0], lim_y[1] = obj.vertices[:, 1].min(), obj.vertices[:, 1].max()
        else:
            return

        self.pause = True
        self.center_x.min = -np.inf
        self.center_x.max = lim_x[1]
        self.center_x.value = np.mean(lim_x)
        self.center_x.min = lim_x[0]

        self.center_y.min = -np.inf
        self.center_y.max = lim_y[1]
        self.center_y.value = np.mean(lim_y)
        self.center_y.min = lim_y[0]

        self.width_x.max = lim_x[1] - lim_x[0]
        self.width_x.value = self.width_x.max / 2.0
        self.width_x.min = 0

        self.width_y.max = lim_y[1] - lim_y[0]
        self.width_y.min = 0

        self.pause = False
        self.width_y.value = self.width_y.max / 2.0


def plot_profile_data_selection(
    entity,
    field_list,
    uncertainties=None,
    selection={},
    resolution=None,
    plot_legend=False,
    ax=None,
    color=[0, 0, 0],
):

    locations = entity.vertices

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    pos = ax.get_position()
    xx, yy = [], []
    threshold = 1e-14
    for key, values in selection.items():

        for line in values:

            if entity.get_data(key):
                ind = np.where(entity.get_data(key)[0].values == line)[0]
            else:
                continue
            if len(ind) == 0:
                continue

            if resolution is not None:
                dwn_ind = filter_xy(locations[ind, 0], locations[ind, 1], resolution,)

                ind = ind[dwn_ind]

            xyLocs = locations[ind, :]

            if np.std(xyLocs[:, 0]) > np.std(xyLocs[:, 1]):
                dist = xyLocs[:, 0].copy()
            else:
                dist = xyLocs[:, 1].copy()

            dist -= dist.min()
            order = np.argsort(dist)
            legend = []

            c_increment = [(1 - c) / (len(field_list) + 1) for c in color]

            for ii, field in enumerate(field_list):
                if (
                    entity.get_data(field)
                    and entity.get_data(field)[0].values is not None
                ):
                    values = entity.get_data(field)[0].values[ind][order]

                    xx.append(dist[order][~np.isnan(values)])
                    yy.append(values[~np.isnan(values)])

                    if uncertainties is not None:
                        ax.errorbar(
                            xx[-1],
                            yy[-1],
                            yerr=uncertainties[ii][0] * np.abs(yy[-1])
                            + uncertainties[ii][1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
                        )
                    else:
                        ax.plot(
                            xx[-1],
                            yy[-1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
                        )
                    legend.append(field)

                    threshold = np.max([threshold, np.percentile(np.abs(yy[-1]), 2)])

            if plot_legend:
                ax.legend(legend, loc=3, bbox_to_anchor=(0, -0.25), ncol=3)

            if xx and yy:
                format_labels(
                    np.hstack(xx),
                    np.hstack(yy),
                    ax,
                    labels=["Distance (m)", "Fields"],
                    aspect="auto",
                )

    return ax, threshold


def plot_plan_data_selection(entity, data, **kwargs):

    locations = entity.vertices
    if "resolution" not in kwargs.keys():
        resolution = 0
    else:
        resolution = kwargs["resolution"]

    if "indices" in kwargs.keys():
        indices = kwargs["indices"]
        if isinstance(indices, np.ndarray) and np.all(indices == False):
            indices = None
    else:
        indices = None

    line_selection = None
    contour_set = None
    values = None
    ax = None
    out = None

    if isinstance(getattr(data, "values", None), np.ndarray):
        if not isinstance(data.values[0], str):
            values = data.values.copy()
            values[(values > 1e-18) * (values < 2e-18)] = np.nan
            values[values == -99999] = np.nan

    color_norm = None
    if "color_norm" in kwargs.keys():
        color_norm = kwargs["color_norm"]

    window = None
    if "window" in kwargs.keys():
        window = kwargs["window"]

    if data is not None and data.entity_type.color_map is not None:
        new_cmap = data.entity_type.color_map.values
        map_vals = new_cmap["Value"].copy()
        cmap = colors.ListedColormap(
            np.c_[
                new_cmap["Red"] / 255, new_cmap["Green"] / 255, new_cmap["Blue"] / 255,
            ]
        )
        color_norm = colors.BoundaryNorm(map_vals, cmap.N)
    else:
        cmap = "Spectral_r"

    if "ax" not in kwargs.keys():
        plt.figure(figsize=(8, 8))
        ax = plt.subplot()
    else:
        ax = kwargs["ax"]

    if isinstance(entity, Grid2D) and values is not None:
        x = entity.centroids[:, 0].reshape(entity.shape, order="F")
        y = entity.centroids[:, 1].reshape(entity.shape, order="F")
        indices = filter_xy(x, y, resolution, window=window)

        ind_x, ind_y = (
            np.any(indices, axis=1),
            np.any(indices, axis=0),
        )

        X = x[ind_x, :][:, ind_y]
        Y = y[ind_x, :][:, ind_y]

        if values is not None:
            values = values.reshape(entity.shape, order="F")
            values[indices == False] = np.nan
            values = values[ind_x, :][:, ind_y]

        if np.any(values):
            out = ax.pcolormesh(
                X, Y, values, cmap=cmap, norm=color_norm, shading="auto"
            )

        if "contours" in kwargs.keys() and np.any(values):
            contour_set = ax.contour(
                X, Y, values, levels=kwargs["contours"], colors="k", linewidths=1.0
            )

    elif isinstance(entity, (Points, Curve, Surface)):
        if indices is None:
            indices = filter_xy(
                entity.vertices[:, 0], entity.vertices[:, 1], resolution, window=window,
            )

        x, y = entity.vertices[:, 0], entity.vertices[:, 1]
        X, Y = x[indices], y[indices]
        if values is not None:
            values = values[indices]

        if "marker_size" not in kwargs.keys():
            marker_size = 5
        else:
            marker_size = kwargs["marker_size"]

        out = ax.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

        if "contours" in kwargs.keys() and np.any(values):
            ind = ~np.isnan(values)
            contour_set = ax.tricontour(
                X[ind],
                Y[ind],
                values[ind],
                levels=kwargs["contours"],
                colors="k",
                linewidths=1.0,
            )

    else:
        print(
            "Sorry, 'plot=True' option only implemented for Grid2D, Points, Surface and Curve objects"
        )
        out = None

    if "zoom_extent" in kwargs.keys() and kwargs["zoom_extent"] and np.any(values):
        ind = ~np.isnan(values.ravel())
        x = X.ravel()[ind]
        y = Y.ravel()[ind]
        if ind.sum() > 0:
            format_labels(x, y, ax)
            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([y.min(), y.max()])
    elif np.any(x) and np.any(y):
        format_labels(x, y, ax)
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])

    if (
        "colorbar" in kwargs.keys()
        and values[~np.isnan(values)].min() != values[~np.isnan(values)].max()
    ):
        plt.colorbar(out, ax=ax)

    line_selection = np.zeros_like(indices, dtype=bool)
    if "highlight_selection" in kwargs.keys():
        for key, values in kwargs["highlight_selection"].items():

            if not np.any(entity.get_data(key)):
                continue

            for line in values:
                ind = np.where(entity.get_data(key)[0].values == line)[0]
                x, y, values = (
                    locations[ind, 0],
                    locations[ind, 1],
                    entity.get_data(key)[0].values[ind],
                )
                ind_line = filter_xy(x, y, resolution, window=window)
                ax.scatter(x[ind_line], y[ind_line], marker_size * 2, "k", marker="+")

                line_selection[ind[ind_line]] = True

    return ax, out, indices, line_selection, contour_set


def plot_em_data_widget(h5file):
    workspace = Workspace(h5file)

    curves = [
        entity.parent.name + "." + entity.name
        for entity in workspace.all_objects()
        if isinstance(entity, Curve)
    ]
    names = [name for name in sorted(curves)]

    def get_parental_child(parental_name):

        parent, child = parental_name.split(".")

        parent_entity = workspace.get_entity(parent)[0]

        children = [entity for entity in parent_entity.children if entity.name == child]
        return children

    def find_value(labels, strings):
        value = None
        for name in labels:
            for string in strings:
                if string.lower() in name.lower():
                    value = name
        return value

    def plot_profiles(entity_name, groups, line_field, lines, scale, threshold):

        fig = plt.figure(figsize=(12, 12))
        entity = get_parental_child(entity_name)[0]

        ax = plt.subplot()
        colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for group, color in zip(groups, colors):

            prop_group = entity.get_property_group(group)

            if prop_group is not None:
                fields = [
                    entity.workspace.get_entity(uid)[0].name
                    for uid in prop_group.properties
                ]

                ax, _ = plot_profile_data_selection(
                    prop_group.parent,
                    fields,
                    selection={line_field: lines},
                    ax=ax,
                    color=color,
                )

        ax.grid(True)

        plt.yscale(scale, linthreshy=10.0 ** threshold)

    def updateList(_):
        entity = get_parental_child(objects.value)[0]
        data_list = entity.get_data_list()
        obj = get_parental_child(objects.value)[0]

        options = [pg.name for pg in obj.property_groups]
        options = [option for option in sorted(options)]
        groups.options = options
        groups.value = [groups.options[0]]
        line_field.options = data_list
        line_field.value = find_value(data_list, ["line"])

        if line_field.value is None:
            line_ids = []
            value = []
        else:
            line_ids = np.unique(entity.get_data(line_field.value)[0].values)
            value = [line_ids[0]]

        lines.options = line_ids
        lines.value = value

    objects = Dropdown(options=names, value=names[0], description="Object:",)

    obj = get_parental_child(objects.value)[0]

    order = np.sort(obj.vertices[:, 0])

    entity = get_parental_child(objects.value)[0]

    data_list = entity.get_data_list()
    line_field = Dropdown(
        options=data_list,
        value=find_value(data_list, ["line"]),
        description="Lines field",
    )

    options = [pg.name for pg in obj.property_groups]
    options = [option for option in sorted(options)]
    groups = SelectMultiple(options=options, value=[options[0]], description="Data: ",)

    if line_field.value is None:
        line_list = []
        value = []
    else:

        line_list = np.unique(entity.get_data(line_field.value)[0].values)
        value = [line_list[0]]

    lines = SelectMultiple(options=line_list, value=value, description="Data: ")

    objects.observe(updateList, names="value")

    scale = Dropdown(
        options=["linear", "symlog"], value="symlog", description="Scaling",
    )

    threshold = FloatSlider(
        min=-16,
        max=-1,
        value=-12,
        steps=0.5,
        description="Log-linear threshold",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    apps = VBox([objects, line_field, lines, groups, scale, threshold])
    layout = HBox(
        [
            apps,
            interactive_output(
                plot_profiles,
                {
                    "entity_name": objects,
                    "groups": groups,
                    "line_field": line_field,
                    "lines": lines,
                    "scale": scale,
                    "threshold": threshold,
                },
            ),
        ]
    )
    return layout
