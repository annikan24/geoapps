import json
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, Octree, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import (
    Dropdown,
    HBox,
    Label,
    Layout,
    Text,
    VBox,
    Checkbox,
    FloatText,
    ToggleButton,
    IntText,
    Widget,
)

from geoapps.base import BaseApplication, working_copy
from geoapps.plotting import plot_plan_data_selection, PlotSelection2D
from geoapps.utils import find_value, rotate_xy, geophysical_systems
from geoapps.selection import ObjectDataSelection, LineOptions


class ChannelOptions(BaseApplication):
    """
    Options for data channels
    """

    def __init__(self, key, description, **kwargs):

        self._active = Checkbox(value=False, indent=True, description="Active",)
        self._label = widgets.Text(description=description)
        self._channel_selection = Dropdown(description="Associated Data:")
        self._channel_selection.header = key

        self._uncertainties = widgets.Text(description="Error (%, floor)")
        self._offsets = widgets.Text(description="Offsets (x,y,z)")

        self._widget = VBox(
            [
                self._active,
                self._label,
                self._channel_selection,
                self._uncertainties,
                self._offsets,
            ]
        )

        super().__init__(**kwargs)

    @property
    def active(self):
        return self._active

    @property
    def label(self):
        return self._label

    @property
    def channel_selection(self):
        return self._channel_selection

    @property
    def uncertainties(self):
        return self._uncertainties

    @property
    def offsets(self):
        return self._offsets

    @property
    def widget(self):
        return self._widget


class SensorOptions(BaseApplication):
    """
    Define the receiver spatial parameters
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.selection = ObjectDataSelection(**kwargs)
        self._objects = self.selection.objects
        self._value = self.selection.data

        self._value.description = "Height Channel"

        self._offset = Text(description="[dx, dy, dz]", value="0, 0, 0")

        self._constant = FloatText(description="Constant elevation (m)",)
        if "offset" in kwargs.keys():
            self._offset.value = kwargs["value"]

        self._options = {
            "(x, y, z) + offset(x,y,z)": self._offset,
            "(x, y, topo + radar) + offset(x,y,z)": VBox([self._offset, self._value]),
        }
        self._options_button = widgets.RadioButtons(
            options=[
                "(x, y, z) + offset(x,y,z)",
                "(x, y, topo + radar) + offset(x,y,z)",
            ],
            description="Define by:",
        )

        def update_options(_):
            self.update_options()

        self._options_button.observe(update_options)
        self._widget = VBox(
            [self._options_button, self._options[self._options_button.value]]
        )

        super().__init__(**kwargs)

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list)

        self.update_list()

        if "value" in kwargs.keys() and kwargs["value"] in self._value.options:
            self._value.value = kwargs["value"]

    @property
    def objects(self):
        return self._objects

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    @property
    def options_button(self):
        return self._options_button

    def update_list(self):
        if self.workspace.get_entity(self.objects.value):
            obj = self.workspace.get_entity(self.objects.value)[0]
            data_list = obj.get_data_list() + [""]

            self._value.options = [
                name for name in data_list if "visual" not in name.lower()
            ]

            self._value.value = ""

    def update_options(self):
        self._widget.children = [
            self._options_button,
            self._options[self._options_button.value],
        ]

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class TopographyOptions(BaseApplication):
    """
    Define the topography used by the inversion
    """

    def __init__(self, **kwargs):

        self.selection = ObjectDataSelection(**kwargs)
        self._objects = self.selection.objects
        self._value = self.selection.data
        self._offset = FloatText(description="Vertical offset (m)")
        self._constant = FloatText(description="Constant elevation (m)",)

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list, names="value")

        super().__init__(**kwargs)

        self.objects.value = find_value(self.objects.options, ["topo", "dem", "dtm"])

        self._options = {
            "Object": self.selection.widget,
            "Drape Height": self.offset,
            "Constant": self.constant,
            "None": widgets.Label("No topography"),
        }
        self._options_button = widgets.RadioButtons(
            options=["Object", "Drape Height", "Constant"], description="Define by:",
        )

        def update_options(_):
            self.update_options()

        self._options_button.observe(update_options)
        self._widget = VBox(
            [self._options_button, self._options[self._options_button.value]]
        )

    @property
    def panel(self):
        return self._panel

    @property
    def constant(self):
        return self._constant

    @property
    def objects(self):
        return self._objects

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    @property
    def options_button(self):
        return self._options_button

    def update_list(self):
        w_s = Workspace(self.h5file)
        if w_s.get_entity(self.objects.value):
            data_list = w_s.get_entity(self.objects.value)[0].get_data_list()
            self._value.options = [
                name for name in data_list if "visual" not in name.lower()
            ] + ["Vertices"]
            self._value.value = find_value(
                self._value.options, ["dem", "topo", "dtm"], default="Vertices"
            )

    def update_options(self):
        self._widget.children = [
            self._options_button,
            self._options[self._options_button.value],
        ]

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class MeshOctreeOptions(BaseApplication):
    """
    Widget used for the creation of an octree meshes
    """

    def __init__(self, **kwargs):
        self._core_cell_size = widgets.Text(
            value="25, 25, 25", description="Smallest cells",
        )
        self._octree_levels_topo = widgets.Text(
            value="0, 0, 0, 2", description="Layers below topo",
        )
        self._octree_levels_obs = widgets.Text(
            value="5, 5, 5, 5", description="Layers below data",
        )
        self._depth_core = FloatText(value=500, description="Minimum depth (m)",)
        self._padding_distance = widgets.Text(
            value="0, 0, 0, 0, 0, 0", description="Padding [W,E,N,S,D,U] (m)",
        )

        self._max_distance = FloatText(
            value=1000, description="Max triangulation length",
        )

        self._widget = widgets.VBox(
            [
                Label("Octree Mesh"),
                self._core_cell_size,
                self._octree_levels_topo,
                self._octree_levels_obs,
                self._depth_core,
                self._padding_distance,
                self._max_distance,
            ]
        )

        super().__init__(**kwargs)

    @property
    def core_cell_size(self):
        return self._core_cell_size

    @property
    def depth_core(self):
        return self._depth_core

    @property
    def max_distance(self):
        return self._max_distance

    @property
    def octree_levels_obs(self):
        return self._octree_levels_obs

    @property
    def octree_levels_topo(self):
        return self._octree_levels_topo

    @property
    def padding_distance(self):
        return self._padding_distance

    @property
    def widget(self):
        return self._widget


class Mesh1DOptions(BaseApplication):
    """
    Widget used for the creation of a 1D mesh
    """

    def __init__(self, **kwargs):
        self._hz_expansion = FloatText(value=1.05, description="Expansion factor:",)
        self._hz_min = FloatText(value=10.0, description="Smallest cell (m):",)
        self._n_cells = FloatText(value=25.0, description="Number of cells:",)

        super().__init__(**kwargs)

        self.cell_count = Label(f"Max depth: {self.count_cells():.2f} m")

        def update_hz_count(_):
            self.update_hz_count()

        self.n_cells.observe(update_hz_count)
        self.hz_expansion.observe(update_hz_count)
        self.hz_min.observe(update_hz_count)

        self._widget = VBox(
            [
                Label("1D Mesh"),
                self.hz_min,
                self.hz_expansion,
                self.n_cells,
                self.cell_count,
            ]
        )

    def count_cells(self):
        return (
            self.hz_min.value * self.hz_expansion.value ** np.arange(self.n_cells.value)
        ).sum()

    def update_hz_count(self):
        self.cell_count.value = f"Max depth: {self.count_cells():.2f} m"

    @property
    def hz_expansion(self):
        """

        """
        return self._hz_expansion

    @property
    def hz_min(self):
        """

        """
        return self._hz_min

    @property
    def n_cells(self):
        """

        """
        return self._n_cells

    @property
    def widget(self):
        return self._widget


class ModelOptions(BaseApplication):
    """
    Widgets for the selection of model options
    """

    def __init__(self, name_list, **kwargs):

        self._options = widgets.RadioButtons(
            options=["Model", "Value"], value="Value", disabled=False,
        )

        def update_panel(_):
            self.update_panel()

        self._options.observe(update_panel, names="value")
        self._list = widgets.Dropdown(description="3D Model", options=name_list)
        self._value = FloatText(description="Units")
        self._description = Label()
        self._widget = widgets.VBox(
            [self._description, widgets.VBox([self._options, self._value])]
        )

        super().__init__(**kwargs)

        for key, value in kwargs.items():
            if key == "units":
                self._value.description = value
            elif key == "label":
                self._description.value = value

    def update_panel(self):

        if self._options.value == "Model":
            self._widget.children[1].children = [self._options, self._list]
            self._widget.children[1].children[1].layout.visibility = "visible"
        elif self._options.value == "Value":
            self._widget.children[1].children = [self._options, self._value]
            self._widget.children[1].children[1].layout.visibility = "visible"
        else:
            self._widget.children[1].children[1].layout.visibility = "hidden"

    @property
    def description(self):
        return self._description

    @property
    def list(self):
        return self._list

    @property
    def options(self):
        return self._options

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class InversionOptions(BaseApplication):
    """
    Collection of widgets controlling the inversion parameters
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        model_list = []
        for obj in self.workspace.all_objects():
            if isinstance(obj, (BlockModel, Octree, Surface)):
                for data in obj.children:
                    if (
                        getattr(data, "values", None) is not None
                        and data.name != "Visual Parameters"
                    ):
                        model_list += [data.name]

        self._output_name = widgets.Text(
            value="Inversion_", description="Save to:", disabled=False
        )

        self._chi_factor = FloatText(
            value=1, description="Target misfit", disabled=False
        )
        self._uncert_mode = widgets.RadioButtons(
            options=[
                "Estimated (%|data| + background)",
                r"User input (\%|data| + floor)",
            ],
            value=r"User input (\%|data| + floor)",
            disabled=False,
        )
        self._lower_bound = widgets.Text(value=None, description="Lower bound value",)
        self._upper_bound = widgets.Text(value=None, description="Upper bound value",)

        self._ignore_values = widgets.Text(value="<0", tooltip="Dummy value",)
        self._max_iterations = IntText(value=10, description="Max beta Iterations")
        self._max_cg_iterations = IntText(value=10, description="Max CG Iterations")
        self._tol_cg = FloatText(value=1e-4, description="CG Tolerance")

        self._beta_start_options = widgets.RadioButtons(
            options=["value", "ratio"],
            value="ratio",
            description="Starting tradeoff (beta):",
        )
        self._beta_start = FloatText(value=1e2, description="phi_d/phi_m")

        def initial_beta_change(_):
            self.initial_beta_change()

        self._beta_start_options.observe(initial_beta_change)

        self._beta_start_panel = HBox([self._beta_start_options, self._beta_start])
        self._optimization = VBox(
            [
                self._max_iterations,
                self._chi_factor,
                self._beta_start_panel,
                self._max_cg_iterations,
                self._tol_cg,
            ]
        )
        self._starting_model = ModelOptions(model_list)
        self._susceptibility_model = ModelOptions(
            model_list, units="SI", label="Background susceptibility"
        )
        self._susceptibility_model.options.options = ["None", "Model", "Value"]
        self._reference_model = ModelOptions(model_list)
        self._reference_model.options.options = [
            "None",
            "Best-fitting halfspace",
            "Model",
            "Value",
        ]

        def update_ref(_):
            self.update_ref()

        self.reference_model.options.observe(update_ref)

        self._alphas = widgets.Text(
            value="1, 1, 1, 1", description="Scaling alpha_(s, x, y, z)",
        )
        self._norms = widgets.Text(
            value="2, 2, 2, 2", description="Norms p_(s, x, y, z)",
        )

        def check_max_iterations(_):
            self.check_max_iterations()

        self._norms.observe(check_max_iterations)

        self._mesh = MeshOctreeOptions()
        self.inversion_options = {
            "output name": self._output_name,
            "uncertainties": self._uncert_mode,
            "starting model": self._starting_model.widget,
            "background susceptibility": self._susceptibility_model.widget,
            "regularization": VBox(
                [self._reference_model.widget, self._alphas, self._norms]
            ),
            "upper-lower bounds": VBox([self._upper_bound, self._lower_bound]),
            "mesh": self.mesh.widget,
            "ignore values (<0 = no negatives)": self._ignore_values,
            "optimization": self._optimization,
        }
        self.option_choices = widgets.Dropdown(
            options=list(self.inversion_options.keys()),
            value=list(self.inversion_options.keys())[0],
            disabled=False,
        )

        def inversion_option_change(_):
            self.inversion_option_change()

        self.option_choices.observe(inversion_option_change)

        self._widget = widgets.VBox(
            [
                widgets.HBox([widgets.Label("Inversion Options")]),
                widgets.HBox(
                    [
                        self.option_choices,
                        self.inversion_options[self.option_choices.value],
                    ],
                ),
            ],
            layout=Layout(width="100%"),
        )

        super().__init__(**kwargs)

    def check_max_iterations(self):
        if not all([val == 2 for val in string_2_list(self._norms.value)]):
            self._max_iterations.value = 30
        else:
            self._max_iterations.value = 10

    def inversion_option_change(self):
        self._widget.children[1].children = [
            self.option_choices,
            self.inversion_options[self.option_choices.value],
        ]

    def initial_beta_change(self):
        if self._beta_start_options.value == "ratio":
            self._beta_start.description = "phi_d/phi_m"
        else:
            self._beta_start.description = ""

    @property
    def alphas(self):
        return self._alphas

    @property
    def beta_start(self):
        return self._beta_start

    @property
    def beta_start_options(self):
        return self._beta_start_options

    @property
    def chi_factor(self):
        return self._chi_factor

    @property
    def ignore_values(self):
        return self._ignore_values

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def max_cg_iterations(self):
        return self._max_cg_iterations

    @property
    def mesh(self):
        return self._mesh

    @property
    def norms(self):
        return self._norms

    @property
    def output_name(self):
        return self._output_name

    @property
    def reference_model(self):
        return self._reference_model

    @property
    def starting_model(self):
        return self._starting_model

    @property
    def susceptibility_model(self):
        return self._susceptibility_model

    @property
    def tol_cg(self):
        return self._tol_cg

    @property
    def uncert_mode(self):
        return self._uncert_mode

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Pre-defined application layout
        """
        return self._widget

    def update_ref(self):
        alphas = string_2_list(self.alphas.value)
        if self.reference_model.options.value == "None":
            alphas[0] = 0.0
        else:
            alphas[0] = 1.0
        self.alphas.value = ", ".join(list(map(str, alphas)))


def get_inversion_output(h5file, group_name):
    """
    Recover an inversion iterations from a ContainerGroup comments.
    """
    workspace = Workspace(h5file)
    out = {"time": [], "iteration": [], "phi_d": [], "phi_m": [], "beta": []}

    if workspace.get_entity(group_name):
        group = workspace.get_entity(group_name)[0]

        for comment in group.comments.values:
            if "Iteration" in comment["Author"]:
                out["iteration"] += [np.int(comment["Author"].split("_")[1])]
                out["time"] += [comment["Date"]]

                values = json.loads(comment["Text"])

                out["phi_d"] += [np.float(values["phi_d"])]
                out["phi_m"] += [np.float(values["phi_m"])]
                out["beta"] += [np.float(values["beta"])]

        if len(out["iteration"]) > 0:
            out["iteration"] = np.hstack(out["iteration"])
            ind = np.argsort(out["iteration"])
            out["iteration"] = out["iteration"][ind]
            out["phi_d"] = np.hstack(out["phi_d"])[ind]
            out["phi_m"] = np.hstack(out["phi_m"])[ind]
            out["time"] = np.hstack(out["time"])[ind]

    return out


def plot_convergence_curve(h5file):
    """

    """
    workspace = Workspace(h5file)

    names = [
        group.name
        for group in workspace.all_groups()
        if isinstance(group, ContainerGroup)
    ]

    objects = widgets.Dropdown(
        options=names, value=names[0], description="Inversion Group:",
    )

    def plot_curve(objects):

        inversion = workspace.get_entity(objects)[0]
        result = None
        if getattr(inversion, "comments", None) is not None:
            if inversion.comments.values is not None:
                result = get_inversion_output(workspace.h5file, objects)
                iterations = result["iteration"]
                phi_d = result["phi_d"]
                phi_m = result["phi_m"]

                ax1 = plt.subplot()
                ax2 = ax1.twinx()
                ax1.plot(iterations, phi_d, linewidth=3, c="k")
                ax1.set_xlabel("Iterations")
                ax1.set_ylabel(r"$\phi_d$", size=16)
                ax2.plot(iterations, phi_m, linewidth=3, c="r")
                ax2.set_ylabel(r"$\phi_m$", size=16)

        return result

    interactive_plot = widgets.interactive(plot_curve, objects=objects)

    return interactive_plot


def inversion_defaults():
    """
    Get defaults for gravity, magnetics and EM1D inversions
    """
    defaults = {
        "units": {"Gravity": "g/cc", "Magnetics": "SI", "EM1D": "S/m"},
        "property": {
            "Gravity": "density",
            "Magnetics": "effective susceptibility",
            "EM1D": "conductivity",
        },
        "reference_value": {"Gravity": 0.0, "Magnetics": 0.0, "EM1D": 1e-3},
        "starting_value": {"Gravity": 1e-4, "Magnetics": 1e-4, "EM1D": 1e-3},
    }

    return defaults


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [np.float(val) for val in string.split(",") if len(val) > 0]


class InversionApp(BaseApplication):
    def __init__(self, **kwargs):

        kwargs = working_copy(**kwargs)

        super().__init__(**kwargs)

        # Load all known em systems
        self.em_system_specs = geophysical_systems.parameters()
        self._data_count = (Label("Data Count: 0", tooltip="Keep <1500 for speed"),)
        self._forward_only = Checkbox(
            value=False,
            description="Forward only",
            tooltip="Forward response of reference model",
        )
        self._inducing_field = widgets.Text(
            description="Inducing Field [Amp, Inc, Dec]",
        )
        self._inversion_parameters = InversionOptions(**kwargs)
        self.mesh_octree = MeshOctreeOptions(**kwargs)
        self.mesh_1D = Mesh1DOptions(**kwargs)
        self._run = ToggleButton(
            value=False, description="Run SimPEG", button_style="danger", icon="check"
        )
        self._starting_channel = (IntText(value=None, description="Starting Channel"),)
        self._system = Dropdown(
            options=["Magnetics", "Gravity"] + list(self.em_system_specs.keys()),
            description="Survey Type: ",
        )
        self._write = ToggleButton(
            value=False,
            description="Write input",
            button_style="warning",
            tooltip="Write json input file",
            icon="check",
        )
        self.plot_selection = PlotSelection2D(
            select_multiple=True, add_groups=True, **kwargs
        )
        self.selection = self.plot_selection.selection

        # Link the data extent with the octree mesh param
        def update_octree_param(_):
            self.update_octree_param()

        for item in self.plot_selection.__dict__.values():
            if isinstance(item, Widget):
                item.observe(update_octree_param)

        dsep = os.path.sep
        if self.h5file is not None:
            self._inv_dir = (
                dsep.join(os.path.dirname(os.path.abspath(self.h5file)).split(dsep))
                + dsep
            )
        else:
            self._inv_dir = os.getcwd() + dsep

        def run_trigger(_):
            self.run_trigger()

        self.run.observe(run_trigger)

        def write_trigger(_):
            self.write_trigger()

        self.write.observe(write_trigger)

        def system_observer(_):
            self.system_observer()

        self.system.observe(system_observer, names="value")

        def update_options(_):
            self.update_options()

        for item in self.inversion_parameters.__dict__.values():
            if isinstance(item, Widget):
                item.observe(update_options)
            elif isinstance(item, BaseApplication):
                for val in item.__dict__.values():
                    if isinstance(val, Widget):
                        val.observe(update_options)

        super().__init__(**kwargs)

        self.plot_selection.widget.layout = Layout(width="60%")

        def object_observer(_):
            self.object_observer()

        self.selection.objects.observe(object_observer, names="value")

        def update_component_panel(_):
            self.update_component_panel()

        self.selection.data.observe(update_component_panel, names="value")

        self.data_channel_choices = widgets.Dropdown()

        def data_channel_choices_observer(_):
            self.data_channel_choices_observer()

        self.data_channel_choices.observe(data_channel_choices_observer, names="value")
        self.data_channel_panel = widgets.VBox([self.data_channel_choices])
        self.survey_type_panel = VBox([self.system])

        self.topography = TopographyOptions(**kwargs)

        # Define bird parameters
        sub_kwargs = kwargs.copy()
        if "objects" in sub_kwargs.keys():
            del sub_kwargs["objects"]

        self.sensor = SensorOptions(objects=self.selection.objects, **sub_kwargs)
        self.lines = LineOptions(
            objects=self.selection.objects,
            select_multiple_lines=True,
            find_value="line",
            **sub_kwargs,
        )

        def update_selection(_):
            self.update_selection()

        self.lines.lines.observe(update_selection, names="value")

        # SPATIAL PARAMETERS DROPDOWN
        self.spatial_options = {
            "Topography": self.topography.widget,
            "Receivers": self.sensor.widget,
            "Line ID": self.lines.widget,
        }
        self.spatial_choices = widgets.Dropdown(
            options=list(self.spatial_options.keys()),
            value=list(self.spatial_options.keys())[0],
            disabled=False,
        )

        def spatial_option_change(_):
            self.spatial_option_change()

        self.spatial_choices.observe(spatial_option_change)

        self.spatial_panel = VBox(
            [self.spatial_choices, self.spatial_options[self.spatial_choices.value]]
        )

        self._widget = VBox(
            [
                HBox(
                    [
                        self.plot_selection.widget,
                        VBox(
                            [
                                VBox([Label(""), self.survey_type_panel]),
                                VBox([Label("Components"), self.data_channel_panel]),
                            ],
                            layout=Layout(width="40%"),
                        ),
                    ]
                ),
                VBox([Label("Spatial Information"), self.spatial_panel]),
                self.inversion_parameters.widget,
                self.forward_only,
                self.write,
                self.run,
            ]
        )

        self.object_observer()
        self.inversion_parameters.update_ref()

    @property
    def data_count(self):
        """

        """
        return self._data_count

    @property
    def forward_only(self):
        """

        """
        return self._forward_only

    @property
    def inducing_field(self):
        """

        """
        return self._inducing_field

    @property
    def inversion_parameters(self):
        """

        """
        return self._inversion_parameters

    @property
    def run(self):
        """

        """
        return self._run

    @property
    def starting_channel(self):
        """

        """
        return self._starting_channel

    @property
    def system(self):
        """

        """
        return self._system

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Pre-defined application layout
        """
        return self._widget

    @property
    def write(self):
        """

        """
        return self._write

    def run_trigger(self):
        """

        """
        if self.run.value:
            if self.system.value in ["Gravity", "Magnetics"]:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python -m geoapps.pf_inversion "'
                    + self._inv_dir
                    + f'\\{self.inversion_parameters.output_name.value}.json"'
                )
            else:
                os.system(
                    "start cmd.exe @cmd /k "
                    + 'python -m geoapps.em1d_inversion "'
                    + self._inv_dir
                    + f'\\{self.inversion_parameters.output_name.value}.json"'
                )
            self.run.value = False
            self.run.button_style = ""

    # Check parameters change
    def update_options(self):
        self.write.button_style = "warning"
        self.run.button_style = "danger"

    def system_observer(self):
        """
        Change the application on change of system
        """
        if self.system.value in ["Magnetics", "Gravity"]:
            start_channel = 0
            if self.system.value == "Magnetics":
                data_type_list = ["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"]
                labels = ["tmi", "bxx", "bxy", "bxz", "byy", "byz", "bzz"]

            else:
                data_type_list = ["gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
                labels = ["gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]

            tx_offsets = [[0, 0, 0]]
            uncertainties = [[0, 1]] * len(data_type_list)

            system_specs = {}
            for key in data_type_list:
                system_specs[key] = key

            # Remove line_id from choices
            self.spatial_choices.options = list(self.spatial_options.keys())[:2]

            # Switch mesh options
            self.inversion_parameters._mesh = self.mesh_octree
            self.inversion_parameters.inversion_options[
                "mesh"
            ] = self.mesh_octree.widget

            self.inversion_parameters.reference_model.options.options = [
                "None",
                "Model",
                "Value",
            ]
            self.inversion_parameters.reference_model.options.value = "Value"
            flag = self.system.value

            self.inversion_parameters.lower_bound.value = ""
            self.inversion_parameters.upper_bound.value = ""
            self.inversion_parameters.ignore_values.value = "-99999"

        else:
            tx_offsets = self.em_system_specs[self.system.value]["tx_offsets"]
            self.sensor.offset.value = ", ".join(
                [
                    str(offset)
                    for offset in self.em_system_specs[self.system.value]["bird_offset"]
                ]
            )
            uncertainties = self.em_system_specs[self.system.value]["uncertainty"]

            # if start_channel is None:
            start_channel = self.em_system_specs[self.system.value][
                "channel_start_index"
            ]
            if self.em_system_specs[self.system.value]["type"] == "time":
                labels = ["Time (s)"] * len(
                    self.em_system_specs[self.system.value]["channels"]
                )
            else:
                labels = ["Frequency (Hz)"] * len(
                    self.em_system_specs[self.system.value]["channels"]
                )

            system_specs = {}
            for key, time in self.em_system_specs[self.system.value][
                "channels"
            ].items():
                system_specs[key] = f"{time:.5e}"

            self.spatial_choices.options = list(self.spatial_options.keys())

            self.inversion_parameters.reference_model.options.options = [
                "Best-fitting halfspace",
                "Model",
                "Value",
            ]
            self.inversion_parameters.reference_model.options.value = (
                "Best-fitting halfspace"
            )
            self.inversion_parameters.lower_bound.value = "1e-5"
            self.inversion_parameters.upper_bound.value = "10"
            self.inversion_parameters.ignore_values.value = "<0"
            # Switch mesh options
            self.inversion_parameters._mesh = self.mesh_1D
            self.inversion_parameters.inversion_options["mesh"] = self.mesh_1D.widget
            flag = "EM1D"

        self.inversion_parameters.reference_model.value.description = inversion_defaults()[
            "units"
        ][
            flag
        ]
        self.inversion_parameters.reference_model.value.value = inversion_defaults()[
            "reference_value"
        ][flag]
        self.inversion_parameters.reference_model.description.value = (
            "Reference " + inversion_defaults()["property"][flag]
        )
        self.inversion_parameters.starting_model.value.description = inversion_defaults()[
            "units"
        ][
            flag
        ]
        self.inversion_parameters.starting_model.value.value = inversion_defaults()[
            "starting_value"
        ][flag]
        self.inversion_parameters.starting_model.description.value = (
            "Starting " + inversion_defaults()["property"][flag]
        )

        self.spatial_choices.value = self.spatial_choices.options[0]

        def channel_setter(caller):

            channel = caller["owner"]
            data_widget = self.data_channel_choices.data_channel_options[channel.header]

            entity = self.workspace.get_entity(self.selection.objects.value)[0]
            if channel.value is None or not entity.get_data(channel.value):
                data_widget.children[0].value = False
                if self.system.value in ["Magnetics", "Gravity"]:
                    data_widget.children[3].value = "0, 1"
            else:
                data_widget.children[0].value = True
                if self.system.value in ["Magnetics", "Gravity"]:
                    values = entity.get_data(channel.value)[0].values
                    if values is not None and isinstance(values[0], float):
                        data_widget.children[
                            3
                        ].value = f"0, {np.percentile(values[values > 2e-18], 5):.2f}"

            # Trigger plot update
            if self.data_channel_choices.value == channel.header:
                self.plot_selection.plotting_data = channel
                self.plot_selection.refresh.value = False
                self.plot_selection.refresh.value = True

        data_channel_options = {}
        for ind, (key, channel) in enumerate(system_specs.items()):
            if ind + 1 < start_channel:
                continue

            if len(tx_offsets) > 1:
                offsets = tx_offsets[ind]
            else:
                offsets = tx_offsets[0]

            channel_options = ChannelOptions(
                key,
                labels[ind],
                uncertainties=", ".join(
                    [str(uncert) for uncert in uncertainties[ind][:2]]
                ),
                offsets=", ".join([str(offset) for offset in offsets]),
            )

            channel_options.channel_selection.observe(channel_setter, names="value")

            data_channel_options[key] = channel_options.widget

            if self.system.value not in ["Magnetics", "Gravity"]:
                data_channel_options[key].children[1].value = channel
            else:
                data_channel_options[key].children[1].layout.visibility = "hidden"
                data_channel_options[key].children[4].layout.visibility = "hidden"

        if len(data_channel_options) > 0:
            self.data_channel_choices.options = list(data_channel_options.keys())
            self.data_channel_choices.value = list(data_channel_options.keys())[0]
            self.data_channel_choices.data_channel_options = data_channel_options
            self.data_channel_panel.children = [
                self.data_channel_choices,
                data_channel_options[self.data_channel_choices.value],
            ]

        self.update_component_panel()

        if (
            self.system.value not in ["Magnetics", "Gravity"]
            and self.em_system_specs[self.system.value]["type"] == "frequency"
        ):
            self.inversion_parameters.option_choices.options = list(
                self.inversion_parameters.inversion_options.keys()
            )
        else:
            self.inversion_parameters.option_choices.options = [
                key
                for key in self.inversion_parameters.inversion_options.keys()
                if key != "background susceptibility"
            ]

        self.write.button_style = "warning"
        self.run.button_style = "danger"

        if self.system.value == "Magnetics":
            self.survey_type_panel.children = [self.system, self.inducing_field]
        else:
            self.survey_type_panel.children = [self.system]

    def object_observer(self):

        self.plot_selection.resolution.indices = None

        if self.workspace.get_entity(self.selection.objects.value):
            obj = self.workspace.get_entity(self.selection.objects.value)[0]
            data_list = obj.get_data_list()
            # lines.update_list()
            # topography.update_list()

            for aem_system, specs in self.em_system_specs.items():
                if any([specs["flag"] in channel for channel in data_list]):
                    self.system.value = aem_system

            self.system_observer()

            if hasattr(self.data_channel_choices, "data_channel_options"):
                for (
                    key,
                    data_widget,
                ) in self.data_channel_choices.data_channel_options.items():
                    data_widget.children[2].options = self.selection.data.options
                    value = find_value(self.selection.data.options, [key])
                    data_widget.children[2].value = value

            self.write.button_style = "warning"
            self.run.button_style = "danger"

    def get_data_list(self, entity):
        groups = [p_g.name for p_g in entity.property_groups]
        data_list = []
        if self.selection.data.value is not None:
            for component in self.selection.data.value:
                if component in groups:
                    data_list += [
                        self.workspace.get_entity(data)[0].name
                        for data in entity.get_property_group(component).properties
                    ]
                elif component in entity.get_data_list():
                    data_list += [component]
        return data_list

    def update_component_panel(self):
        if self.workspace.get_entity(self.selection.objects.value):
            entity = self.workspace.get_entity(self.selection.objects.value)[0]
            data_list = self.get_data_list(entity)

            if hasattr(self.data_channel_choices, "data_channel_options"):
                for (
                    key,
                    data_widget,
                ) in self.data_channel_choices.data_channel_options.items():
                    data_widget.children[2].options = data_list
                    value = find_value(data_list, [key])
                    data_widget.children[2].value = value

    def data_channel_choices_observer(self):
        if hasattr(
            self.data_channel_choices, "data_channel_options"
        ) and self.data_channel_choices.value in (
            self.data_channel_choices.data_channel_options.keys()
        ):
            data_widget = self.data_channel_choices.data_channel_options[
                self.data_channel_choices.value
            ]
            self.data_channel_panel.children = [self.data_channel_choices, data_widget]

            if (
                self.workspace.get_entity(self.selection.objects.value)
                and data_widget.children[2].value is None
            ):
                entity = self.workspace.get_entity(self.selection.objects.value)[0]
                data_list = self.get_data_list(entity)
                value = find_value(data_list, [self.data_channel_choices.value])
                data_widget.children[2].value = value

        self.write.button_style = "warning"
        self.run.button_style = "danger"

    def spatial_option_change(self):
        self.spatial_panel.children = [
            self.spatial_choices,
            self.spatial_options[self.spatial_choices.value],
        ]
        self.write.button_style = "warning"
        self.run.button_style = "danger"

    def update_octree_param(self):
        dl = self.plot_selection.resolution.value
        self.mesh_octree.core_cell_size.value = f"{dl/2:.0f}, {dl/2:.0f}, {dl/2:.0f}"
        self.mesh_octree.depth_core.value = np.ceil(
            np.min([self.plot_selection.width.value, self.plot_selection.height.value])
            / 2.0
        )
        self.mesh_octree.padding_distance.value = ", ".join(
            list(
                map(
                    str,
                    [
                        np.ceil(self.plot_selection.width.value / 2),
                        np.ceil(self.plot_selection.width.value / 2),
                        np.ceil(self.plot_selection.height.value / 2),
                        np.ceil(self.plot_selection.height.value / 2),
                        0,
                        0,
                    ],
                )
            )
        )
        self.plot_selection.resolution.indices = None
        self.write.button_style = "warning"
        self.run.button_style = "danger"

    def update_selection(self):
        self.plot_selection.highlight_selection = {
            self.lines.data.value: self.lines.lines.value
        }
        self.plot_selection.refresh.value = False
        self.plot_selection.refresh.value = True

    def write_trigger(self):
        if self.write.value is False:
            return

        input_dict = {
            "out_group": self.inversion_parameters.output_name.value,
            "workspace": self.h5file,
            "save_to_geoh5": self.h5file,
        }
        if self.system.value in ["Gravity", "Magnetics"]:
            input_dict["inversion_type"] = self.system.value.lower()

            if input_dict["inversion_type"] == "magnetics":
                input_dict["inducing_field_aid"] = string_2_list(
                    self.inducing_field.value
                )
            # Octree mesh parameters
            input_dict["core_cell_size"] = string_2_list(
                self.mesh_octree.core_cell_size.value
            )
            input_dict["octree_levels_topo"] = string_2_list(
                self.mesh_octree.octree_levels_topo.value
            )
            input_dict["octree_levels_obs"] = string_2_list(
                self.mesh_octree.octree_levels_obs.value
            )
            input_dict["depth_core"] = {"value": self.mesh_octree.depth_core.value}
            input_dict["max_distance"] = self.mesh_octree.max_distance.value
            p_d = string_2_list(self.mesh_octree.padding_distance.value)
            input_dict["padding_distance"] = [
                [p_d[0], p_d[1]],
                [p_d[2], p_d[3]],
                [p_d[4], p_d[5]],
            ]

        else:
            input_dict["system"] = self.system.value
            input_dict["lines"] = {
                self.lines.data.value: [str(line) for line in self.lines.lines.value]
            }

            input_dict["mesh 1D"] = [
                self.mesh_1D.hz_min.value,
                self.mesh_1D.hz_expansion.value,
                self.mesh_1D.n_cells.value,
            ]
        input_dict["chi_factor"] = self.inversion_parameters.chi_factor.value
        input_dict["max_iterations"] = self.inversion_parameters.max_iterations.value
        input_dict[
            "max_cg_iterations"
        ] = self.inversion_parameters.max_cg_iterations.value

        if self.inversion_parameters.beta_start_options.value == "value":
            input_dict["initial_beta"] = self.inversion_parameters.beta_start.value
        else:
            input_dict[
                "initial_beta_ratio"
            ] = self.inversion_parameters.beta_start.value

        input_dict["tol_cg"] = self.inversion_parameters.tol_cg.value
        input_dict["ignore_values"] = self.inversion_parameters.ignore_values.value
        input_dict["resolution"] = self.plot_selection.resolution.value
        input_dict["window"] = {
            "center": [
                self.plot_selection.center_x.value,
                self.plot_selection.center_y.value,
            ],
            "size": [self.plot_selection.width.value, self.plot_selection.height.value],
            "azimuth": self.plot_selection.azimuth.value,
        }
        input_dict["alphas"] = string_2_list(self.inversion_parameters.alphas.value)
        input_dict["reference_model"] = {
            self.inversion_parameters.reference_model.widget.children[1]
            .children[0]
            .value.lower(): self.inversion_parameters.reference_model.widget.children[1]
            .children[1]
            .value
        }
        input_dict["starting_model"] = {
            self.inversion_parameters.starting_model.widget.children[1]
            .children[0]
            .value.lower(): self.inversion_parameters.starting_model.widget.children[1]
            .children[1]
            .value
        }

        if self.inversion_parameters.susceptibility_model.options.value != "None":
            input_dict["susceptibility"] = {
                self.inversion_parameters.susceptibility_model.widget.children[1]
                .children[0]
                .value.lower(): self.inversion_parameters.susceptibility_model.widget.children[
                    1
                ]
                .children[1]
                .value
            }

        input_dict["model_norms"] = string_2_list(self.inversion_parameters.norms.value)

        if len(self.inversion_parameters.lower_bound.value) > 1:
            input_dict["lower_bound"] = string_2_list(
                self.inversion_parameters.lower_bound.value
            )

        if len(self.inversion_parameters.upper_bound.value) > 1:
            input_dict["upper_bound"] = string_2_list(
                self.inversion_parameters.upper_bound.value
            )

        input_dict["data"] = {}
        input_dict["data"]["type"] = "GA_object"
        input_dict["data"]["name"] = self.selection.objects.value

        if hasattr(self.data_channel_choices, "data_channel_options"):
            channel_param = {}

            for (
                key,
                data_widget,
            ) in self.data_channel_choices.data_channel_options.items():
                if data_widget.children[0].value is False:
                    continue

                channel_param[key] = {}
                channel_param[key]["name"] = data_widget.children[2].value
                channel_param[key]["uncertainties"] = string_2_list(
                    data_widget.children[3].value
                )
                channel_param[key]["offsets"] = string_2_list(
                    data_widget.children[4].value
                )

                if self.system.value not in ["Gravity", "Magnetics"]:
                    channel_param[key]["value"] = string_2_list(
                        data_widget.children[1].value
                    )
                if (
                    self.system.value in ["Gravity", "Magnetics"]
                    and self.plot_selection.azimuth.value != 0
                    and key not in ["tmi", "gz"]
                ):
                    print(
                        f"Gradient data with rotated window is currently not supported"
                    )
                    self.run.button_style = "danger"
                    return

            input_dict["data"]["channels"] = channel_param

        input_dict["uncertainty_mode"] = self.inversion_parameters.uncert_mode.value

        if self.sensor.options_button.value == "(x, y, z) + offset(x,y,z)":
            input_dict["receivers_offset"] = {
                "constant": string_2_list(self.sensor.offset.value)
            }
        else:
            input_dict["receivers_offset"] = {
                "radar_drape": string_2_list(self.sensor.offset.value)
                + [self.sensor.value.value]
            }

        if self.topography.options_button.value == "Object":
            if self.topography.objects.value is None:
                input_dict["topography"] = None
            else:
                input_dict["topography"] = {
                    "GA_object": {
                        "name": self.topography.objects.value,
                        "data": self.topography.value.value,
                    }
                }
        elif self.topography.options_button.value == "Drape Height":
            input_dict["topography"] = {"drapped": self.topography.offset.value}
        else:
            input_dict["topography"] = {"constant": self.topography.constant.value}

        if self.forward_only.value:
            input_dict["forward_only"] = []

        checks = [key for key, val in input_dict.items() if val is None]

        if len(list(input_dict["data"]["channels"].keys())) == 0:
            checks += ["'Channel' for at least one data component."]

        if len(checks) > 0:
            print(f"Required value for {checks}")
            self.run.button_style = "danger"
        else:
            self.write.button_style = ""
            file = self._inv_dir + f"{self.inversion_parameters.output_name.value}.json"
            with open(file, "w") as f:
                json.dump(input_dict, f, indent=4)
            self.run.button_style = "success"

        self.write.value = False
        self.write.button_style = ""
        self.run.button_style = "success"

    # for attr, item in kwargs.items():
    #     try:
    #         if hasattr(w_l[attr], "options") and item not in w_l[attr].options:
    #             continue
    #         w_l[attr].value = item
    #     except KeyError:
    #         continue
    # for attr, item in kwargs.items():
    #     try:
    #         if getattr(self.selection, attr, None) is not None:
    #             widget = getattr(self.selection, attr)
    #             if hasattr(widget, "options") and item not in widget.options:
    #                 continue
    #             widget.value = item
    #     except AttributeError:
    #         continue