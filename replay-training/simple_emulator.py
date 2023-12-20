"""
Create and train a "simple" emulator:
- 1 degree resolution
- assumes 6 hour timestep (see the temporal slicing command in .preprocess method, skipping by 2)
- only propagate a few variables
- only maintain a few vertical levels
- keep the hidden dimension/network size small

Purpose is to have the training pipeline working.
"""

import dataclasses
import xarray as xr
from jax import tree_util

from graphcast.graphcast import ModelConfig, TaskConfig
from graphcast.data_utils import extract_inputs_targets_forcings

class ReplayEmulator1Degree:

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp",
    )
    forcing_variables = (
        "land",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    all_variables = tuple() # this is created in __init__
    pressure_levels = (
        100,
        500,
        1000,
    )
    input_duration = "12h"

    # model config options
    mesh_size = 2
    latent_size = 32
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    mesh2grid_edge_normalization_factor = 0.6180338738074472

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0

    def __init__(self):

        ds = xr.open_zarr(
            "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr",
            storage_options={"token": "anon"},
        )
        levels = ds["pfull"].sel(
            pfull=list(self.pressure_levels),
            method="nearest",
        )
        self.model_config = ModelConfig(
            resolution=1.0,
            mesh_size=self.mesh_size,
            latent_size=self.latent_size,
            gnn_msg_steps=self.gnn_msg_steps,
            hidden_layers=self.hidden_layers,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
        )
        self.task_config = TaskConfig(
            input_variables=self.input_variables,
            target_variables=self.target_variables,
            forcing_variables=self.forcing_variables,
            pressure_levels=levels,
            input_duration=self.input_duration,
        )

        self.all_variables = tuple(set(
            self.input_variables + self.target_variables + self.forcing_variables
        ))

        self.norm = {}
        self.norm["mean"], self.norm["std"], self.norm["stddiff"] = self.load_normalization()
        ds.close()


    def preprocess(self, xds, n_steps, batch_index=0):
        """Prepare dataset for GraphCast

        Note:
            This assumes a 6 hour timestep with 3 hourly data because of the first line

            >>> bds = xds.isel(time=slice(None, (n_steps+1)*2+1, 2))

            which skips timesteps by 2

        Args:
            xds (xarray.Dataset): with replay data
            n_steps (int): number of timesteps made by the model during training
            batch_index (int, optional): this index of this batch

        Returns:
            bds (xarray.Dataset): this batch of data
        """

        # slice out this batch from time
        bds = xds.isel(time=slice(None, (n_steps+1)*2+1, 2))

        # select our vertical levels
        bds = bds.sel(pfull=list(self.pressure_levels), method="nearest")

        # only grab variables we care about
        myvars = list(x for x in self.all_variables if x in xds)
        bds = bds[myvars]

        bds = bds.rename({
            "pfull": "level",
            "grid_xt": "lon",
            "grid_yt": "lat",
            "time": "datetime",
        })

        # unclear if this is necessary for computation
        bds = bds.sortby("lat", ascending=True)

        bds["time"] = (bds.datetime - bds.datetime[0])
        bds = bds.swap_dims({"datetime": "time"}).reset_coords()
        bds = bds.expand_dims({
            "batch": [batch_index],
        })
        bds = bds.set_coords(["datetime", "cftime", "ftime"])
        return bds


    def get_training_batches(self, xds, batch_indices, n_steps):
        """Get a dataset with all the batches of data necessary for training

        Note:
            This uses :meth:`preprocess` and therefore assumes a 6 hour timestep
            with 3 hourly data.

        Args:
            xds (xarray.Dataset): the Replay dataset
            batch_indices (list): with the time index of data to grab
            n_steps (int): number of timesteps made by the model during training

        Returns:
            inputs, targets, forcings (xarray.Dataset): with new dimension "batch"
                and appropriate fields for each dataset, based on the variables in :attr:`task_config`
        """

        inputs = []
        targets = []
        forcings = []
        for index in batch_indices:
            batch = self.preprocess(
                xds.isel(time=slice(index, index+(n_steps+1)*2+1)),
                n_steps=n_steps,
                batch_index=index,
            )
            i, t, f = extract_inputs_targets_forcings(
                batch,
                target_lead_times=slice("6h", f"{n_steps*6}h"),
                **dataclasses.asdict(self.task_config),
            )
            inputs.append(i)
            targets.append(t)
            forcings.append(f)

        inputs = xr.concat(inputs, dim="batch")
        targets = xr.concat(targets, dim="batch")
        forcings = xr.concat(forcings, dim="batch")
        return inputs, targets, forcings



    def load_normalization(self, **kwargs):
        """Load the normalization fields into memory

        Returns:
            mean_by_level, stddev_by_level, diffs_stddev_by_level (xarray.Dataset): with normalization fields
        """

        def open_normalization(fname, **kwargs):
            xds = xr.open_zarr(fname, **kwargs)
            myvars = list(x for x in self.all_variables if x in xds)
            xds = xds[myvars]
            xds = xds.load()
            xds = xds.rename({"pfull": "level"})
            return xds

        mean_by_level = open_normalization("../zstores/ufs-hr1/1.00-degree/normalization/mean_by_level.zarr/")
        stddev_by_level = open_normalization("../zstores/ufs-hr1/1.00-degree/normalization/stddev_by_level.zarr/")
        diffs_stddev_by_level = open_normalization("../zstores/ufs-hr1/1.00-degree/normalization/diffs_stddev_by_level.zarr/")

        # hacky, just copying these from graphcast demo to get moving
        mean_by_level['year_progress'] = 0.49975101137533784
        mean_by_level['year_progress_sin'] = -0.0019232822626236157
        mean_by_level['year_progress_cos'] = 0.01172127404282719
        mean_by_level['day_progress'] = 0.49861110098039113
        mean_by_level['day_progress_sin'] = -1.0231613285011715e-08
        mean_by_level['day_progress_cos'] = 2.679492657383283e-08

        stddev_by_level['year_progress'] = 0.29067483157079654
        stddev_by_level['year_progress_sin'] = 0.7085840482846367
        stddev_by_level['year_progress_cos'] = 0.7055264413169846
        stddev_by_level['day_progress'] = 0.28867401335991755
        stddev_by_level['day_progress_sin'] = 0.7071067811865475
        stddev_by_level['day_progress_cos'] = 0.7071067888988349

        diffs_stddev_by_level['year_progress'] = 0.024697753562180874
        diffs_stddev_by_level['year_progress_sin'] = 0.0030342521761048467
        diffs_stddev_by_level['year_progress_cos'] = 0.0030474038590028816
        diffs_stddev_by_level['day_progress'] = 0.4330127018922193
        diffs_stddev_by_level['day_progress_sin'] = 0.9999999974440369
        diffs_stddev_by_level['day_progress_cos'] = 1.0

        return mean_by_level, stddev_by_level, diffs_stddev_by_level


    def _tree_flatten(self):
        """Pack up everything needed to remake this object.
        Since this class is static, we don't really need anything now, but that will change if we
        set the class attributes with a yaml file.
        In that case the yaml filename will needto be added to the aux_data bit

        See `here <https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree>`_
        for reference.
        """
        children = tuple()
        aux_data = dict() # in the future could be {"config_filename": self.config_filename}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

tree_util.register_pytree_node(
    ReplayEmulator1Degree,
    ReplayEmulator1Degree._tree_flatten,
    ReplayEmulator1Degree._tree_unflatten,
)
