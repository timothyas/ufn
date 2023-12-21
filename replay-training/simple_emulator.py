"""
Create and train a "simple" emulator:
- 1 degree resolution
- assumes 6 hour timestep (see the temporal slicing command in .preprocess method, skipping by 2)
- only propagate a few variables
- only maintain a few vertical levels
- keep the hidden dimension/network size small

Purpose is to have the training pipeline working.
"""

import warnings
import dataclasses
import pandas as pd
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


    def preprocess(self, xds, batch_index=0):
        """Prepare a single batch for GraphCast

        Args:
            xds (xarray.Dataset): with replay data
            batch_index (int, optional): the index of this batch

        Returns:
            bds (xarray.Dataset): this batch of data
        """

        # select our vertical levels
        bds = xds.sel(pfull=list(self.pressure_levels), method="nearest")

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


    def get_training_batches(self,
        xds,
        n_batches,
        batch_size,
        delta_t,
        target_lead_time="6h"
        ):
        """Get a dataset with all the batches of data necessary for training

        Note:
            Here we're using target_lead_time as a single value, see graphcast.data_utils.extract ... where it could be multi valued but here I'm using it to compute the total time per batch so it's more useful this way

        Note:
            It's really unclear how the graphcast.data_utils.extract... function used in this method creates samples/batches... it's also unclear how optax expects the data in order to do minibatches.

        Args:
            xds (xarray.Dataset): the Replay dataset
            n_batches (int): number of training batches to grab
            batch_size (int): number of samples viewed per mini batch
            delta_t (Timedeltalike): timestep of the desired emulator, e.g. "3h" or "6h"
            target_lead_times (str or slice, optional): the lead time to use in the cost function, see graphcast.data_utils.extract_input_target_lead_times

        Returns:
            inputs, targets, forcings (xarray.Dataset): with new dimension "batch"
                and appropriate fields for each dataset, based on the variables in :attr:`task_config`

        Example:

            Create training 100 batches, each batch has a single sample,
            where each sample is made up of error from a 12h forecast,
            where the emulator operates on 6 hour timesteps


            >>> gufs = ReplayEmulator1Degree()
            >>> xds = #... replay data
            >>> inputs, targets, forcings = gufs.get_training_batches(
                    xds=xds,
                    n_batches=100,
                    batch_size=1,
                    delta_t="6h",
                    target_lead_time="12h",
                )
        """

        inputs = []
        targets = []
        forcings = []

        if batch_size > 1:
            warnings.warn("it's not clear how the batch/sample time slices are defined in graphcast or how they are used by optax")

        delta_t = pd.Timedelta(delta_t)
        input_duration = pd.Timedelta(self.input_duration)

        time_per_sample = target_lead_time + input_duration
        time_per_batch = batch_size * time_per_sample

        # create a new time vector with desired delta_t
        new_time = pd.date_range(
            start=xds["time"].isel(time=0).values,
            end=xds["time"].isel(time=-1).values,
            freq=delta_t,
            inclusive="both",
        )
        batch_initial_times = pd.date_range(
            start=new_time[0],
            end=new_time[-1],
            freq=time_per_batch,
            inclusive="both",
        )
        if n_batches > len(batch_initial_times)-1:
            n_batches = len(batch_initial_times)-1
            warnings.warn(f"There's less data than the number of batches requested, reducing n_batches to {n_batches}")

        for i in range(n_batches):

            timestamps_in_this_batch = pd.date_range(
                start=batch_initial_times[i],
                end=batch_initial_times[i+1],
                freq=delta_t,
                inclusive="both",
            )

            batch = self.preprocess(
                xds.sel(time=timestamps_in_this_batch),
                batch_index=i,
            )

            i, t, f = extract_inputs_targets_forcings(
                batch,
                target_lead_times=target_lead_time,
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
