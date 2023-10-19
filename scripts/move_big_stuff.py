
import os
import numpy as np
import xarray as xr
import dask.array as darray
from zarr import NestedDirectoryStore
from datetime import datetime, timedelta
import pandas as pd
from shutil import rmtree

from UFS2ARCO import FV3Dataset
from UFS2ARCO.replay import cached_replay_path_1degree as replay_path

from timer import Timer

def get_full_cycles():

    cycles = pd.date_range(start="1994-01-01", end="1999-06-13T06:00:00", freq="6h")
    return xr.DataArray(cycles, coords={"cycles": cycles}, coords="cycles")

def cycles2time(cycles):
    """assume cycles are ordered"""
    end = cycles[-1] + timedelta(hours=3)
    time = pd.date_range(start=cycles[0], end=end, freq="3h")
    iau_time = time - timedelta(hours=6)
    return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time")

def get_full_time():
    """Make the time vector, although only shape, chunks, and dtype matter, lets get values right"""

    # the range of time covered by replay dates
    time = pd.date_range(start="1994-01-01", end="1999-06-13T09:00:00", freq="3h")

    # now take into account IAU
    iau_time = time - timedelta(hours=6)

    # chunk it, 1 chunk per time
    dummy = darray.from_array(iau_time, chunks=1)
    xtime = xr.DataArray(
            dummy,
            coords={"time": dummy},
            dims="time",
            attrs={
                "long_name": "time",
                "axis": "T",
                },
            )
    return xtime

def remove_time(xds):
    single = xds.isel(time=0)
    for key in xds.data_vars:
        if "time" in key:
            del single[key]

    for key in xds.coords:
        if "time" in key:
            del single[key]
    return single


def make_dummy_dataset():

    localtime = Timer()

    localtime.start("Reading single dataset")
    replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")
    date = datetime(year=1994, month=1, day=1, hour=0)
    xds = replay.open_dataset(date, fsspec_kwargs={"s3":{"anon": True}}, engine="h5netcdf")
    xds = xds.reset_coords()
    localtime.stop()

    localtime.start("details")
    # Just drop cftime and ftime for now, remove vars we don't care about
    xds = xds.drop(["ftime", "cftime"])
    data_vars = [x for x in replay.data_vars if x in xds]
    xds = xds[data_vars]

    # get a no-time dataset
    single = remove_time(xds)

    # make an empty container, copy all non-time coordinates
    dds = xr.Dataset()
    for key in single.coords:
        dds[key] = xds[key]
    localtime.stop()

    # now make empty arrays for each data variable, using the full time vector
    localtime.start("get full time")
    xtime = get_full_time()
    dds["time"] = xtime
    localtime.stop()

    localtime.start("making big container")
    for key in single.data_vars:

        dims = ("time",) + single[key].dims
        chunks = tuple(replay.chunks_out[k] for k in dims)
        dds[key] = xr.DataArray(
                data=darray.zeros(
                    shape=(len(xtime),)+single[key].shape,
                    chunks=chunks,
                    dtype=single[key].dtype,
                    ),
                coords={"time": xtime, **{d: single[d] for d in single[key].dims}},
                dims=dims,
                attrs=single[key].attrs.copy(),
                )
        print(f"\t ... done with {key}")

    localtime.stop()

    # TODO: now make containers for cftime and ftime
    localtime.start("Storing")
    store = NestedDirectoryStore(path=replay.forecast_path)
    dds.to_zarr(store, compute=False)
    localtime.stop()
    return

def store_a_slice(my_id):

    # this should be a class attribute or something
    n_nodes = 5

    xcycles = get_full_cycles()
    xtime = get_full_time()
    splits = np.linspace(0, len(xcycles), n_nodes)
    cycle_slices = [slice(int(st), int(ed)) for st,ed in zip(splits[:-1], splits[1:])]
    my_cycles = xcycles.isel(cycles=cycle_slices[my_id])
    # this isn't necessary
    #my_time = cycles2time(my_cycles)


    replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")

    for t in my_cycles:
        date = datetime(year=int(t.dt.year), month=int(t.dt.month), day=int(t.dt.day), hour=int(t.dt.hour))
        ds = replay.open_dataset(date, fsspec_kwargs={"s3":{"anon": True}}, engine="h5netcdf")

        #TODO: figure out the region within xtime based on the values in ds.time
        replay.store_dataset(ds, region={"time": tslice})



if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Initializing job")

    localtime.start("Make Dummy Dataset")
    make_dummy_dataset()
    localtime.stop()

    # determine how to split up the time vector based on number of nodes used
    xtime = get_full_time()
    n_nodes = 5
    splits = np.linspace(0, len(xtime), n_nodes+1)
    slices = [slice(int(st), int(ed)) for st,ed in zip(splits[:-1], splits[1:])]




    #replay = FV3Dataset(path_in=replay_path, config_filename="config-replay.yaml")
    #kw = {"mode": "w"}
    #year = 1994
    #month = 2
    #for day in range(28,30):
    #    for hour in [0, 6, 12, 18]:

    #        try:
    #            date = datetime(year=year, month=month, day=day, hour=hour)
    #            valid_time = True

    #        except ValueError:

    #            # this should only happen when we ask for an invalid date, like Feb 30
    #            valid_time = False
    #            print(f" ... skipping year={year}, month={month}, day={day}, hour={hour}")

    #        if valid_time:
    #            localtime.start(f"Pulling {str(date)}")
    #            ds = replay.open_dataset(date, fsspec_kwargs={"s3":{"anon": True}}, engine="h5netcdf")
    #            replay.store_dataset(ds, **kw)
    #            kw = {"mode": "a", "append_dim":"time"}
    #            localtime.stop()

    walltime.stop("Total Walltime")
