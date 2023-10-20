from os.path import join
from datetime import datetime, timedelta
import yaml

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darray
from zarr import NestedDirectoryStore

from UFS2ARCO import FV3Dataset

from timer import Timer

class ReplayMover1Degree():
    """

    Note:
        Currently this makes the unnecessary but easy-to-implement assumption that we want forecast_hours 0 & 3.
        This assumption is key to the hard coded end date and timedelta used to make :attr:`xtime`.
        It should also be confirmed that this assumption does not impact the "ftime" variable, that is right now
        being created in the container dataset. The values should be overwritten when the data are actually
        generated, but who knows.
    """


    n_jobs = None

    forecast_hours = None
    file_prefixes = None

    @property
    def xcycles(self):
        cycles = pd.date_range(start="1994-01-01", end="1999-06-13T06:00:00", freq="6h")
        return xr.DataArray(cycles, coords={"cycles": cycles}, dims="cycles")


    @property
    def xtime(self):
        time = pd.date_range(start="1994-01-01", end="1999-06-13T09:00:00", freq="3h")
        iau_time = time - timedelta(hours=6)
        return xr.DataArray(iau_time, coords={"time": iau_time}, dims="time", attrs={"long_name": "time", "axis": "T"})


    @property
    def splits(self):
        """The indices used to split all cycles across :attr:`n_jobs`"""
        return [int(x) for x in np.linspace(0, len(self.xcycles), self.n_jobs+1)]

    @property
    def ods_kwargs(self):
        return {"fsspec_kwargs": {"s3": {"anon": True}}, "engine":"h5netcdf"}


    def my_cycles(self, job_id):
        slices = [slice(st, ed) for st, ed in zip(self.splits[:-1], self.splits[1:])]
        return self.xcycles.isel(cycles=slices[job_id])


    def __init__(self, n_jobs, config_filename, component="fv3"):
        self.n_jobs = n_jobs
        self.config_filename = config_filename

        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)

        name = f"{component.upper()}Dataset" # i.e., FV3Dataset ... maybe an unnecessary generalization at this stage
        self.forecast_hours = config[name]["forecast_hours"]
        self.file_prefixes = config[name]["file_prefixes"]

        assert tuple(self.forecast_hours) == (0, 3)


    def run(self, job_id):
        """Make this essentially a function that can run completely independently of other objects"""

        replay = FV3Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        for cycle in self.my_cycles(job_id):

            cycle_date = self.npdate2datetime(cycle)
            xds = replay.open_dataset(cycle_date, **self.ods_kwargs)

            indices = np.array([list(self.xtime.values).index(t) for t in xds.time.values])
            tslice = slice(indices.min(), indices.max()+1)

            replay.store_dataset(
                    xds,
                    region={
                        "time": tslice,
                        "pfull": slice(None, None),
                        "grid_yt": slice(None, None),
                        "grid_xt": slice(None, None),
                        },
                    )


    def store_container(self):
        """Create an empty container that has the write shape, chunks, and dtype for each variable"""

        localtime = Timer()

        replay = FV3Dataset(path_in=self.cached_path, config_filename=self.config_filename)

        localtime.start("Reading Single Dataset")
        cycle = self.npdate2datetime(self.xcycles[0])
        xds = replay.open_dataset(cycle, **self.ods_kwargs)
        xds = xds.reset_coords()
        localtime.stop()

        xds = xds.drop(["ftime", "cftime"])
        data_vars = [x for x in replay.data_vars if x in xds]
        xds = xds[data_vars]

        # Make a container, starting with coordinates
        single = self.remove_time(xds)
        dds = xr.Dataset()
        for key in single.coords:
            dds[key] = xds[key]

        localtime.start("Making container for the dataset")
        dds["time"] = self.xtime
        dds = self.add_time_coords(dds, replay._time2cftime)
        for key in single.data_vars:

            dims = ("time",) + single[key].dims
            chunks = tuple(replay.chunks_out[k] for k in dims)
            shape = (len(dds["time"]),) + single[key].shape

            dds[key] = xr.DataArray(
                    data=darray.zeros(
                        shape=shape,
                        chunks=chunks,
                        dtype=single[key].dtype,
                        ),
                    coords={"time": dds["time"], **{d: single[d] for d in single[key].dims}},
                    dims=dims,
                    attrs=single[key].attrs.copy(),
                )
            print(f"\t ... done with {key}")

        localtime.stop()

        localtime.start("Storing to zarr")
        store = NestedDirectoryStore(path=replay.forecast_path)
        dds.to_zarr(store, compute=False)
        localtime.stop()

    @staticmethod
    def path(date, forecast_hours, file_prefixes):
        """Construct path to 1 degree replay data

        Args:
            date (datetime): with the DA cycle to grab

        Returns:
            paths (list of str): with paths to s3 buckets
        """

        upper = "s3://noaa-ufs-gefsv13replay-pds/1deg"
        this_dir = f"{date.year:04d}/{date.month:02d}/{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}"
        files = []
        for fp in file_prefixes:
            for fhr in forecast_hours:
                files.append(
                        f"{fp}{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}_fhr{fhr:02d}_control")
        return [join(upper, this_dir, this_file) for this_file in files]


    @staticmethod
    def cached_path(date, forecast_hours, file_prefixes):
        """there has to be a better way to do this"""
        upper = "simplecache::s3://noaa-ufs-gefsv13replay-pds/1deg"
        this_dir = f"{date.year:04d}/{date.month:02d}/{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}"
        files = []
        for fp in file_prefixes:
            for fhr in forecast_hours:
                files.append(
                        f"{fp}{date.year:04d}{date.month:02d}{date.day:02d}{date.hour:02d}_fhr{fhr:02d}_control")
        return [join(upper, this_dir, this_file) for this_file in files]


    @staticmethod
    def npdate2datetime(npdate):
        return datetime(
                    year=int(npdate.dt.year),
                    month=int(npdate.dt.month),
                    day=int(npdate.dt.day),
                    hour=int(npdate.dt.hour),
                )


    @staticmethod
    def remove_time(xds):
        single = xds.isel(time=0)
        for key in xds.data_vars:
            if "time" in key:
                del single[key]

        for key in xds.coords:
            if "time" in key:
                del single[key]
        return single

    @staticmethod
    def add_time_coords(xds, time2cftime):
        """add ftime and cftime to container

        This is a bit dirty, passing a static method from another class as a function arg... so it goes.
        """
        ftime = np.array(
                [
                    (np.timedelta64(timedelta(hours=-6)), np.timedelta64(timedelta(hours=-3)))
                        for _ in range(len(xds["time"])//2)
                    ]
                ).flatten()

        xds["ftime"] = xr.DataArray(
                ftime,
                coords=xds["time"].coords,
                dims=xds["time"].dims,
                attrs={
                    "axis": "T",
                    "description": "time passed since forecast initialization",
                    "long_name": "forecast_time",
                    },
                )
        xds["cftime"] = xr.DataArray(
                time2cftime(xds["time"]),
                coords=xds["time"].coords,
                dims=xds["time"].dims,
                attrs={
                    "calendar_type": "JULIAN",
                    "cartesian_axis": "T",
                    "long_name": "time",
                    },
                )
        xds = xds.set_coords(["ftime", "cftime"])
        return xds

