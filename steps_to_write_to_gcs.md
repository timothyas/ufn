# What it took to write directly to Google Cloud Storage

After testing the whole pipeline locally, I wanted to use python to write
directly to the cloud bucket.
This involved:

1. Creating a Service Account (SA) which is an account for the computer basically, to
   access the bucket. Even though I was given write permissions, I did not have
   permissions to create the SA, so I had to ask the admin.
   Go
[here](https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-console)
and also [here is a nice overview of service
accounts](https://cloud.google.com/iam/docs/service-account-overview)

2. Someone with NODD ended up creating an SA for me, and sent me the json file
   that describes it over kiteworks. I do not see SA anywhere on my own account.
3. Verify read access with:

Example using Google python API
```python
from google.cloud import storage

storage_client = storage.Client.from_service_account_json("/path/to/sa-file.json")
list(storage_client.list_blobs("noaa-ufs-gefsv13replay"))
# returns [] because nothing is there
```

Example using gcsfs, more relevant for xarray
```python
import gcsfs
fs = gcsfs.GCSFileSystem(token="/path/to/sa-file.json")
fs.ls("noaa-ufs-gefsv13replay")
# returns []
```

4. Verify write access:

```python
import xarray as xr
ds = xr.open_zarr("/path/to/existing/test/dataset.zarr")
path_out = "ufs-hr1/1.00-degree/03h-freq/test-zarr/fv3.zarr")
ds.to_zarr(
    f"gcs://noaa-ufs-gefsv13replay/{path_out}",
    storage_options={
        "token": "/path/to/sa-file.json",
    },
)
```

Done.
