import numpy as np
import h5py
import xarray as xr

class IAccessor():
    def read_instant(self, ds, t_instant, region: tuple=None):
        raise NotImplementedError

    def read_window(self, ds, t_instant, window_size):
        raise NotImplementedError

    def get_spatial_shape(self, ds):
        raise NotImplementedError

class AccessorNumpy(IAccessor):
    def read_instant(self, ds, t_instant, region: tuple=None):
        if region == None:
            return ds[t_instant]
        else:
            x, y = region
            return ds[t_instant, x, y]

    def read_window(self, ds, t_instant, window_size):
        return ds[t_instant:t_instant+window_size]

    def filter_by_region(self, ds, x, y):
        return ds[:, x[0]:x[1], y[0]:y[1]]

    def get_spatial_shape(self, ds):
        return ds.shape[1:]

class AccessorNetcdf(IAccessor):
    def read_instant(self, ds, t_instant):
        return ds[t_instant]

    def get_spatial_shape(self, ds):
        return ds.shape[1:]

class AccessorXArray(IAccessor):
    def read_instant(self, ds, t_instant):
        return ds[t_instant]
