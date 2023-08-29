import numpy as np
import math
import os.path
import netCDF4 as nc
from core.data.accessor import *

class DatasetManager:
    def __init__(self, json_config):
        self.config = json_config

    def loadDataset(self, load_range):        
        self.data_path = self.config["dataset"]
        self.start, self.end = self.config["time_range"]        
        subregion = self.config["subregion"]
        self.lat_min, self.lat_max = subregion["lat_min"], subregion["lat_max"]
        self.lon_min, self.lon_max = subregion["lon_min"], subregion["lon_max"]
        
        data_path = self.data_path
        if not os.path.isfile(data_path):
            raise Exception("Data Path " + data_path + " is empty.")

        elif data_path[-4:] == '.npy':
            self.ds = np.load(data_path, mmap_mode='c')
            stride = self.config["compacting_factor"]
            self.ds = self.ds[:, self.lat_min:self.lat_max:stride, 
                self.lon_min: self.lon_max:stride]
            self.accessor = AccessorNumpy()
        elif data_path[-3:] == '.nc':
            self.load_netcdf(load_range)            
        else:
            self.ds = xr.load_dataset(data_path).sortby('time')
            self.accessor = AccessorXArray()

    def load_netcdf(self, load_range):
        start, end = load_range
        self.ds = nc.Dataset(self.data_path)
        ds_attribute = self.config["target_attribute"]
        stride = self.config["compacting_factor"]        
        self.ds = np.array(self.ds[ds_attribute][
            start:end, 
            self.lat_min:self.lat_max:stride, 
            self.lon_min: self.lon_max:stride
        ])
        #self.ds = self.ds[self.start:self.end]
        self.accessor = AccessorNumpy()

    def read_instant(self, t_instant):
        return self.accessor.read_instant(self.ds, t_instant)

    def read_window(self, t_instant, window_size):
        return self.accessor.read_window(self.ds, t_instant, window_size)

    def read_all_data(self):
        return self.ds

    def filter_by_region(self, x, y):
        self.ds = self.accessor.filter_by_region(self.ds, x, y)

    def filter_by_date(self, ds, lower_date, upper_date):
        filtered_dataset = ds.sel(time=slice(lower_date, upper_date), drop=True)
        filtered_dataset['rain'] = filtered_dataset['rain'].fillna(0)
        return filtered_dataset

    def loadTemperatureDataset(self, dataPath):
        with h5py.File(dataPath) as f:
            dataset = f['real'][...]
        return dataset

    def get_data_from_tile(self, dataset: np.array, tile):
        sx, sy = tile.get_start_coordinate()
        ex, ey = tile.get_end_coordinate()
        return dataset[:, sx:ex+1, sy:ey+1]

    def get_spatial_shape(self):
        return self.accessor.get_spatial_shape(self.ds)

    def filter_frame_by_query_region(self, dataset, x1, x2):
        data_window = dataset[x1[0]:x2[0], x1[1]:x2[1]]
        return data_window