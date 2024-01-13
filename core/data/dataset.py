import h5py

class Dataset():
    source = ''
    def __init__(self, source: str, dimensions: dict, title: str):
        #Dimensions: Dict {"time":(start, end)}
        self.source = source
        self.dimensions = dimensions
        self.title = title

    def get_data(self):
        with h5py.File(self.source) as f:
            time = self.dimensions["time"]
            lat  = self.dimensions["lat"]
            long = self.dimensions["long"]
            return f['data'][...][range(time[0], time[1]+1),
                                   range(lat[0], lat[1]+1),
                                    range(long[0], long[1] + 1)]
