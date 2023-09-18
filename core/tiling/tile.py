class Tile():

    bounds = {}
    def __init__(self, id, coordinates, 
                 centroid_coordinates, centroid_series, offset):
        self.id = id
        self.start_coord, self.end_coord = coordinates
        self.centroid_coordinates = centroid_coordinates
        self.centroid_series = centroid_series
        self.offset = offset

    def get_start_relative_coordinate(self):
        return self.start_coord

    def get_start_abs_coordinate(self):
        abs_start = [sum(x) for x in zip(self.start_coord, self.offset)]
        return abs_start

    def get_end_relative_coordinate(self):
        return self.end_coord

    def get_end_abs_coordinate(self):
        abs_end = [sum(x) for x in zip(self.end_coord, self.offset)]
        return abs_end


    def get_centroid_coordinates(self):
        return self.centroid_coordinates

    def get_centroid_series(self):
        return self.centroid_series