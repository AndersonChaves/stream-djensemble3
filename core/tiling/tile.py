class Tile():

    bounds = {}
    def __init__(self, id, coordinates, 
                 centroid_coordinates, centroid_series):
        self.id = id
        self.start_coord, self.end_coord = coordinates
        self.centroid_coordinates = centroid_coordinates
        self.centroid_series = centroid_series

    def get_start_coordinate(self):
        return self.start_coord

    def get_end_coordinate(self):
        return self.end_coord

    def get_centroid_coordinates(self):
        return self.centroid_coordinates

    def get_centroid_series(self):
        return self.centroid_series