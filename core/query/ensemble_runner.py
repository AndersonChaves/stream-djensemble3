import numpy as np
from core.tiling.tile import Tile
from core.ensemble import Ensemble
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnsembleRunner():
    def __init__(self, config):
        self.config = config
        self.query_config = self.config["query"]
    
    def run(self, ensemble: Ensemble, 
                         input_dataset: np.array):
        ensemble.run(input_dataset)
        self.prediction = self._compose_prediction(ensemble, input_dataset)        

    def get_prediction(self):
        return self.prediction

    def get_query_start(self):
        start = \
            int(self.query_config["start"][0] / self.config["data_source"]["compacting_factor"]),\
            int(self.query_config["start"][1] / self.config["data_source"]["compacting_factor"])
        return start
    
    def get_query_end(self):
        end = \
            int(self.query_config["end"][0] / self.config["data_source"]["compacting_factor"]),\
            int(self.query_config["end"][1] / self.config["data_source"]["compacting_factor"])

        return end

    def _compose_prediction(self, ensemble: Ensemble, 
                           input_dataset: np.ndarray) -> np.ndarray:
        prediction_length = 1
        start = self.get_query_start()
        end = self.get_query_end()
        query_size_lat = end[0] - start[0]
        query_size_lon = end[1] - start[1]
        query_predicted_series = np.zeros((prediction_length, query_size_lat, query_size_lon))

        for tile in ensemble.get_tiles():        
            self._compose_predicted_frame(query_predicted_series, 
                (start, end), tile, ensemble.get_tile_predicted_frame(tile))
        return query_predicted_series

    def _compose_predicted_frame(self, resulting_array: np.ndarray, 
                                query_endpoints: tuple[tuple], 
                                tile: Tile,
                                tile_prediction: np.array):

        tile_lat = [tile.get_start_coordinate()[0], tile.get_end_coordinate()[0]]
        tile_long = [tile.get_start_coordinate()[1], tile.get_end_coordinate()[1]]

        query_lat  = query_endpoints[0][0], query_endpoints[1][0]-1#query endpoints are always open interval, so -1
        query_long = query_endpoints[0][1], query_endpoints[1][1]-1 # If change this must change result declaration

        # Tile coordinates are always absolute coordinates with respect to the input
        #if self.config["query"]["cluster_query_window"]:
        #    tile_lat[0]  = tile_lat[0] + query_lat[0]
        #    tile_lat[1]  = tile_lat[1] + query_lat[0]
        #    tile_long[0] = tile_long[0] + query_long[0]
        #    tile_long[1] = tile_long[1] + query_long[0]


        # Get from prediction, area corresponding to query
        ##1. Get intersecting coordinates relative to tile
        intersection_lat = max(tile_lat[0] , query_lat[0]), min(tile_lat[1] , query_lat[1])
        intersection_long = max(tile_long[0], query_long[0]), min(tile_long[1], query_long[1])

        ##2. Get data corresponding to coordinates found
        i_lat = intersection_lat[0] - tile_lat[0], intersection_lat[1] - tile_lat[0]
        i_lon = intersection_long[0] - tile_long[0], intersection_long[1] - tile_long[0]

        #print("DEBUG Compose: shape of tile_prediction is ", tile_prediction.shape)
        if len(tile_prediction.shape) == 3:
            intersecting_data = tile_prediction[-1, i_lat[0]:i_lat[1] + 1, i_lon[0]:i_lon[1] + 1]
        else:
            intersecting_data = tile_prediction[i_lat[0]:i_lat[1]+1, i_lon[0]:i_lon[1]+1]

        ##3. Get intersecting coordinates corresponding to query
        lat = intersection_lat[0] - query_lat[0], intersection_lat[1] - query_lat[0]
        lon = intersection_long[0] - query_long[0], intersection_long[1] - query_long[0]

        ##5. Attribute data to query array in corresponding coordinates
        resulting_array[:, lat[0]:lat[1]+1, lon[0]:lon[1]+1] = intersecting_data