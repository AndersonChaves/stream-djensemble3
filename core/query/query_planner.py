import numpy as np
import core.utils as ut
from core.tiling.tiling import Tiling
from core.tiling.tile import Tile
from core.ensemble import Ensemble
from core.models.models_manager import ModelsManager

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class QueryPlanner():
    def __init__(self, config):
        self.config = config

    def define_ensemble(self, tiling: Tiling, 
                        candidate_models, data_window: np.ndarray) -> Ensemble:

        error_estimative = self.get_error_estimative(
            tiling, data_window, candidate_models)
        ensemble = self.get_lower_cost_combination(error_estimative)
        return ensemble

    def get_error_estimative(self, tiling: Tiling, 
                             data_window, candidate_models):
        error_estimative = {}
        for tile in tiling.tiles:
            logger.debug("----Estimating error for tile "+ str(tile.id))                        
            error_estimative[tile] = self.rank_models_for_tile(
                  data_window, tile, candidate_models)
        return error_estimative

    def rank_models_for_tile(self, dataset,
                                    tile: Tile, candidate_models):
        error_estimative = {}
        data_from_tile_region = self.get_data_from_tile(dataset, tile)
        for learner in candidate_models:
            error_est = learner.execute_eef(data_from_tile_region, tile)
            if error_est < 0:
                raise(Exception("Error - Cef Value is negative"))
            error_estimative[learner] = error_est
        return error_estimative


    def get_lower_cost_combination(self, error_estimative):
        ensemble = Ensemble()
        for tile_id in error_estimative.keys():
            best_model, best_error = None, float('inf')
            for model, error in error_estimative[tile_id].items():
                if error < best_error:
                    best_model = model
                    best_error = error                    
            ensemble.add_item(tile_id, best_model, best_error)
        return ensemble    

    def get_candidate_models_list(self):
        self.temporal_models_path = self.config["temporal_models_path"]
        self.convolutional_models_path = self.config["convolutional_models_path"]
        temporal_models_names = ut.get_names_of_models_in_dir(self.temporal_models_path)
        convolutional_models_names = ut.get_names_of_models_in_dir(self.convolutional_models_path)
        return temporal_models_names + convolutional_models_names    

    def get_data_from_tile(self, dataset: np.array, tile):
        sx, sy = tile.get_start_coordinate()
        ex, ey = tile.get_end_coordinate()
        return dataset[:, sx:ex+1, sy:ey+1]